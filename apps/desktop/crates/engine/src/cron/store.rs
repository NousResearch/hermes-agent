use super::*;
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Duration, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tempfile::NamedTempFile;

const MAX_JOBS: usize = 1000;
const MAX_RUNS: usize = 5000;
const MAX_BYTES: u64 = 32 * 1024 * 1024;
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
    profile: PathBuf,
    owner: Arc<Owner>,
}
#[derive(Debug)]
struct Owner {
    id: String,
    _lock: Option<File>,
}
#[derive(Serialize, Deserialize, Default)]
struct Document {
    #[serde(default)]
    jobs: Vec<Job>,
    #[serde(default)]
    runs: Vec<Run>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}
static NEXT: AtomicU64 = AtomicU64::new(1);
fn id() -> String {
    format!(
        "{:016x}{:08x}{:08x}",
        Utc::now().timestamp_nanos_opt().unwrap_or_default() as u64,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
fn private_dir(path: &Path) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        ensure!(
            !metadata.file_type().is_symlink() && metadata.is_dir(),
            "Scheduler directory is not a regular directory"
        );
    }
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
fn lock_file(path: &Path) -> Result<File> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        ensure!(
            !metadata.file_type().is_symlink() && metadata.is_file(),
            "Scheduler lock must be a regular file"
        );
    }
    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}
impl Store {
    pub fn open(profile_root: &Path) -> Result<Self> {
        let profile = profile_root
            .canonicalize()
            .context("Could not open profile directory")?;
        let root = profile.join("cron");
        private_dir(&root)?;
        private_dir(&root.join("runs"))?;
        let lock = lock_file(&root.join(".runner.lock"))?;
        let owns = lock.try_lock_exclusive().is_ok();
        let store = Self {
            root,
            profile,
            owner: Arc::new(Owner {
                id: id(),
                _lock: owns.then_some(lock),
            }),
        };
        if owns {
            store.transaction(|document| {
                recover(document, Utc::now());
                Ok(())
            })?;
        }
        Ok(store)
    }
    pub fn profile_root(&self) -> &Path {
        &self.profile
    }
    pub fn owns_scheduler(&self) -> bool {
        self.owner._lock.is_some()
    }
    fn transaction<T>(&self, operation: impl FnOnce(&mut Document) -> Result<T>) -> Result<T> {
        let lock = lock_file(&self.root.join(".jobs.lock"))?;
        lock.lock_exclusive()?;
        let mut document = self.read()?;
        let value = operation(&mut document)?;
        ensure!(
            document.jobs.len() <= MAX_JOBS && document.runs.len() <= MAX_RUNS + 32,
            "Scheduler store exceeds its limit"
        );
        let bytes = serde_json::to_vec_pretty(&document)?;
        ensure!(
            bytes.len() as u64 <= MAX_BYTES,
            "Scheduled jobs store is full; remove older run records first"
        );
        let mut temp = NamedTempFile::new_in(&self.root)?;
        temp.write_all(&bytes)?;
        temp.write_all(b"\n")?;
        temp.as_file().sync_all()?;
        temp.persist(self.root.join("jobs.json"))
            .context("Could not save scheduled jobs")?;
        Ok(value)
    }
    fn read(&self) -> Result<Document> {
        let path = self.root.join("jobs.json");
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Document::default());
            }
            Err(error) => return Err(error.into()),
        };
        ensure!(
            !fs::symlink_metadata(&path)?.file_type().is_symlink() && file.metadata()?.is_file(),
            "Scheduled jobs file must not be a symbolic link"
        );
        ensure!(
            file.metadata()?.len() <= MAX_BYTES,
            "Scheduled jobs file is too large"
        );
        let mut bytes = Vec::new();
        file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= MAX_BYTES,
            "Scheduled jobs file is too large"
        );
        serde_json::from_slice(&bytes)
            .context("Could not read scheduled jobs; the existing file was left untouched")
    }
    pub fn jobs(&self) -> Result<Vec<Job>> {
        let document = self.read()?;
        let mut jobs = document.jobs;
        for job in &mut jobs {
            if document
                .runs
                .iter()
                .any(|run| run.job_id == job.id && run.status.live())
            {
                job.state = "running".into();
            }
        }
        Ok(jobs)
    }
    pub fn runs(&self, job_id: &str) -> Result<Vec<Run>> {
        let mut runs: Vec<_> = self
            .read()?
            .runs
            .into_iter()
            .filter(|run| run.job_id == job_id)
            .collect();
        runs.reverse();
        Ok(runs)
    }
    pub fn create(&self, draft: Draft, now: DateTime<Utc>) -> Result<Job> {
        let job = build(draft, now, None)?;
        self.transaction(|document| {
            ensure!(document.jobs.len() < MAX_JOBS, "Too many scheduled jobs");
            document.jobs.push(job.clone());
            Ok(job)
        })
    }
    pub fn update(&self, id: &str, draft: Draft, now: DateTime<Utc>) -> Result<Job> {
        self.transaction(|document| {
            let job = document
                .jobs
                .iter_mut()
                .find(|job| job.id == id)
                .context("Scheduled job not found")?;
            *job = build(draft, now, Some(job))?;
            Ok(job.clone())
        })
    }
    pub fn pause(&self, id: &str, now: DateTime<Utc>) -> Result<()> {
        self.transaction(|document| {
            let job = find(document, id)?;
            job.enabled = false;
            job.state = "paused".into();
            job.paused_at = Some(stamp(now));
            job.next_run_at = None;
            Ok(())
        })
    }
    pub fn resume(&self, id: &str, now: DateTime<Utc>) -> Result<()> {
        self.transaction(|document| {
            let job = find(document, id)?;
            ensure!(
                !exhausted(job),
                "Repeat limit reached; use Trigger now for an extra run"
            );
            let next = first(&job.schedule, now, job.timezone())?
                .context("This one-shot has completed; use Trigger now for an extra run")?;
            job.enabled = true;
            job.state = "scheduled".into();
            job.paused_at = None;
            job.next_run_at = Some(stamp(next));
            Ok(())
        })
    }
    pub fn remove(&self, id: &str) -> Result<()> {
        self.transaction(|document| {
            ensure!(
                document.jobs.iter().any(|job| job.id == id),
                "Scheduled job not found"
            );
            document.jobs.retain(|job| job.id != id);
            for run in &mut document.runs {
                if run.job_id == id && run.status.live() {
                    run.cancel_requested = true;
                }
            }
            Ok(())
        })
    }
    pub fn cancel(&self, run_id: &str) -> Result<()> {
        self.transaction(|document| {
            let run = document
                .runs
                .iter_mut()
                .find(|run| run.id == run_id)
                .context("Run not found")?;
            if run.status.live() {
                run.cancel_requested = true;
            }
            Ok(())
        })
    }
    pub fn cancellation_requested(&self, run_id: &str) -> Result<bool> {
        Ok(self
            .read()?
            .runs
            .iter()
            .find(|run| run.id == run_id)
            .is_none_or(|run| run.cancel_requested))
    }
    pub fn claim_due(&self, now: DateTime<Utc>) -> Result<Vec<Claim>> {
        if !self.owns_scheduler() {
            return Ok(Vec::new());
        }
        let snapshot = self.read()?;
        if !snapshot.jobs.iter().any(|job| {
            job.enabled
                && job
                    .next_run_at
                    .as_deref()
                    .and_then(|value| instant(value).ok())
                    .is_some_and(|at| at <= now)
        }) {
            return Ok(Vec::new());
        }
        self.transaction(|document| {
            let mut claims = Vec::new();
            let already_running = document.runs.iter().filter(|run| run.status.live()).count();
            for index in 0..document.jobs.len() {
                if claims.len() + already_running >= 2 {
                    break;
                }
                let job = &document.jobs[index];
                if !job.enabled
                    || job.paused_at.is_some()
                    || exhausted(job)
                    || document
                        .runs
                        .iter()
                        .any(|run| run.job_id == job.id && run.status.live())
                {
                    continue;
                }
                let Some(at) = job
                    .next_run_at
                    .as_deref()
                    .and_then(|value| instant(value).ok())
                else {
                    continue;
                };
                if at > now {
                    continue;
                }
                let lateness = (now - at).num_seconds();
                let grace = if job.grace_minutes == Some(0) {
                    30
                } else {
                    grace_seconds(job, at)?
                };
                if lateness > grace && (job.grace_minutes == Some(0) || job.catch_up != Some(true))
                {
                    let job = &mut document.jobs[index];
                    job.last_status = Some("skipped".into());
                    job.last_error =
                        Some("Missed occurrence was outside the catch-up window".into());
                    job.next_run_at = job.schedule.next_after(now, job.timezone())?.map(stamp);
                    if job.schedule.is_once() {
                        job.enabled = false;
                        job.state = "completed".into();
                    }
                    continue;
                }
                let claim = claim(document, index, false, now, &self.owner.id, &self.root)?;
                claims.push(claim);
            }
            prune(document);
            Ok(claims)
        })
    }
    pub fn claim_now(&self, id: &str, now: DateTime<Utc>) -> Result<Claim> {
        ensure!(
            self.owns_scheduler(),
            "Another window owns this profile's scheduler"
        );
        self.transaction(|document| {
            let index = document
                .jobs
                .iter()
                .position(|job| job.id == id)
                .context("Scheduled job not found")?;
            ensure!(
                !document
                    .runs
                    .iter()
                    .any(|run| run.job_id == id && run.status.live()),
                "This job already has a running occurrence"
            );
            let claim = claim(document, index, true, now, &self.owner.id, &self.root)?;
            prune(document);
            Ok(claim)
        })
    }
    pub fn begin(&self, claim: &Claim, now: DateTime<Utc>) -> Result<()> {
        self.transaction(|document| {
            let run = document
                .runs
                .iter_mut()
                .find(|run| run.id == claim.run.id)
                .context("Run no longer exists")?;
            ensure!(
                run.owner == self.owner.id && run.status == RunStatus::Claimed,
                "Run claim is stale"
            );
            ensure!(!run.cancel_requested, "Run was cancelled before starting");
            run.status = RunStatus::Running;
            run.started_at = Some(stamp(now));
            Ok(())
        })
    }
    pub fn finish(&self, claim: &Claim, outcome: Outcome, now: DateTime<Utc>) -> Result<()> {
        self.transaction(|document| {
            let run = document
                .runs
                .iter_mut()
                .find(|run| run.id == claim.run.id)
                .context("Run no longer exists")?;
            ensure!(run.owner == self.owner.id, "Run owner changed");
            if !run.status.live() {
                return Ok(());
            }
            run.finished_at = Some(stamp(now));
            match outcome {
                Outcome::Completed { output } => {
                    run.status = RunStatus::Completed;
                    run.output = bounded(&output, 262144);
                    run.error = None;
                }
                Outcome::Failed { error } => {
                    run.status = RunStatus::Failed;
                    run.error = Some(bounded(&error, 4096));
                }
                Outcome::Cancelled { reason } => {
                    run.status = RunStatus::Cancelled;
                    run.error = Some(bounded(&reason, 4096));
                }
            }
            if let Some(job) = document.jobs.iter_mut().find(|job| job.id == run.job_id) {
                settle(job, run, now);
            }
            Ok(())
        })
    }
}
fn bounded(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}
fn find<'a>(document: &'a mut Document, id: &str) -> Result<&'a mut Job> {
    document
        .jobs
        .iter_mut()
        .find(|job| job.id == id)
        .context("Scheduled job not found")
}
fn exhausted(job: &Job) -> bool {
    job.repeat
        .times
        .is_some_and(|times| times > 0 && job.repeat.completed >= times)
}
fn first(schedule: &Schedule, now: DateTime<Utc>, zone: &str) -> Result<Option<DateTime<Utc>>> {
    if let Schedule::Once { run_at, .. } = schedule {
        let at = instant(run_at)?;
        ensure!(
            (now - at).num_seconds() <= 120,
            "One-shot time is in the past"
        );
        Ok(Some(at))
    } else {
        schedule.next_after(now, zone)
    }
}
fn build(mut draft: Draft, now: DateTime<Utc>, previous: Option<&Job>) -> Result<Job> {
    ensure!(
        !draft.prompt.trim().is_empty() && draft.prompt.len() <= 65536,
        "Prompt is required (maximum 64 KiB)"
    );
    ensure!(draft.name.len() <= 240, "Name is too long");
    ensure!(
        draft.model.as_ref().is_none_or(|model| model.len() <= 256),
        "Model ID is too long"
    );
    ensure!(
        draft.deliver == "local",
        "This native scheduler currently delivers to This desktop"
    );
    ensure!(
        draft.grace_minutes.is_none_or(|minutes| minutes <= 1440),
        "Catch-up grace must be between 0 and 1440 minutes"
    );
    schedule::zone(&draft.timezone)?;
    let schedule = Schedule::parse(&draft.schedule, now, &draft.timezone)?;
    let workdir = draft
        .workdir
        .map(|path| path.canonicalize())
        .transpose()
        .context("Working directory does not exist")?;
    if let Some(path) = &workdir {
        ensure!(path.is_dir(), "Working directory must be a folder");
    }
    let times = draft
        .repeat
        .filter(|times| *times > 0)
        .or_else(|| schedule.is_once().then_some(1));
    let changed = previous.is_none_or(|job| {
        !same_timing(&job.schedule, &schedule) || job.timezone() != draft.timezone
    });
    let completed = previous.map_or(0, |job| job.repeat.completed);
    let next = if draft.paused {
        None
    } else if !changed {
        previous.and_then(|job| job.next_run_at.clone())
    } else {
        first(&schedule, now, &draft.timezone)?.map(stamp)
    };
    let name = if draft.name.trim().is_empty() {
        draft.prompt.chars().take(50).collect()
    } else {
        draft.name.trim().into()
    };
    if draft
        .model
        .as_ref()
        .is_some_and(|model| model.trim().is_empty())
    {
        draft.model = None;
    }
    if draft.model.is_none() {
        draft.provider = None;
    }
    let mut job = Job {
        id: previous.map_or_else(id, |job| job.id.clone()),
        name,
        prompt: draft.prompt.trim().into(),
        schedule,
        repeat: Repeat { times, completed },
        enabled: !draft.paused,
        state: if draft.paused { "paused" } else { "scheduled" }.into(),
        timezone: Some(draft.timezone),
        grace_minutes: draft.grace_minutes,
        catch_up: draft.catch_up,
        deliver: draft.deliver,
        workdir,
        model: draft.model,
        provider: draft.provider,
        created_at: previous.map_or_else(|| stamp(now), |job| job.created_at.clone()),
        next_run_at: next,
        last_run_at: previous.and_then(|job| job.last_run_at.clone()),
        last_status: previous.and_then(|job| job.last_status.clone()),
        last_error: previous.and_then(|job| job.last_error.clone()),
        paused_at: draft.paused.then(|| stamp(now)),
        definition_revision: previous.map_or(1, |job| job.definition_revision + 1),
        extra: previous.map_or_else(BTreeMap::new, |job| job.extra.clone()),
    };
    if exhausted(&job) {
        job.enabled = false;
        job.state = "completed".into();
        job.next_run_at = None;
    }
    Ok(job)
}
fn same_timing(a: &Schedule, b: &Schedule) -> bool {
    match (a, b) {
        (Schedule::Once { run_at: a, .. }, Schedule::Once { run_at: b, .. }) => a == b,
        (Schedule::Interval { minutes: a, .. }, Schedule::Interval { minutes: b, .. }) => a == b,
        (Schedule::Cron { expr: a, .. }, Schedule::Cron { expr: b, .. }) => a == b,
        _ => false,
    }
}
fn grace_seconds(job: &Job, at: DateTime<Utc>) -> Result<i64> {
    if let Some(minutes) = job.grace_minutes {
        return Ok(i64::from(minutes) * 60);
    }
    if job.schedule.is_once() {
        return Ok(120);
    }
    let period = job
        .schedule
        .next_after(at, job.timezone())?
        .map_or(240, |next| (next - at).num_seconds());
    Ok((period / 2).clamp(120, 7200))
}
fn claim(
    document: &mut Document,
    index: usize,
    manual: bool,
    now: DateTime<Utc>,
    owner: &str,
    root: &Path,
) -> Result<Claim> {
    let job = &mut document.jobs[index];
    let run_id = id();
    let scheduled_instant = if manual {
        None
    } else {
        job.next_run_at.clone()
    };
    if !manual {
        let next = if let (Some(seconds), Some(at)) = (
            job.schedule.interval_seconds(),
            scheduled_instant.as_deref().and_then(|at| instant(at).ok()),
        ) {
            let steps = ((now - at).num_seconds() / seconds).max(0) + 1;
            Some(at + Duration::seconds(steps * seconds))
        } else {
            job.schedule.next_after(now, job.timezone())?
        };
        job.next_run_at = next.map(stamp);
    }
    let run = Run {
        id: run_id.clone(),
        job_id: job.id.clone(),
        status: RunStatus::Claimed,
        claimed_at: stamp(now),
        started_at: None,
        finished_at: None,
        scheduled_instant,
        manual,
        repeat_counted: false,
        owner: owner.into(),
        definition_revision: job.definition_revision,
        session_dir: root.join("runs").join(&run_id),
        output: String::new(),
        error: None,
        cancel_requested: false,
    };
    let claim = Claim {
        job: job.clone(),
        run: run.clone(),
    };
    document.runs.push(run);
    Ok(claim)
}
fn settle(job: &mut Job, run: &mut Run, now: DateTime<Utc>) {
    job.last_run_at = Some(stamp(now));
    job.last_status = Some(
        match run.status {
            RunStatus::Completed => "ok",
            RunStatus::Cancelled | RunStatus::Interrupted => "cancelled",
            _ => "error",
        }
        .into(),
    );
    job.last_error = run.error.clone();
    if !run.manual && !run.repeat_counted {
        job.repeat.completed = job.repeat.completed.saturating_add(1);
        run.repeat_counted = true;
    }
    if !run.manual && (exhausted(job) || job.schedule.is_once()) {
        job.enabled = false;
        job.next_run_at = None;
        job.state = "completed".into();
    } else if !job.enabled {
        job.state = "paused".into();
    } else {
        job.state = if run.status == RunStatus::Failed {
            "error"
        } else {
            "scheduled"
        }
        .into();
    }
}
fn recover(document: &mut Document, now: DateTime<Utc>) {
    for run in &mut document.runs {
        if !run.status.live() {
            continue;
        }
        let started = run.status == RunStatus::Running;
        run.status = RunStatus::Interrupted;
        run.finished_at = Some(stamp(now));
        run.error = Some(
            if started {
                "Previous scheduler stopped during execution; this occurrence will not be repeated"
            } else {
                "Previous scheduler stopped before dispatch; pending occurrence restored"
            }
            .into(),
        );
        if let Some(job) = document.jobs.iter_mut().find(|job| job.id == run.job_id) {
            if started {
                settle(job, run, now);
            } else if !run.manual
                && job.enabled
                && job.definition_revision == run.definition_revision
            {
                if let Some(slot) = &run.scheduled_instant {
                    if job.next_run_at.as_ref().is_none_or(|next| slot < next) {
                        job.next_run_at = Some(slot.clone());
                    }
                }
            }
        }
    }
}
fn prune(document: &mut Document) {
    if document.runs.len() > MAX_RUNS {
        let mut remove = document.runs.len() - MAX_RUNS;
        document.runs.retain(|run| {
            if remove > 0 && !run.status.live() {
                remove -= 1;
                false
            } else {
                true
            }
        });
    }
}
