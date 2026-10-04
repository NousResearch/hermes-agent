//! The "update in progress" marker: the Desktop launch gate and the
//! cross-process update lock shared with `hermes_cli/update_lock.py` and the
//! Electron gate (`apps/desktop/electron/update-marker.ts`).
//!
//! Parse, liveness, claim and litter rules are the LP-LOCK round-2 contract
//! that `hermes_cli/update_lock.py` implements identically; keep them in step.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::update::UPDATE_EXIT_CONCURRENT;

/// RAII guard that owns the "update in progress" marker (see
/// `paths::update_in_progress_marker`). Created at the top of `run_update`;
/// its `Drop` releases the claim on EVERY exit path — success, early
/// `return Err`, or a panic that unwinds through `run_update` — so a crashed
/// or aborted updater can never permanently strand the marker and block
/// future desktop launches.
///
/// Marker contract (C1, shared byte-for-byte with `hermes_cli/update_lock.py`
/// and the Electron gate), UTF-8, `\n` line ends, trailing newline:
///
/// ```text
/// <pid>
/// <started_at unix seconds>
/// ct:<owner process creation time, unix seconds, 3 decimals>
/// delegate:<pid> ct:<ct>          (optional; written by `hermes update` / the hand-off scripts)
/// run:<id>                        (optional; the Desktop update run, A7 rule 6)
/// ```
///
/// Every decision-and-mutation (reclaim, adopt, release, rewrite) runs while holding the
/// kernel lock on the never-deleted sidecar `<marker>.lock` (A7 rule 1); only the exclusive
/// first publish is lock-free. `tests/fixtures/update_marker_corpus.json` pins parse, liveness
/// and release byte-for-byte across every implementation.
///
/// Line 3 is absent in legacy (v1) markers. The creation time makes the pid
/// a process IDENTITY, so liveness never has to fall back to an age ceiling
/// (a live v2 owner is live however long it runs) and a recycled pid is not
/// mistaken for the original owner.
///
/// The marker is also the cross-process update lock: `hermes update` claims
/// the same file so a dashboard-spawned update and this updater can't mutate
/// one checkout at the same time. `acquire` therefore publishes its claim
/// exclusively (a fully written tmp file hard-linked onto the path) and
/// REFUSES when a live foreign owner holds it — the pre-fix clobber is what
/// let a dashboard `hermes update` keep running while install-mode bootstrap
/// rewrote the tree underneath it.
pub(crate) struct UpdateMarkerGuard {
    path: PathBuf,
    /// We published or adopted a claim (released per A7 rule 5 on `complete`/`Drop`).
    claimed: bool,
}

/// Why `UpdateMarkerGuard::acquire` did not produce a claim.
pub(crate) enum AcquireError {
    /// A live update (or a claim being published right now: pid 0) holds it.
    Busy(MarkerOwner),
    /// The marker cannot be written at all; the user-facing message.
    Unwritable(String),
}

/// Age ceiling for markers whose owner identity cannot be confirmed: legacy
/// (v1, no `ct:` line) markers, and v2 markers whose live pid's creation
/// time cannot be read. Without a creation time a live pid may be a
/// recycled one, so age is the only bound. Matches the v1 rule in
/// apps/desktop/electron/update-marker.ts and hermes_cli/update_lock.py.
const UPDATE_MARKER_MAX_AGE_SECS: u64 = 20 * 60;

/// Creation-time tolerance when matching a recorded `ct:` against the live
/// process (Linux derives it from whole-second btime plus clock ticks).
const MARKER_CT_TOLERANCE_SECS: f64 = 2.0;

/// A 0-byte marker this young is a claim between its exclusive create and
/// its first write (the non-hard-link fallback): live, never deleted.
const EMPTY_MARKER_GRACE: Duration = Duration::from_secs(5);

/// The pid + age of a confirmed-live update holding the marker. pid 0 is a
/// claim still being written (a fresh 0-byte marker).
pub(crate) struct MarkerOwner {
    pub(crate) pid: u32,
    pub(crate) age_secs: u64,
    /// R6: the marker's owner is dead but the install's checkout lock is still held (a process
    /// that update started is still mutating the checkout), so the marker is kept. pid is 0.
    pub(crate) held: bool,
}

/// Parsed marker body.
struct MarkerRecord {
    pid: u32,
    started_at: u64,
    ct: Option<f64>,
    /// The `ct:` value exactly as written (a rewrite keeps it byte-identical).
    ct_text: Option<String>,
    /// The first well-formed `delegate:<pid> ct:<ct>` line at line 4 or later.
    delegate: Option<(u32, f64)>,
    delegate_ct_text: Option<String>,
    /// Every well-formed `run:<id>` line at line 4 or later (the first is THE run id).
    runs: Vec<String>,
    /// Lines 1-2 plus the `ct:` line when present.
    #[cfg_attr(not(test), allow(dead_code))]
    identity: Vec<String>,
}

impl MarkerRecord {
    /// A7 rule 5 rewrite: `pid` owns the claim from the original `started_at`; run lines kept.
    fn canonical(&self, pid: u32, ct_text: Option<&str>) -> Vec<u8> {
        let mut body = format!("{pid}\n{}\n", self.started_at);
        if let Some(ct) = ct_text {
            body.push_str(&format!("ct:{ct}\n"));
        }
        for run in &self.runs {
            body.push_str(&format!("run:{run}\n"));
        }
        body.into_bytes()
    }
}

fn is_ascii_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

/// `[0-9]+(\.[0-9]+)?` as a float; anything else is `None`.
fn parse_ct_value(text: &str) -> Option<f64> {
    let (whole, frac) = match text.split_once('.') {
        Some((whole, frac)) => (whole, Some(frac)),
        None => (text, None),
    };
    if !is_ascii_digits(whole) || frac.is_some_and(|frac| !is_ascii_digits(frac)) {
        return None;
    }
    text.parse().ok()
}

/// `delegate:<pid> ct:<ct>` with exactly one space.
fn parse_delegate(line: &str) -> Option<(u32, f64, &str)> {
    let (pid, ct) = line.strip_prefix("delegate:")?.split_once(' ')?;
    if !is_ascii_digits(pid) {
        return None;
    }
    let ct = ct.strip_prefix("ct:")?;
    Some((pid.parse().ok()?, parse_ct_value(ct)?, ct))
}

/// `run:<[A-Za-z0-9._-]{1,128}>` (A7 rule 6).
fn parse_run(line: &str) -> Option<&str> {
    let id = line.strip_prefix("run:")?;
    let ok = (1..=128).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    ok.then_some(id)
}

/// Parse a marker body (A2 + A7 rule 7; `tests/fixtures/update_marker_corpus.json`). `None` =
/// malformed (dead): line 1 (a u32 pid) and line 2 must be ASCII digits only after dropping one
/// BOM, one trailing `\r` per line and surrounding spaces/tabs. A line 3 that is not a
/// well-formed `ct:` makes a v1 marker. Lines 4+ are tagged: the first well-formed delegate line
/// and every well-formed run line count; anything else is ignored.
fn parse_marker(raw: &[u8]) -> Option<MarkerRecord> {
    let text = String::from_utf8_lossy(raw);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let lines: Vec<&str> = text
        .split('\n')
        .map(|line| {
            line.strip_suffix('\r')
                .unwrap_or(line)
                .trim_matches(|c| c == ' ' || c == '\t')
        })
        .collect();
    let (pid_line, started_line) = (*lines.first()?, *lines.get(1)?);
    if !is_ascii_digits(pid_line) || !is_ascii_digits(started_line) {
        return None;
    }
    let pid = pid_line.parse().ok()?;
    let started_at = started_line.parse().ok()?;
    let ct_line = lines.get(2).copied().unwrap_or("");
    let ct_text = ct_line
        .strip_prefix("ct:")
        .filter(|ct| parse_ct_value(ct).is_some());
    let mut identity = vec![pid_line.to_string(), started_line.to_string()];
    if ct_text.is_some() {
        identity.push(ct_line.to_string());
    }
    let tagged = lines.get(3..).unwrap_or(&[]);
    let delegate = tagged.iter().find_map(|line| parse_delegate(line));
    Some(MarkerRecord {
        pid,
        started_at,
        ct: ct_text.and_then(parse_ct_value),
        ct_text: ct_text.map(str::to_string),
        delegate: delegate.map(|(pid, ct, _)| (pid, ct)),
        delegate_ct_text: delegate.map(|(_, _, text)| text.to_string()),
        runs: tagged
            .iter()
            .filter_map(|line| parse_run(line))
            .map(str::to_string)
            .collect(),
        identity,
    })
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The process facts a marker verdict depends on: who we are, the clock, and how to probe a pid.
/// Production uses `World::real`; the shared corpus injects its own process table.
struct World<'a> {
    pid: u32,
    /// Our own creation time; `None` only when it cannot be probed.
    ct: Option<f64>,
    now: u64,
    alive: &'a dyn Fn(u32) -> bool,
    ct_of: &'a dyn Fn(u32) -> Option<f64>,
}

impl World<'static> {
    fn real() -> Self {
        let pid = std::process::id();
        World {
            pid,
            ct: process_creation_time(pid),
            now: unix_now_secs(),
            alive: &pid_is_alive,
            ct_of: &process_creation_time,
        }
    }
}

/// Our own pid is our incarnation only within this epsilon (A7 rule 4, R12): the `ct:` we
/// wrote is our exact creation time to 3 decimals, so anything further is a previous process
/// that had our pid — never "close enough".
const OWN_CT_EPSILON_SECS: f64 = 0.005;

/// The process `pid` is alive AND is the process that recorded `recorded_ct` (A1, A7 rule 4).
/// Our own pid must match our creation time exactly; a no-`ct` claim naming our pid is a
/// previous incarnation (dead). Any other pid: only a confirmed creation-time match is
/// unbounded; a v1 record, or a live pid whose creation time cannot be read, is live only while
/// the marker is within the v1 age ceiling.
fn identity_live(pid: u32, recorded_ct: Option<f64>, age_secs: u64, world: &World) -> bool {
    if pid == 0 {
        return false;
    }
    // We are alive by definition: only our incarnation is in question.
    if pid == world.pid {
        return match (recorded_ct, world.ct) {
            (Some(recorded), Some(own)) => (recorded - own).abs() <= OWN_CT_EPSILON_SECS,
            // Degraded: we cannot read our own creation time, so we write v1 claims ourselves.
            (None, None) => true,
            _ => false,
        };
    }
    if !(world.alive)(pid) {
        return false;
    }
    let within_ceiling = age_secs <= UPDATE_MARKER_MAX_AGE_SECS;
    let Some(recorded) = recorded_ct else {
        return within_ceiling;
    };
    match (world.ct_of)(pid) {
        Some(actual) => (recorded - actual).abs() <= MARKER_CT_TOLERANCE_SECS,
        None => within_ceiling,
    }
}

/// The live holder of a parsed marker, if any: the owner (lines 1-3) or, failing that, the
/// delegate, both aged by the marker's `started_at`.
fn marker_live_holder(record: &MarkerRecord, world: &World) -> Option<MarkerOwner> {
    let age_secs = world.now.saturating_sub(record.started_at);
    if identity_live(record.pid, record.ct, age_secs, world) {
        return Some(MarkerOwner {
            pid: record.pid,
            age_secs,
            held: false,
        });
    }
    match record.delegate {
        Some((pid, ct)) if identity_live(pid, Some(ct), age_secs, world) => Some(MarkerOwner {
            pid,
            age_secs,
            held: false,
        }),
        _ => None,
    }
}

/// What releasing our claim does to the marker bytes (A7 rule 5).
#[derive(Debug, PartialEq)]
enum Release {
    Keep,
    Delete,
    Rewrite(Vec<u8>),
}

/// A7 rule 5. The owner (lines 1-3 are our exact incarnation) deletes the marker regardless of
/// delegate lines, unless a LIVE delegate other than us exists: that delegate then inherits the
/// claim (rewritten with it as owner). A delegate (us) drops its line while the owner lives,
/// else deletes. A marker not naming us is kept.
fn release_decision(raw: &[u8], world: &World) -> Release {
    let Some(record) = parse_marker(raw) else {
        return Release::Keep;
    };
    let age_secs = world.now.saturating_sub(record.started_at);
    let owner_live = identity_live(record.pid, record.ct, age_secs, world);
    let delegate_live = record
        .delegate
        .is_some_and(|(pid, ct)| identity_live(pid, Some(ct), age_secs, world));
    let delegate_pid = record.delegate.map(|(pid, _)| pid);
    if record.pid == world.pid && owner_live {
        return match delegate_pid {
            Some(pid) if pid != world.pid && delegate_live => {
                Release::Rewrite(record.canonical(pid, record.delegate_ct_text.as_deref()))
            }
            _ => Release::Delete,
        };
    }
    if delegate_pid == Some(world.pid) && delegate_live {
        return if owner_live {
            Release::Rewrite(record.canonical(record.pid, record.ct_text.as_deref()))
        } else {
            Release::Delete
        };
    }
    Release::Keep
}

/// How long a marker mutation waits for another process's hold on the marker mutex.
const MUTEX_WAIT: Duration = Duration::from_secs(10);

/// `<marker>.lock`: the A7 rule 1 sidecar. Never deleted; holding its kernel lock serializes
/// every marker decision-and-mutation across Python, this updater and the hand-off scripts.
fn marker_mutex_path(path: &Path) -> PathBuf {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!("{name}.lock"))
}

/// A held marker mutex; the kernel lock is released when the file closes (also on a crash).
struct MarkerMutex {
    _file: std::fs::File,
}

/// A7 rule 1: hold the sidecar's kernel lock (POSIX `flock`; Windows an exclusive, share-none
/// open — the same lock the PowerShell hand-off takes) for at most `MUTEX_WAIT`.
/// `ErrorKind::WouldBlock` = another process held it the whole time.
fn lock_marker(path: &Path) -> std::io::Result<MarkerMutex> {
    let sidecar = marker_mutex_path(path);
    let deadline = std::time::Instant::now() + MUTEX_WAIT;
    loop {
        if let Some(file) = try_lock_sidecar(&sidecar)? {
            return Ok(MarkerMutex { _file: file });
        }
        if std::time::Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "another process holds the update marker mutex",
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(unix)]
fn try_lock_sidecar(sidecar: &Path) -> std::io::Result<Option<std::fs::File>> {
    use std::os::unix::io::AsRawFd;

    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(sidecar)
    {
        Ok(file) => file,
        // flock needs no write access: a sidecar another user created still serializes us.
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
            std::fs::OpenOptions::new().read(true).open(sidecar)?
        }
        Err(err) => return Err(err),
    };
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(Some(file));
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        Some(code) if code == libc::EWOULDBLOCK || code == libc::EINTR => Ok(None),
        _ => Err(err),
    }
}

#[cfg(windows)]
fn try_lock_sidecar(sidecar: &Path) -> std::io::Result<Option<std::fs::File>> {
    use std::os::windows::fs::OpenOptionsExt;
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_SHARING_VIOLATION: i32 = 32;

    let opened = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .share_mode(0)
        .open(sidecar);
    let opened = match opened {
        Err(err) if err.raw_os_error() == Some(ERROR_ACCESS_DENIED) => std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(sidecar),
        other => other,
    };
    match opened {
        Ok(file) => Ok(Some(file)),
        Err(err) if err.raw_os_error() == Some(ERROR_SHARING_VIOLATION) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Delete the marker only if its bytes still equal `expected` (the bytes a verdict was reached
/// on). Callers hold the marker mutex; the compare additionally spares a claim a non-locking
/// legacy writer published meanwhile. `Ok(true)` when the file is gone because of us.
fn remove_if_unchanged(path: &Path, expected: &[u8]) -> std::io::Result<bool> {
    match std::fs::read(path) {
        Ok(current) if current == expected => match std::fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(err) => {
                tracing::warn!(?path, %err, "could not remove update marker");
                Err(err)
            }
        },
        _ => Ok(false),
    }
}

fn compare_and_delete(path: &Path, expected: &[u8]) -> bool {
    remove_if_unchanged(path, expected).unwrap_or(false)
}

/// Replace the marker with `body` (tmp + rename) while its bytes are still `expected`.
fn replace_if_unchanged(path: &Path, expected: &[u8], body: &[u8]) -> std::io::Result<bool> {
    let tmp = tmp_sibling(path);
    let swapped = write_new_file(&tmp, body).and_then(|()| {
        if std::fs::read(path)? != expected {
            return Ok(false);
        }
        std::fs::rename(&tmp, path).map(|()| true)
    });
    let _ = std::fs::remove_file(&tmp);
    swapped
}

/// What is on disk at the marker path.
enum MarkerState {
    Absent,
    /// A live holder, or a dead marker kept because its checkout lock is held (`held`, R6).
    Live(MarkerOwner),
    /// Dead / malformed / recycled / past the ceiling, compare-and-deleted.
    /// Carries the error when the marker could not be read or removed (it
    /// is then still on disk).
    Dead(Option<std::io::Error>),
}

/// Read and judge the marker; a dead verdict REMOVES it. The CALLER HOLDS the marker mutex
/// (A7 rule 1), so the verdict and the delete are one critical section: a claim published by a
/// lock-respecting process after our read can no longer be deleted on a stale verdict (R3).
/// A 0-byte marker younger than `EMPTY_MARKER_GRACE` is a claim being written: live as pid 0.
/// R6: a dead marker is never deleted while `install_root`'s checkout lock is held; it is then
/// reported `held` (pid 0) — the update it names exited, but a process it started still runs.
fn inspect_marker_locked(path: &Path, install_root: &Path, world: &World) -> MarkerState {
    let raw = match std::fs::read(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return MarkerState::Absent,
        // Present but unreadable: no verdict, and nothing we may delete.
        Err(err) => return MarkerState::Dead(Some(err)),
    };
    let live = if raw.is_empty() {
        empty_marker_age(path)
            .filter(|age| *age < EMPTY_MARKER_GRACE)
            .map(|age| MarkerOwner {
                pid: 0,
                age_secs: age.as_secs(),
                held: false,
            })
    } else {
        parse_marker(&raw).and_then(|record| marker_live_holder(&record, world))
    };
    match live {
        Some(owner) => MarkerState::Live(owner),
        None if checkout_lock_held(install_root) => MarkerState::Live(MarkerOwner {
            pid: 0,
            age_secs: parse_marker(&raw)
                .map_or(0, |record| world.now.saturating_sub(record.started_at)),
            held: true,
        }),
        None => MarkerState::Dead(remove_if_unchanged(path, &raw).err()),
    }
}

/// Age of the marker by mtime; a future mtime counts as brand new.
fn empty_marker_age(path: &Path) -> Option<Duration> {
    let modified = std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()?;
    Some(
        SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default(),
    )
}

/// Read the marker (under its mutex) and report a live holder, if any; a dead marker is
/// removed. Test-only view of `inspect_marker_locked`; the install root is the marker's
/// `hermes-agent` sibling, as in production (`<HERMES_HOME>/hermes-agent`).
#[cfg(test)]
fn live_marker_owner(path: &Path) -> Option<MarkerOwner> {
    let _mutex = lock_marker(path).ok()?;
    let install_root = path.with_file_name("hermes-agent");
    match inspect_marker_locked(path, &install_root, &World::real()) {
        MarkerState::Live(owner) => Some(owner),
        _ => None,
    }
}

/// True when the on-disk marker's owner identity is THIS incarnation (A7 rule 4): our pid AND
/// our creation time, never just a pid a previous process also had.
///
/// Liveness of anything else is deliberately NOT consulted: the exit-2 self-heal below needs
/// exactly one fact — is this our claim — because a `hermes update` child that refuses over
/// OUR claim is a handoff-recognition failure in a stale checkout, not a concurrent update.
fn marker_owned_by_self(path: &Path) -> bool {
    let Some(record) = std::fs::read(path).ok().and_then(|raw| parse_marker(&raw)) else {
        return false;
    };
    let world = World::real();
    record.pid == world.pid && identity_live(record.pid, record.ct, 0, &world)
}

/// The exit-2 heal decision (#75788), extracted so the contract is testable.
///
/// True only when ALL hold: the child exited with the concurrent-update
/// refusal code, the on-disk marker names THIS process, and nothing holds
/// `install_root`'s checkout lock. That combination means the child refused
/// over its own parent's claim — a stale checkout without handoff
/// recognition — so dropping the claim and retrying once is safe. Any other
/// owner (live foreign updater, garbage, missing marker), any other exit
/// code, or a held checkout lock (R6: another update's process is still
/// mutating the install, so the refusal is legitimate) must leave the
/// refusal untouched.
pub(crate) fn should_heal_self_marker_refusal(
    exit_code: Option<i32>,
    marker_path: &Path,
    install_root: &Path,
) -> bool {
    exit_code == Some(UPDATE_EXIT_CONCURRENT)
        && marker_owned_by_self(marker_path)
        && !checkout_lock_held(install_root)
}

/// `hermes_cli/update_lock.py::CHECKOUT_LOCK_NAME` (no checkout) / `GIT_CHECKOUT_LOCK_NAME`.
const CHECKOUT_LOCK_NAME: &str = ".hermes-update.lock";
const GIT_CHECKOUT_LOCK_NAME: &str = "hermes-update.lock";

/// `update_lock.py::_WINDOWS_LOCK_OFFSET`: the byte msvcrt locks, far past the holder record.
#[cfg(windows)]
const WINDOWS_LOCK_OFFSET: u32 = 1 << 20;

/// `update_lock.py::_git_common_dir`: the repository's common git dir, read from disk. `.git`
/// is the dir itself, or a `gitdir: <path>` file (linked worktree, submodule) whose target may
/// name the shared dir in `commondir`. `None` when `root` is no checkout.
fn git_common_dir(root: &Path) -> Option<PathBuf> {
    let read = |path: &Path| -> Option<String> {
        let text = std::fs::read_to_string(path).ok()?;
        Some(
            text.strip_prefix('\u{feff}')
                .unwrap_or(&text)
                .trim()
                .to_string(),
        )
    };
    let dot = root.join(".git");
    let mut gitdir = if dot.is_dir() {
        dot
    } else if dot.is_file() {
        // An absolute target replaces `root`.
        root.join(read(&dot)?.strip_prefix("gitdir:")?.trim())
    } else {
        return None;
    };
    let common = gitdir.join("commondir");
    if common.is_file() {
        gitdir = gitdir.join(read(&common)?);
    }
    Some(normalize_lexically(&gitdir))
}

/// `os.path.normpath`: drop `.` and fold `..` without touching the filesystem.
fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => out.push(".."),
            },
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `update_lock.py::checkout_lock_path`: the install's checkout kernel lock file.
fn checkout_lock_path(install_root: &Path) -> PathBuf {
    match git_common_dir(install_root) {
        Some(common) => common.join(GIT_CHECKOUT_LOCK_NAME),
        None => install_root.join(CHECKOUT_LOCK_NAME),
    }
}

/// `update_lock.py::checkout_lock_held`: true while some process holds the checkout kernel
/// lock. The probe takes the lock for one try on a fresh open and drops it; a missing file is
/// not held. This updater never takes the checkout lock itself, so it is never the holder.
#[cfg(unix)]
fn checkout_lock_held(install_root: &Path) -> bool {
    use std::os::unix::io::AsRawFd;

    let Ok(file) = std::fs::File::open(checkout_lock_path(install_root)) else {
        return false;
    };
    let fd = file.as_raw_fd();
    if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        return false;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EWOULDBLOCK)
}

/// Windows: msvcrt's `LK_NBLCK` is a `LockFile` byte-range lock on one byte at
/// `WINDOWS_LOCK_OFFSET`; `LockFileEx` on the same byte conflicts with it. As in `_try_lock`,
/// a lock that cannot be taken is held.
#[cfg(windows)]
fn checkout_lock_held(install_root: &Path) -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, UnlockFileEx, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let Ok(file) = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .open(checkout_lock_path(install_root))
    else {
        return false;
    };
    let handle = file.as_raw_handle();
    let at_offset = || unsafe {
        let mut overlapped: OVERLAPPED = std::mem::zeroed();
        overlapped.Anonymous.Anonymous.Offset = WINDOWS_LOCK_OFFSET;
        overlapped
    };
    unsafe {
        let mut overlapped = at_offset();
        let flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
        if LockFileEx(handle, flags, 0, 1, 0, &mut overlapped) == 0 {
            return true;
        }
        let mut overlapped = at_offset();
        UnlockFileEx(handle, 0, 1, 0, &mut overlapped);
    }
    false
}

#[cfg(not(any(unix, windows)))]
fn checkout_lock_held(_install_root: &Path) -> bool {
    false
}

/// Process creation time as unix seconds, comparable with the `ct:` values
/// `hermes_cli/update_lock.py` records. `None` when it cannot be probed.
#[cfg(target_os = "linux")]
fn process_creation_time(pid: u32) -> Option<f64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 22 (starttime, clock ticks since boot). The comm field may hold
    // spaces, so count from the closing paren: the field after it is #3.
    let after_comm = &stat[stat.rfind(')')? + 1..];
    let ticks: f64 = after_comm.split_whitespace().nth(22 - 3)?.parse().ok()?;
    let btime: f64 = std::fs::read_to_string("/proc/stat")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("btime "))?
        .trim()
        .parse()
        .ok()?;
    let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if clk_tck <= 0 {
        return None;
    }
    Some(btime + ticks / clk_tck as f64)
}

#[cfg(target_os = "macos")]
fn process_creation_time(pid: u32) -> Option<f64> {
    // proc_pidinfo(PROC_PIDTBSDINFO) reads the same kernel start time as
    // sysctl kern.proc.pid's kp_proc.p_starttime.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let written = unsafe {
        libc::proc_pidinfo(
            pid as libc::c_int,
            libc::PROC_PIDTBSDINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size,
        )
    };
    if written != size {
        return None;
    }
    Some(info.pbi_start_tvsec as f64 + info.pbi_start_tvusec as f64 / 1_000_000.0)
}

#[cfg(windows)]
fn process_creation_time(pid: u32) -> Option<f64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    const UNIX_EPOCH_AS_FILETIME: u64 = 116_444_736_000_000_000;
    let empty = || FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut creation, mut exit, mut kernel, mut user) = (empty(), empty(), empty(), empty());
    let ok = unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            // Access denied included: `pid_is_alive` calls such a pid alive, and an unreadable
            // creation time leaves it only the v1 age ceiling (`identity_live`).
            return None;
        }
        let ok = GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user);
        CloseHandle(handle);
        ok
    };
    if ok == 0 {
        return None;
    }
    let ticks = (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime);
    Some(ticks.checked_sub(UNIX_EPOCH_AS_FILETIME)? as f64 / 10_000_000.0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn process_creation_time(_pid: u32) -> Option<f64> {
    None
}

/// `ERROR_ACCESS_DENIED` (winerror.h); spelled out so the decision below is testable off Windows.
#[cfg(any(windows, test))]
const WIN32_ERROR_ACCESS_DENIED: u32 = 5;

/// Liveness verdict when `OpenProcess` on a pid fails with `GetLastError() == err`.
///
/// Access denied means the process EXISTS but belongs to another user or runs elevated: alive,
/// as `hermes_cli/_early_recovery._pid_is_running` decides — racing an elevated updater is worse
/// than postponing. Its creation time is then unreadable too (`process_creation_time` is
/// `None`), so `identity_live` bounds it by the v1 age ceiling, never forever. Any other failure
/// (`ERROR_INVALID_PARAMETER` for a pid that does not exist, ...) is dead.
#[cfg(any(windows, test))]
fn liveness_from_open_error(err: u32) -> bool {
    err == WIN32_ERROR_ACCESS_DENIED
}

/// True when a process with `pid` currently exists.
#[cfg(windows)]
fn pid_is_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // pid 0 is the System Idle Process, never an updater (Python: `pid <= 0` is dead).
    if pid == 0 {
        return false;
    }
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return liveness_from_open_error(GetLastError());
        }
        let mut code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        ok != 0 && code == STILL_ACTIVE as u32
    }
}

#[cfg(not(windows))]
fn pid_is_alive(pid: u32) -> bool {
    // pid 0 is the caller's process GROUP, not a process: kill(0, 0) always
    // succeeds, so a marker corrupted to "0" would read as alive forever.
    if pid == 0 {
        return false;
    }
    // signal 0 delivers nothing; it only probes existence/permission.
    // ESRCH => dead. EPERM => alive but owned by another user.
    //
    // kill(pid, 0) alone is not a reliable liveness probe: it also succeeds
    // for a ZOMBIE — a process that has exited but whose parent has not yet
    // reaped it. A crashed updater lingering as a zombie would read as alive
    // and hold a stale marker "live" for the whole age ceiling (#77259). On
    // Linux the process state is directly observable via /proc; fall back to
    // signal 0 when /proc is unavailable (e.g. a container without procfs).
    #[cfg(target_os = "linux")]
    {
        if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            // Field 3 is the state; the comm field in parens may contain
            // spaces, so anchor on the closing paren instead of splitting.
            if let Some(comm_end) = stat.rfind(')') {
                let state = stat[comm_end + 1..].split_whitespace().next().unwrap_or("");
                if state == "Z" {
                    return false;
                }
            }
        }
    }
    // macOS has no /proc; `ps -o stat=` reports the same state field ('Z' for
    // a zombie). Only consulted after the marker's pid answered signal 0, so
    // the spawn cost is paid exactly when a stale-marker zombie is the
    // question. A failed or empty probe falls through to the signal-0
    // verdict (fail-open, matching the EPERM rule below).
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .arg("-o")
            .arg("stat=")
            .arg("-p")
            .arg(pid.to_string())
            .output()
        {
            let state = String::from_utf8_lossy(&output.stdout);
            if state.trim_start().starts_with('Z') {
                return false;
            }
        }
    }
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// A unique tmp sibling `<marker name>.<own pid>.<nanos>-<seq>.tmp`. The pid
/// is the first dot component after the marker name so `sweep_tmp_litter`
/// can tell a dead claimant's leftovers from a live one's in-flight file.
fn tmp_sibling(path: &Path) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!("{name}.{}.{nanos}-{seq}.tmp", std::process::id()))
}

/// Exclusively create `path` and write + fsync `body` into it.
fn write_new_file(path: &Path, body: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(body)?;
    file.sync_all()
}

/// m10: delete `<marker name>.<pid>.tmp` / `<marker name>.<pid>.<any>.tmp`
/// siblings whose `<pid>` is no longer alive — the leftovers of a claimant
/// that died between writing its tmp file and removing it.
fn sweep_tmp_litter(path: &Path) {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return;
    };
    let prefix = format!("{}.", name.to_string_lossy());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let entry_name = entry.file_name();
        let entry_name = entry_name.to_string_lossy();
        let Some(rest) = entry_name
            .strip_prefix(prefix.as_str())
            .and_then(|rest| rest.strip_suffix(".tmp"))
        else {
            continue;
        };
        let pid = rest.split('.').next().unwrap_or("");
        let dead = is_ascii_digits(pid) && !pid.parse::<u32>().is_ok_and(pid_is_alive);
        if dead {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Outcome of one claim attempt.
enum Publish {
    Claimed,
    Exists,
}

/// A3: publish `body` at `path` only if nothing is there. The body is fully
/// written and fsynced in a tmp sibling first, then hard-linked onto the
/// path, so a reader never sees a torn claim. A filesystem without hard
/// links falls back to exclusive create + write + fsync (whose brief 0-byte
/// window readers treat as live). `Err` = the marker cannot be written.
fn publish_claim(path: &Path, body: &[u8]) -> std::io::Result<Publish> {
    let tmp = tmp_sibling(path);
    let linked = write_new_file(&tmp, body).and_then(|()| std::fs::hard_link(&tmp, path));
    let _ = std::fs::remove_file(&tmp);
    match linked {
        Ok(()) => return Ok(Publish::Claimed),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => return Ok(Publish::Exists),
        Err(err) => tracing::debug!(?path, %err, "hard-link claim unavailable; using create_new"),
    }
    match write_new_file(path, body) {
        Ok(()) => Ok(Publish::Claimed),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(Publish::Exists),
        Err(err) => {
            // Remove our torn file only while it still holds a prefix of
            // what we were writing (a reader may already have reaped it and
            // someone else re-claimed the path).
            if let Ok(current) = std::fs::read(path) {
                if body.starts_with(&current) {
                    compare_and_delete(path, &current);
                }
            }
            Err(err)
        }
    }
}

fn unwritable(path: &Path, err: &std::io::Error) -> AcquireError {
    AcquireError::Unwritable(format!(
        "Cannot lock this install for the update: {} is not writable ({err}). \
         Run the update as the user that owns the install.",
        path.display()
    ))
}

impl UpdateMarkerGuard {
    /// Claim the marker, or report why not.
    ///
    /// A7 rule 2: the claim is an exclusive publish (`publish_claim`) with no lock; an existing
    /// marker is never truncated and overwritten. If one exists, the verdict and any mutation
    /// happen under the marker mutex (rule 1): a dead holder (or unparseable bytes — including a
    /// no-`ct` claim naming our own pid, a previous incarnation, rule 4) is removed and the
    /// claim re-published inside the same hold; a live foreign holder is `Busy`; a claim that is
    /// our exact incarnation is adopted.
    ///
    /// R6: a dead marker is NOT removed while `install_root`'s checkout lock is held — a process
    /// the dead update started is still mutating the install — and the refusal is `Busy` with a
    /// `held` owner.
    ///
    /// A marker that cannot be written at all is `Unwritable`: the update
    /// refuses rather than run unserialized against other updaters (m8).
    pub(crate) fn acquire(path: PathBuf, install_root: &Path) -> Result<Self, AcquireError> {
        let world = World::real();
        let mut body = format!("{}\n{}\n", world.pid, world.now);
        if let Some(ct) = world.ct {
            body.push_str(&format!("ct:{ct:.3}\n"));
        }
        if let Some(parent) = path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                return Err(unwritable(&path, &err));
            }
        }
        sweep_tmp_litter(&path);
        let claimed = |path: PathBuf| Self {
            path,
            claimed: true,
        };
        for _ in 0..3 {
            match publish_claim(&path, body.as_bytes()) {
                Ok(Publish::Claimed) => return Ok(claimed(path)),
                Ok(Publish::Exists) => {}
                Err(err) => {
                    tracing::warn!(?path, %err, "could not create update-in-progress marker");
                    return Err(unwritable(&path, &err));
                }
            }
            let _mutex = match lock_marker(&path) {
                Ok(mutex) => mutex,
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    return Err(AcquireError::Busy(MarkerOwner {
                        pid: 0,
                        age_secs: 0,
                        held: false,
                    }))
                }
                Err(err) => return Err(unwritable(&marker_mutex_path(&path), &err)),
            };
            let world = World::real();
            match inspect_marker_locked(&path, install_root, &world) {
                MarkerState::Live(owner) if owner.pid == world.pid => return Ok(claimed(path)),
                MarkerState::Live(owner) => return Err(AcquireError::Busy(owner)),
                MarkerState::Dead(Some(err)) => return Err(unwritable(&path, &err)),
                MarkerState::Absent | MarkerState::Dead(None) => {
                    match publish_claim(&path, body.as_bytes()) {
                        Ok(Publish::Claimed) => return Ok(claimed(path)),
                        // A non-locking exclusive creator won the path: judge it.
                        Ok(Publish::Exists) => {}
                        Err(err) => return Err(unwritable(&path, &err)),
                    }
                }
            }
        }
        // The path kept changing under three attempts: refuse rather than run unserialized.
        tracing::warn!(?path, "update marker changed under three claim attempts");
        Err(AcquireError::Busy(MarkerOwner {
            pid: 0,
            age_secs: 0,
            held: false,
        }))
    }

    /// Release the marker as soon as every mutating stage has completed.
    ///
    /// The updater still owns a Tauri/Cocoa event loop while it relaunches the
    /// desktop, and that loop can outlive `app.exit(0)`. Relying on `Drop`
    /// alone therefore leaves a *successful* update looking active — a live
    /// pid holding a fresh marker — which blocks desktop startup and every
    /// other updater. Idempotent: `Drop` still runs and tolerates an
    /// already-removed (or handed-over) marker.
    ///
    /// A7 rule 5 under the marker mutex (`release_decision`): our claim is removed regardless of
    /// delegate lines, except that a LIVE `hermes update` delegate inherits it (the marker is
    /// rewritten with that delegate as owner) instead of losing it.
    pub(crate) fn complete(&self) {
        if !self.claimed {
            return;
        }
        let _mutex = match lock_marker(&self.path) {
            Ok(mutex) => mutex,
            Err(err) => {
                tracing::warn!(path = ?self.path, %err, "could not lock the update marker to release it");
                return;
            }
        };
        let Ok(raw) = std::fs::read(&self.path) else {
            return;
        };
        match release_decision(&raw, &World::real()) {
            Release::Keep => {}
            Release::Delete => {
                compare_and_delete(&self.path, &raw);
            }
            Release::Rewrite(body) => {
                if let Err(err) = replace_if_unchanged(&self.path, &raw, &body) {
                    tracing::warn!(path = ?self.path, %err, "could not hand the update marker to its delegate");
                }
            }
        }
    }
}

impl Drop for UpdateMarkerGuard {
    fn drop(&mut self) {
        self.complete();
    }
}

#[cfg(test)]
#[path = "marker_tests.rs"]
mod tests;
