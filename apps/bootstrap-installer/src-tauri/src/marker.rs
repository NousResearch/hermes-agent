//! The "update in progress" marker: the Desktop launch gate and the
//! cross-process update lock shared with `hermes_cli/update_lock.py` and the
//! Electron gate (`apps/desktop/electron/update-marker.ts`).

use std::path::{Path, PathBuf};

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
/// delegate:<pid> ct:<ct>          (optional; written by `hermes update`)
/// ```
///
/// Line 3 is absent in legacy (v1) markers. The creation time makes the pid
/// a process IDENTITY, so liveness never has to fall back to an age ceiling
/// (a live v2 owner is live however long it runs) and a recycled pid is not
/// mistaken for the original owner.
///
/// The marker is also the cross-process update lock: `hermes update` claims
/// the same file so a dashboard-spawned update and this updater can't mutate
/// one checkout at the same time. `acquire` therefore claims with an atomic
/// exclusive create and REFUSES when a live foreign owner holds it — the
/// pre-fix clobber is what let a dashboard `hermes update` keep running while
/// install-mode bootstrap rewrote the tree underneath it.
pub(crate) struct UpdateMarkerGuard {
    path: PathBuf,
    /// Identity lines (pid, started_at, optional ct) of the claim we hold.
    /// `None` when we hold no claim (the marker could not be written), so
    /// release must not touch whatever is at the path.
    claim: Option<Vec<String>>,
}

/// Age ceiling for LEGACY (v1, no `ct:` line) markers only: without a
/// creation time a live pid may be a recycled one, so age is the only bound.
/// v2 markers have no age ceiling. Matches the v1 rule in
/// apps/desktop/electron/update-marker.ts and hermes_cli/update_lock.py.
const UPDATE_MARKER_MAX_AGE_SECS: u64 = 20 * 60;

/// Creation-time tolerance when matching a recorded `ct:` against the live
/// process (Linux derives it from whole-second btime plus clock ticks).
const MARKER_CT_TOLERANCE_SECS: f64 = 2.0;

/// The pid + age of a confirmed-live update holding the marker.
pub(crate) struct MarkerOwner {
    pub(crate) pid: u32,
    pub(crate) age_secs: u64,
}

/// Parsed marker body.
struct MarkerRecord {
    pid: u32,
    started_at: u64,
    ct: Option<f64>,
    delegate: Option<(u32, Option<f64>)>,
    /// Lines 1-2 plus the `ct:` line when present: what an owner compares
    /// before deleting (a delegate line may come and go underneath it).
    identity: Vec<String>,
}

fn parse_marker(raw: &str) -> Option<MarkerRecord> {
    let lines: Vec<&str> = raw.lines().map(str::trim).collect();
    let pid = lines.first()?.parse::<u32>().ok()?;
    let started_line = lines.get(1).copied().unwrap_or("");
    let started_at = started_line.parse().unwrap_or(0);
    let mut identity = vec![lines[0].to_string(), started_line.to_string()];
    let ct = lines
        .get(2)
        .and_then(|line| line.strip_prefix("ct:"))
        .and_then(|value| value.trim().parse::<f64>().ok());
    if ct.is_some() {
        identity.push(lines[2].to_string());
    }
    // The delegate is line 4; a v1 body has no ct line, so accept it on the
    // first line after the identity block.
    let delegate = lines
        .get(identity.len())
        .and_then(|line| line.strip_prefix("delegate:"))
        .and_then(|rest| {
            let mut parts = rest.split_whitespace();
            let pid = parts.next()?.parse::<u32>().ok()?;
            let ct = parts
                .next()
                .and_then(|token| token.strip_prefix("ct:"))
                .and_then(|value| value.parse::<f64>().ok());
            Some((pid, ct))
        });
    Some(MarkerRecord {
        pid,
        started_at,
        ct,
        delegate,
        identity,
    })
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The process `pid` is alive AND (when a creation time was recorded) is the
/// same process that recorded it. A creation time we cannot probe counts as
/// a match: a live pid is never declared dead on missing evidence.
fn process_identity_alive(pid: u32, recorded_ct: Option<f64>) -> bool {
    if !pid_is_alive(pid) {
        return false;
    }
    match (recorded_ct, process_creation_time(pid)) {
        (Some(recorded), Some(actual)) => (recorded - actual).abs() <= MARKER_CT_TOLERANCE_SECS,
        _ => true,
    }
}

/// The live holder of a parsed marker, if any: the owner (lines 1-3) or,
/// failing that, the delegate (line 4).
fn marker_live_holder(record: &MarkerRecord) -> Option<MarkerOwner> {
    let age_secs = unix_now_secs().saturating_sub(record.started_at);
    let within_v1_ceiling = record.ct.is_some() || age_secs <= UPDATE_MARKER_MAX_AGE_SECS;
    if within_v1_ceiling && process_identity_alive(record.pid, record.ct) {
        return Some(MarkerOwner {
            pid: record.pid,
            age_secs,
        });
    }
    match record.delegate {
        Some((pid, ct)) if process_identity_alive(pid, ct) => Some(MarkerOwner { pid, age_secs }),
        _ => None,
    }
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

/// Delete the marker only if its bytes still equal `expected` — the bytes a
/// verdict was reached on, or the bytes we wrote. A marker replaced since
/// (a new claim, an added delegate line) belongs to someone else's verdict.
/// Returns true when the file is gone afterwards because of us.
fn compare_and_delete(path: &Path, expected: &[u8]) -> bool {
    match std::fs::read(path) {
        Ok(current) if current == expected => match std::fs::remove_file(path) {
            Ok(()) => true,
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(?path, %err, "could not remove update marker");
                }
                false
            }
        },
        _ => false,
    }
}

/// Read the marker and report a live holder, if any. `None` for every "no
/// live update" case — absent, unreadable, malformed, dead owner and
/// delegate, a recycled pid (creation-time mismatch), or a v1 marker past the
/// legacy age ceiling — matching `readLiveUpdateMarker` in the Electron gate.
/// Never panics.
///
/// A marker judged dead is REMOVED here by compare-and-delete (only the exact
/// bytes judged), mirroring `read_live_update` in `hermes_cli/update_lock.py`:
/// a crashed updater whose `Drop` never ran must not wedge every later
/// acquire (#77259).
///
/// Self-PID is returned so `acquire` can adopt the desktop's pre-written claim
/// without refreshing its acquisition time (#74761). A foreign live pid (e.g.
/// a dashboard-spawned `hermes update`) still blocks.
fn live_marker_owner(path: &Path) -> Option<MarkerOwner> {
    let raw = std::fs::read(path).ok()?;
    let live = std::str::from_utf8(&raw)
        .ok()
        .and_then(parse_marker)
        .and_then(|record| marker_live_holder(&record));
    if live.is_none() {
        compare_and_delete(path, &raw);
    }
    live
}

/// True when the on-disk marker names THIS process as its owner.
///
/// A raw read is used instead of `live_marker_owner` on purpose: that
/// helper folds in age and liveness policy (and, since the #74761
/// adoption work, self-ownership handling has changed shape more than
/// once). The exit-2 self-heal below needs exactly one raw fact — does
/// the marker name our PID — because a `hermes update` child that
/// refuses over OUR marker is a handoff-recognition failure in a stale
/// checkout, not a real concurrent update.
fn marker_owned_by_self(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| {
            raw.lines()
                .next()
                .and_then(|line| line.trim().parse::<u32>().ok())
        })
        == Some(std::process::id())
}

/// The exit-2 heal decision (#75788), extracted so the contract is testable.
///
/// True only when BOTH hold: the child exited with the concurrent-update
/// refusal code, AND the on-disk marker names THIS process. That combination
/// means the child refused over its own parent's claim — a stale checkout
/// without handoff recognition — so dropping the claim and retrying once is
/// safe. Any other owner (live foreign updater, garbage, missing marker) or
/// any other exit code must leave the refusal untouched.
pub(crate) fn should_heal_self_marker_refusal(exit_code: Option<i32>, marker_path: &Path) -> bool {
    exit_code == Some(UPDATE_EXIT_CONCURRENT) && marker_owned_by_self(marker_path)
}

/// True when a process with `pid` currently exists.
#[cfg(windows)]
fn pid_is_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            // Either the pid is gone or we lack rights to open it. A pid we
            // can't inspect is treated as dead so an unopenable straggler
            // can't wedge every future update.
            return false;
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

impl UpdateMarkerGuard {
    /// Claim the marker, or report the live updater that already owns it.
    ///
    /// The claim is an atomic exclusive create (`create_new`); an existing
    /// marker is never truncated and overwritten. If one exists and its
    /// holder is dead (or its bytes are unparseable) it is removed by
    /// compare-and-delete and the create is retried ONCE; a live foreign
    /// holder is a refusal. A marker naming our own live process (the
    /// desktop pre-writes our pid, #74761) is adopted verbatim so a retry
    /// cannot reset its age.
    ///
    /// Best-effort only when the marker cannot be CREATED at all (read-only
    /// or missing home dir): the update proceeds with no claim — the gate
    /// degrades to "no marker => proceed", the pre-marker behaviour — and
    /// release is a no-op. That is safe because this marker is the
    /// Desktop-side launch gate; `hermes update` still takes the checkout
    /// lock itself.
    pub(crate) fn acquire(path: PathBuf) -> Result<Self, MarkerOwner> {
        let pid = std::process::id();
        let mut body = format!("{pid}\n{}\n", unix_now_secs());
        if let Some(ct) = process_creation_time(pid) {
            body.push_str(&format!("ct:{ct:.3}\n"));
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut retried = false;
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok(Self::write_claim(path, file, &body)),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(err) => {
                    tracing::warn!(?path, %err, "could not create update-in-progress marker");
                    return Ok(Self { path, claim: None });
                }
            }
            // A dead or unparseable marker is compare-and-deleted by the read.
            match live_marker_owner(&path) {
                Some(owner) if owner.pid == pid => {
                    let claim = std::fs::read_to_string(&path)
                        .ok()
                        .and_then(|raw| parse_marker(&raw))
                        .filter(|record| record.pid == pid)
                        .map(|record| record.identity);
                    return Ok(Self { path, claim });
                }
                Some(owner) => return Err(owner),
                None if retried => {
                    // Lost the race twice: something else is claiming this
                    // path right now. Refuse rather than run unserialized.
                    tracing::warn!(?path, "update marker changed under two claim attempts");
                    return Err(MarkerOwner {
                        pid: 0,
                        age_secs: 0,
                    });
                }
                None => retried = true,
            }
        }
    }

    /// Write our body into the file `create_new` just made for us.
    fn write_claim(path: PathBuf, mut file: std::fs::File, body: &str) -> Self {
        use std::io::Write;
        let written = file.write_all(body.as_bytes()).and_then(|()| file.sync_all());
        drop(file);
        match written {
            Ok(()) => Self {
                claim: parse_marker(body).map(|record| record.identity),
                path,
            },
            Err(err) => {
                tracing::warn!(?path, %err, "could not write update-in-progress marker");
                // Remove our torn file only while it still holds a prefix of
                // what we were writing (a reader may already have reaped it
                // and someone else re-claimed the path).
                if let Ok(current) = std::fs::read(&path) {
                    if body.as_bytes().starts_with(&current) {
                        compare_and_delete(&path, &current);
                    }
                }
                Self { path, claim: None }
            }
        }
    }

    /// Release the marker as soon as every mutating stage has completed.
    ///
    /// The updater still owns a Tauri/Cocoa event loop while it relaunches the
    /// desktop, and that loop can outlive `app.exit(0)`. Relying on `Drop`
    /// alone therefore leaves a *successful* update looking active — a live
    /// pid holding a fresh marker — which blocks desktop startup and every
    /// other updater. Idempotent: `Drop` still runs and tolerates an
    /// already-removed marker.
    ///
    /// Compare-and-delete only: the marker is removed only while its
    /// identity lines (1-3) are still ours. A `delegate:` line naming a LIVE
    /// `hermes update` leaves the marker to that delegate; a dead delegate's
    /// line does not stop us from releasing our own claim.
    pub(crate) fn complete(&self) {
        let Some(claim) = &self.claim else {
            return;
        };
        let Ok(raw) = std::fs::read(&self.path) else {
            return;
        };
        let Some(record) = std::str::from_utf8(&raw).ok().and_then(parse_marker) else {
            return;
        };
        if &record.identity != claim {
            return;
        }
        if let Some((delegate_pid, delegate_ct)) = record.delegate {
            if process_identity_alive(delegate_pid, delegate_ct) {
                return;
            }
        }
        compare_and_delete(&self.path, &raw);
    }
}

impl Drop for UpdateMarkerGuard {
    fn drop(&mut self) {
        self.complete();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_marker_guard_writes_then_removes_on_drop() {
        let dir = unique_tmp_dir("marker-guard");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        {
            let _g = UpdateMarkerGuard::acquire(marker.clone())
                .unwrap_or_else(|_| panic!("no live owner => acquire must succeed"));
            assert!(marker.exists(), "marker must exist while the guard is held");
            let body = std::fs::read_to_string(&marker).unwrap();
            let pid_line = body.lines().next().unwrap();
            assert_eq!(
                pid_line.trim().parse::<u32>().unwrap(),
                std::process::id(),
                "marker records our pid so the desktop can probe liveness"
            );
            assert_eq!(body.lines().count(), 3, "marker is pid + started_at + ct lines");
            assert!(body.ends_with('\n'), "contract C1 bodies end with a newline");
            let ct = parse_marker(&body).and_then(|record| record.ct);
            assert!(ct.is_some(), "a v2 claim records the owner's creation time");
        }

        assert!(
            !marker.exists(),
            "Drop must remove the marker on every exit path (incl. early return / panic unwind)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn update_marker_guard_drop_is_quiet_when_already_gone() {
        let dir = unique_tmp_dir("marker-guard-gone");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("no live owner => acquire must succeed"));
        // Simulate an external cleanup (e.g. the desktop pruned a marker it
        // judged stale) before our guard drops — Drop must not panic.
        std::fs::remove_file(&marker).unwrap();
        drop(guard);

        assert!(!marker.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Spawn a short-lived sibling process whose pid stands in for a foreign
    /// updater. Same-process double-acquire no longer models contention: since
    /// #74761 `acquire` treats our own pid as adoptable (desktop pre-writes it),
    /// so a second acquire in *this* process would succeed.
    fn spawn_foreign_holder() -> std::process::Child {
        #[cfg(windows)]
        {
            std::process::Command::new("timeout")
                .args(["/t", "30", "/nobreak"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn foreign marker holder")
        }
        #[cfg(not(windows))]
        {
            std::process::Command::new("sleep")
                .arg("30")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn foreign marker holder")
        }
    }

    #[test]
    fn acquire_refuses_while_a_live_updater_owns_the_marker() {
        let dir = unique_tmp_dir("marker-contended");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        // A live *foreign* updater holds it. We must NOT clobber the marker and
        // run concurrently over the same checkout — that race is what let a
        // dashboard `hermes update` and install-mode bootstrap mutate one tree
        // at once. Own-pid markers are adoptable (#74761), so the foreign pid
        // must be a real sibling process.
        let mut foreign = spawn_foreign_holder();
        let foreign_pid = foreign.id();
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::fs::write(&marker, format!("{foreign_pid}\n{started_at}")).unwrap();

        let owner = UpdateMarkerGuard::acquire(marker.clone())
            .err()
            .expect("acquire must be refused while a foreign updater is live");
        assert_eq!(owner.pid, foreign_pid);

        // The refused guard must not delete the live owner's marker.
        assert!(marker.exists(), "refused acquire must leave the marker intact");
        let _ = foreign.kill();
        let _ = foreign.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn acquire_adopts_a_marker_prewritten_with_our_own_pid() {
        // #74761: desktop writeUpdateMarker(hermesHome, child.pid) races ahead
        // of UpdateMarkerGuard::acquire. The marker names US; refusing it made
        // every in-app desktop update loop forever. Adopt it without resetting
        // the holder age, so a wedged updater still reaches the stale ceiling.
        let dir = unique_tmp_dir("marker-own-pid");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .saturating_sub(2);
        std::fs::write(&marker, format!("{}\n{started_at}", std::process::id())).unwrap();

        let guard = UpdateMarkerGuard::acquire(marker.clone()).unwrap_or_else(|owner| {
            panic!(
                "own-pid pre-write must be adoptable, got foreign owner pid={}",
                owner.pid
            )
        });
        assert!(marker.exists(), "adopted guard must own the marker");
        let body = std::fs::read_to_string(&marker).unwrap();
        assert_eq!(
            body.lines().next().unwrap().trim().parse::<u32>().unwrap(),
            std::process::id(),
            "acquire keeps the adopted marker owner"
        );
        assert_eq!(
            body.lines().nth(1).unwrap().trim().parse::<u64>().unwrap(),
            started_at,
            "adopting an own-pid marker must preserve its original holder age"
        );
        drop(guard);
        assert!(
            !marker.exists(),
            "Drop must still clear the marker we adopted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- exit-2 self-marker heal (#75788) --------------------------------
    // The deadlock: the updater holds the marker with its own PID; a stale
    // checkout's `hermes update` reads it as a live foreign update and exits
    // 2; the generic retry deliberately skips exit 2 — so the refusal loops
    // forever. These tests pin the heal decision's full contract. On
    // merge-base product code (no heal) the decision function does not exist
    // and the refusal is terminal — the A/B run proves that.

    #[test]
    fn self_owned_marker_plus_exit_2_heals() {
        let dir = unique_tmp_dir("heal-self-owned");
        let marker = dir.join(".hermes-update-in-progress");
        std::fs::write(&marker, format!("{}\n123\n", std::process::id())).unwrap();

        assert!(
            should_heal_self_marker_refusal(Some(UPDATE_EXIT_CONCURRENT), &marker),
            "a child refusing over OUR marker is the #75788 deadlock — must heal"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foreign_owned_marker_never_heals() {
        let dir = unique_tmp_dir("heal-foreign");
        let marker = dir.join(".hermes-update-in-progress");
        // A live sibling process stands in for a genuinely concurrent updater.
        let mut foreign = spawn_foreign_holder();
        std::fs::write(&marker, format!("{}\n123\n", foreign.id())).unwrap();

        assert!(
            !should_heal_self_marker_refusal(Some(UPDATE_EXIT_CONCURRENT), &marker),
            "a foreign owner is a REAL concurrent update — the refusal must stand"
        );
        let _ = foreign.kill();
        let _ = foreign.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_or_garbage_marker_never_heals() {
        let dir = unique_tmp_dir("heal-garbage");
        let missing = dir.join("never-written");
        assert!(
            !should_heal_self_marker_refusal(Some(UPDATE_EXIT_CONCURRENT), &missing),
            "no marker on disk = the child refused over something else entirely"
        );

        let garbage = dir.join(".hermes-update-in-progress");
        std::fs::write(&garbage, "not-a-pid\n123\n").unwrap();
        assert!(
            !should_heal_self_marker_refusal(Some(UPDATE_EXIT_CONCURRENT), &garbage),
            "an unparseable marker must not be treated as ours"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_exit_2_outcomes_never_heal() {
        let dir = unique_tmp_dir("heal-wrong-exit");
        let marker = dir.join(".hermes-update-in-progress");
        std::fs::write(&marker, format!("{}\n123\n", std::process::id())).unwrap();

        for code in [Some(0), Some(1), Some(3), None] {
            assert!(
                !should_heal_self_marker_refusal(code, &marker),
                "heal is exit-2-only; exit {code:?} must keep its normal path"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn heal_end_to_end_marker_lifecycle() {
        // The full deadlock-and-heal sequence with a REAL marker guard, as
        // run_update executes it: acquire (marker written with our pid) →
        // child exits 2 refusing our own claim → heal decision fires →
        // complete() drops the claim → the retry's precondition (no marker,
        // or a marker the child can now claim) holds.
        let dir = unique_tmp_dir("heal-e2e");
        let marker = dir.join(".hermes-update-in-progress");

        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("no live owner => acquire must succeed"));
        assert!(marker.exists(), "updater holds the marker during the child run");

        // Stale child refused over our claim:
        assert!(should_heal_self_marker_refusal(
            Some(UPDATE_EXIT_CONCURRENT),
            &marker
        ));

        // The heal drops the claim exactly as run_update does:
        guard.complete();
        assert!(
            !marker.exists(),
            "claim dropped — the one retry now runs with the marker absent"
        );

        // And with the marker gone the heal can never fire twice (the retry's
        // own exit 2, e.g. a genuinely still-running Hermes, stays terminal).
        assert!(!should_heal_self_marker_refusal(
            Some(UPDATE_EXIT_CONCURRENT),
            &marker
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn acquire_reclaims_a_marker_owned_by_a_dead_pid() {
        let dir = unique_tmp_dir("marker-dead-pid");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        // pid 1 exists everywhere, so fabricate a dead one: a very large pid
        // that no live process owns. A crashed updater must never wedge every
        // future update.
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::fs::write(&marker, format!("4294967294\n{started_at}")).unwrap();

        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("a dead owner must not block acquisition"));
        let body = std::fs::read_to_string(&marker).unwrap();
        assert_eq!(
            body.lines().next().unwrap().trim().parse::<u32>().unwrap(),
            std::process::id(),
            "reclaiming rewrites the marker with our pid"
        );
        drop(guard);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn acquire_reclaims_a_marker_past_the_age_ceiling() {
        let dir = unique_tmp_dir("marker-stale-age");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        // LEGACY v1 marker (no ct line): our own live pid, but started past
        // the ceiling. Without a creation time the pid may be recycled, so
        // age still bounds a v1 claim. (v2 has no ceiling — see
        // v2_live_owner_is_never_aged_out.)
        let long_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .saturating_sub(UPDATE_MARKER_MAX_AGE_SECS + 60);
        std::fs::write(&marker, format!("{}\n{long_ago}", std::process::id())).unwrap();

        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("a marker past the ceiling must be reclaimable"));
        drop(guard);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_marker_owner_removes_stale_marker_with_dead_pid() {
        // The core self-heal of #77259: a marker whose owner is gone must be
        // REMOVED on read (like read_live_update in update_lock.py), not just
        // ignored — otherwise the stale bytes keep failing every acquire
        // until the 20-minute age ceiling expires.
        let dir = unique_tmp_dir("marker-read-dead");
        let marker = dir.join(".hermes-update-in-progress");
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // i32::MAX: beyond every platform's pid_max, and positive even when
        // narrowed to a 32-bit pid_t — unlike 4294967294, which wraps to -2
        // on macOS and probes process group 2 instead of a pid.
        std::fs::write(&marker, format!("2147483647\n{started_at}")).unwrap();

        assert!(live_marker_owner(&marker).is_none());
        assert!(
            !marker.exists(),
            "a dead owner's marker must be self-healed (removed) on read"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_marker_owner_removes_marker_past_the_age_ceiling() {
        let dir = unique_tmp_dir("marker-read-stale-age");
        let marker = dir.join(".hermes-update-in-progress");
        // LEGACY v1 marker (no ct line): our own live pid, but started past
        // the ceiling: age alone must stale a v1 claim, and the stale file
        // must not survive to wedge the next run.
        let long_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .saturating_sub(UPDATE_MARKER_MAX_AGE_SECS + 60);
        std::fs::write(&marker, format!("{}\n{long_ago}", std::process::id())).unwrap();

        assert!(live_marker_owner(&marker).is_none());
        assert!(
            !marker.exists(),
            "past-ceiling marker must be removed on read"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_marker_owner_removes_malformed_marker() {
        // A torn write (garbage pid line) is not a live update either; leaving
        // it would wedge every future acquire the same way a dead pid does.
        let dir = unique_tmp_dir("marker-read-malformed");
        let marker = dir.join(".hermes-update-in-progress");
        std::fs::write(&marker, "not-a-pid\n").unwrap();

        assert!(live_marker_owner(&marker).is_none());
        assert!(
            !marker.exists(),
            "an unparseable marker must not wedge future updates"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_marker_owner_keeps_live_foreign_marker() {
        // The self-heal must NOT delete a live updater's marker — that would
        // let two updaters mutate one checkout concurrently.
        let mut foreign = spawn_foreign_holder();
        let dir = unique_tmp_dir("marker-read-live-foreign");
        let marker = dir.join(".hermes-update-in-progress");
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::fs::write(&marker, format!("{}\n{started_at}", foreign.id())).unwrap();

        let owner = live_marker_owner(&marker).expect("live foreign holder must be reported");
        assert_eq!(owner.pid, foreign.id());
        assert!(
            marker.exists(),
            "a live owner's marker must be left intact (no clobbering)"
        );
        let _ = foreign.kill();
        let _ = foreign.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_marker_owner_keeps_own_live_marker() {
        // #74761: the desktop pre-writes the marker with OUR pid. That claim
        // must be reported (so `acquire` can adopt it without refreshing its
        // age), never deleted as stale — deleting it would break the desktop
        // handoff that pre-claims the lock for us.
        let dir = unique_tmp_dir("marker-read-own-live");
        let marker = dir.join(".hermes-update-in-progress");
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::fs::write(&marker, format!("{}\n{started_at}", std::process::id())).unwrap();

        let owner = live_marker_owner(&marker)
            .expect("our own live pid must be reported for acquire to adopt");
        assert_eq!(owner.pid, std::process::id());
        assert!(
            marker.exists(),
            "our own live marker must be kept for adoption"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pid_is_alive_true_for_self() {
        assert!(pid_is_alive(std::process::id()));
    }

    #[test]
    fn pid_is_alive_false_for_unusable_pid() {
        // i32::MAX is beyond every platform's pid_max, and stays positive
        // when narrowed to a 32-bit pid_t (unlike 4294967294 -> -2 on macOS,
        // which would probe process group 2): it can never name a live pid.
        assert!(!pid_is_alive(2147483647));
    }

    #[test]
    fn pid_is_alive_false_for_pid_zero() {
        // pid 0 means the caller's process GROUP to kill(2), so kill(0, 0)
        // always succeeds. Without the guard, a marker corrupted to "0" would
        // read as a live owner forever.
        assert!(!pid_is_alive(0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pid_is_alive_false_for_zombie() {
        // The kill(pid, 0) false positive behind #77259: a process that has
        // exited but is still in the table as a zombie (parent hasn't reaped
        // it yet) reads as "alive" via signal 0. /proc shows state 'Z', which
        // must count as dead so a crashed updater can't hold the marker past
        // its death.
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                // Child: exit immediately, staying unreaped (a zombie).
                libc::_exit(0);
            }
            // Parent: do NOT waitpid yet — the child must linger as a zombie.
            // Poll until it actually reaches state 'Z' so the assertion below
            // can't race the child's exit.
            let mut became_zombie = false;
            for _ in 0..20 {
                if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                    if let Some(comm_end) = stat.rfind(')') {
                        if stat[comm_end + 1..].split_whitespace().next() == Some("Z") {
                            became_zombie = true;
                            break;
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            assert!(became_zombie, "child never reached zombie state");

            assert!(
                !pid_is_alive(pid as u32),
                "a zombie must not count as a live marker owner"
            );
            // Reap the zombie so the test process doesn't leak children.
            let mut status: libc::c_int = 0;
            libc::waitpid(pid, &mut status, 0);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn pid_is_alive_false_for_zombie() {
        // Same false positive as the Linux branch, probed the macOS way: the
        // child exits, the parent does not reap it, and `ps -o stat=` must
        // report state 'Z' (or 'Z+'), which counts as dead.
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                libc::_exit(0);
            }
            let mut became_zombie = false;
            for _ in 0..20 {
                if let Ok(output) = std::process::Command::new("ps")
                    .arg("-o")
                    .arg("stat=")
                    .arg("-p")
                    .arg(pid.to_string())
                    .output()
                {
                    let state = String::from_utf8_lossy(&output.stdout);
                    if state.trim_start().starts_with('Z') {
                        became_zombie = true;
                        break;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            assert!(became_zombie, "child never reached zombie state");

            assert!(
                !pid_is_alive(pid as u32),
                "a zombie must not count as a live marker owner"
            );
            // Reap the zombie so the test process doesn't leak children.
            let mut status: libc::c_int = 0;
            libc::waitpid(pid, &mut status, 0);
        }
    }

    #[test]
    fn completed_update_releases_marker_before_guard_drop() {
        let dir = unique_tmp_dir("marker-complete");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("no live owner => acquire must succeed"));
        guard.complete();

        assert!(
            !marker.exists(),
            "a successful update must unblock desktop startup before relaunch/exit"
        );
        drop(guard);
        assert!(!marker.exists(), "Drop stays idempotent after completion");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- marker contract C1 (v2: ct identity, CAS claim/delete, delegate) ----

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn v2_body(pid: u32, started_at: u64, ct: f64) -> String {
        format!("{pid}\n{started_at}\nct:{ct:.3}\n")
    }

    fn ct_of(pid: u32) -> f64 {
        process_creation_time(pid).expect("creation-time probe must work on the test host")
    }

    #[test]
    fn v2_live_owner_is_never_aged_out() {
        // V3: a v2 owner that is alive (pid AND creation time match) stays
        // live however long it has run. The old 20-minute ceiling stole the
        // lock from a slow but healthy update.
        let mut foreign = spawn_foreign_holder();
        let dir = unique_tmp_dir("marker-v2-old-live");
        let marker = dir.join(".hermes-update-in-progress");
        let body = v2_body(foreign.id(), now_secs() - 25 * 60, ct_of(foreign.id()));
        std::fs::write(&marker, &body).unwrap();

        let owner = UpdateMarkerGuard::acquire(marker.clone())
            .err()
            .expect("a live v2 owner must not be reclaimed by age");
        assert_eq!(owner.pid, foreign.id());
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), body);
        let _ = foreign.kill();
        let _ = foreign.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn v2_marker_with_mismatched_creation_time_is_reclaimed() {
        // V22: a live pid whose creation time differs from the recorded one is
        // a recycled pid, not the owner — the marker is dead.
        let dir = unique_tmp_dir("marker-v2-ct-mismatch");
        let marker = dir.join(".hermes-update-in-progress");
        let me = std::process::id();
        std::fs::write(&marker, v2_body(me, now_secs(), ct_of(me) + 100.0)).unwrap();

        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("a recycled-pid marker must be reclaimable"));
        let body = std::fs::read_to_string(&marker).unwrap();
        let record = parse_marker(&body).expect("reclaimed marker must parse");
        assert_eq!(record.pid, me);
        let ct = record.ct.expect("our claim records a creation time");
        assert!((ct - ct_of(me)).abs() <= MARKER_CT_TOLERANCE_SECS);
        drop(guard);
        assert!(!marker.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Child half of `second_claimant_process_is_refused_by_create_new`:
    /// claims the marker from a SEPARATE process and reports via exit code.
    #[test]
    #[ignore = "spawned by second_claimant_process_is_refused_by_create_new"]
    fn marker_claim_child_helper() {
        let Some(path) = std::env::var_os("HERMES_TEST_MARKER_CLAIM_PATH") else {
            return;
        };
        let code = match UpdateMarkerGuard::acquire(PathBuf::from(path)) {
            Ok(guard) if guard.claim.is_some() => {
                // Leave the claim on disk, as a still-running owner would.
                std::mem::forget(guard);
                0
            }
            Ok(_) => 4,
            Err(_) => 3,
        };
        std::process::exit(code);
    }

    fn run_claim_child(marker: &Path) -> (i32, u32) {
        let module = module_path!();
        let test_name = format!(
            "{}::marker_claim_child_helper",
            module.split_once("::").map_or(module, |(_, rest)| rest)
        );
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test_name.as_str(), "--exact", "--ignored", "--nocapture"])
            .env("HERMES_TEST_MARKER_CLAIM_PATH", marker)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn claim child");
        let pid = child.id();
        let status = child.wait_with_output().unwrap().status;
        (status.code().unwrap_or(-1), pid)
    }

    #[test]
    fn second_claimant_process_is_refused_by_create_new() {
        // V4: the claim is an exclusive create, so a second updater process
        // racing a fresh claim is refused and never truncates our bytes.
        let dir = unique_tmp_dir("marker-exclusive");
        let marker = dir.join(".hermes-update-in-progress");
        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
        let ours = std::fs::read(&marker).unwrap();

        let (code, _) = run_claim_child(&marker);
        assert_eq!(code, 3, "second claimant must be refused while our claim is live");
        assert_eq!(std::fs::read(&marker).unwrap(), ours, "refusal must not touch our bytes");

        drop(guard);
        assert!(!marker.exists());
        let (code, child_pid) = run_claim_child(&marker);
        assert_eq!(code, 0, "after release the next claimant must succeed");
        assert!(std::fs::read_to_string(&marker)
            .unwrap()
            .starts_with(&format!("{child_pid}\n")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compare_and_delete_spares_changed_bytes() {
        // V4: a delete decided on earlier bytes must not remove a marker that
        // was replaced since (a new claim, or our claim plus a delegate line).
        let dir = unique_tmp_dir("marker-cas");
        let marker = dir.join(".hermes-update-in-progress");
        std::fs::write(&marker, "2147483647\n1\n").unwrap();
        assert!(!compare_and_delete(&marker, b"2147483647\n0\n"));
        assert!(marker.exists(), "changed bytes must survive compare-and-delete");
        assert!(compare_and_delete(&marker, b"2147483647\n1\n"));
        assert!(!marker.exists());

        // The guard's release is compare-and-delete too: a marker whose
        // identity lines are no longer ours is someone else's claim.
        let mut foreign = spawn_foreign_holder();
        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
        let theirs = v2_body(foreign.id(), now_secs(), ct_of(foreign.id()));
        std::fs::write(&marker, &theirs).unwrap();
        drop(guard);
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), theirs);
        let _ = foreign.kill();
        let _ = foreign.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_delegate_keeps_a_dead_owner_marker_live() {
        // Rule 6: `hermes update` running under a claim appends line 4; the
        // marker is live while EITHER the owner or the delegate is.
        let mut foreign = spawn_foreign_holder();
        let dir = unique_tmp_dir("marker-delegate");
        let marker = dir.join(".hermes-update-in-progress");
        let delegate = format!("delegate:{} ct:{:.3}\n", foreign.id(), ct_of(foreign.id()));
        let body = format!("{}{delegate}", v2_body(2147483647, now_secs(), 1.0));
        std::fs::write(&marker, &body).unwrap();

        let owner = live_marker_owner(&marker).expect("live delegate keeps the marker live");
        assert_eq!(owner.pid, foreign.id());
        assert!(UpdateMarkerGuard::acquire(marker.clone()).is_err());
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), body);
        std::fs::remove_file(&marker).unwrap();

        // Our own claim with a live delegate line: release leaves it to the
        // delegate; once the delegate is dead, release removes it.
        let guard = UpdateMarkerGuard::acquire(marker.clone())
            .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
        let mut ours = std::fs::read_to_string(&marker).unwrap();
        ours.push_str(&delegate);
        std::fs::write(&marker, &ours).unwrap();
        guard.complete();
        assert!(marker.exists(), "a live delegate owns the marker after our release");
        let _ = foreign.kill();
        let _ = foreign.wait();
        guard.complete();
        assert!(!marker.exists(), "a dead delegate's line must not strand our claim");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "hermes-marker-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }
}
