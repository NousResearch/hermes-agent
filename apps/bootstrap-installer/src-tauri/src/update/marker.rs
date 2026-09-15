/// RAII guard that owns the "update in progress" marker (see
/// `paths::update_in_progress_marker`). Created at the top of `run_update`;
/// its `Drop` removes the marker on EVERY exit path — success, early
/// `return Err`, or a panic that unwinds through `run_update` — so a crashed
/// or aborted updater can never permanently strand the marker and block
/// future desktop launches. The marker payload is `{pid}\n{started_at_unix}`
/// so the desktop's launch gate can detect a stale marker (dead PID / past a
/// hard ceiling) and self-heal rather than wait forever.
///
/// The marker is also the cross-process update lock: `hermes update` claims
/// the same file (see `hermes_cli/update_lock.py`) so a dashboard-spawned
/// update and this updater can't mutate one checkout at the same time.
/// `acquire` therefore REFUSES when a live foreign owner holds it rather than
/// overwriting — the pre-fix clobber is what let a dashboard `hermes update`
/// keep running while install-mode bootstrap rewrote the tree underneath it.
struct UpdateMarkerGuard {
    path: PathBuf,
    /// False when a live foreign updater already owns the marker: we hold no
    /// claim, so `Drop` must not delete their marker.
    owned: bool,
}

/// Never treat a marker older than this as a live update. Mirrors
/// UPDATE_MARKER_MAX_AGE_MS in apps/desktop/electron/update-marker.ts and
/// UPDATE_MARKER_MAX_AGE_SECONDS in hermes_cli/update_lock.py — all three read
/// this one file, so a shorter ceiling in any of them would steal a lock the
/// others still consider live.
const UPDATE_MARKER_MAX_AGE_SECS: u64 = 20 * 60;

/// The pid + age of a confirmed-live update holding the marker.
struct MarkerOwner {
    pid: u32,
    age_secs: u64,
}

/// Read the marker and report a live owner, if any. `None` for every "no live
/// update" case — absent, unreadable, malformed, dead pid, or past the ceiling
/// — matching `readLiveUpdateMarker` in the Electron gate. Never panics.
///
/// Self-PID is returned so `acquire` can adopt the desktop's pre-written claim
/// without refreshing its acquisition time (#74761). A foreign live pid (e.g.
/// a dashboard-spawned `hermes update`) still blocks.
fn live_marker_owner(path: &Path) -> Option<MarkerOwner> {
    let raw = std::fs::read_to_string(path).ok()?;
    let mut lines = raw.lines();
    let pid: u32 = lines.next()?.trim().parse().ok()?;
    let started_at: u64 = lines.next().unwrap_or("").trim().parse().unwrap_or(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let age_secs = now.saturating_sub(started_at);
    if age_secs > UPDATE_MARKER_MAX_AGE_SECS || !pid_is_alive(pid) {
        return None;
    }
    Some(MarkerOwner { pid, age_secs })
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
fn should_heal_self_marker_refusal(exit_code: Option<i32>, marker_path: &Path) -> bool {
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
    // signal 0 delivers nothing; it only probes existence/permission.
    // ESRCH => dead. EPERM => alive but owned by another user.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

impl UpdateMarkerGuard {
    /// Claim the marker, or report the live updater that already owns it.
    ///
    /// Writing is best-effort: a write failure must NOT abort the update (the
    /// gate degrades to "no marker => proceed", i.e. exactly the pre-marker
    /// behavior), so we log and carry on with a guard that still attempts
    /// cleanup of whatever may exist at the path.
    fn acquire(path: PathBuf) -> Result<Self, MarkerOwner> {
        let pid = std::process::id();
        if let Some(owner) = live_marker_owner(&path) {
            if owner.pid == pid {
                // Repeated acquisition in this process is intentionally
                // re-entrant because the desktop may have pre-written our pid.
                // The desktop races ahead and pre-writes our pid. Adopt that
                // claim verbatim: rewriting started_at here lets retries reset
                // a wedged updater's age before the stale ceiling can clear it.
                return Ok(Self { path, owned: true });
            }
            return Err(owner);
        }
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(err) = std::fs::write(&path, format!("{pid}\n{started_at}")) {
            tracing::warn!(?path, %err, "could not write update-in-progress marker");
        }
        Ok(Self { path, owned: true })
    }

    /// Release the marker as soon as every mutating stage has completed.
    ///
    /// The updater still owns a Tauri/Cocoa event loop while it relaunches the
    /// desktop, and that loop can outlive `app.exit(0)`. Relying on `Drop`
    /// alone therefore leaves a *successful* update looking active — a live
    /// pid holding a fresh marker — which blocks desktop startup and every
    /// other updater for the full age ceiling. Idempotent: `Drop` still runs
    /// and tolerates an already-removed marker.
    fn complete(&self) {
        if !self.owned {
            return;
        }
        if let Err(err) = std::fs::remove_file(&self.path) {
            if err.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = ?self.path, %err, "could not remove completed update marker");
            }
        }
    }
}

impl Drop for UpdateMarkerGuard {
    fn drop(&mut self) {
        self.complete();
    }
}
