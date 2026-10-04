//! `marker.rs` tests (split out to keep the module under the size gate).

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
        assert_eq!(
            body.lines().count(),
            3,
            "marker is pid + started_at + ct lines"
        );
        assert!(
            body.ends_with('\n'),
            "contract C1 bodies end with a newline"
        );
        let ct = parse_marker(body.as_bytes()).and_then(|record| record.ct);
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

    let owner = busy(UpdateMarkerGuard::acquire(marker.clone()));
    assert_eq!(owner.pid, foreign_pid);

    // The refused guard must not delete the live owner's marker.
    assert!(
        marker.exists(),
        "refused acquire must leave the marker intact"
    );
    let _ = foreign.kill();
    let _ = foreign.wait();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn own_pid_v1_prewrite_is_a_previous_incarnation_and_reclaimed() {
    // A7 rule 4 / R12: a no-ct claim naming our pid cannot prove it is THIS process (an old
    // Desktop pre-writes `<our pid>\n<ts>`, but so did any earlier process that had our pid).
    // It is dead: reclaimed under the mutex and replaced by our exact v2 claim.
    let dir = unique_tmp_dir("marker-own-pid");
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join(".hermes-update-in-progress");
    let me = std::process::id();
    std::fs::write(&marker, format!("{me}\n{}", now_secs() - 2)).unwrap();

    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("a previous incarnation's claim never blocks us"));
    let record = parse_marker(&std::fs::read(&marker).unwrap()).unwrap();
    assert_eq!(record.pid, me);
    assert!((record.ct.expect("our claim is v2") - ct_of(me)).abs() <= OWN_CT_EPSILON_SECS);
    drop(guard);
    assert!(!marker.exists(), "Drop clears our claim");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn own_pid_with_a_creation_time_6ms_off_is_not_us() {
    // R12: the 2 s foreign tolerance never applies to our own pid.
    let dir = unique_tmp_dir("marker-own-pid-6ms");
    let marker = dir.join(".hermes-update-in-progress");
    let me = std::process::id();
    std::fs::write(&marker, v2_body(me, now_secs(), ct_of(me) - 0.006)).unwrap();
    assert!(live_marker_owner(&marker).is_none());
    assert!(
        !marker.exists(),
        "a previous incarnation's claim is reclaimed"
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
    let me = std::process::id();
    std::fs::write(&marker, v2_body(me, 123, ct_of(me))).unwrap();

    assert!(
        should_heal_self_marker_refusal(Some(UPDATE_EXIT_CONCURRENT), &marker),
        "a child refusing over OUR marker is the #75788 deadlock — must heal"
    );
    // R12: our pid without our creation time is a previous incarnation's claim, not ours.
    std::fs::write(&marker, format!("{me}\n123\n")).unwrap();
    assert!(!should_heal_self_marker_refusal(
        Some(UPDATE_EXIT_CONCURRENT),
        &marker
    ));
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
    let me = std::process::id();
    std::fs::write(&marker, v2_body(me, 123, ct_of(me))).unwrap();

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
    assert!(
        marker.exists(),
        "updater holds the marker during the child run"
    );

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
    // A claim that is our exact incarnation is reported (so `acquire` adopts it), never
    // deleted as stale.
    let dir = unique_tmp_dir("marker-read-own-live");
    let marker = dir.join(".hermes-update-in-progress");
    let me = std::process::id();
    std::fs::write(&marker, v2_body(me, now_secs(), ct_of(me))).unwrap();

    let owner =
        live_marker_owner(&marker).expect("our own live pid must be reported for acquire to adopt");
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

    // a live v2 owner must not be reclaimed by age
    let owner = busy(UpdateMarkerGuard::acquire(marker.clone()));
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
    let record = parse_marker(body.as_bytes()).expect("reclaimed marker must parse");
    assert_eq!(record.pid, me);
    let ct = record.ct.expect("our claim records a creation time");
    assert!((ct - ct_of(me)).abs() <= MARKER_CT_TOLERANCE_SECS);
    drop(guard);
    assert!(!marker.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Child half of `second_claimant_process_is_refused_by_exclusive_publish`:
/// claims the marker from a SEPARATE process and reports via exit code.
#[test]
#[ignore = "spawned by second_claimant_process_is_refused_by_exclusive_publish"]
fn marker_claim_child_helper() {
    let Some(path) = std::env::var_os("HERMES_TEST_MARKER_CLAIM_PATH") else {
        return;
    };
    let code = match UpdateMarkerGuard::acquire(PathBuf::from(path)) {
        Ok(guard) if guard.claimed => {
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
fn second_claimant_process_is_refused_by_exclusive_publish() {
    // V4: the claim is an exclusive publish, so a second updater process
    // racing a fresh claim is refused and never truncates our bytes.
    let dir = unique_tmp_dir("marker-exclusive");
    let marker = dir.join(".hermes-update-in-progress");
    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
    let ours = std::fs::read(&marker).unwrap();

    let (code, _) = run_claim_child(&marker);
    assert_eq!(
        code, 3,
        "second claimant must be refused while our claim is live"
    );
    assert_eq!(
        std::fs::read(&marker).unwrap(),
        ours,
        "refusal must not touch our bytes"
    );

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
    assert!(
        marker.exists(),
        "changed bytes must survive compare-and-delete"
    );
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

    // A7 rule 5: our release with a LIVE delegate hands the claim to it (rewritten with the
    // delegate as owner, started_at and run line kept); a dead delegate's line never strands
    // our claim.
    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
    let mut ours = std::fs::read_to_string(&marker).unwrap();
    let started = ours.lines().nth(1).unwrap().to_string();
    ours.push_str(&delegate);
    ours.push_str("run:desk-1\n");
    std::fs::write(&marker, &ours).unwrap();
    guard.complete();
    assert_eq!(
        std::fs::read_to_string(&marker).unwrap(),
        format!(
            "{}\n{started}\nct:{:.3}\nrun:desk-1\n",
            foreign.id(),
            ct_of(foreign.id())
        ),
        "the live delegate inherits the claim"
    );
    guard.complete();
    assert!(
        marker.exists(),
        "a handed-over claim is no longer ours to release"
    );
    let _ = foreign.kill();
    let _ = foreign.wait();
    std::fs::remove_file(&marker).unwrap();

    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
    let mut ours = std::fs::read_to_string(&marker).unwrap();
    ours.push_str(&delegate); // the delegate is dead now
    std::fs::write(&marker, &ours).unwrap();
    guard.complete();
    assert!(
        !marker.exists(),
        "a dead delegate's line must not strand our claim"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- LP-LOCK round 2: shared parse / liveness / claim / litter rules ----

#[test]
fn parse_matrix_follows_the_shared_contract() {
    let parse = |body: &str| parse_marker(body.as_bytes());
    for body in [
        "\u{feff}123\n456\nct:1.5\n",
        "123\r\n456\r\nct:1.5\r\n",
        " 123\t\n\t456 \n ct:1.5 \n",
    ] {
        let record = parse(body).unwrap_or_else(|| panic!("{body:?} must parse"));
        assert_eq!(
            (record.pid, record.started_at, record.ct),
            (123, 456, Some(1.5)),
            "{body:?}"
        );
        assert_eq!(record.identity, ["123", "456", "ct:1.5"], "{body:?}");
    }
    for malformed in [
        "123\n1700000000.5\n", // fractional started_at
        "123\nabc\n",          // garbage line 2
        "123\n\n",             // empty line 2
        "123\n",               // missing line 2
        "123",
        "1_0\n456\n",
        "+5\n456\n",
        "\u{663}\n456\n", // non-ASCII digit
        "",
    ] {
        assert!(
            parse(malformed).is_none(),
            "{malformed:?} must be malformed"
        );
    }
    for v1 in [
        "123\n456\nct:abc\n",
        "123\n456\nct: 1.5\n",
        "123\n456\nct:1.\n",
        "123\n456\n",
    ] {
        let record = parse(v1).unwrap_or_else(|| panic!("{v1:?} must parse"));
        assert_eq!(record.ct, None, "{v1:?} is a v1 marker");
        assert_eq!(record.identity.len(), 2, "{v1:?}");
    }
    // A delegate is read from line 4 only, in its exact shape.
    let line3 = parse("123\n456\ndelegate:5 ct:1.0\n").unwrap();
    assert_eq!(
        (line3.ct, line3.delegate),
        (None, None),
        "line 3 is never a delegate"
    );
    assert_eq!(
        parse("123\n456\nct:1.5\ndelegate:5 ct:1.0\n")
            .unwrap()
            .delegate,
        Some((5, 1.0))
    );
    for bad in [
        "delegate:5  ct:1.0",
        "delegate:5 ct:abc",
        "delegate:5",
        "delegate:x ct:1.0",
    ] {
        let body = format!("123\n456\nct:1.5\n{bad}\n");
        assert_eq!(parse(&body).unwrap().delegate, None, "{bad:?}");
    }
}

#[test]
fn malformed_started_at_marker_is_dead_and_deleted() {
    // Our own live pid, but a fractional line 2: malformed, so dead.
    let dir = unique_tmp_dir("marker-fractional");
    let marker = dir.join(".hermes-update-in-progress");
    std::fs::write(
        &marker,
        format!("{}\n{}.5\n", std::process::id(), now_secs()),
    )
    .unwrap();
    assert!(live_marker_owner(&marker).is_none());
    assert!(
        !marker.exists(),
        "a malformed marker is compare-and-deleted"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn world_with<'a>(
    now: u64,
    alive: &'a dyn Fn(u32) -> bool,
    ct_of: &'a dyn Fn(u32) -> Option<f64>,
) -> World<'a> {
    let me = std::process::id();
    World {
        pid: me,
        ct: process_creation_time(me),
        now,
        alive,
        ct_of,
    }
}

#[test]
fn unreadable_creation_time_gets_the_v1_age_ceiling() {
    // A1: a live foreign pid whose creation time cannot be probed is live only within the v1
    // ceiling; a confirmed match has no age limit.
    let mut foreign = spawn_foreign_holder();
    let pid = foreign.id();
    let started = now_secs() - 30 * 60;
    let record = parse_marker(v2_body(pid, started, ct_of(pid)).as_bytes()).unwrap();
    let alive = |pid: u32| pid_is_alive(pid);
    let unreadable = |_: u32| None;
    let at = |mins: u64| started + mins * 60;
    let holder = |now, probe: &dyn Fn(u32) -> Option<f64>| {
        marker_live_holder(&record, &world_with(now, &alive, probe)).map(|owner| owner.pid)
    };
    assert_eq!(
        holder(at(5), &unreadable),
        Some(pid),
        "5 min, ct unreadable: live"
    );
    assert_eq!(
        holder(at(25), &unreadable),
        None,
        "25 min, ct unreadable: dead"
    );
    let matching = |pid: u32| process_creation_time(pid);
    assert_eq!(
        holder(at(25), &matching),
        Some(pid),
        "matching ct: live at any age"
    );
    let mismatched = |pid: u32| process_creation_time(pid).map(|ct| ct + 100.0);
    assert_eq!(
        holder(at(5), &mismatched),
        None,
        "ct mismatch: recycled pid"
    );

    // The delegate is aged by the same marker age.
    let delegated = format!(
        "2147483647\n{started}\nct:1.000\ndelegate:{pid} ct:{:.3}\n",
        ct_of(pid)
    );
    let record = parse_marker(delegated.as_bytes()).unwrap();
    let delegate_holder = |now| {
        marker_live_holder(&record, &world_with(now, &alive, &unreadable)).map(|owner| owner.pid)
    };
    assert_eq!(delegate_holder(at(5)), Some(pid));
    assert_eq!(delegate_holder(at(25)), None);
    let _ = foreign.kill();
    let _ = foreign.wait();
}

#[test]
fn fresh_empty_marker_is_a_claim_in_flight() {
    // A3: a 0-byte marker younger than 5 s is a claimant between its
    // exclusive create and its write — live, and never deleted.
    let dir = unique_tmp_dir("marker-empty-fresh");
    let marker = dir.join(".hermes-update-in-progress");
    std::fs::write(&marker, "").unwrap();
    assert_eq!(busy(UpdateMarkerGuard::acquire(marker.clone())).pid, 0);
    assert_eq!(
        std::fs::read(&marker).unwrap(),
        b"",
        "a fresh empty marker is not deleted"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn old_empty_marker_is_reclaimed_and_claim_leaves_no_tmp() {
    let dir = unique_tmp_dir("marker-empty-old");
    let marker = dir.join(".hermes-update-in-progress");
    let file = std::fs::File::create(&marker).unwrap();
    file.set_modified(SystemTime::now() - Duration::from_secs(10))
        .unwrap();
    drop(file);
    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("a 10 s old empty marker is dead"));
    let body = std::fs::read_to_string(&marker).unwrap();
    assert!(body.starts_with(&format!("{}\n", std::process::id())));
    let litter: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(
        litter.is_empty(),
        "the publish tmp file must be removed: {litter:?}"
    );
    drop(guard);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn own_exact_v2_claim_is_adopted_verbatim_and_released() {
    // Our exact incarnation's claim (any age) is adopted byte-for-byte; complete() releases it.
    let me = std::process::id();
    let dir = unique_tmp_dir("marker-own-v2");
    let marker = dir.join(".hermes-update-in-progress");
    let body = v2_body(me, now_secs() - 25 * 60, ct_of(me));
    std::fs::write(&marker, &body).unwrap();
    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("own v2 marker must be adopted"));
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), body);
    drop(guard);
    assert!(!marker.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

fn unwritable_message(result: Result<UpdateMarkerGuard, AcquireError>) -> String {
    match result {
        Err(AcquireError::Unwritable(msg)) => msg,
        Err(AcquireError::Busy(owner)) => {
            panic!("expected Unwritable, got Busy pid {}", owner.pid)
        }
        Ok(_) => panic!("expected Unwritable, acquire succeeded"),
    }
}

#[test]
fn marker_under_a_regular_file_is_unwritable() {
    // m8: never proceed unclaimed.
    let dir = unique_tmp_dir("marker-under-file");
    let not_a_dir = dir.join("home");
    std::fs::write(&not_a_dir, "x").unwrap();
    let marker = not_a_dir.join(".hermes-update-in-progress");
    let msg = unwritable_message(UpdateMarkerGuard::acquire(marker.clone()));
    assert!(msg.starts_with(&format!(
        "Cannot lock this install for the update: {} is not writable (",
        marker.display()
    )));
    assert!(msg.ends_with("). Run the update as the user that owns the install."));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn marker_in_a_read_only_dir_is_unwritable() {
    use std::os::unix::fs::PermissionsExt;
    if unsafe { libc::geteuid() } == 0 {
        return; // root ignores directory permissions
    }
    // No marker yet, and a dead marker that cannot be removed: both refuse.
    for existing in [None, Some(format!("2147483647\n{}\n", now_secs()))] {
        let dir = unique_tmp_dir("marker-read-only");
        let marker = dir.join(".hermes-update-in-progress");
        if let Some(body) = &existing {
            std::fs::write(&marker, body).unwrap();
        }
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let result = UpdateMarkerGuard::acquire(marker.clone());
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            unwritable_message(result).contains("is not writable"),
            "{existing:?}"
        );
        assert_eq!(std::fs::read_to_string(&marker).ok(), existing);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn claim_sweeps_dead_claimants_tmp_litter() {
    // m10: a claimant that died mid-publish leaves its tmp sibling.
    let dir = unique_tmp_dir("marker-litter");
    let marker = dir.join(".hermes-update-in-progress");
    let me = std::process::id();
    let dead = [
        ".hermes-update-in-progress.2147483647.123.tmp",
        ".hermes-update-in-progress.2147483647.tmp",
    ];
    let kept = [
        format!(".hermes-update-in-progress.{me}.123.tmp"),
        "other.2147483647.123.tmp".to_string(),
    ];
    for name in dead
        .iter()
        .map(|n| n.to_string())
        .chain(kept.iter().cloned())
    {
        std::fs::write(dir.join(name), "").unwrap();
    }
    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("fresh acquire must succeed"));
    for name in dead {
        assert!(!dir.join(name).exists(), "{name} belongs to a dead pid");
    }
    for name in &kept {
        assert!(dir.join(name).exists(), "{name} must be kept");
    }
    drop(guard);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The live owner a refused acquire reports; panics on any other outcome.
fn busy(result: Result<UpdateMarkerGuard, AcquireError>) -> MarkerOwner {
    match result {
        Err(AcquireError::Busy(owner)) => owner,
        Err(AcquireError::Unwritable(msg)) => panic!("expected Busy, got Unwritable: {msg}"),
        Ok(_) => panic!("expected Busy, acquire succeeded"),
    }
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

// ---- A7: marker mutex (rule 1) and the shared corpus (rule 7) ----

/// Child half of `marker_mutations_wait_for_the_mutex_another_process_holds`: holds the
/// marker mutex from a SEPARATE process for 1.5 s.
#[test]
#[ignore = "spawned by marker_mutations_wait_for_the_mutex_another_process_holds"]
fn marker_mutex_child_helper() {
    let Some(path) = std::env::var_os("HERMES_TEST_MARKER_MUTEX_PATH") else {
        return;
    };
    let path = PathBuf::from(path);
    let _mutex = lock_marker(&path).expect("child takes the free mutex");
    std::fs::write(path.with_extension("held"), b"").unwrap();
    std::thread::sleep(Duration::from_millis(1500));
}

#[test]
fn marker_mutations_wait_for_the_mutex_another_process_holds() {
    // R3: a reclaim decided on stale bytes cannot delete a claim published after its read,
    // because every decision-and-mutation runs under one kernel lock. Here another process
    // holds that lock (as a paused reader would): our reclaim of a dead marker waits for it.
    let dir = unique_tmp_dir("marker-mutex");
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join(".hermes-update-in-progress");
    std::fs::write(&marker, "0\n0\n").unwrap();
    let module = module_path!();
    let test_name = format!(
        "{}::marker_mutex_child_helper",
        module.split_once("::").map_or(module, |(_, rest)| rest)
    );
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test_name.as_str(), "--exact", "--ignored", "--nocapture"])
        .env("HERMES_TEST_MARKER_MUTEX_PATH", &marker)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn mutex child");
    let held = marker.with_extension("held");
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !held.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "mutex child never held the lock"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let started = std::time::Instant::now();
    let guard = UpdateMarkerGuard::acquire(marker.clone())
        .unwrap_or_else(|_| panic!("the dead marker is reclaimed once the mutex is free"));
    assert!(
        started.elapsed() >= Duration::from_millis(900),
        "the reclaim ran while another process held the marker mutex"
    );
    assert_eq!(
        parse_marker(&std::fs::read(&marker).unwrap()).unwrap().pid,
        std::process::id()
    );
    assert!(
        marker_mutex_path(&marker).exists(),
        "the sidecar is never deleted"
    );
    drop(guard);
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
}

const CORPUS: &str = include_str!("../../../../tests/fixtures/update_marker_corpus.json");

fn corpus_world<'a>(
    corpus: &serde_json::Value,
    case: &serde_json::Value,
    pid_key: &str,
    ct_key: &str,
    table: &'a std::collections::HashMap<u32, Option<f64>>,
    alive: &'a dyn Fn(u32) -> bool,
    ct_of: &'a dyn Fn(u32) -> Option<f64>,
) -> World<'a> {
    let pid = case
        .get(pid_key)
        .unwrap_or(&corpus["our_pid"])
        .as_u64()
        .unwrap() as u32;
    let ct = case.get(ct_key).unwrap_or(&corpus["our_ct"]).as_f64();
    let _ = table;
    World {
        pid,
        ct,
        now: corpus["now"].as_u64().unwrap(),
        alive,
        ct_of,
    }
}

fn corpus_table(
    corpus: &serde_json::Value,
    case: &serde_json::Value,
    pid_key: &str,
    ct_key: &str,
) -> std::collections::HashMap<u32, Option<f64>> {
    let mut table: std::collections::HashMap<u32, Option<f64>> = case["live"]
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|(pid, ct)| Some((pid.parse::<u32>().ok()?, ct.as_f64())))
        .collect();
    let me = case
        .get(pid_key)
        .unwrap_or(&corpus["our_pid"])
        .as_u64()
        .unwrap() as u32;
    table.insert(me, case.get(ct_key).unwrap_or(&corpus["our_ct"]).as_f64());
    table
}

#[test]
fn shared_marker_corpus_judge_cases() {
    let corpus: serde_json::Value = serde_json::from_str(CORPUS).unwrap();
    assert_eq!(corpus["own_ct_epsilon"].as_f64(), Some(OWN_CT_EPSILON_SECS));
    assert_eq!(
        corpus["ct_tolerance"].as_f64(),
        Some(MARKER_CT_TOLERANCE_SECS)
    );
    assert_eq!(
        corpus["v1_max_age"].as_u64(),
        Some(UPDATE_MARKER_MAX_AGE_SECS)
    );
    let cases = corpus["judge"].as_array().unwrap();
    assert!(cases.len() >= 30);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let table = corpus_table(&corpus, case, "our_pid", "our_ct");
        let alive = |pid: u32| pid != 0 && table.contains_key(&pid);
        let ct_of = |pid: u32| table.get(&pid).copied().flatten();
        let world = corpus_world(&corpus, case, "our_pid", "our_ct", &table, &alive, &ct_of);
        let raw = case["text"].as_str().unwrap().as_bytes();
        let (verdict, owner, run) = match parse_marker(raw) {
            None => ("malformed", None, None),
            Some(record) => {
                let run = record.runs.first().cloned();
                match marker_live_holder(&record, &world) {
                    None => ("dead", None, run),
                    Some(holder) => {
                        let age = world.now.saturating_sub(record.started_at);
                        let ours = (record.pid == world.pid
                            && identity_live(record.pid, record.ct, age, &world))
                            || record.delegate.is_some_and(|(pid, ct)| {
                                pid == world.pid && identity_live(pid, Some(ct), age, &world)
                            });
                        (if ours { "ours" } else { "live" }, Some(holder.pid), run)
                    }
                }
            }
        };
        let expect = &case["expect"];
        assert_eq!(
            verdict,
            expect["verdict"].as_str().unwrap(),
            "{name}: verdict"
        );
        assert_eq!(
            owner.map(u64::from),
            expect["owner"].as_u64(),
            "{name}: owner"
        );
        assert_eq!(run.as_deref(), expect["run"].as_str(), "{name}: run");
    }
}

#[test]
fn shared_marker_corpus_release_cases() {
    let corpus: serde_json::Value = serde_json::from_str(CORPUS).unwrap();
    for case in corpus["release"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let table = corpus_table(&corpus, case, "releaser_pid", "releaser_ct");
        let alive = |pid: u32| pid != 0 && table.contains_key(&pid);
        let ct_of = |pid: u32| table.get(&pid).copied().flatten();
        let world = corpus_world(
            &corpus,
            case,
            "releaser_pid",
            "releaser_ct",
            &table,
            &alive,
            &ct_of,
        );
        let expect = &case["expect"];
        let got = release_decision(case["text"].as_str().unwrap().as_bytes(), &world);
        let want = match expect["action"].as_str().unwrap() {
            "keep" => Release::Keep,
            "delete" => Release::Delete,
            "rewrite" => Release::Rewrite(expect["text"].as_str().unwrap().as_bytes().to_vec()),
            other => panic!("{name}: unknown action {other}"),
        };
        assert_eq!(got, want, "{name}");
    }
}
