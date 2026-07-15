//! Update orchestration.
//!
//! Driven when the installer is launched as `Hermes-Setup.exe --update` (see
//! `AppMode` in lib.rs). The desktop app hands off to us — it exits, then we:
//!
//!   1. wait for the old Hermes desktop process to fully exit (so both the
//!      venv shim and packaged app.asar are free; otherwise `hermes update`
//!      or repair bootstrap can race locked files),
//!   2. run `hermes update --yes --gateway` (Python/repo update; this does NOT
//!      rebuild apps/desktop by design — see cmd_update in hermes_cli/main.py),
//!   3. run `hermes desktop --build-only` (the rebuild step update skips),
//!   4. launch the freshly-built desktop (reuses bootstrap::launch logic).
//!
//! We reuse the `BootstrapEvent` channel + the existing progress UI by
//! emitting a synthetic multi-stage manifest (handoff → update → rebuild, plus
//! an install stage on macOS). To the frontend an update looks like a short
//! bootstrap, broken into the real operations run_update performs so the user
//! sees discrete steps (with the live log underneath) instead of one bar.
//!
//! Cross-platform note: `hermes update` already handles macOS/Linux (git/pip).
//! The only OS-specific bits here are the venv shim path (resolve_hermes) and
//! the no-window creation flag — both already cfg-gated. Keep new logic
//! OS-agnostic so the mac/linux port stays "fill in the paths".

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::events::{BootstrapEvent, LogStream, StageInfo, StageState};

/// `hermes update` exit code meaning "another hermes process is holding the
/// venv shim open / dirty precondition" — see _cmd_update_impl in
/// hermes_cli/main.py (sys.exit(2)). We surface a targeted message for this.
const UPDATE_EXIT_CONCURRENT: i32 = 2;

/// How long to wait for the old desktop process to release files under the
/// install tree before giving up and letting `hermes update`'s own guard decide.
const DESKTOP_EXIT_WAIT: Duration = Duration::from_secs(20);
const DESKTOP_EXIT_POLL: Duration = Duration::from_millis(500);

/// Guards against concurrent update runs. The frontend kicks `startUpdate()`
/// from a mount effect, which can fire more than once (React strict-mode
/// double-invokes effects in dev; a window reload or stray re-init can do it
/// in prod). Two `run_update` tasks racing on `git stash` corrupt the working
/// tree — one stashes the changes the other then can't find. Exactly one task
/// may hold this flag at a time.
static UPDATE_RUNNING: AtomicBool = AtomicBool::new(false);

/// Frontend → Rust: kick off the update flow. Mirrors `start_bootstrap`'s
/// fire-and-forget shape; progress arrives on the `bootstrap` event channel.
#[tauri::command]
pub async fn start_update(app: AppHandle) -> Result<(), String> {
    // Re-entrancy guard (see UPDATE_RUNNING). compare_exchange lets exactly one
    // caller flip false→true; any concurrent caller no-ops instead of spawning
    // a second racing update.
    if UPDATE_RUNNING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        // Already running: re-emit the manifest so a duplicate startUpdate()
        // call (which resets the frontend store) can recover its stage list.
        let target_app = if cfg!(target_os = "macos") {
            target_app_from_args(std::env::args().skip(1))
        } else {
            None
        };
        emit(
            &app,
            BootstrapEvent::Manifest {
                stages: update_stages(target_app.is_some()),
                protocol_version: None,
            },
        );
        return Ok(());
    }
    tokio::spawn(async move {
        if let Err(err) = run_update(app.clone()).await {
            // run_update already emits a Failed event on the paths that matter;
            // this catches anything that escaped. Emit defensively.
            emit(
                &app,
                BootstrapEvent::Failed {
                    stage: None,
                    error: format!("{err:#}"),
                },
            );
        }
        UPDATE_RUNNING.store(false, Ordering::SeqCst);
    });
    Ok(())
}

/// RAII guard that owns the "update in progress" marker (see
/// `paths::update_in_progress_marker`). Created at the top of `run_update`;
/// its `Drop` removes the marker on EVERY exit path — success, early
/// `return Err`, or a panic that unwinds through `run_update` — so a crashed
/// or aborted updater can never permanently strand the marker and block
/// future desktop launches. The marker payload is `{pid}\n{started_at_unix}`
/// so the desktop's launch gate can detect a stale marker (dead PID / past a
/// hard ceiling) and self-heal rather than wait forever.
struct UpdateMarkerGuard {
    path: PathBuf,
}

impl UpdateMarkerGuard {
    /// Write the marker. Best-effort: a write failure must NOT abort the
    /// update (the gate degrades to "no marker => proceed", i.e. exactly the
    /// pre-fix behavior), so we log and carry on with a guard that still
    /// attempts cleanup of whatever may exist at the path.
    fn acquire(path: PathBuf) -> Self {
        let pid = std::process::id();
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
        Self { path }
    }
}

impl Drop for UpdateMarkerGuard {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.path) {
            if err.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = ?self.path, %err, "could not remove update-in-progress marker");
            }
        }
    }
}

async fn run_update(app: AppHandle) -> Result<()> {
    let hermes_home = crate::paths::hermes_home();
    let install_root = hermes_home.join("hermes-agent");

    // Mutual exclusion (#50238): publish an "update in progress" marker for the
    // entire duration of this update. A desktop instance the user relaunches
    // mid-update consults this before spawning its own local backend — without
    // it, that backend re-locks the venv shim, our `force_kill_other_hermes`
    // straggler-cleanup kills it, and the relaunch/kill cycle loops. The guard
    // removes the marker on every exit path (incl. early returns / panics).
    let _update_marker = UpdateMarkerGuard::acquire(crate::paths::update_in_progress_marker());

    let update_branch = update_branch_from_args(std::env::args().skip(1))
        .or_else(|| option_env_string("BUILD_PIN_BRANCH"))
        .unwrap_or_else(|| "main".to_string());
    let target_app = if cfg!(target_os = "macos") {
        target_app_from_args(std::env::args().skip(1))
    } else {
        None
    };

    let hermes = resolve_hermes(&install_root).ok_or_else(|| {
        let msg = format!(
            "Could not find the hermes CLI under {}. Is Hermes installed? \
             Re-run the installer to repair the install.",
            install_root.display()
        );
        emit(
            &app,
            BootstrapEvent::Failed {
                stage: None,
                error: msg.clone(),
            },
        );
        anyhow!(msg)
    })?;

    // Synthetic manifest so the existing progress UI renders our stages.
    emit(
        &app,
        BootstrapEvent::Manifest {
            stages: update_stages(target_app.is_some()),
            protocol_version: None,
        },
    );

    // ---- stage 1: wait for the old desktop to die ------------------------
    // The desktop exec'd us then called app.exit(), but process teardown is
    // async on Windows. If it still holds the venv shim, `hermes update`
    // aborts with exit 2. If it still holds the packaged app.asar,
    // install.ps1's repair/re-clone path cannot move/remove the install tree.
    // Give both handles a bounded window to clear. Surfaced as its own stage
    // (rather than a silent pre-step) so a slow close / force-kill reads as
    // real progress instead of a frozen first bar.
    let started = Instant::now();
    emit_stage(&app, "handoff", StageState::Running, None, None);
    wait_for_install_locks_free(&install_root, &app, "handoff").await;
    emit_stage(
        &app,
        "handoff",
        StageState::Succeeded,
        Some(started.elapsed().as_millis() as u64),
        None,
    );

    // ---- stage 2: hermes update -----------------------------------------
    // Pass --branch so `hermes update` targets the branch this installer was
    // built/pinned against (BUILD_PIN_BRANCH), NOT its built-in default of
    // `main`. The install was a detached-HEAD checkout of a specific commit;
    // without --branch, `hermes update` switches the checkout to `main` (a
    // divergent branch that may not even have the desktop CLI command), then
    // reports "already up to date" against the wrong branch. The desktop
    // detected the update against this same branch, so we must update against
    // it too.
    emit_log(
        &app,
        Some("update"),
        LogStream::Stdout,
        &format!("[update] updating against branch {update_branch}"),
    );
    let child_env = update_child_env(&install_root);
    let mut update_args: Vec<String> =
        vec!["update".into(), "--yes".into(), "--gateway".into()];
    // --force skips `hermes update`'s Windows running-exe guard (which would
    // `sys.exit(2)` and dead-end the handoff). By contract the desktop has
    // already exited and waited for the install locks to clear before launching
    // us, and wait_for_install_locks_free below force-kills any straggler — so by the
    // time `hermes update` runs there is no legitimate hermes.exe to protect,
    // and the guard would only produce a false "Hermes is still running" stop.
    //
    // NOTE: --force does NOT bypass the venv-python holder guard (that needs
    // an explicit `--force-venv`, which we deliberately do not pass). Our lock
    // probe only checks the hermes.exe shim and app.asar, so an external venv
    // python holding a native .pyd (a user terminal, an unmanaged gateway)
    // could still be alive here — mutating the venv under it would strand the
    // install half-updated. If that guard fires, it exits 2 and the match arm
    // below surfaces the correct "close all Hermes windows" message.
    update_args.push("--force".into());
    update_args.push("--branch".into());
    update_args.push(update_branch);

    emit_stage(&app, "update", StageState::Running, None, None);
    let started = Instant::now();
    let mut update = run_streamed(
        &app,
        &hermes,
        &update_args,
        &install_root,
        &child_env,
        Some("update"),
    )
    .await?;

    // Retry-once for the update-boundary crash. `hermes update` lazily imports
    // the FRESHLY PULLED modules, but the dependency-install step still runs the
    // already-in-memory pre-pull code for one invocation. A release that changed
    // an updater-path contract across that boundary (e.g. #39780's `_UvResult`,
    // whose `__iter__` injected a bool into the argv and crashed Windows
    // `list2cmdline` with `TypeError: sequence item 1: expected str instance,
    // bool found`, fixed in #39820) therefore kills the FIRST update on the
    // parked population — even though the fix is already on disk by then. A
    // second `hermes update` runs clean because the now-current module is loaded
    // from the start. Rather than make the parked user click Update twice (and
    // stare at a scary crash first), retry once automatically. Skip the retry
    // for the concurrent-instance guard (exit 2) — that's a "close Hermes" state
    // a retry can't fix.
    if !matches!(update.exit_code, Some(0) | Some(UPDATE_EXIT_CONCURRENT)) {
        emit_log(
            &app,
            Some("update"),
            LogStream::Stdout,
            "[update] first update attempt failed; retrying once (the fix it just \
             pulled loads on the second run)…",
        );
        update = run_streamed(
            &app,
            &hermes,
            &update_args,
            &install_root,
            &child_env,
            Some("update"),
        )
        .await?;
    }
    let update_ms = started.elapsed().as_millis() as u64;

    match update.exit_code {
        Some(0) => {
            emit_stage(&app, "update", StageState::Succeeded, Some(update_ms), None);
        }
        Some(code) if code == UPDATE_EXIT_CONCURRENT => {
            let msg = "Hermes is still running. Close all Hermes windows and try \
                       the update again."
                .to_string();
            emit_stage(
                &app,
                "update",
                StageState::Failed,
                Some(update_ms),
                Some(msg.clone()),
            );
            emit(
                &app,
                BootstrapEvent::Failed {
                    stage: Some("update".into()),
                    error: msg.clone(),
                },
            );
            return Err(anyhow!(msg));
        }
        other => {
            let msg = format!(
                "hermes update failed (exit {:?}). See {} for details.",
                other,
                crate::paths::hermes_home()
                    .join("logs")
                    .join("update.log")
                    .display()
            );
            emit_stage(
                &app,
                "update",
                StageState::Failed,
                Some(update_ms),
                Some(msg.clone()),
            );
            emit(
                &app,
                BootstrapEvent::Failed {
                    stage: Some("update".into()),
                    error: msg.clone(),
                },
            );
            return Err(anyhow!(msg));
        }
    }

    // ---- stage 3: hermes desktop --build-only ----------------------------
    // `hermes update` deliberately does NOT build apps/desktop (it installs
    // repo-root deps with --workspaces=false). This is the rebuild it skips.
    emit_stage(&app, "rebuild", StageState::Running, None, None);
    let started = Instant::now();
    let rebuild_args: Vec<String> = vec!["desktop".into(), "--build-only".into()];
    let mut rebuild = run_streamed(
        &app,
        &hermes,
        &rebuild_args,
        &install_root,
        &child_env,
        Some("rebuild"),
    )
    .await?;

    // Retry-once: the first `--build-only` can return nonzero on a still-settling
    // post-update tree or a network-blocked Electron fetch that our self-heal
    // repaired mid-run. A second attempt then builds clean off the healed dist
    // (the content-hash stamp makes it a near-no-op when the first actually
    // succeeded). Without this the updater bails here and never reaches the
    // relaunch below — the app updates but doesn't restart. Matches the
    // retry-once `hermes update` already does above, and `hermes update`'s own
    // desktop rebuild in cmd_update.
    if rebuild_needs_retry(rebuild.exit_code) {
        emit_log(
            &app,
            Some("rebuild"),
            LogStream::Stdout,
            "[rebuild] first desktop rebuild failed; retrying once (a self-healed \
             Electron download builds clean on the second run)…",
        );
        rebuild = run_streamed(
            &app,
            &hermes,
            &rebuild_args,
            &install_root,
            &child_env,
            Some("rebuild"),
        )
        .await?;
    }
    let rebuild_ms = started.elapsed().as_millis() as u64;

    if rebuild.exit_code != Some(0) {
        let msg = format!(
            "Rebuilding the desktop app failed (exit {:?}). The update was \
             applied but the app could not be rebuilt; run `hermes desktop` \
             from a terminal to see the error.",
            rebuild.exit_code
        );
        emit_stage(
            &app,
            "rebuild",
            StageState::Failed,
            Some(rebuild_ms),
            Some(msg.clone()),
        );
        emit(
            &app,
            BootstrapEvent::Failed {
                stage: Some("rebuild".into()),
                error: msg.clone(),
            },
        );
        return Err(anyhow!(msg));
    }
    emit_stage(&app, "rebuild", StageState::Succeeded, Some(rebuild_ms), None);

    let launch_target = if let Some(target_app) = target_app {
        let started = Instant::now();
        emit_stage(&app, "install", StageState::Running, None, None);
        match install_macos_app_update(&app, &install_root, &target_app).await {
            Ok(installed_app) => {
                emit_stage(
                    &app,
                    "install",
                    StageState::Succeeded,
                    Some(started.elapsed().as_millis() as u64),
                    None,
                );
                Some(installed_app)
            }
            Err(err) => {
                let msg = format!("{err:#}");
                emit_stage(
                    &app,
                    "install",
                    StageState::Failed,
                    Some(started.elapsed().as_millis() as u64),
                    Some(msg.clone()),
                );
                emit(
                    &app,
                    BootstrapEvent::Failed {
                        stage: Some("install".into()),
                        error: msg.clone(),
                    },
                );
                return Err(anyhow!(msg));
            }
        }
    } else {
        None
    };

    // ---- done: signal complete, then launch the fresh desktop ------------
    emit(
        &app,
        BootstrapEvent::Complete {
            install_root: install_root.to_string_lossy().into_owned(),
            marker: None,
        },
    );

    if let Some(target_app) = launch_target {
        if let Err(err) = launch_macos_app_and_exit(&app, &target_app).await {
            emit_log(
                &app,
                None,
                LogStream::Stderr,
                &format!("[update] could not auto-launch desktop: {err}. Launch Hermes manually."),
            );
        }
    } else if let Err(err) =
        crate::bootstrap::launch_hermes_desktop(app.clone(), install_root.to_string_lossy().into_owned()).await
    {
        // Launch failed: don't hard-fail the update (it succeeded); surface a
        // log line so the success screen can still tell the user to launch
        // manually.
        emit_log(
            &app,
            None,
            LogStream::Stdout,
            &format!("[update] could not auto-launch desktop: {err}. Launch Hermes manually."),
        );
    }

    Ok(())
}

/// Poll until the venv shim AND packaged desktop app bundle are no longer locked
/// (Windows) or a bounded timeout elapses. On non-Windows this is a short fixed
/// grace since file locking isn't the failure mode there.
pub(crate) async fn wait_for_install_locks_free(install_root: &Path, app: &AppHandle, stage: &str) {
    let lock_targets = install_lock_probe_paths(install_root);
    let deadline = Instant::now() + DESKTOP_EXIT_WAIT;

    emit_log(app, Some(stage), LogStream::Stdout, "[handoff] waiting for Hermes to exit…");

    loop {
        let locked = locked_paths(&lock_targets);
        if locked.is_empty() {
            return;
        }
        if Instant::now() >= deadline {
            // Last resort: a backend hermes.exe (or the desktop Hermes.exe
            // itself) is still holding one of the update-sensitive files. The
            // desktop should have reaped its tree before handing off, but
            // SIGTERM races / detached grandchildren / AV handles can leave a
            // straggler. Rather than "proceed anyway" straight into uv's
            // "Access is denied" or install.ps1's locked app.asar failure,
            // force-kill every Hermes.exe except ourselves, then give the OS a
            // beat to unload the image.
            emit_log(
                app,
                Some(stage),
                LogStream::Stdout,
                &format!(
                    "[handoff] Hermes still holding install files ({}); force-killing stragglers…",
                    format_locked_paths(&locked)
                ),
            );
            force_kill_other_hermes();
            tokio::time::sleep(Duration::from_millis(800)).await;
            let locked_after_kill = locked_paths(&lock_targets);
            if locked_after_kill.is_empty() {
                emit_log(
                    app,
                    Some(stage),
                    LogStream::Stdout,
                    "[handoff] install files freed after force-kill",
                );
            } else {
                emit_log(
                    app,
                    Some(stage),
                    LogStream::Stdout,
                    &format!(
                        "[handoff] install files still locked ({}); proceeding (--force + quarantine will handle it)",
                        format_locked_paths(&locked_after_kill)
                    ),
                );
            }
            return;
        }
        tokio::time::sleep(DESKTOP_EXIT_POLL).await;
    }
}

fn install_lock_probe_paths(install_root: &Path) -> Vec<PathBuf> {
    let mut paths = vec![venv_hermes(install_root)];
    paths.extend(desktop_app_payload_paths(install_root));
    paths
}

fn desktop_app_payload_paths(install_root: &Path) -> Vec<PathBuf> {
    let release = install_root.join("apps").join("desktop").join("release");
    if cfg!(target_os = "windows") {
        vec![
            release.join("win-unpacked").join("resources").join("app.asar"),
            release.join("win-arm64-unpacked").join("resources").join("app.asar"),
        ]
    } else if cfg!(target_os = "macos") {
        let mac_arches = if cfg!(target_arch = "x86_64") {
            ["mac-x64", "mac-arm64", "mac"]
        } else {
            ["mac-arm64", "mac-x64", "mac"]
        };
        mac_arches
            .into_iter()
            .map(|arch| release.join(arch).join("Hermes.app").join("Contents").join("Resources").join("app.asar"))
            .collect()
    } else {
        vec![release.join("linux-unpacked").join("resources").join("app.asar")]
    }
}

fn locked_paths(paths: &[PathBuf]) -> Vec<PathBuf> {
    paths.iter().filter(|p| is_locked(p)).cloned().collect()
}

fn format_locked_paths(paths: &[PathBuf]) -> String {
    paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
}

/// Force-kill any `hermes.exe` other than this process. Windows-only; a no-op
/// elsewhere (POSIX has no mandatory-lock contention). We can't selectively
/// target "the backend" by PID here — the desktop already exited and we never
/// knew its children — so we kill the whole `hermes.exe` image tree via
/// taskkill, excluding our own PID.
///
/// Safe w.r.t. our own update child: this runs inside the install-lock wait,
/// which completes BEFORE we spawn `venv\Scripts\hermes.exe update`. And a
/// desktop the user relaunches mid-update will NOT have spawned a backend —
/// `startHermes()` in the desktop gates local-backend startup on our
/// update-in-progress marker and parks until we finish (#50238). So the only
/// hermes.exe images here are stragglers from the old desktop — exactly what
/// we want gone. (`/FI PID ne <self>` also spares this Tauri process, though it
/// isn't named hermes.exe.)
fn force_kill_other_hermes() {
    if !cfg!(target_os = "windows") {
        return;
    }
    #[cfg(target_os = "windows")]
    {
        let my_pid = std::process::id();
        // /FI excludes our own PID; /T kills the tree; /F forces.
        let _ = std::process::Command::new("taskkill")
            .args([
                "/F",
                "/T",
                "/IM",
                "hermes.exe",
                "/FI",
                &format!("PID ne {my_pid}"),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

/// Best-effort lock probe: try to open the file for read+write. On Windows an
/// exclusively-held running .exe refuses the open with a sharing violation.
/// On Unix this almost always succeeds (no mandatory locking), which is fine —
/// the venv-shim contention is a Windows-only problem.
fn is_locked(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    match std::fs::OpenOptions::new().read(true).write(true).open(path) {
        Ok(_) => false,
        Err(_) => true,
    }
}

/// Whether the `desktop --build-only` rebuild should be retried once. Any
/// non-success exit qualifies: the common cause is a transient first-attempt
/// failure (still-settling tree / self-healed Electron download) that a clean
/// second run resolves.
fn rebuild_needs_retry(exit_code: Option<i32>) -> bool {
    exit_code != Some(0)
}

/// Spawn `hermes <args>` from `cwd`, stream stdout/stderr as Log events on the
/// bootstrap channel, and return the exit code. Mirrors powershell::run_script
/// but for an arbitrary command (no install.ps1 -File wrapping).
async fn run_streamed(
    app: &AppHandle,
    program: &Path,
    args: &[String],
    cwd: &Path,
    envs: &[(String, OsString)],
    stage: Option<&str>,
) -> Result<CmdResult> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        cmd.env(key, value);
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW = 0x08000000 — no flashing console behind the GUI.
        cmd.creation_flags(0x0800_0000);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow!("spawning {} {:?}: {e}", program.display(), args))?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    let mut out = BufReader::new(stdout).lines();
    let mut err = BufReader::new(stderr).lines();

    let stage_owned = stage.map(|s| s.to_string());
    loop {
        tokio::select! {
            line = out.next_line() => match line {
                Ok(Some(l)) => emit_log(app, stage_owned.as_deref(), LogStream::Stdout, &l),
                Ok(None) => break,
                Err(e) => { tracing::warn!("stdout read error: {e}"); break; }
            },
            line = err.next_line() => match line {
                Ok(Some(l)) => emit_log(app, stage_owned.as_deref(), LogStream::Stderr, &l),
                Ok(None) => {}
                Err(e) => { tracing::warn!("stderr read error: {e}"); }
            },
        }
    }
    while let Ok(Some(l)) = out.next_line().await {
        emit_log(app, stage_owned.as_deref(), LogStream::Stdout, &l);
    }
    while let Ok(Some(l)) = err.next_line().await {
        emit_log(app, stage_owned.as_deref(), LogStream::Stderr, &l);
    }

    let status = child.wait().await.map_err(|e| anyhow!("waiting for child: {e}"))?;
    Ok(CmdResult {
        exit_code: status.code(),
    })
}

struct CmdResult {
    exit_code: Option<i32>,
}

/// Path to the venv hermes shim under an install root, regardless of existence.
fn venv_hermes(install_root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        install_root.join("venv").join("Scripts").join("hermes.exe")
    } else {
        install_root.join("venv").join("bin").join("hermes")
    }
}

/// Resolve the hermes CLI to drive. Prefer the venv shim in the install we
/// just updated; fall back to `hermes` on PATH.
fn resolve_hermes(install_root: &Path) -> Option<PathBuf> {
    let shim = venv_hermes(install_root);
    if shim.exists() {
        return Some(shim);
    }
    // PATH fallback. which-style probe via env, kept dependency-free.
    let exe = if cfg!(target_os = "windows") { "hermes.exe" } else { "hermes" };
    if let Ok(path) = std::env::var("PATH") {
        let sep = if cfg!(target_os = "windows") { ';' } else { ':' };
        for dir in path.split(sep) {
            let cand = Path::new(dir).join(exe);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

fn update_child_env(install_root: &Path) -> Vec<(String, OsString)> {
    let hermes_home = crate::paths::hermes_home();
    let mut envs = vec![(
        "HERMES_HOME".to_string(),
        hermes_home.as_os_str().to_os_string(),
    )];
    if let Some(path) = path_with_prepended_entries(&[
        hermes_home.join("node").join("bin"),
        venv_bin_dir(install_root),
    ]) {
        envs.push(("PATH".to_string(), path));
    }
    envs
}

fn venv_bin_dir(install_root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        install_root.join("venv").join("Scripts")
    } else {
        install_root.join("venv").join("bin")
    }
}

fn path_with_prepended_entries(entries: &[PathBuf]) -> Option<OsString> {
    let mut parts: Vec<PathBuf> = entries.to_vec();
    if let Some(existing) = env::var_os("PATH") {
        parts.extend(env::split_paths(&existing));
    }
    env::join_paths(parts).ok()
}

fn update_branch_from_args<I, S>(args: I) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    arg_value_from_args(args, "--branch")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn target_app_from_args<I, S>(args: I) -> Option<PathBuf>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    arg_value_from_args(args, "--target-app")
        .map(PathBuf::from)
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("app"))
}

fn arg_value_from_args<I, S>(args: I, name: &str) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut iter = args.into_iter().map(|s| s.as_ref().to_string()).peekable();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next();
        }
        if let Some(value) = arg.strip_prefix(&format!("{name}=")) {
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(target_os = "macos")]
async fn install_macos_app_update(
    app: &AppHandle,
    install_root: &Path,
    target_app: &Path,
) -> Result<PathBuf> {
    if target_app.extension().and_then(|e| e.to_str()) != Some("app") {
        return Err(anyhow!(
            "refusing to install update into non-app path: {}",
            target_app.display()
        ));
    }

    let rebuilt_app = crate::bootstrap::resolve_hermes_desktop_app(install_root).ok_or_else(|| {
        anyhow!(
            "desktop rebuild succeeded but no Hermes.app was found under {}",
            install_root.join("apps").join("desktop").join("release").display()
        )
    })?;

    let same = match (rebuilt_app.canonicalize(), target_app.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => rebuilt_app == target_app,
    };
    if same {
        emit_log(
            app,
            Some("install"),
            LogStream::Stdout,
            &format!(
                "[update] rebuilt app is already the launch target: {}",
                target_app.display()
            ),
        );
        return Ok(target_app.to_path_buf());
    }

    emit_log(
        app,
        Some("install"),
        LogStream::Stdout,
        &format!(
            "[update] installing rebuilt app {} -> {}",
            rebuilt_app.display(),
            target_app.display()
        ),
    );

    if let Some(parent) = target_app.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let tmp = PathBuf::from(format!("{}.hermes-update-new", target_app.display()));
    let old = PathBuf::from(format!("{}.hermes-update-old", target_app.display()));
    remove_dir_if_exists(&tmp).await;
    remove_dir_if_exists(&old).await;

    let ditto = Command::new("/usr/bin/ditto")
        .arg(&rebuilt_app)
        .arg(&tmp)
        .current_dir(crate::paths::hermes_home())
        .status()
        .await
        .map_err(|e| anyhow!("running ditto: {e}"))?;
    if !ditto.success() {
        return Err(anyhow!(
            "ditto failed while copying updated app into {}",
            tmp.display()
        ));
    }

    // Atomic-as-possible swap with rollback. Extracted so the invariant
    // (target is never left deleted-with-no-replacement) can be unit-tested
    // without ditto / a real .app bundle.
    swap_in_new_bundle(&tmp, target_app, &old).await?;

    let _ = Command::new("/usr/bin/xattr")
        .arg("-dr")
        .arg("com.apple.quarantine")
        .arg(target_app)
        .current_dir(crate::paths::hermes_home())
        .status()
        .await;

    Ok(target_app.to_path_buf())
}

/// Move a freshly-staged bundle (`tmp`) into place at `target`, parking any
/// existing bundle at `old` so the move can succeed (macOS `rename` won't
/// overwrite a non-empty directory).
///
/// Invariant: on ANY failure path, `target` is left pointing at a working
/// bundle — either the original (rolled back from `old`) or untouched — and we
/// never delete the running app with no replacement in place. The staged `tmp`
/// copy is cleaned up on failure.
async fn swap_in_new_bundle(tmp: &Path, target: &Path, old: &Path) -> Result<()> {
    let moved_old = if target.exists() {
        if let Err(err) = tokio::fs::rename(target, old).await {
            // Could not move the existing app aside. Leave it untouched and
            // bail — a failed update must not brick the install.
            remove_dir_if_exists(tmp).await;
            return Err(anyhow!(
                "could not move existing app aside at {} (leaving it in place): {err}",
                target.display()
            ));
        }
        true
    } else {
        false
    };
    if let Err(err) = tokio::fs::rename(tmp, target).await {
        // Restore the original app from the backup so the user keeps a working
        // install, and clean up the staged copy.
        if moved_old {
            let _ = tokio::fs::rename(old, target).await;
        }
        remove_dir_if_exists(tmp).await;
        return Err(anyhow!("installing updated app at {}: {err}", target.display()));
    }
    remove_dir_if_exists(old).await;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
async fn install_macos_app_update(
    _app: &AppHandle,
    _install_root: &Path,
    target_app: &Path,
) -> Result<PathBuf> {
    Ok(target_app.to_path_buf())
}

async fn remove_dir_if_exists(path: &Path) {
    if path.exists() {
        let _ = tokio::fs::remove_dir_all(path).await;
    }
}

#[cfg(target_os = "macos")]
async fn launch_macos_app_and_exit(app: &AppHandle, target_app: &Path) -> Result<()> {
    crate::bootstrap::open_macos_app_detached(target_app)
        .map_err(|e| anyhow!("launching {}: {e}", target_app.display()))?;
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    app.exit(0);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
async fn launch_macos_app_and_exit(_app: &AppHandle, _target_app: &Path) -> Result<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Event helpers — keep emit shape identical to bootstrap.rs so the UI is reused
// ---------------------------------------------------------------------------

fn stage_info(name: &str, title: &str) -> StageInfo {
    StageInfo {
        name: name.to_string(),
        title: title.to_string(),
        category: "update".to_string(),
        needs_user_input: false,
    }
}

/// The synthetic update manifest. Mirrors the real operations `run_update`
/// performs so the progress UI shows them as discrete steps (with the live log
/// underneath) instead of one monolithic bar. `include_install` adds the macOS
/// app-swap stage. Both the happy path and the re-entrancy guard build the
/// manifest here so the two can never drift apart.
fn update_stages(include_install: bool) -> Vec<StageInfo> {
    let mut stages = vec![
        stage_info("handoff", "Preparing to update"),
        stage_info("update", "Downloading the latest version"),
        stage_info("rebuild", "Rebuilding the desktop app"),
    ];
    if include_install {
        stages.push(stage_info("install", "Installing the update"));
    }
    stages
}

// option_env! only accepts string literals, so the build-time pins are read
// by their literal names here. Mirrors bootstrap.rs's helper of the same name
// (kept local rather than shared because option_env! can't be parameterized).
fn option_env_string(key: &str) -> Option<String> {
    let val = match key {
        "BUILD_PIN_COMMIT" => option_env!("BUILD_PIN_COMMIT"),
        "BUILD_PIN_BRANCH" => option_env!("BUILD_PIN_BRANCH"),
        _ => None,
    };
    val.map(|s| s.to_string())
}

fn emit(app: &AppHandle, event: BootstrapEvent) {
    if let Err(e) = app.emit(BootstrapEvent::CHANNEL, &event) {
        tracing::warn!(?e, "failed to emit update event");
    }
}

fn emit_stage(
    app: &AppHandle,
    name: &str,
    state: StageState,
    duration_ms: Option<u64>,
    error: Option<String>,
) {
    tracing::info!(stage = %name, ?state, ?duration_ms, ?error, "update stage");
    emit(
        app,
        BootstrapEvent::Stage {
            name: name.to_string(),
            state,
            duration_ms,
            result: None,
            error,
        },
    );
}

fn emit_log(app: &AppHandle, stage: Option<&str>, stream: LogStream, line: &str) {
    match stage {
        Some(s) => tracing::info!(target: "bootstrap.log", stage = %s, "{line}"),
        None => tracing::info!(target: "bootstrap.log", "{line}"),
    }
    emit(
        app,
        BootstrapEvent::Log {
            stage: stage.map(|s| s.to_string()),
            line: line.to_string(),
            stream,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venv_hermes_is_under_install_root() {
        let root = Path::new("/x/hermes-agent");
        let shim = venv_hermes(root);
        assert!(shim.starts_with(root));
        assert!(shim.to_string_lossy().contains("venv"));
    }

    #[test]
    fn missing_file_is_not_locked() {
        assert!(!is_locked(Path::new("/nonexistent/does/not/exist/xyz")));
    }

    #[test]
    fn lock_probe_paths_include_desktop_app_payload() {
        let root = Path::new("/x/hermes-agent");
        let probes = install_lock_probe_paths(root);

        assert!(
            probes.iter().any(|p| p == &venv_hermes(root)),
            "venv shim remains part of the update lock probe"
        );
        assert!(
            probes.iter().any(|p| p.ends_with(Path::new("resources/app.asar"))),
            "packaged app.asar must be probed so repair/re-clone waits for the old desktop to exit"
        );
    }

    #[test]
    fn locked_paths_ignores_missing_payloads() {
        let root = Path::new("/nonexistent/hermes-agent");
        let probes = install_lock_probe_paths(root);

        assert!(locked_paths(&probes).is_empty());
    }

    #[test]
    fn update_marker_guard_writes_then_removes_on_drop() {
        let dir = unique_tmp_dir("marker-guard");
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join(".hermes-update-in-progress");

        {
            let _g = UpdateMarkerGuard::acquire(marker.clone());
            assert!(marker.exists(), "marker must exist while the guard is held");
            let body = std::fs::read_to_string(&marker).unwrap();
            let pid_line = body.lines().next().unwrap();
            assert_eq!(
                pid_line.trim().parse::<u32>().unwrap(),
                std::process::id(),
                "marker records our pid so the desktop can probe liveness"
            );
            assert_eq!(body.lines().count(), 2, "marker is pid + started_at lines");
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

        let guard = UpdateMarkerGuard::acquire(marker.clone());
        // Simulate an external cleanup (e.g. the desktop pruned a marker it
        // judged stale) before our guard drops — Drop must not panic.
        std::fs::remove_file(&marker).unwrap();
        drop(guard);

        assert!(!marker.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_update_branch_from_space_or_equals_args() {
        assert_eq!(
            update_branch_from_args(["--update", "--branch", "bb/test"]),
            Some("bb/test".to_string())
        );
        assert_eq!(
            update_branch_from_args(["--update", "--branch=main"]),
            Some("main".to_string())
        );
        assert_eq!(update_branch_from_args(["--update"]), None);
    }

    #[test]
    fn update_manifest_leads_with_handoff_and_gates_install() {
        let base = update_stages(false);
        assert_eq!(
            base.first().map(|s| s.name.as_str()),
            Some("handoff"),
            "the lock-wait must surface as the first visible step"
        );
        assert!(
            base.iter().any(|s| s.name == "update") && base.iter().any(|s| s.name == "rebuild"),
            "update + rebuild remain distinct stages"
        );
        assert!(
            base.iter().all(|s| s.name != "install"),
            "no app-swap stage unless an install target was passed"
        );

        let with_install = update_stages(true);
        assert_eq!(
            with_install.last().map(|s| s.name.as_str()),
            Some("install"),
            "the macOS app-swap is the final stage when present"
        );
        assert_eq!(
            with_install.len(),
            base.len() + 1,
            "include_install adds exactly one stage"
        );
    }

    #[test]
    fn rebuild_retries_only_on_failure() {
        assert!(!rebuild_needs_retry(Some(0)), "a clean rebuild must not retry");
        assert!(rebuild_needs_retry(Some(1)), "a failed rebuild retries once");
        assert!(
            rebuild_needs_retry(None),
            "a killed/signalled rebuild (no exit code) retries once"
        );
    }

    #[test]
    fn parses_only_app_targets() {
        assert_eq!(
            target_app_from_args(["--update", "--target-app", "/Applications/Hermes.app"]),
            Some(PathBuf::from("/Applications/Hermes.app"))
        );
        assert_eq!(target_app_from_args(["--target-app", "/tmp/not-an-app"]), None);
    }

    // Helpers for the swap tests: make a throwaway dir tree we can rename.
    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "hermes-swap-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    fn write_marker(dir: &Path, contents: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("marker.txt"), contents).unwrap();
    }

    #[tokio::test]
    async fn swap_installs_new_bundle_and_cleans_up() {
        let base = unique_tmp_dir("ok");
        let target = base.join("Hermes.app");
        let tmp = base.join("Hermes.app.hermes-update-new");
        let old = base.join("Hermes.app.hermes-update-old");
        write_marker(&target, "OLD");
        write_marker(&tmp, "NEW");

        swap_in_new_bundle(&tmp, &target, &old).await.unwrap();

        // New bundle is now at target; staging + backup dirs are gone.
        assert_eq!(
            std::fs::read_to_string(target.join("marker.txt")).unwrap(),
            "NEW"
        );
        assert!(!tmp.exists(), "staged copy should be cleaned up");
        assert!(!old.exists(), "backup should be cleaned up on success");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn swap_failure_never_leaves_target_missing() {
        // Regression guard for the catastrophic path: the move-aside of the
        // existing app fails AND the staged bundle can't be installed. The
        // buggy version deleted `target` when move-aside failed and then
        // skipped rollback, bricking the install. The fixed version must leave
        // the original app intact on disk.
        //
        // Trigger both failures deterministically:
        //  - `old` is a NON-EMPTY dir  -> rename(target, old) fails
        //  - `tmp` does not exist       -> rename(tmp, target) fails
        let base = unique_tmp_dir("fail");
        let target = base.join("Hermes.app");
        let tmp = base.join("Hermes.app.hermes-update-new"); // intentionally absent
        let old = base.join("Hermes.app.hermes-update-old");
        write_marker(&target, "OLD");
        write_marker(&old, "OCCUPIED"); // non-empty => rename(target,old) fails

        let result = swap_in_new_bundle(&tmp, &target, &old).await;

        assert!(result.is_err(), "swap should fail when neither move can complete");
        assert!(target.exists(), "original app must NOT be deleted on failure");
        assert_eq!(
            std::fs::read_to_string(target.join("marker.txt")).unwrap(),
            "OLD",
            "original app contents must be intact after a failed swap"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn swap_rolls_back_when_install_step_fails() {
        // Move-aside succeeds but installing the staged bundle fails (tmp
        // absent). The original must be rolled back from `old` to `target`.
        let base = unique_tmp_dir("rollback");
        let target = base.join("Hermes.app");
        let tmp = base.join("Hermes.app.hermes-update-new"); // absent
        let old = base.join("Hermes.app.hermes-update-old");
        write_marker(&target, "OLD");

        let result = swap_in_new_bundle(&tmp, &target, &old).await;

        assert!(result.is_err());
        assert!(target.exists(), "original must be restored after failed install");
        assert_eq!(
            std::fs::read_to_string(target.join("marker.txt")).unwrap(),
            "OLD"
        );
        assert!(!old.exists(), "backup should be rolled back, not left behind");
        let _ = std::fs::remove_dir_all(&base);
    }
}
