//! Update orchestration.
//!
//! Driven when the installer is launched as `Hermes-Setup.exe --update` (see
//! `AppMode` in lib.rs). The desktop app hands off to us — it exits, then we:
//!
//!   1. stop any install-scoped Windows locker processes, then wait for the
//!      old Hermes desktop process to fully exit (so the venv shim is free;
//!      otherwise `hermes update` aborts with exit code 2),
//!   2. run `hermes update --yes --gateway` (Python/repo update; this does NOT
//!      rebuild apps/desktop by design — see cmd_update in hermes_cli/main.py),
//!   3. run `hermes desktop --build-only` (the rebuild step update skips),
//!   4. launch the freshly-built desktop (reuses bootstrap::launch logic).
//!
//! We reuse the `BootstrapEvent` channel + the existing progress UI by
//! emitting a synthetic two-stage manifest ("update", "rebuild"). To the
//! frontend an update looks like a short bootstrap.
//!
//! Cross-platform note: `hermes update` already handles macOS/Linux (git/pip).
//! The only OS-specific bits here are the venv shim path (resolve_hermes) and
//! the no-window creation flag — both already cfg-gated. Keep new logic
//! OS-agnostic so the mac/linux port stays "fill in the paths".

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::events::{BootstrapEvent, StageInfo, StageState};

/// `hermes update` exit code meaning "another hermes process is holding the
/// venv shim open / dirty precondition" — see _cmd_update_impl in
/// hermes_cli/main.py (sys.exit(2)). We surface a targeted message for this.
const UPDATE_EXIT_CONCURRENT: i32 = 2;

/// How long to wait for the old desktop process to release the venv shim
/// before giving up and letting `hermes update`'s own guard decide.
const DESKTOP_EXIT_WAIT: Duration = Duration::from_secs(20);
const DESKTOP_EXIT_POLL: Duration = Duration::from_millis(500);

/// Frontend → Rust: kick off the update flow. Mirrors `start_bootstrap`'s
/// fire-and-forget shape; progress arrives on the `bootstrap` event channel.
#[tauri::command]
pub async fn start_update(app: AppHandle) -> Result<(), String> {
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
    });
    Ok(())
}

async fn run_update(app: AppHandle) -> Result<()> {
    let hermes_home = crate::paths::hermes_home();
    let install_root = hermes_home.join("hermes-agent");

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

    // Synthetic manifest so the existing progress UI renders our two stages.
    emit(
        &app,
        BootstrapEvent::Manifest {
            stages: vec![
                stage_info("update", "Updating Hermes"),
                stage_info("rebuild", "Rebuilding the desktop app"),
            ],
            protocol_version: None,
        },
    );

    // ---- pre-step: wait for the old desktop to die -----------------------
    reap_windows_update_lockers(&install_root, &app).await;
    // The desktop exec'd us then called app.exit(), but process teardown is
    // async on Windows. If it still holds the venv shim, `hermes update`
    // aborts with exit 2. Give it a bounded window to clear.
    wait_for_venv_free(&install_root, &app).await;

    // ---- stage 1: hermes update -----------------------------------------
    // Pass --branch so `hermes update` targets the branch this installer was
    // built/pinned against (BUILD_PIN_BRANCH), NOT its built-in default of
    // `main`. The install was a detached-HEAD checkout of a specific commit;
    // without --branch, `hermes update` switches the checkout to `main` (a
    // divergent branch that may not even have the desktop CLI command), then
    // reports "already up to date" against the wrong branch. The desktop
    // detected the update against this same branch, so we must update against
    // it too.
    let pin_branch = option_env_string("BUILD_PIN_BRANCH");
    let mut update_args: Vec<&str> = vec!["update", "--yes", "--gateway"];
    if let Some(b) = pin_branch.as_deref() {
        update_args.push("--branch");
        update_args.push(b);
    }

    emit_stage(&app, "update", StageState::Running, None, None);
    let started = Instant::now();
    let update = run_streamed(&app, &hermes, &update_args, &install_root, Some("update")).await?;
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

    // ---- stage 2: hermes desktop --build-only ----------------------------
    // `hermes update` deliberately does NOT build apps/desktop (it installs
    // repo-root deps with --workspaces=false). This is the rebuild it skips.
    emit_stage(&app, "rebuild", StageState::Running, None, None);
    let started = Instant::now();
    let rebuild = run_streamed(
        &app,
        &hermes,
        &["desktop", "--build-only"],
        &install_root,
        Some("rebuild"),
    )
    .await?;
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
    emit_stage(
        &app,
        "rebuild",
        StageState::Succeeded,
        Some(rebuild_ms),
        None,
    );

    // ---- done: signal complete, then launch the fresh desktop ------------
    emit(
        &app,
        BootstrapEvent::Complete {
            install_root: install_root.to_string_lossy().into_owned(),
            marker: None,
        },
    );

    // Reuse the same detached-launch + app.exit(0) used post-install.
    if let Err(err) = crate::bootstrap::launch_hermes_desktop(
        app.clone(),
        install_root.to_string_lossy().into_owned(),
    )
    .await
    {
        // Launch failed: don't hard-fail the update (it succeeded); surface a
        // log line so the success screen can still tell the user to launch
        // manually.
        emit_log(
            &app,
            None,
            &format!("[update] could not auto-launch desktop: {err}. Launch Hermes manually."),
        );
    }

    Ok(())
}

/// Poll until the venv shim is no longer locked (Windows) or a bounded timeout
/// elapses. On non-Windows this is a short fixed grace since file locking
/// isn't the failure mode there.
async fn wait_for_venv_free(install_root: &Path, app: &AppHandle) {
    let shim = venv_hermes(install_root);
    let deadline = Instant::now() + DESKTOP_EXIT_WAIT;

    emit_log(app, Some("update"), "[update] waiting for Hermes to exit…");

    loop {
        if !is_locked(&shim) {
            return;
        }
        if Instant::now() >= deadline {
            emit_log(
                app,
                Some("update"),
                "[update] timed out waiting for Hermes to exit; proceeding anyway",
            );
            return;
        }
        tokio::time::sleep(DESKTOP_EXIT_POLL).await;
    }
}

#[cfg(not(target_os = "windows"))]
async fn reap_windows_update_lockers(_install_root: &Path, _app: &AppHandle) {}

#[cfg(target_os = "windows")]
async fn reap_windows_update_lockers(install_root: &Path, app: &AppHandle) {
    emit_log(
        app,
        Some("update"),
        "[update] stopping stale Windows locker processes…",
    );
    match run_windows_locker_cleanup(install_root).await {
        Ok(output) => {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                emit_log(app, Some("update"), line);
            }
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                emit_log(app, Some("update"), &format!("stderr: {line}"));
            }
            if !output.status.success() {
                emit_log(
                    app,
                    Some("update"),
                    &format!(
                        "[update] locker cleanup exited {:?}; continuing with lock wait",
                        output.status.code()
                    ),
                );
            }
        }
        Err(err) => emit_log(
            app,
            Some("update"),
            &format!("[update] locker cleanup failed: {err}; continuing with lock wait"),
        ),
    }
}

#[cfg(target_os = "windows")]
async fn run_windows_locker_cleanup(install_root: &Path) -> Result<std::process::Output> {
    let script = windows_locker_cleanup_script(
        install_root,
        crate::bootstrap::resolve_hermes_desktop_exe(install_root).as_deref(),
    );

    let mut cmd = Command::new("powershell.exe");
    cmd.arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-Command")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW = 0x08000000 — keep the GUI update path silent.
        cmd.creation_flags(0x0800_0000);
    }

    cmd.output()
        .await
        .map_err(|e| anyhow!("running locker cleanup PowerShell: {e}"))
}

fn windows_locker_cleanup_script(install_root: &Path, desktop_exe: Option<&Path>) -> String {
    let venv_scripts = install_root.join("venv").join("Scripts");
    let targets = [
        ("python.exe", Some(venv_scripts.join("python.exe"))),
        ("pythonw.exe", Some(venv_scripts.join("pythonw.exe"))),
        ("hermes.exe", Some(venv_scripts.join("hermes.exe"))),
        ("Hermes.exe", desktop_exe.map(Path::to_path_buf)),
    ];

    let mut lines = vec![
        "$ErrorActionPreference = 'Continue'".to_string(),
        "function Get-MatchingPids($name, $target) {".to_string(),
        "  if ([string]::IsNullOrWhiteSpace($target) -or -not (Test-Path -LiteralPath $target)) { return @() }".to_string(),
        "  $fullTarget = [System.IO.Path]::GetFullPath($target)".to_string(),
        "  Get-CimInstance Win32_Process -Filter (\"Name='\" + $name + \"'\") |".to_string(),
        "    Where-Object { $_.ExecutablePath -and ([System.IO.Path]::GetFullPath($_.ExecutablePath) -ieq $fullTarget) } |".to_string(),
        "    ForEach-Object { $_.ProcessId }".to_string(),
        "}".to_string(),
        "$pids = @()".to_string(),
    ];

    for (name, path) in targets {
        match path {
            Some(path) => lines.push(format!(
                "$pids += Get-MatchingPids '{}' '{}'",
                name,
                ps_single_quote(&path)
            )),
            None => lines.push(format!("$pids += Get-MatchingPids '{}' ''", name)),
        }
    }

    lines.extend([
        "$pids = @($pids | Where-Object { $_ } | Sort-Object -Unique)".to_string(),
        "if ($pids.Count -eq 0) {".to_string(),
        "  Write-Output '[update] no Windows locker processes found'".to_string(),
        "  exit 0".to_string(),
        "}".to_string(),
        "foreach ($pid in $pids) {".to_string(),
        "  try {".to_string(),
        "    $proc = Get-Process -Id $pid -ErrorAction Stop".to_string(),
        "    Write-Output ((\"[update] stopping PID {0} ({1})\" -f $pid, $proc.ProcessName))".to_string(),
        "  } catch {".to_string(),
        "    Write-Output ((\"[update] stopping PID {0}\" -f $pid))".to_string(),
        "  }".to_string(),
        "  & taskkill /PID $pid /T /F | Out-Null".to_string(),
        "  if ($LASTEXITCODE -eq 0) {".to_string(),
        "    Write-Output ((\"[update] taskkill succeeded for PID {0}\" -f $pid))".to_string(),
        "  } else {".to_string(),
        "    Write-Output ((\"[update] taskkill failed for PID {0} (exit {1})\" -f $pid, $LASTEXITCODE))".to_string(),
        "  }".to_string(),
        "}".to_string(),
        "exit 0".to_string(),
    ]);

    lines.join("\n")
}

fn ps_single_quote(path: &Path) -> Cow<'_, str> {
    let rendered = path.to_string_lossy();
    if rendered.contains('\'') {
        Cow::Owned(rendered.replace('\'', "''"))
    } else {
        rendered
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
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    {
        Ok(_) => false,
        Err(_) => true,
    }
}

/// Spawn `hermes <args>` from `cwd`, stream stdout/stderr as Log events on the
/// bootstrap channel, and return the exit code. Mirrors powershell::run_script
/// but for an arbitrary command (no install.ps1 -File wrapping).
async fn run_streamed(
    app: &AppHandle,
    program: &Path,
    args: &[&str],
    cwd: &Path,
    stage: Option<&str>,
) -> Result<CmdResult> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

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
                Ok(Some(l)) => emit_log(app, stage_owned.as_deref(), &l),
                Ok(None) => break,
                Err(e) => { tracing::warn!("stdout read error: {e}"); break; }
            },
            line = err.next_line() => match line {
                Ok(Some(l)) => emit_log(app, stage_owned.as_deref(), &format!("stderr: {l}")),
                Ok(None) => {}
                Err(e) => { tracing::warn!("stderr read error: {e}"); }
            },
        }
    }
    while let Ok(Some(l)) = out.next_line().await {
        emit_log(app, stage_owned.as_deref(), &l);
    }
    while let Ok(Some(l)) = err.next_line().await {
        emit_log(app, stage_owned.as_deref(), &format!("stderr: {l}"));
    }

    let status = child
        .wait()
        .await
        .map_err(|e| anyhow!("waiting for child: {e}"))?;
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
    let exe = if cfg!(target_os = "windows") {
        "hermes.exe"
    } else {
        "hermes"
    };
    if let Ok(path) = std::env::var("PATH") {
        let sep = if cfg!(target_os = "windows") {
            ';'
        } else {
            ':'
        };
        for dir in path.split(sep) {
            let cand = Path::new(dir).join(exe);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
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

fn emit_log(app: &AppHandle, stage: Option<&str>, line: &str) {
    match stage {
        Some(s) => tracing::info!(target: "bootstrap.log", stage = %s, "{line}"),
        None => tracing::info!(target: "bootstrap.log", "{line}"),
    }
    emit(
        app,
        BootstrapEvent::Log {
            stage: stage.map(|s| s.to_string()),
            line: line.to_string(),
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
    fn powershell_quote_escapes_single_quotes() {
        let rendered = ps_single_quote(Path::new(r"C:\Users\O'Neil\hermes-agent"));
        assert_eq!(rendered, "C:\\Users\\O''Neil\\hermes-agent");
    }

    #[test]
    fn windows_locker_cleanup_targets_install_scoped_processes() {
        let root = Path::new(r"C:\Users\me\AppData\Local\hermes\hermes-agent");
        let python = root.join("venv").join("Scripts").join("python.exe");
        let desktop = root
            .join("apps")
            .join("desktop")
            .join("release")
            .join("win-unpacked")
            .join("Hermes.exe");

        let script = windows_locker_cleanup_script(root, Some(&desktop));

        assert!(script.contains("python.exe"));
        assert!(script.contains("pythonw.exe"));
        assert!(script.contains("hermes.exe"));
        assert!(script.contains("Hermes.exe"));
        assert!(script.contains("Get-CimInstance Win32_Process"));
        assert!(script.contains("taskkill /PID $pid /T /F"));
        assert!(script.contains(ps_single_quote(&python).as_ref()));
        assert!(script.contains(ps_single_quote(&desktop).as_ref()));
    }
}
