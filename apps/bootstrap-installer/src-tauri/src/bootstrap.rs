//! Bootstrap orchestration — North Forge lightweight tier.
//!
//! Upstream Hermes drove `install.ps1` through a `-Manifest` / per-stage
//! `-Json` protocol. `bootstrap-north-forge.ps1` has none of that: it is one
//! monolithic run that prints human text and exits 0 / 1. So this module
//! runs it ONCE, streams stdout/stderr line-by-line over the Tauri
//! `bootstrap` channel (indeterminate progress + the collapsible log panel
//! own the UI), and emits exactly one `Complete` or `Failed`.
//!
//! The audited script (`scripts/bootstrap-north-forge.ps1`,
//! `scripts/nf-preflight.ps1`) is run UNMODIFIED — we only pass `-RepoRoot`.
//!
//! Lifecycle:
//!   1. `start_bootstrap` (Tauri command) → spawns the worker task.
//!   2. Worker resolves the checkout (arg override, else auto-detect).
//!   3. Worker runs `bootstrap-north-forge.ps1 -RepoRoot <root>`.
//!   4. Exit 0 → `Complete`. Non-zero / spawn failure / cancel → `Failed`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::{mpsc, Mutex};

use crate::events::{BootstrapEvent, LogStream};
use crate::powershell::{self, StreamSink};
use crate::repo;
use crate::AppState;

// ---------------------------------------------------------------------------
// Public Tauri commands
// ---------------------------------------------------------------------------

/// Frontend → Rust: kick off the bootstrap.
#[derive(Debug, Deserialize)]
pub struct StartBootstrapArgs {
    /// The North Forge checkout to bootstrap. `null` → auto-detect
    /// (env override → installer-relative walk-up → drive scan).
    pub repo_root: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BootstrapStatus {
    pub running: bool,
    pub completed: bool,
    pub repo_root: Option<String>,
    pub data_dir: Option<String>,
    pub last_error: Option<String>,
}

/// Handle stored in AppState while a bootstrap run is in flight.
pub struct BootstrapHandle {
    pub cancel_tx: mpsc::Sender<()>,
    pub started_at: Instant,
    pub status: BootstrapStatus,
}

#[tauri::command]
pub async fn start_bootstrap(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    args: StartBootstrapArgs,
) -> Result<(), String> {
    let mut guard = state.bootstrap.lock().await;
    if let Some(h) = guard.as_ref() {
        if h.status.running {
            return Err("Bootstrap is already running".into());
        }
    }

    let (cancel_tx, cancel_rx) = mpsc::channel::<()>(1);
    *guard = Some(BootstrapHandle {
        cancel_tx,
        started_at: Instant::now(),
        status: BootstrapStatus {
            running: true,
            completed: false,
            repo_root: None,
            data_dir: None,
            last_error: None,
        },
    });
    drop(guard);

    let app_for_task = app.clone();
    let state_for_task = state.inner().clone();
    let cancel_rx = Arc::new(Mutex::new(Some(cancel_rx)));

    tokio::spawn(async move {
        let result = run_bootstrap(app_for_task, args, cancel_rx).await;

        let mut guard = state_for_task.bootstrap.lock().await;
        if let Some(h) = guard.as_mut() {
            h.status.running = false;
            match &result {
                Ok(done) => {
                    h.status.completed = true;
                    h.status.repo_root = Some(done.repo_root.clone());
                    h.status.data_dir = Some(done.data_dir.clone());
                    h.status.last_error = None;
                }
                Err(err) => {
                    h.status.completed = false;
                    h.status.last_error = Some(err.to_string());
                }
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub async fn cancel_bootstrap(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let guard = state.bootstrap.lock().await;
    if let Some(h) = guard.as_ref() {
        let _ = h.cancel_tx.try_send(());
    }
    Ok(())
}

#[tauri::command]
pub async fn get_bootstrap_status(
    state: State<'_, Arc<AppState>>,
) -> Result<BootstrapStatus, String> {
    let guard = state.bootstrap.lock().await;
    Ok(match guard.as_ref() {
        Some(h) => BootstrapStatus {
            running: h.status.running,
            completed: h.status.completed,
            repo_root: h.status.repo_root.clone(),
            data_dir: h.status.data_dir.clone(),
            last_error: h.status.last_error.clone(),
        },
        None => BootstrapStatus {
            running: false,
            completed: false,
            repo_root: None,
            data_dir: None,
            last_error: None,
        },
    })
}

/// Success-screen "Launch": open a North Forge terminal. Prefers the
/// drive-root `Start North Forge.lnk` the bootstrap writes; falls back to
/// running `north-forge.cmd` in a fresh console. Then closes the installer.
#[tauri::command]
pub async fn launch_north_forge(app: AppHandle, repo_root: String) -> Result<(), String> {
    let root = PathBuf::from(&repo_root);
    let launcher = root.join(repo::LAUNCHER_REL);
    if !launcher.is_file() {
        return Err(format!(
            "north-forge.cmd not found at {}. The checkout may be incomplete.",
            launcher.display()
        ));
    }

    // Drive-root shortcut, if the bootstrap wrote one (<drive>:\Start North Forge.lnk).
    let drive_lnk = root
        .components()
        .next()
        .map(|c| PathBuf::from(c.as_os_str()).join("\\").join("Start North Forge.lnk"))
        .filter(|p| p.is_file());

    let spawn_target = drive_lnk.clone().unwrap_or_else(|| launcher.clone());
    tracing::info!(?spawn_target, "launching North Forge");

    // `cmd /C start "" <target>` opens it in its own console window and
    // returns immediately, so the installer can exit without killing it.
    let mut cmd = std::process::Command::new("cmd");
    cmd.arg("/C")
        .arg("start")
        .arg("North Forge")
        .arg("/D")
        .arg(&root)
        .arg(&spawn_target)
        .current_dir(&root);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NEW_CONSOLE (0x10) so the launched shell gets its own window.
        cmd.creation_flags(0x0000_0010);
    }

    cmd.spawn()
        .map_err(|e| format!("failed to launch {}: {e}", spawn_target.display()))?;

    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    app.exit(0);
    Ok(())
}

// ---------------------------------------------------------------------------
// Bootstrap implementation
// ---------------------------------------------------------------------------

struct BootstrapDone {
    repo_root: String,
    data_dir: String,
}

async fn run_bootstrap(
    app: AppHandle,
    args: StartBootstrapArgs,
    cancel_rx_holder: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
) -> Result<BootstrapDone> {
    // 1. Resolve the checkout.
    let info = match args.repo_root.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(picked) => repo::describe(Some(PathBuf::from(picked)), "picked"),
        None => repo::detect_repo(),
    };

    let fail = |app: &AppHandle, msg: String| -> anyhow::Error {
        emit_event(app, BootstrapEvent::Failed { error: msg.clone() });
        anyhow!(msg)
    };

    let Some(repo_root) = info.repo_root.clone() else {
        return Err(fail(
            &app,
            "Couldn't find a North Forge checkout. Use \"Choose location\" to \
             pick the drive or folder that holds it."
                .to_string(),
        ));
    };
    let Some(script) = info.bootstrap_script.clone() else {
        return Err(fail(
            &app,
            format!("{repo_root}\\scripts\\bootstrap-north-forge.ps1 is missing — the checkout looks incomplete."),
        ));
    };
    let data_dir = info.data_dir.clone().unwrap_or_default();
    let script_path = PathBuf::from(&script);
    let repo_path = PathBuf::from(&repo_root);

    tracing::info!(%repo_root, %script, "north forge bootstrap starting");
    emit_event(
        &app,
        BootstrapEvent::Started {
            script: script.clone(),
            repo_root: repo_root.clone(),
        },
    );

    // 2. Stream the one script run.
    let app_for_out = app.clone();
    let app_for_err = app.clone();
    let sink = StreamSink {
        on_stdout_line: Box::new(move |line: &str| {
            emit_event(
                &app_for_out,
                BootstrapEvent::Log {
                    line: line.to_string(),
                    stream: LogStream::Stdout,
                },
            );
            tracing::info!(target: "bootstrap.log", "{line}");
        }),
        on_stderr_line: Box::new(move |line: &str| {
            emit_event(
                &app_for_err,
                BootstrapEvent::Log {
                    line: line.to_string(),
                    stream: LogStream::Stderr,
                },
            );
            tracing::warn!(target: "bootstrap.log", "stderr: {line}");
        }),
    };

    let mut cancel_rx = cancel_rx_holder.lock().await.take();
    let script_args = vec!["-RepoRoot".to_string(), repo_root.clone()];
    let result = powershell::run_script(
        &script_path,
        &script_args,
        sink,
        Some(repo_path.as_path()),
        &mut cancel_rx,
    )
    .await;
    *cancel_rx_holder.lock().await = cancel_rx;

    let result = match result {
        Ok(r) => r,
        Err(err) => {
            return Err(fail(
                &app,
                format!("could not run bootstrap-north-forge.ps1: {err:#}"),
            ));
        }
    };

    if result.killed {
        return Err(fail(&app, "Bootstrap cancelled.".to_string()));
    }

    if result.exit_code != Some(0) {
        let tail = tail_lines(&result.stderr, &result.stdout, 12);
        return Err(fail(
            &app,
            format!(
                "bootstrap-north-forge.ps1 exited with code {}.{}",
                result
                    .exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                if tail.is_empty() {
                    String::new()
                } else {
                    format!("\n\n{tail}")
                }
            ),
        ));
    }

    // 3. Done. Recompute paths from disk so the success screen shows what
    // actually landed.
    let done_info = repo::describe(Some(repo_path.clone()), "picked");
    let data_dir = done_info.data_dir.unwrap_or(data_dir);
    let venv_dir = done_info.venv_dir.unwrap_or_default();

    emit_event(
        &app,
        BootstrapEvent::Complete {
            repo_root: repo_root.clone(),
            venv_dir,
            data_dir: data_dir.clone(),
            launcher_cmd: done_info.launcher_cmd,
        },
    );

    Ok(BootstrapDone {
        repo_root,
        data_dir,
    })
}

/// Last `n` non-empty lines of stderr, else stdout — for the failure message.
fn tail_lines(stderr: &str, stdout: &str, n: usize) -> String {
    let src = if stderr.trim().is_empty() { stdout } else { stderr };
    let lines: Vec<&str> = src.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

fn emit_event(app: &AppHandle, event: BootstrapEvent) {
    match &event {
        BootstrapEvent::Started { repo_root, .. } => {
            tracing::info!(%repo_root, "bootstrap started");
        }
        BootstrapEvent::Complete { data_dir, .. } => {
            tracing::info!(%data_dir, "bootstrap complete");
        }
        BootstrapEvent::Failed { error } => {
            tracing::error!(%error, "bootstrap FAILED");
        }
        BootstrapEvent::Log { .. } => {}
    }
    if let Err(e) = app.emit(BootstrapEvent::CHANNEL, &event) {
        tracing::warn!(?e, "failed to emit bootstrap event");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_lines_prefers_stderr_and_bounds_length() {
        let stderr = "e1\n\ne2\ne3\ne4";
        let stdout = "o1\no2";
        assert_eq!(tail_lines(stderr, stdout, 2), "e3\ne4");
        assert_eq!(tail_lines("   \n\n", stdout, 5), "o1\no2");
        assert_eq!(tail_lines("", "", 5), "");
    }
}
