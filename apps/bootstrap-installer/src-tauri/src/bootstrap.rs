//! Bootstrap orchestration.
//!
//! Direct port of `runBootstrap` from `apps/desktop/electron/bootstrap-runner.cjs`.
//! Drives install.ps1 / install.sh stage-by-stage, emits progress events
//! over the Tauri `bootstrap` channel, writes a forensic log to
//! HERMES_HOME/logs/bootstrap-<timestamp>.log.
//!
//! Lifecycle:
//!   1. `start_bootstrap` (Tauri command) → spawns the worker task.
//!   2. Worker resolves install script (dev/cache/download).
//!   3. Worker builds a native manifest → emits `manifest` event.
//!   4. Worker iterates stages, calling `install.ps1 -Stage NAME -NonInteractive -Json`.
//!   5. On success → `complete`. On any stage failure → `failed`. On cancel → `failed`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager, State};
use tokio::sync::{mpsc, Mutex};

use crate::events::{BootstrapEvent, LogStream, StageState};
use crate::install_script::{self, Pin, ScriptKind, ScriptSource};
use crate::powershell::{self, StreamSink};
use crate::AppState;

// ---------------------------------------------------------------------------
// Public Tauri commands
// ---------------------------------------------------------------------------

/// Frontend → Rust: kick off the install.
#[derive(Debug, Deserialize)]
pub struct StartBootstrapArgs {
    /// Optional override for the commit pin. Defaults to the build-time
    /// pin baked in via `BUILD_PIN_COMMIT`.
    pub commit: Option<String>,
    /// Optional override for the branch pin. Defaults to `BUILD_PIN_BRANCH`.
    pub branch: Option<String>,
    /// Include Stage-Desktop (build apps/desktop) in the manifest. The
    /// signed bootstrap installer passes true; the deprecated Electron-side
    /// bootstrap-runner passes false to avoid building-while-running.
    #[serde(default = "default_true")]
    pub include_desktop: bool,
    /// Optional override for HERMES_HOME. Tests use this; production
    /// almost always falls back to the OS default.
    pub hermes_home: Option<String>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct BootstrapStatus {
    pub running: bool,
    pub completed: bool,
    pub install_root: Option<String>,
    pub last_error: Option<String>,
}

/// Handle stored in AppState while a bootstrap run is in flight. Carries
/// the cancellation channel and the most recent terminal status so the
/// frontend can re-query after a window refresh.
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
    let handle = BootstrapHandle {
        cancel_tx,
        started_at: Instant::now(),
        status: BootstrapStatus {
            running: true,
            completed: false,
            install_root: None,
            last_error: None,
        },
    };
    *guard = Some(handle);
    drop(guard);

    let app_for_task = app.clone();
    let state_for_task = state.inner().clone();
    let args_for_task = args;
    let cancel_rx = Arc::new(Mutex::new(Some(cancel_rx)));

    tokio::spawn(async move {
        let result = run_bootstrap(app_for_task.clone(), args_for_task, cancel_rx).await;

        // Reflect terminal state into AppState so get_bootstrap_status()
        // can serve it after the task exits.
        let mut guard = state_for_task.bootstrap.lock().await;
        if let Some(h) = guard.as_mut() {
            h.status.running = false;
            match &result {
                Ok(install_root) => {
                    h.status.completed = true;
                    h.status.install_root = Some(install_root.clone());
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
            install_root: h.status.install_root.clone(),
            last_error: h.status.last_error.clone(),
        },
        None => BootstrapStatus {
            running: false,
            completed: false,
            install_root: None,
            last_error: None,
        },
    })
}

/// Spawn the locally-built Hermes desktop binary, then close the installer
/// window. Caller resolves the binary path from `install_root`.
///
/// Returns Err with a human-readable message if the binary doesn't exist
/// (e.g. when Stage-Desktop was skipped) so the frontend can present
/// actionable failure UI rather than silently doing nothing.
#[tauri::command]
pub async fn launch_hermes_desktop(
    app: AppHandle,
    install_root: String,
) -> Result<(), String> {
    let install_root = PathBuf::from(install_root);
    let exe_path = resolve_hermes_desktop_exe(&install_root).ok_or_else(|| {
        format!(
            "Couldn't find a built Hermes desktop at {}. The desktop build step \
             may have been skipped or failed. Run `hermes desktop` from a \
             terminal to build and launch it.",
            install_root.join("apps").join("desktop").join("release").display()
        )
    })?;

    tracing::info!(?exe_path, "launching Hermes desktop");

    // Detach from us — the installer is about to exit. On macOS launch the
    // bundle through LaunchServices instead of exec'ing Contents/MacOS/Hermes
    // directly; this matches user double-click/open behavior and avoids cwd /
    // quarantine oddities after a self-update rebuild.
    let mut cmd = desktop_launch_command(&exe_path, &install_root);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS = 0x00000008
        cmd.creation_flags(0x0000_0008);
    }

    cmd.spawn().map_err(|e| {
        format!(
            "failed to launch {}: {e}",
            exe_path.display()
        )
    })?;

    // Give Windows ~150ms to actually start the new process before we exit.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    // Exit the installer cleanly. Tauri's process plugin gives us the
    // right hook regardless of platform.
    app.exit(0);
    Ok(())
}

/// Walks the well-known electron-builder unpacked-app paths under
/// `install_root`. Mirrors the resolver in `cmd_gui` (apps/desktop/release/
/// <os>-unpacked/<exe>).
pub(crate) fn resolve_hermes_desktop_exe(install_root: &std::path::Path) -> Option<PathBuf> {
    let release_dir = install_root.join("apps").join("desktop").join("release");
    let candidates: &[(&str, &str)] = if cfg!(target_os = "windows") {
        &[
            ("win-unpacked", "Hermes.exe"),
            ("win-arm64-unpacked", "Hermes.exe"),
        ]
    } else if cfg!(target_os = "macos") {
        &[
            ("mac/Hermes.app/Contents/MacOS", "Hermes"),
            ("mac-arm64/Hermes.app/Contents/MacOS", "Hermes"),
        ]
    } else {
        &[("linux-unpacked", "hermes")]
    };
    for (subdir, exe) in candidates {
        let p = release_dir.join(subdir).join(exe);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub(crate) fn resolve_hermes_desktop_app(install_root: &std::path::Path) -> Option<PathBuf> {
    let exe = resolve_hermes_desktop_exe(install_root)?;
    #[cfg(target_os = "macos")]
    {
        // .../Hermes.app/Contents/MacOS/Hermes -> .../Hermes.app
        let app = exe.parent()?.parent()?.parent()?.to_path_buf();
        if app.extension().and_then(|e| e.to_str()) == Some("app") && app.is_dir() {
            return Some(app);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        return Some(exe);
    }
    #[allow(unreachable_code)]
    None
}

/// True when a prior install completed (bootstrap-complete marker present) AND a
/// launchable desktop app exists on disk. Used by the installer's launcher fast
/// path so a bare re-open just opens Hermes instead of re-running setup.
pub(crate) fn hermes_is_installed(install_root: &std::path::Path) -> bool {
    install_root.join(".hermes-bootstrap-complete").exists()
        && resolve_hermes_desktop_exe(install_root).is_some()
}

/// Spawn the already-built desktop app, detached. Returns Err if no built app
/// exists or the spawn fails, so the caller can fall back to showing the
/// installer UI.
pub(crate) fn spawn_installed_desktop(install_root: &std::path::Path) -> std::io::Result<()> {
    let exe = resolve_hermes_desktop_exe(install_root).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "no built Hermes desktop app")
    })?;
    let mut cmd = desktop_launch_command_std(&exe, install_root);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS = 0x00000008 — keep the desktop alive after the
        // installer exits, mirroring launch_hermes_desktop. Kept correct here
        // even though the only caller is macOS-gated today, so future reuse on
        // Windows doesn't reintroduce the relaunch race.
        cmd.creation_flags(0x0000_0008);
    }
    cmd.spawn().map(|_child| ())
}

#[cfg(target_os = "macos")]
pub(crate) fn open_macos_app_detached(app_bundle: &std::path::Path) -> std::io::Result<()> {
    let mut cmd = std::process::Command::new("/usr/bin/open");
    cmd.arg(app_bundle);
    cmd.current_dir(crate::paths::hermes_home());
    cmd.spawn().map(|_child| ())
}

#[cfg(target_os = "macos")]
fn app_bundle_for_exe(exe: &std::path::Path) -> Option<PathBuf> {
    let app = exe.parent()?.parent()?.parent()?.to_path_buf();
    if app.extension().and_then(|e| e.to_str()) == Some("app") && app.is_dir() {
        Some(app)
    } else {
        None
    }
}

fn desktop_launch_command(
    exe_path: &std::path::Path,
    install_root: &std::path::Path,
) -> tokio::process::Command {
    #[cfg(target_os = "macos")]
    {
        if let Some(app_bundle) = app_bundle_for_exe(exe_path) {
            let mut cmd = tokio::process::Command::new("/usr/bin/open");
            cmd.arg(app_bundle);
            cmd.current_dir(crate::paths::hermes_home());
            return cmd;
        }
    }

    let mut cmd = tokio::process::Command::new(exe_path);
    cmd.current_dir(exe_path.parent().unwrap_or(install_root));
    cmd
}

fn desktop_launch_command_std(
    exe_path: &std::path::Path,
    install_root: &std::path::Path,
) -> std::process::Command {
    #[cfg(target_os = "macos")]
    {
        if let Some(app_bundle) = app_bundle_for_exe(exe_path) {
            let mut cmd = std::process::Command::new("/usr/bin/open");
            cmd.arg(app_bundle);
            cmd.current_dir(crate::paths::hermes_home());
            return cmd;
        }
    }

    let mut cmd = std::process::Command::new(exe_path);
    cmd.current_dir(exe_path.parent().unwrap_or(install_root));
    cmd
}

// ---------------------------------------------------------------------------
// Bootstrap implementation
// ---------------------------------------------------------------------------

async fn run_bootstrap(
    app: AppHandle,
    args: StartBootstrapArgs,
    cancel_rx_holder: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
) -> Result<String> {
    let kind = ScriptKind::for_current_os();

    let pin = Pin {
        commit: args.commit.or_else(|| option_env_string("BUILD_PIN_COMMIT")),
        branch: args.branch.or_else(|| option_env_string("BUILD_PIN_BRANCH")),
    };

    tracing::info!(
        ?pin,
        kind = ?kind,
        include_desktop = args.include_desktop,
        "bootstrap starting"
    );

    let app_for_log = app.clone();
    let emit_log = move |line: &str| {
        emit_event(
            &app_for_log,
            BootstrapEvent::Log {
                stage: None,
                line: line.to_string(),
                stream: LogStream::Stdout,
            },
        );
        // Bump to info-level so the line shows in bootstrap-installer.log
        // under the default INFO filter. Previously this was debug! which
        // got dropped on the floor, leaving us blind whenever install.ps1
        // failed — the log only had the "bootstrap starting" banner.
        tracing::info!(target: "bootstrap.log", "{line}");
    };

    // 1. Resolve install.ps1
    let script = install_script::resolve(kind, &pin, &emit_log)
        .await
        .map_err(|e| {
            let msg = format!("resolve install script failed: {e:#}");
            emit_event(
                &app,
                BootstrapEvent::Failed {
                    stage: None,
                    error: msg.clone(),
                },
            );
            anyhow!(msg)
        })?;

    let source_note = match &script.source {
        ScriptSource::DevCheckout => "dev checkout",
        ScriptSource::Bundled => "bundled",
        ScriptSource::Cached => "cached",
        ScriptSource::Downloaded => "downloaded",
    };
    emit_log(&format!(
        "[bootstrap] script {} via {}",
        script.path.display(),
        source_note
    ));
    let bundled_scripts = install_script::bundled_script_manifest()
        .into_iter()
        .map(|resource| {
            format!(
                "{}={}b:{}",
                resource.filename,
                resource.size_bytes,
                &resource.sha256[..12]
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    emit_log(&format!("[bootstrap] bundled scripts [{bundled_scripts}]"));

    // 2. Build manifest natively. The script remains the executor for stages
    // that have not reached Rust parity, but bootstrapping no longer needs a
    // shell process just to learn the stage list.
    let manifest_args = build_pin_args(&script);
    let manifest = crate::orchestrator::native_bootstrap_manifest(kind, args.include_desktop);
    emit_log(&format!(
        "[bootstrap] native manifest generated: stages={}",
        manifest.stages.len()
    ));

    let hermes_home_for_report = args
        .hermes_home
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(crate::paths::hermes_home);
    let install_state = crate::orchestrator::probe_install_state(&hermes_home_for_report);
    let rust_plan = crate::orchestrator::build_stage_plan(&manifest.stages, args.include_desktop);
    let bundled_tools_dir = bootstrap_tools_resource_dir(&app);
    if let Some(path) = &bundled_tools_dir {
        emit_log(&format!("[bootstrap] bundled tool archives at {}", path.display()));
    }
    emit_log(&crate::orchestrator::summarize_plan(
        &install_state,
        &rust_plan,
    ));

    emit_event(
        &app,
        BootstrapEvent::Manifest {
            stages: manifest.stages.clone(),
            protocol_version: manifest.protocol_version,
        },
    );

    // 3. Iterate stages.
    for stage in &manifest.stages {
        // Skip Stage-Desktop unless explicitly requested. install.ps1 may
        // or may not include it in the manifest depending on the flag we
        // pass, but if it slipped in, gate client-side too.
        if !args.include_desktop && stage.name.eq_ignore_ascii_case("desktop") {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(0),
                    result: None,
                    error: Some("skipped by include_desktop=false".into()),
                },
            );
            continue;
        }

        if cancellation_signalled(&cancel_rx_holder).await {
            let err = "bootstrap cancelled by user".to_string();
            emit_event(
                &app,
                BootstrapEvent::Failed {
                    stage: Some(stage.name.clone()),
                    error: err.clone(),
                },
            );
            return Err(anyhow!(err));
        }

        let started = Instant::now();
        emit_event(
            &app,
            BootstrapEvent::Stage {
                name: stage.name.clone(),
                state: StageState::Running,
                duration_ms: None,
                result: None,
                error: None,
            },
        );

        let hermes_home = args
            .hermes_home
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(crate::paths::hermes_home);
        let install_root = hermes_home.join("hermes-agent");
        if let Some(frame) = crate::orchestrator::interactive_stage_skip_result(stage) {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(frame),
                    error: None,
                },
            );
            continue;
        }

        if should_defer_git_stage_for_archive_install(&stage.name, &install_root) {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(crate::events::StageResultPayload {
                        stage: stage.name.clone(),
                        ok: true,
                        skipped: true,
                        reason: Some("deferred until repository archive fallback is needed".into()),
                        data: Some(serde_json::json!({
                            "installRoot": install_root,
                            "nativeRepositoryArchive": true,
                        })),
                    }),
                    error: None,
                },
            );
            continue;
        }

        if let Some(frame) =
            crate::orchestrator::satisfied_tool_stage_skip_result_from_env(stage, &hermes_home)
        {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(frame),
                    error: None,
                },
            );
            continue;
        }

        if let Some(frame) = crate::orchestrator::python_stage_skip_result(stage, &hermes_home) {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(frame),
                    error: None,
                },
            );
            continue;
        }

        if let Some(frame) = crate::orchestrator::platform_sdks_skip_result(stage, &hermes_home) {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(frame),
                    error: None,
                },
            );
            continue;
        }

        if let Some(frame) =
            crate::orchestrator::node_deps_skip_result_from_env(stage, &hermes_home)
        {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(frame),
                    error: None,
                },
            );
            continue;
        }

        if let Some(frame) =
            crate::orchestrator::desktop_stage_skip_result(stage, &install_root)
        {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Skipped,
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    result: Some(frame),
                    error: None,
                },
            );
            continue;
        }

        if !cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("prerequisites") {
            match crate::orchestrator::install_unix_node_runtime_stage(&hermes_home).await {
                Ok(data) => emit_log(&format!(
                    "[bootstrap] native Unix Node preparation succeeded: {data}"
                )),
                Err(err) => emit_log(&format!(
                    "[bootstrap] warning: native Unix Node preparation failed; \
                     prerequisites script will handle Node fallback: {err}"
                )),
            }
        }

        let native_stage_result = {
            if stage.name.eq_ignore_ascii_case("bootstrap-marker") {
                Some(crate::orchestrator::write_bootstrap_marker(
                    &install_root,
                    pin.commit.as_deref(),
                    pin.branch.as_deref(),
                ))
            } else if is_config_template_stage(&stage.name) {
                Some(crate::orchestrator::configure_templates(
                    &hermes_home,
                    &install_root,
                ))
            } else if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("path") {
                Some(crate::orchestrator::configure_windows_path_stage(
                    &hermes_home,
                    &install_root,
                ))
            } else if !cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("path") {
                Some(crate::orchestrator::configure_unix_path_stage(&install_root))
            } else if stage.name.eq_ignore_ascii_case("complete") {
                Some(crate::orchestrator::write_install_method_stamp(&hermes_home))
            } else if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("uv") {
                Some(
                    crate::orchestrator::install_windows_uv_runtime_stage(
                        &hermes_home,
                        bundled_tools_dir.as_deref(),
                    )
                    .await,
                )
            } else if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("git") {
                Some(
                    crate::orchestrator::install_windows_git_runtime_stage(
                        &hermes_home,
                        bundled_tools_dir.as_deref(),
                    )
                    .await,
                )
            } else if stage.name.eq_ignore_ascii_case("python") {
                Some(crate::orchestrator::install_python_runtime_stage(&hermes_home))
            } else if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("node") {
                Some(
                    crate::orchestrator::install_windows_node_runtime_stage(
                        &hermes_home,
                        bundled_tools_dir.as_deref(),
                    )
                    .await,
                )
            } else if stage.name.eq_ignore_ascii_case("venv") {
                Some(crate::orchestrator::create_python_venv_stage(
                    &install_root,
                    &hermes_home,
                ))
            } else if is_python_dependencies_stage(&stage.name) {
                Some(crate::orchestrator::sync_python_dependencies_stage(
                    &install_root,
                    &hermes_home,
                ))
            } else if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("node-deps") {
                Some(crate::orchestrator::install_windows_node_dependencies_stage(
                    &install_root,
                    &hermes_home,
                ))
            } else if stage.name.eq_ignore_ascii_case("platform-sdks") {
                Some(crate::orchestrator::install_platform_sdks_stage(
                    &hermes_home,
                    &install_root,
                ))
            } else if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("desktop") {
                Some(crate::orchestrator::build_windows_desktop_stage(
                    &install_root,
                    &hermes_home,
                ))
            } else if should_try_native_repository_archive(&stage.name, &install_root) {
                Some(
                    crate::orchestrator::install_repository_archive_fresh(
                        &install_root,
                        pin.commit.as_deref(),
                        pin.branch.as_deref(),
                    )
                    .await,
                )
            } else {
                None
            }
        };

        if let Some(native_stage_result) = native_stage_result {
            match native_stage_result {
                Ok(data) => {
                    if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("desktop") {
                        if let Err(err) = create_windows_desktop_shortcuts(&install_root) {
                            tracing::warn!(?err, "failed to create desktop shortcuts via manager");
                            emit_log(&format!(
                                "[bootstrap] warning: could not create desktop shortcuts: {err}"
                            ));
                        }
                    }
                    emit_event(
                        &app,
                        BootstrapEvent::Stage {
                            name: stage.name.clone(),
                            state: StageState::Succeeded,
                            duration_ms: Some(started.elapsed().as_millis() as u64),
                            result: Some(crate::events::StageResultPayload {
                                stage: stage.name.clone(),
                                ok: true,
                                skipped: false,
                                reason: None,
                                data: Some(data),
                            }),
                            error: None,
                        },
                    );
                    continue;
                }
                Err(err) => {
                    let err = err.to_string();
                    if should_fallback_repository_archive(&stage.name, &install_root) {
                        emit_log(&format!(
                            "[bootstrap] warning: native repository archive failed; \
                             falling back to script-backed repository stage: {err}"
                        ));
                        if cfg!(target_os = "windows") {
                            let git_args =
                                stage_script_args("git", &manifest_args, args.include_desktop);
                            let git_result = run_install_script(
                                &app,
                                &script.path,
                                &git_args,
                                args.hermes_home.as_deref(),
                                &[],
                                None,
                                Some("git".to_string()),
                            )
                            .await?;
                            let git_ok = powershell::parse_stage_result(&git_result.stdout)
                                .map(|frame| frame.ok)
                                .unwrap_or(false);
                            if git_ok && git_result.exit_code == Some(0) {
                                emit_log(
                                    "[bootstrap] Git fallback stage succeeded; \
                                     continuing repository stage",
                                );
                            } else {
                                let fallback_err = format!(
                                    "native repository archive failed ({err}); \
                                     fallback Git stage failed with exit {:?}",
                                    git_result.exit_code
                                );
                                emit_event(
                                    &app,
                                    BootstrapEvent::Stage {
                                        name: stage.name.clone(),
                                        state: StageState::Failed,
                                        duration_ms: Some(started.elapsed().as_millis() as u64),
                                        result: None,
                                        error: Some(fallback_err.clone()),
                                    },
                                );
                                emit_event(
                                    &app,
                                    BootstrapEvent::Failed {
                                        stage: Some(stage.name.clone()),
                                        error: fallback_err.clone(),
                                    },
                                );
                                return Err(anyhow!(fallback_err));
                            }
                        }
                    } else if should_fallback_native_stage(&stage.name, &install_root) {
                        emit_log(&format!(
                            "[bootstrap] warning: native {} stage failed; \
                             falling back to script-backed stage: {err}",
                            stage.name
                        ));
                    } else {
                        emit_event(
                            &app,
                            BootstrapEvent::Stage {
                                name: stage.name.clone(),
                                state: StageState::Failed,
                                duration_ms: Some(started.elapsed().as_millis() as u64),
                                result: None,
                                error: Some(err.clone()),
                            },
                        );
                        emit_event(
                            &app,
                            BootstrapEvent::Failed {
                                stage: Some(stage.name.clone()),
                                error: err.clone(),
                            },
                        );
                        return Err(anyhow!(err));
                    }
                }
            }
        }

        let stage_args = stage_script_args(&stage.name, &manifest_args, args.include_desktop);
        let stage_extra_env = stage_script_extra_env(&stage.name, &install_root);

        // Each stage gets its own cancel receiver because tokio::select!
        // in run_script consumes it. Take/return through the Arc<Mutex>.
        let local_cancel_rx = cancel_rx_holder.lock().await.take();

        let stage_result = run_install_script(
            &app,
            &script.path,
            &stage_args,
            args.hermes_home.as_deref(),
            &stage_extra_env,
            local_cancel_rx,
            Some(stage.name.clone()),
        )
        .await?;

        let duration_ms = started.elapsed().as_millis() as u64;

        if stage_result.killed {
            emit_event(
                &app,
                BootstrapEvent::Stage {
                    name: stage.name.clone(),
                    state: StageState::Failed,
                    duration_ms: Some(duration_ms),
                    result: None,
                    error: Some("cancelled by user".into()),
                },
            );
            emit_event(
                &app,
                BootstrapEvent::Failed {
                    stage: Some(stage.name.clone()),
                    error: "cancelled by user".into(),
                },
            );
            return Err(anyhow!("cancelled by user"));
        }

        let result_frame = powershell::parse_stage_result(&stage_result.stdout);

        match result_frame {
            None => {
                let err = format!(
                    "install.ps1 -Stage {} produced no JSON result frame (exit={:?})",
                    stage.name, stage_result.exit_code
                );
                emit_event(
                    &app,
                    BootstrapEvent::Stage {
                        name: stage.name.clone(),
                        state: StageState::Failed,
                        duration_ms: Some(duration_ms),
                        result: None,
                        error: Some(err.clone()),
                    },
                );
                emit_event(
                    &app,
                    BootstrapEvent::Failed {
                        stage: Some(stage.name.clone()),
                        error: err.clone(),
                    },
                );
                return Err(anyhow!(err));
            }
            Some(frame) if frame.ok && frame.skipped => {
                emit_event(
                    &app,
                    BootstrapEvent::Stage {
                        name: stage.name.clone(),
                        state: StageState::Skipped,
                        duration_ms: Some(duration_ms),
                        result: Some(frame),
                        error: None,
                    },
                );
            }
            Some(frame) if frame.ok => {
                if cfg!(target_os = "windows") && stage.name.eq_ignore_ascii_case("desktop") {
                    let hermes_home = args
                        .hermes_home
                        .as_ref()
                        .map(PathBuf::from)
                        .unwrap_or_else(crate::paths::hermes_home);
                    let install_root = hermes_home.join("hermes-agent");
                    if let Err(err) = create_windows_desktop_shortcuts(&install_root) {
                        tracing::warn!(?err, "failed to create desktop shortcuts via manager");
                        emit_log(&format!(
                            "[bootstrap] warning: could not create desktop shortcuts: {err}"
                        ));
                    }
                }
                emit_event(
                    &app,
                    BootstrapEvent::Stage {
                        name: stage.name.clone(),
                        state: StageState::Succeeded,
                        duration_ms: Some(duration_ms),
                        result: Some(frame),
                        error: None,
                    },
                );
            }
            Some(frame) => {
                let err = frame
                    .reason
                    .clone()
                    .unwrap_or_else(|| format!("exit code {:?}", stage_result.exit_code));
                emit_event(
                    &app,
                    BootstrapEvent::Stage {
                        name: stage.name.clone(),
                        state: StageState::Failed,
                        duration_ms: Some(duration_ms),
                        result: Some(frame),
                        error: Some(err.clone()),
                    },
                );
                emit_event(
                    &app,
                    BootstrapEvent::Failed {
                        stage: Some(stage.name.clone()),
                        error: err.clone(),
                    },
                );
                return Err(anyhow!(err));
            }
        }
    }

    // 4. Resolve install_root. install.ps1 doesn't (yet) report this back
    // explicitly; we infer it from $HermesHome which Stage-Repository clones
    // the repo INTO at $HermesHome\hermes-agent. Mirrors hermes_constants.
    let hermes_home = args
        .hermes_home
        .clone()
        .unwrap_or_else(|| crate::paths::hermes_home().to_string_lossy().into_owned());
    let install_root = PathBuf::from(&hermes_home).join("hermes-agent");

    // Copy ourselves to HERMES_HOME/hermes-setup.exe so the desktop app can
    // re-invoke us with `--update` and shortcuts have a stable target. This is
    // a one-shot install concern; an `--update` re-invocation no-ops because
    // we're already running from that path. Best-effort — a failure here must
    // not fail an otherwise-successful install.
    if let Err(err) = crate::paths::copy_self_to_hermes_home() {
        tracing::warn!(?err, "failed to copy installer into HERMES_HOME (non-fatal)");
        emit_log(&format!(
            "[bootstrap] warning: could not stage updater binary: {err}"
        ));
    }

    let hermes_home_path = PathBuf::from(&hermes_home);
    if !record_manager_install_metadata(&hermes_home_path) {
        emit_log("[bootstrap] warning: could not record manager install metadata");
    }

    emit_event(
        &app,
        BootstrapEvent::Complete {
            install_root: install_root.to_string_lossy().into_owned(),
            marker: Some(serde_json::json!({
                "pinnedCommit": pin.commit,
                "pinnedBranch": pin.branch,
            })),
        },
    );

    Ok(install_root.to_string_lossy().into_owned())
}

fn stage_script_args(stage_name: &str, manifest_args: &[String], include_desktop: bool) -> Vec<String> {
    let mut stage_args = vec![
        "-Stage".to_string(),
        stage_name.to_string(),
        "-NonInteractive".to_string(),
        "-Json".to_string(),
    ];
    stage_args.extend(manifest_args.iter().cloned());
    if include_desktop {
        stage_args.push("-IncludeDesktop".to_string());
    }
    if cfg!(target_os = "windows") && stage_name.eq_ignore_ascii_case("desktop") {
        stage_args.push("-SkipDesktopShortcuts".to_string());
    }
    stage_args
}

fn is_config_template_stage(stage_name: &str) -> bool {
    stage_name.eq_ignore_ascii_case("config-templates") || stage_name.eq_ignore_ascii_case("config")
}

fn should_use_native_repository_archive(install_root: &std::path::Path) -> bool {
    !install_root.exists()
}

fn should_defer_git_stage_for_archive_install(
    stage_name: &str,
    install_root: &std::path::Path,
) -> bool {
    cfg!(target_os = "windows")
        && stage_name.eq_ignore_ascii_case("git")
        && should_use_native_repository_archive(install_root)
}

fn should_try_native_repository_archive(stage_name: &str, install_root: &std::path::Path) -> bool {
    stage_name.eq_ignore_ascii_case("repository")
        && should_use_native_repository_archive(install_root)
}

fn should_fallback_repository_archive(stage_name: &str, install_root: &std::path::Path) -> bool {
    should_try_native_repository_archive(stage_name, install_root)
}

fn should_fallback_native_stage(stage_name: &str, install_root: &std::path::Path) -> bool {
    should_fallback_repository_archive(stage_name, install_root)
        || stage_name.eq_ignore_ascii_case("venv")
        || (cfg!(target_os = "windows") && stage_name.eq_ignore_ascii_case("uv"))
        || (cfg!(target_os = "windows") && stage_name.eq_ignore_ascii_case("git"))
        || stage_name.eq_ignore_ascii_case("python")
        || (cfg!(target_os = "windows") && stage_name.eq_ignore_ascii_case("node"))
        || is_python_dependencies_stage(stage_name)
        || stage_name.eq_ignore_ascii_case("platform-sdks")
        || (cfg!(target_os = "windows") && stage_name.eq_ignore_ascii_case("node-deps"))
        || (cfg!(target_os = "windows") && stage_name.eq_ignore_ascii_case("desktop"))
}

fn is_python_dependencies_stage(stage_name: &str) -> bool {
    stage_name.eq_ignore_ascii_case("dependencies") || stage_name.eq_ignore_ascii_case("python-deps")
}

fn stage_script_extra_env(
    stage_name: &str,
    install_root: &std::path::Path,
) -> Vec<(&'static str, &'static str)> {
    if stage_name.eq_ignore_ascii_case("prerequisites")
        && should_use_native_repository_archive(install_root)
    {
        vec![("HERMES_NATIVE_REPOSITORY_ARCHIVE", "1")]
    } else {
        Vec::new()
    }
}

fn create_windows_desktop_shortcuts(install_root: &std::path::Path) -> anyhow::Result<()> {
    let target_exe = resolve_hermes_desktop_exe(install_root).ok_or_else(|| {
        anyhow!(
            "desktop build succeeded but no Hermes.exe was found under {}",
            install_root.join("apps").join("desktop").join("release").display()
        )
    })?;
    let programs_dir = default_windows_programs_dir();
    let desktop_dir = default_windows_desktop_dir();
    let icon_exists = target_exe
        .parent()
        .map(|parent| parent.join("resources").join("icon.ico").is_file())
        .unwrap_or(false);
    let plans = hermes_manager::platform::plan_windows_shortcuts(
        &target_exe,
        &programs_dir,
        &desktop_dir,
        icon_exists,
    );
    hermes_manager::platform::write_windows_shortcuts(&plans)
        .map_err(|err| anyhow!("manager write-shortcuts failed: {err}"))
}

fn default_windows_programs_dir() -> PathBuf {
    std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Microsoft")
        .join("Windows")
        .join("Start Menu")
        .join("Programs")
}

fn default_windows_desktop_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Desktop")
}

fn record_manager_install_metadata(hermes_home: &std::path::Path) -> bool {
    match hermes_manager::commands::install_metadata(hermes_home) {
        Ok(()) => {
            tracing::info!(
                hermes_home = %hermes_home.display(),
                "manager install metadata recorded"
            );
            true
        }
        Err(err) => {
            tracing::warn!(
                hermes_home = %hermes_home.display(),
                ?err,
                "failed to record manager install metadata"
            );
            false
        }
    }
}

async fn cancellation_signalled(holder: &Arc<Mutex<Option<mpsc::Receiver<()>>>>) -> bool {
    let mut guard = holder.lock().await;
    if let Some(rx) = guard.as_mut() {
        rx.try_recv().is_ok()
    } else {
        false
    }
}

fn bootstrap_tools_resource_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .resolve("bootstrap-tools", BaseDirectory::Resource)
        .ok()
        .filter(|path| path.is_dir())
}

async fn run_install_script(
    app: &AppHandle,
    script_path: &std::path::Path,
    args: &[String],
    hermes_home_override: Option<&str>,
    extra_env: &[(&str, &str)],
    cancel_rx: Option<mpsc::Receiver<()>>,
    stage_name: Option<String>,
) -> Result<powershell::ScriptResult> {
    let app_for_stdout = app.clone();
    let stage_for_stdout = stage_name.clone();
    let app_for_stderr = app.clone();
    let stage_for_stderr = stage_name.clone();
    let stage_for_stdout_log = stage_name.clone();
    let stage_for_stderr_log = stage_name.clone();

    let sink = StreamSink {
        on_stdout_line: Box::new(move |line: &str| {
            emit_event(
                &app_for_stdout,
                BootstrapEvent::Log {
                    stage: stage_for_stdout.clone(),
                    line: line.to_string(),
                    stream: LogStream::Stdout,
                },
            );
            // Tee to the rolling installer log so we have a persistent
            // record of every install.ps1 line. Without this, the only
            // log evidence of a failure was the Tauri event stream —
            // which gets discarded the moment the failure route mounts.
            match &stage_for_stdout_log {
                Some(name) => {
                    tracing::info!(target: "bootstrap.log", stage = %name, "{line}")
                }
                None => tracing::info!(target: "bootstrap.log", "{line}"),
            }
        }),
        on_stderr_line: Box::new(move |line: &str| {
            emit_event(
                &app_for_stderr,
                BootstrapEvent::Log {
                    stage: stage_for_stderr.clone(),
                    line: line.to_string(),
                    stream: LogStream::Stderr,
                },
            );
            // stderr-level lines get warn! so they're visually distinct
            // when scrolling through the log later.
            match &stage_for_stderr_log {
                Some(name) => {
                    tracing::warn!(target: "bootstrap.log", stage = %name, "stderr: {line}")
                }
                None => tracing::warn!(target: "bootstrap.log", "stderr: {line}"),
            }
        }),
    };

    powershell::run_script(script_path, args, sink, hermes_home_override, extra_env, cancel_rx)
        .await
        .map_err(|e| {
            tracing::error!(?e, "install script invocation failed");
            anyhow!("install script invocation failed: {e:#}")
        })
}

fn build_pin_args(script: &install_script::ResolvedScript) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(c) = &script.commit {
        out.push("-Commit".to_string());
        out.push(c.clone());
    }
    if let Some(b) = &script.branch {
        out.push("-Branch".to_string());
        out.push(b.clone());
    }
    out
}

fn emit_event(app: &AppHandle, event: BootstrapEvent) {
    // Tee important state transitions to the rolling installer log so
    // bootstrap-installer.log isn't just "starting" + final summary.
    // Log lines (the noisy stuff) handle their own tracing in
    // run_install_script's sink; here we cover the lifecycle frames.
    match &event {
        BootstrapEvent::Manifest { stages, .. } => {
            tracing::info!(
                stage_count = stages.len(),
                names = ?stages.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
                "manifest received"
            );
        }
        BootstrapEvent::Stage {
            name,
            state,
            duration_ms,
            error,
            ..
        } => {
            tracing::info!(
                stage = %name,
                ?state,
                duration_ms = ?duration_ms,
                error = ?error,
                "stage transition"
            );
        }
        BootstrapEvent::Complete { install_root, .. } => {
            tracing::info!(install_root = %install_root, "bootstrap complete");
        }
        BootstrapEvent::Failed { stage, error } => {
            tracing::error!(stage = ?stage, error = %error, "bootstrap FAILED");
        }
        BootstrapEvent::Log { .. } => {
            // Log lines are teed via the sink callbacks in
            // run_install_script — don't double-emit here.
        }
    }
    if let Err(e) = app.emit(BootstrapEvent::CHANNEL, &event) {
        tracing::warn!(?e, "failed to emit bootstrap event");
    }
}

fn option_env_string(key: &str) -> Option<String> {
    // option_env! only accepts literals, so we hardcode the known keys.
    let val = match key {
        "BUILD_PIN_COMMIT" => option_env!("BUILD_PIN_COMMIT"),
        "BUILD_PIN_BRANCH" => option_env!("BUILD_PIN_BRANCH"),
        _ => None,
    };
    val.map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::path::PathBuf;

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "hermes-bootstrap-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    // Build a fake built-desktop release tree at the platform's expected path
    // and return (install_root, expected_app_bundle_or_exe).
    fn make_release_tree(install_root: &Path) -> PathBuf {
        let release = install_root.join("apps").join("desktop").join("release");
        if cfg!(target_os = "macos") {
            let macos_dir = release
                .join("mac-arm64")
                .join("Hermes.app")
                .join("Contents")
                .join("MacOS");
            std::fs::create_dir_all(&macos_dir).unwrap();
            std::fs::write(macos_dir.join("Hermes"), b"#!/bin/sh\n").unwrap();
            macos_dir.parent().unwrap().parent().unwrap().to_path_buf() // .../Hermes.app
        } else if cfg!(target_os = "windows") {
            let dir = release.join("win-unpacked");
            std::fs::create_dir_all(&dir).unwrap();
            let exe = dir.join("Hermes.exe");
            std::fs::write(&exe, b"stub").unwrap();
            exe
        } else {
            let dir = release.join("linux-unpacked");
            std::fs::create_dir_all(&dir).unwrap();
            let exe = dir.join("hermes");
            std::fs::write(&exe, b"stub").unwrap();
            exe
        }
    }

    // The relaunch / install target is derived from the rebuilt desktop app.
    // On macOS this MUST resolve to the .app bundle (what `open` relaunches and
    // what the updater ditto's over /Applications/Hermes.app). A regression in
    // this derivation breaks the post-update auto-relaunch, so guard it.
    #[test]
    fn resolve_hermes_desktop_app_finds_built_bundle() {
        let root = unique_tmp_dir("app-ok");
        let expected = make_release_tree(&root);

        let resolved = resolve_hermes_desktop_app(&root)
            .expect("should resolve the freshly-built desktop app");

        #[cfg(target_os = "macos")]
        {
            assert_eq!(resolved, expected, "must resolve to the .app bundle");
            assert_eq!(
                resolved.extension().and_then(|e| e.to_str()),
                Some("app"),
                "relaunch target must be a .app bundle on macOS"
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(resolved, expected);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_hermes_desktop_app_is_none_without_a_build() {
        let root = unique_tmp_dir("app-none");
        // No release tree created.
        assert!(
            resolve_hermes_desktop_app(&root).is_none(),
            "no resolved app when nothing has been built"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn record_manager_install_metadata_writes_default_manifest() {
        let hermes_home = unique_tmp_dir("manager-metadata");
        let agent_root = hermes_home.join("hermes-agent");
        std::fs::create_dir_all(&agent_root).unwrap();

        assert!(record_manager_install_metadata(&hermes_home));
        assert!(hermes_home
            .join("manager")
            .join("installed-files.json")
            .exists());

        let _ = std::fs::remove_dir_all(&hermes_home);
    }

    #[test]
    fn desktop_stage_args_skip_script_shortcuts_on_windows_bootstrap() {
        let args = stage_script_args(
            "desktop",
            &["-Commit".to_string(), "abc123".to_string()],
            true,
        );

        assert!(args.contains(&"-IncludeDesktop".to_string()));
        assert_eq!(
            args.contains(&"-SkipDesktopShortcuts".to_string()),
            cfg!(target_os = "windows")
        );
    }

    #[test]
    fn git_stage_deferred_only_for_windows_fresh_install() {
        let root = unique_tmp_dir("git-defer");
        let install_root = root.join("hermes-agent");

        assert_eq!(
            should_defer_git_stage_for_archive_install("git", &install_root),
            cfg!(target_os = "windows")
        );
        assert!(should_fallback_repository_archive(
            "repository",
            &install_root
        ));
        assert!(!should_fallback_repository_archive("git", &install_root));

        std::fs::create_dir_all(&install_root).unwrap();
        assert!(!should_defer_git_stage_for_archive_install(
            "git",
            &install_root
        ));
        assert!(!should_defer_git_stage_for_archive_install(
            "repository",
            &install_root
        ));
        assert!(!should_fallback_repository_archive(
            "repository",
            &install_root
        ));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn venv_native_stage_failure_can_fall_back_to_script() {
        let root = unique_tmp_dir("venv-fallback");
        let install_root = root.join("hermes-agent");

        assert!(should_fallback_native_stage("venv", &install_root));
        assert!(should_fallback_native_stage("python", &install_root));
        assert!(should_fallback_native_stage("dependencies", &install_root));
        assert!(should_fallback_native_stage("python-deps", &install_root));
        assert!(should_fallback_native_stage("platform-sdks", &install_root));
        assert_eq!(
            should_fallback_native_stage("node-deps", &install_root),
            cfg!(target_os = "windows")
        );
        assert_eq!(
            should_fallback_native_stage("desktop", &install_root),
            cfg!(target_os = "windows")
        );
        assert_eq!(
            should_fallback_native_stage("node", &install_root),
            cfg!(target_os = "windows")
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn native_archive_signal_is_sent_to_fresh_prerequisites_only() {
        let root = unique_tmp_dir("native-archive-env");
        let install_root = root.join("hermes-agent");

        assert_eq!(
            stage_script_extra_env("prerequisites", &install_root),
            vec![("HERMES_NATIVE_REPOSITORY_ARCHIVE", "1")]
        );
        assert!(stage_script_extra_env("repository", &install_root).is_empty());
        assert!(should_try_native_repository_archive(
            "repository",
            &install_root
        ));
        assert!(should_fallback_repository_archive("repository", &install_root));

        std::fs::create_dir_all(&install_root).unwrap();
        assert!(stage_script_extra_env("prerequisites", &install_root).is_empty());
        assert!(!should_try_native_repository_archive(
            "repository",
            &install_root
        ));
        assert!(!should_fallback_repository_archive("repository", &install_root));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn config_template_stage_accepts_windows_and_unix_names() {
        assert!(is_config_template_stage("config-templates"));
        assert!(is_config_template_stage("config"));
        assert!(is_config_template_stage("CONFIG"));
        assert!(!is_config_template_stage("configure"));
    }
}
