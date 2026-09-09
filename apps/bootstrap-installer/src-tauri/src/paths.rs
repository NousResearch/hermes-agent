//! The installer's own log location + logging setup.
//!
//! North Forge Setup writes its forensic log next to the engine's other
//! logs, under `%LOCALAPPDATA%\hermes\logs\` — BRANDING.md keeps
//! `%LOCALAPPDATA%\hermes` as engine-owned and not renamed. This is only
//! where the *installer* logs; the actual North Forge data dir is a
//! `<checkout>-data` sibling on the chosen drive and is resolved per-run in
//! `repo.rs`.

use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;

/// `%LOCALAPPDATA%\hermes` on Windows, `~/.hermes` elsewhere. Used only for
/// the installer's log directory and the diagnostics footer.
pub fn engine_home() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(local_app_data) = dirs::data_local_dir() {
            return local_app_data.join("hermes");
        }
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".hermes");
    }
    PathBuf::from(".hermes")
}

pub fn log_dir() -> PathBuf {
    engine_home().join("logs")
}

pub fn log_path() -> PathBuf {
    log_dir().join("north-forge-setup.log")
}

/// Initializes tracing to north-forge-setup.log under the engine log dir.
/// Returns a guard that flushes the appender on drop — keep it alive for
/// the lifetime of the process.
pub fn init_logging() -> Option<WorkerGuard> {
    let dir = log_dir();
    if let Err(err) = std::fs::create_dir_all(&dir) {
        // No log dir → log to stderr only. Don't panic; the installer
        // should still be usable on an exotic filesystem.
        eprintln!("[north-forge-setup] could not create log dir {dir:?}: {err}");
        return None;
    }

    let file_appender = tracing_appender::rolling::never(&dir, "north-forge-setup.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::try_from_env("NORTH_FORGE_SETUP_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(true)
        .init();

    Some(guard)
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn get_log_path() -> String {
    log_path().to_string_lossy().into_owned()
}

#[tauri::command]
pub fn open_log_dir(app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let path = log_dir();
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| e.to_string())
}
