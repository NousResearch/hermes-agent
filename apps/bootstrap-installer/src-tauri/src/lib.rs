//! Hermes Setup — Tauri entrypoint.
//!
//! Spawns a single window pointed at the React frontend (apps/bootstrap-installer/src/).
//! All install-time work lives in `bootstrap.rs` and is invoked through the Tauri
//! commands registered at the bottom of `run()`.
//!
//! The Windows-subsystem strip lives on the binary crate (src/main.rs), not
//! here — a crate-level attribute on a lib doesn't propagate to the linker
//! flags of the executable that consumes it.

mod bootstrap;
mod events;
mod install_script;
mod powershell;
mod paths;
mod update;

use std::sync::Arc;
use tokio::sync::Mutex;

/// How the installer was invoked. Resolved once from the process args in
/// `run()` and exposed to the frontend via `get_mode` so it can route to the
/// install flow (first-run onboarding) or the update flow (driven by the
/// desktop app handing off via `Hermes-Setup.exe --update`).
///
/// Bare launch (double-click, first-run) => Install.
/// `--update` (spawned by the desktop's "Update" button) => Update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AppMode {
    Install,
    Update,
}

impl AppMode {
    /// Resolve the mode from an argument iterator. Anything containing the
    /// `--update` flag selects Update; otherwise Install. Kept arg-iterator
    /// generic (not reading `std::env` directly) so it's unit-testable.
    pub fn from_args<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for a in args {
            if a.as_ref() == "--update" {
                return AppMode::Update;
            }
        }
        AppMode::Install
    }
}

/// Process-wide install state, shared across Tauri commands.
///
/// The bootstrap is a one-shot, single-tenant process — we only need one
/// of these per window. `Arc<Mutex<...>>` lets command handlers grab it
/// without lifetime gymnastics.
pub struct AppState {
    pub bootstrap: Mutex<Option<bootstrap::BootstrapHandle>>,
    /// How this process was launched (install vs update). Immutable for the
    /// lifetime of the process; read by the `get_mode` command.
    pub mode: AppMode,
}

impl AppState {
    fn new(mode: AppMode) -> Self {
        Self {
            bootstrap: Mutex::new(None),
            mode,
        }
    }
}

/// Frontend → Rust: which flow should the UI render?
#[tauri::command]
fn get_mode(state: tauri::State<'_, Arc<AppState>>) -> AppMode {
    state.mode
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Tracing → bootstrap-installer.log under HERMES_HOME/logs/ so install
    // failures leave a trail for support. Console output also goes here in
    // debug builds.
    let _guard = paths::init_logging();

    let mode = AppMode::from_args(std::env::args().skip(1));
    tracing::info!(?mode, "Hermes Setup starting");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Arc::new(AppState::new(mode)))
        .invoke_handler(tauri::generate_handler![
            // Mode (install vs update)
            get_mode,
            // Bootstrap lifecycle
            bootstrap::start_bootstrap,
            bootstrap::cancel_bootstrap,
            bootstrap::get_bootstrap_status,
            // Update lifecycle
            update::start_update,
            // Hand-off
            bootstrap::launch_hermes_desktop,
            // Diagnostics
            paths::get_log_path,
            paths::get_hermes_home,
            paths::open_log_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Hermes Setup");
}

#[cfg(test)]
mod tests {
    use super::AppMode;

    #[test]
    fn bare_args_are_install() {
        assert_eq!(AppMode::from_args(Vec::<String>::new()), AppMode::Install);
        assert_eq!(AppMode::from_args(["--foo", "bar"]), AppMode::Install);
    }

    #[test]
    fn update_flag_selects_update() {
        assert_eq!(AppMode::from_args(["--update"]), AppMode::Update);
        assert_eq!(
            AppMode::from_args(["--something", "--update", "--else"]),
            AppMode::Update
        );
    }
}
