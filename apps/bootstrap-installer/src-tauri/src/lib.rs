//! North Forge Setup — Tauri entrypoint.
//!
//! Spawns a single window pointed at the React frontend
//! (apps/bootstrap-installer/src/). All bootstrap work lives in
//! `bootstrap.rs` and is invoked through the Tauri commands registered at
//! the bottom of `run()`.
//!
//! Retargeted from upstream Hermes' `bootstrap-installer`: no `--update`
//! mode (North Forge updates via `git pull` + the launcher's own venv
//! self-heal), no macOS "already installed → relaunch the desktop app"
//! fast path (North Forge is CLI-only), no GitHub download of the install
//! script (it's already in the checkout on the drive).
//!
//! The Windows-subsystem strip lives on the binary crate (src/main.rs), not
//! here — a crate-level attribute on a lib doesn't propagate to the linker
//! flags of the executable that consumes it.

mod bootstrap;
mod events;
mod paths;
mod powershell;
mod repo;

use std::sync::Arc;
use tokio::sync::Mutex;

/// Process-wide install state, shared across Tauri commands.
///
/// The bootstrap is a one-shot, single-tenant process — one of these per
/// window. `Arc<Mutex<...>>` lets command handlers grab it without lifetime
/// gymnastics.
pub struct AppState {
    pub bootstrap: Mutex<Option<bootstrap::BootstrapHandle>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            bootstrap: Mutex::new(None),
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Tracing → north-forge-setup.log under %LOCALAPPDATA%\hermes\logs\ so
    // bootstrap failures leave a trail for support.
    let _guard = paths::init_logging();
    tracing::info!("North Forge Setup starting");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Arc::new(AppState::new()))
        .setup(move |app| {
            use tauri::Manager;
            // The window is created hidden (`"visible": false`) so we can
            // reveal it here without a flash. North Forge has no launcher
            // fast path — every run shows the setup UI.
            match app.get_webview_window("main") {
                Some(win) => {
                    if let Err(err) = win.show() {
                        tracing::error!(?err, "failed to show main setup window");
                    }
                }
                None => {
                    tracing::error!("main setup window not found; UI will not appear");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Checkout / drive resolution
            repo::detect_repo,
            repo::set_repo_root,
            // Bootstrap lifecycle
            bootstrap::start_bootstrap,
            bootstrap::cancel_bootstrap,
            bootstrap::get_bootstrap_status,
            // Hand-off
            bootstrap::launch_north_forge,
            // Diagnostics
            paths::get_log_path,
            paths::open_log_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running North Forge Setup");
}
