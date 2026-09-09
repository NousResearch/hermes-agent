//! Event types streamed from Rust → React over the single `bootstrap`
//! Tauri event channel. The `type` discriminator on each payload is how the
//! frontend routes.
//!
//! North Forge's bootstrap is ONE monolithic `bootstrap-north-forge.ps1`
//! run — no `-Manifest` / per-stage JSON protocol like the upstream Hermes
//! installer had. So there are no `Manifest` / `Stage` events: the UI shows a
//! single indeterminate progress view plus the live log, and we emit `Log`
//! lines as they stream, then exactly one `Complete` or `Failed`.

use serde::Serialize;

/// Which pipe a raw log line came from. Reported as structured metadata so
/// the UI can style stderr subtly rather than mislabeling it as an error:
/// uv/pip/git write normal progress to stderr by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogStream {
    Stdout,
    Stderr,
}

/// The single event channel `bootstrap` emits these. `type` discriminates.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BootstrapEvent {
    /// Sent once when the PowerShell child has been spawned, so the UI can
    /// switch to the progress view even before the first line of output.
    Started {
        script: String,
        #[serde(rename = "repoRoot")]
        repo_root: String,
    },
    /// Raw stdout/stderr line from bootstrap-north-forge.ps1. `stream` tells
    /// the UI which pipe it came from so stderr can be styled subtly instead
    /// of being mislabeled as an error.
    Log { line: String, stream: LogStream },
    /// Sent once when the script exits 0.
    Complete {
        #[serde(rename = "repoRoot")]
        repo_root: String,
        #[serde(rename = "venvDir")]
        venv_dir: String,
        #[serde(rename = "dataDir")]
        data_dir: String,
        #[serde(rename = "launcherCmd", skip_serializing_if = "Option::is_none")]
        launcher_cmd: Option<String>,
    },
    /// Sent once if the run aborts (non-zero exit, spawn failure, or cancel).
    Failed { error: String },
}

impl BootstrapEvent {
    /// Tauri event name. Single channel for all bootstrap events; the
    /// `type` tag tells the renderer how to interpret the payload.
    pub const CHANNEL: &'static str = "bootstrap";
}
