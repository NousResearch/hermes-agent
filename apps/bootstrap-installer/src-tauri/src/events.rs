//! Event types streamed from Rust → React.
//!
//! These mirror `apps/desktop/electron/bootstrap-runner.ts`'s event shape
//! 1:1 so the React installer code can be roughly identical to the Electron
//! install-overlay we'll replace.
//!
//! The Tauri event channel name is `"bootstrap"` for all of these — the
//! `type` discriminator on each payload is how the frontend routes.

use serde::{Deserialize, Serialize};

/// Stage definition as reported by `install.ps1 -Manifest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfo {
    pub name: String,
    pub title: String,
    pub category: String,
    /// `needs_user_input=true` stages run with -NonInteractive and emit
    /// skipped=true; the post-install wizard takes over for those.
    #[serde(rename = "needs_user_input", alias = "needsUserInput")]
    pub needs_user_input: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub stages: Vec<StageInfo>,
    #[serde(rename = "protocol_version", alias = "protocolVersion", default)]
    pub protocol_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResultPayload {
    pub stage: String,
    pub ok: bool,
    #[serde(default)]
    pub skipped: bool,
    #[serde(default)]
    pub reason: Option<String>,
    /// install.ps1 may attach stage-specific structured data here.
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

/// Run-state for a single stage as we transition through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StageState {
    Running,
    Succeeded,
    Skipped,
    Failed,
}

/// Which pipe a raw log line came from. Reported as structured metadata so
/// the UI can style stderr subtly rather than mislabeling it as an error:
/// uv/pip/git/npm write normal progress to stderr by design.
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
    /// Sent once at the start with the full stage list.
    Manifest {
        stages: Vec<StageInfo>,
        #[serde(rename = "protocolVersion")]
        protocol_version: Option<u32>,
    },
    /// Stage state transition. `result` populated only on terminal states.
    Stage {
        name: String,
        state: StageState,
        #[serde(rename = "durationMs", skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<StageResultPayload>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Raw stdout/stderr line from install.ps1 (or our wrapper). `stream`
    /// tells the UI which pipe it came from so stderr can be styled subtly
    /// instead of being mislabeled as an error.
    Log {
        #[serde(skip_serializing_if = "Option::is_none")]
        stage: Option<String>,
        line: String,
        stream: LogStream,
    },
    /// Sent once when all stages complete successfully.
    Complete {
        #[serde(rename = "installRoot")]
        install_root: String,
        marker: Option<serde_json::Value>,
    },
    /// Sent once if the run aborts.
    Failed {
        #[serde(skip_serializing_if = "Option::is_none")]
        stage: Option<String>,
        error: String,
    },
}

impl BootstrapEvent {
    /// Tauri event name. Single channel for all bootstrap events; the
    /// `type` tag tells the renderer how to interpret the payload.
    pub const CHANNEL: &'static str = "bootstrap";
}
