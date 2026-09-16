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

    /// Returns this event with terminal escape bytes removed from `Log`
    /// lines. The webview renders log lines as plain text, so styling and
    /// cursor codes from the install script would show up as mojibake
    /// (#112675).
    pub fn sanitized_for_ui(self) -> Self {
        match self {
            Self::Log {
                stage,
                line,
                stream,
            } => Self::Log {
                stage,
                line: strip_ansi(&line),
                stream,
            },
            other => other,
        }
    }
}

/// Removes ANSI escape sequences from one raw installer log line.
///
/// install.sh (and the git/curl/uv children it drives) emits SGR styling,
/// cursor movement, and OSC title commands even though its stdout is a pipe,
/// not a TTY. The UI's log pane has no terminal emulator, so those bytes must
/// not cross the event boundary. Carriage-return in-place redraws (progress
/// meters) collapse to the last visible frame, which is what a terminal
/// would be left showing.
pub(crate) fn strip_ansi(line: &str) -> String {
    const ESC: char = '\u{1b}';

    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c != ESC {
            out.push(c);
            continue;
        }

        match chars.next() {
            // CSI: parameters (0x30–0x3F) and intermediates (0x20–0x2F),
            // closed by a final byte in 0x40–0x7E.
            Some('[') => {
                for b in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&b) {
                        break;
                    }
                }
            }
            // String sequences (OSC/DCS/PM/APC): run until BEL or the ST
            // terminator (ESC \).
            Some(']' | 'P' | 'X' | '^' | '_') => {
                for b in chars.by_ref() {
                    if b == '\u{07}' {
                        break;
                    }
                    if b == ESC {
                        if chars.peek() == Some(&'\\') {
                            chars.next();
                        }
                        break;
                    }
                }
            }
            // Charset selection ESC ( B and friends carry one trailing byte;
            // every other two-character escape is fully consumed here.
            Some('(') | Some(')') => {
                let _ = chars.next();
            }
            Some(_) | None => {}
        }
    }

    match out.split('\r').filter(|seg| !seg.is_empty()).next_back() {
        Some(seg) => seg.to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_sgr_color_sequences() {
        // The colored checkmark banners from install.sh seen in #112675.
        assert_eq!(
            strip_ansi("\u{1b}[0;32m✓\u{1b}[0m Detected: macos (macos)"),
            "✓ Detected: macos (macos)"
        );
    }

    #[test]
    fn strips_cursor_movement_and_private_modes() {
        assert_eq!(
            strip_ansi("\u{1b}[2K\u{1b}[1GCloning repository…"),
            "Cloning repository…"
        );
        assert_eq!(strip_ansi("down\u{1b}[?25lloading"), "downloading");
    }

    #[test]
    fn strips_osc_title_commands_with_bel_and_st_terminators() {
        assert_eq!(
            strip_ansi("\u{1b}]0;hermes\u{07}Installing Hermes"),
            "Installing Hermes"
        );
        assert_eq!(
            strip_ansi("\u{1b}]2;hermes\u{1b}\\Installing Hermes"),
            "Installing Hermes"
        );
    }

    #[test]
    fn carriage_return_redraws_keep_the_last_frame() {
        // curl-style progress rewrites one line in place with \r.
        assert_eq!(strip_ansi("\r 12%\r 67%\r100%"), "100%");
        assert_eq!(
            strip_ansi("Resolving dependencies…\r"),
            "Resolving dependencies…"
        );
        assert_eq!(strip_ansi("\r\r"), "");
    }

    #[test]
    fn keeps_plain_and_multibyte_text_verbatim() {
        assert_eq!(
            strip_ansi("Installed 12 packages in 1.2s"),
            "Installed 12 packages in 1.2s"
        );
        assert_eq!(strip_ansi("Ready — café ✓ 中文"), "Ready — café ✓ 中文");
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn drops_an_unterminated_sequence_cut_by_the_pipe() {
        assert_eq!(strip_ansi("ok\u{1b}[0;3"), "ok");
        assert_eq!(strip_ansi("ok\u{1b}"), "ok");
    }

    #[test]
    fn sanitized_for_ui_only_touches_log_lines() {
        let log = BootstrapEvent::Log {
            stage: None,
            line: "\u{1b}[1;32mdone\u{1b}[0m\r".to_string(),
            stream: LogStream::Stdout,
        };
        match log.sanitized_for_ui() {
            BootstrapEvent::Log { line, .. } => assert_eq!(line, "done"),
            other => panic!("expected Log, got {other:?}"),
        }

        let stage = BootstrapEvent::Failed {
            stage: None,
            error: "\u{1b}[0;31mfatal\u{1b}[0m".to_string(),
        };
        match stage.sanitized_for_ui() {
            BootstrapEvent::Failed { error, .. } => assert_eq!(error, "\u{1b}[0;31mfatal\u{1b}[0m"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}
