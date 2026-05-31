//! Drives PowerShell (Windows) or bash (Unix) for install.ps1 / install.sh.
//!
//! Port of `spawnPowerShell` from bootstrap-runner.cjs, with the same
//! line-buffered stdout/stderr streaming + cancellation semantics.
//!
//! On Windows we pass `-NoProfile -ExecutionPolicy Bypass -File <script>`.
//! On Unix we shell out to `bash <script>` since install.sh expects bash.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

/// Hooks the caller installs to receive output.
pub struct StreamSink {
    pub on_stdout_line: Box<dyn Fn(&str) + Send + Sync>,
    pub on_stderr_line: Box<dyn Fn(&str) + Send + Sync>,
}

/// Outcome of a script invocation. Mirrors bootstrap-runner.cjs's
/// `{stdout, stderr, code, signal, killed}` shape.
#[derive(Debug)]
pub struct ScriptResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub killed: bool,
}

/// Cancellation signal — `cancel_tx.send(()).await` aborts the running script.
pub type CancelRx = mpsc::Receiver<()>;

/// Spawns install.ps1 / install.sh with the given args and streams output.
///
/// `hermes_home_override` propagates to the child as $HERMES_HOME so the
/// install script writes to the same directory the installer is reading from.
pub async fn run_script(
    script_path: &Path,
    args: &[String],
    sink: StreamSink,
    hermes_home_override: Option<&str>,
    mut cancel_rx: Option<CancelRx>,
) -> Result<ScriptResult> {
    let mut cmd = build_command(script_path, args);

    if let Some(home) = hermes_home_override {
        cmd.env("HERMES_HOME", home);
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // On Windows, avoid spawning a flashing cmd window when we're hosted
    // inside a GUI process. Tauri's main window is already created, so
    // the side-effect console for the child is unwanted.
    #[cfg(target_os = "windows")]
    {
        // CREATE_NO_WINDOW = 0x08000000
        cmd.creation_flags(0x0800_0000);
    }

    let mut child: Child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", script_path.display()))?;

    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    let mut combined_stdout = String::new();
    let mut combined_stderr = String::new();
    let mut killed = false;

    // Loop: poll stdout, stderr, cancel, and child exit concurrently.
    loop {
        tokio::select! {
            line = stdout_reader.next_line() => {
                match line {
                    Ok(Some(l)) => {
                        (sink.on_stdout_line)(&l);
                        combined_stdout.push_str(&l);
                        combined_stdout.push('\n');
                    }
                    Ok(None) => {
                        // EOF on stdout — wait for stderr + exit.
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("stdout read error: {e}");
                        break;
                    }
                }
            }
            line = stderr_reader.next_line() => {
                match line {
                    Ok(Some(l)) => {
                        (sink.on_stderr_line)(&l);
                        combined_stderr.push_str(&l);
                        combined_stderr.push('\n');
                    }
                    Ok(None) => {
                        // stderr EOF — keep draining stdout.
                    }
                    Err(e) => {
                        tracing::warn!("stderr read error: {e}");
                    }
                }
            }
            _ = recv_cancel(&mut cancel_rx) => {
                tracing::warn!("cancellation received — killing child");
                killed = true;
                // best-effort kill; don't propagate errors
                let _ = child.start_kill();
                break;
            }
        }
    }

    // Drain remaining lines after the loop exited.
    while let Ok(Some(l)) = stdout_reader.next_line().await {
        (sink.on_stdout_line)(&l);
        combined_stdout.push_str(&l);
        combined_stdout.push('\n');
    }
    while let Ok(Some(l)) = stderr_reader.next_line().await {
        (sink.on_stderr_line)(&l);
        combined_stderr.push_str(&l);
        combined_stderr.push('\n');
    }

    let status = child
        .wait()
        .await
        .context("waiting for install script to exit")?;

    Ok(ScriptResult {
        stdout: combined_stdout,
        stderr: combined_stderr,
        exit_code: status.code(),
        killed,
    })
}

async fn recv_cancel(rx: &mut Option<CancelRx>) {
    match rx {
        Some(r) => {
            let _ = r.recv().await;
        }
        None => std::future::pending::<()>().await,
    }
}

#[cfg(target_os = "windows")]
fn build_command(script_path: &Path, args: &[String]) -> Command {
    // We want PowerShell 5.1 / 7. install.ps1 uses 5.1-safe syntax everywhere.
    // Prefer `powershell.exe` (5.1 baseline, present on every Windows since 7)
    // over `pwsh.exe` (7+, may not be present).
    let mut cmd = Command::new("powershell.exe");
    cmd.arg("-NoProfile");
    cmd.arg("-ExecutionPolicy").arg("Bypass");
    cmd.arg("-File").arg(script_path);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

#[cfg(not(target_os = "windows"))]
fn build_command(script_path: &Path, args: &[String]) -> Command {
    // install.sh expects bash. /bin/bash is fine on macOS (Apple still
    // ships an old 3.2 bash; install.sh is written to that baseline).
    let mut cmd = Command::new("bash");
    cmd.arg(script_path);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Parses the LAST line of stdout that looks like a JSON object matching
/// the install.ps1 stage-result contract: `{ok: bool, stage: string, ...}`.
///
/// Mirrors `parseStageResult` from bootstrap-runner.cjs. install.ps1 may
/// print info/banner lines before the result frame; we scan from the end.
pub fn parse_stage_result(stdout: &str) -> Option<crate::events::StageResultPayload> {
    for line in stdout.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if value.get("ok").and_then(|v| v.as_bool()).is_some()
                && value.get("stage").and_then(|v| v.as_str()).is_some()
            {
                if let Ok(parsed) =
                    serde_json::from_value::<crate::events::StageResultPayload>(value)
                {
                    return Some(parsed);
                }
            }
        }
    }
    None
}

/// Same logic but for the `-Manifest` payload (the LAST line with a `stages`
/// array). Returns the parsed manifest.
pub fn parse_manifest(stdout: &str) -> Option<crate::events::Manifest> {
    for line in stdout.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if value.get("stages").and_then(|v| v.as_array()).is_some() {
                if let Ok(parsed) = serde_json::from_value::<crate::events::Manifest>(value) {
                    return Some(parsed);
                }
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stage_result_picks_last_json_line() {
        let stdout = r#"
[bootstrap] some info
{"ok": false, "stage": "venv", "reason": "bad python"}
{"ok": true, "stage": "venv"}
final non-json banner
"#;
        let result = parse_stage_result(stdout).unwrap();
        assert_eq!(result.stage, "venv");
        assert!(result.ok);
    }

    #[test]
    fn parse_manifest_finds_stages_array() {
        let stdout = r#"
info line
{"stages": [{"name": "uv", "title": "uv", "category": "prereqs", "needs_user_input": false}], "protocol_version": 1}
"#;
        let m = parse_manifest(stdout).unwrap();
        assert_eq!(m.stages.len(), 1);
        assert_eq!(m.stages[0].name, "uv");
        assert_eq!(m.protocol_version, Some(1));
    }

    #[test]
    fn parse_returns_none_when_no_match() {
        assert!(parse_stage_result("just banner\n").is_none());
        assert!(parse_manifest("just banner\n").is_none());
    }
}
