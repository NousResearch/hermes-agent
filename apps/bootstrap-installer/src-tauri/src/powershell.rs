//! Drives PowerShell (Windows) or bash (Unix) for install.ps1 / install.sh.
//!
//! Port of `spawnPowerShell` from bootstrap-runner.ts, with the same
//! line-buffered stdout/stderr streaming + cancellation semantics.
//!
//! On Windows we pass `-NoProfile -ExecutionPolicy Bypass -File <script>`.
//! On Unix we shell out to `bash <script>` since install.sh expects bash.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::time::timeout;

/// CP1252 mapping for bytes `0x80..=0x9F` (the range that differs from Latin-1).
/// Undefined slots keep the C1 control code points, matching Windows-1252
/// best-fit behavior used by `encoding_rs::WINDOWS_1252`.
const CP1252_80_9F: [char; 32] = [
    '\u{20AC}', // 0x80 €
    '\u{0081}', // 0x81
    '\u{201A}', // 0x82 ‚
    '\u{0192}', // 0x83 ƒ
    '\u{201E}', // 0x84 „
    '\u{2026}', // 0x85 …
    '\u{2020}', // 0x86 †
    '\u{2021}', // 0x87 ‡
    '\u{02C6}', // 0x88 ˆ
    '\u{2030}', // 0x89 ‰
    '\u{0160}', // 0x8A Š
    '\u{2039}', // 0x8B ‹
    '\u{0152}', // 0x8C Œ
    '\u{008D}', // 0x8D
    '\u{017D}', // 0x8E Ž
    '\u{008F}', // 0x8F
    '\u{0090}', // 0x90
    '\u{2018}', // 0x91 ‘
    '\u{2019}', // 0x92 ’
    '\u{201C}', // 0x93 “
    '\u{201D}', // 0x94 ”
    '\u{2022}', // 0x95 •
    '\u{2013}', // 0x96 –
    '\u{2014}', // 0x97 —
    '\u{02DC}', // 0x98 ˜
    '\u{2122}', // 0x99 ™
    '\u{0161}', // 0x9A š
    '\u{203A}', // 0x9B ›
    '\u{0153}', // 0x9C œ
    '\u{009D}', // 0x9D
    '\u{017E}', // 0x9E ž
    '\u{0178}', // 0x9F Ÿ
];

fn decode_cp1252_byte(b: u8) -> char {
    match b {
        0x00..=0x7F => b as char,
        0x80..=0x9F => CP1252_80_9F[(b - 0x80) as usize],
        // 0xA0..=0xFF match Unicode Latin-1 / Windows-1252.
        _ => b as char,
    }
}

/// Decode one stdout/stderr line from a child process.
///
/// Tokio's `BufReader::lines()` requires valid UTF-8 and aborts the line (with
/// `stream did not contain valid UTF-8`) at the first accented byte. Windows
/// PowerShell 5.1 emits localized ParserError text in the console ANSI code
/// page (often CP1252), so Portuguese/Spanish/etc. users only saw a truncated
/// `No` instead of `Não foi fornecido o terminador...` (#67193).
///
/// Prefer UTF-8 when the bytes are valid; otherwise decode as Windows-1252 so
/// both Western-European letters and CP1252-only punctuation (e.g. `0x91` →
/// U+2018) survive rather than disappearing into a read-error warning.
pub(crate) fn decode_console_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => bytes.iter().copied().map(decode_cp1252_byte).collect(),
    }
}

/// Read one line (LF or CRLF) and decode it with [`decode_console_bytes`].
/// Returns `Ok(None)` on EOF with no bytes pending.
pub(crate) async fn read_decoded_line<R>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> std::io::Result<Option<String>>
where
    R: AsyncBufReadExt + Unpin,
{
    // Cancel-safety: `buf` is NOT cleared on entry. When this future is
    // dropped mid-read inside `tokio::select!` (the other stream produced a
    // line first), `read_until` has already appended any consumed bytes to
    // `buf`; the next call resumes and appends the rest of the line. Clearing
    // on entry would silently drop those bytes. We clear only after a full
    // line has been decoded.
    let n = reader.read_until(b'\n', buf).await?;
    if n == 0 && buf.is_empty() {
        return Ok(None);
    }
    // n == 0 with a non-empty buf means EOF cut off an unterminated line
    // (possibly accumulated across cancelled reads) -- emit it.
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
    }
    let line = decode_console_bytes(buf);
    buf.clear();
    Ok(Some(line))
}

/// Hooks the caller installs to receive output.
pub struct StreamSink {
    pub on_stdout_line: Box<dyn Fn(&str) + Send + Sync>,
    pub on_stderr_line: Box<dyn Fn(&str) + Send + Sync>,
}

/// Outcome of a script invocation. Mirrors bootstrap-runner.ts's
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

/// How long a child's pipes get to reach EOF AFTER the child itself has exited.
///
/// This is not a timeout on the child. The clock starts once the process is
/// already gone and everything it wrote is sitting in the pipe buffer, so a
/// 40-minute `uv pip install` is untouched — the grace only covers the final
/// drain.
///
/// It exists because pipe EOF is not the child's to give. The write end of a
/// redirected pipe is handed to the child as an inheritable handle, so every
/// descendant spawned without its own redirection holds a duplicate, and the
/// read side does not see EOF until the last of them closes it. `hermes update`
/// deliberately runs its build steps with stdout inherited, so the tree under a
/// child is arbitrarily deep and not something the caller can enumerate. When
/// one of those descendants is a resident gateway, the pipe stays open for the
/// life of the gateway — and every obligation downstream of the read is
/// stranded with it.
///
/// Same bound `Invoke-HermesStep` grew in `scripts/desktop-update/windows.ps1`
/// (#90455), and the same shape as Go's `exec.Cmd.WaitDelay`.
pub(crate) const DRAIN_GRACE: Duration = Duration::from_secs(20);

/// What [`pump_child`] observed.
pub(crate) struct PumpOutcome {
    pub exit_code: Option<i32>,
    /// The child was killed because the caller cancelled.
    pub killed: bool,
    /// The child exited but its pipes never reached EOF within [`DRAIN_GRACE`],
    /// so the tail of its output was dropped. Callers must say so out loud: a
    /// silently truncated log is indistinguishable from one that was empty.
    pub abandoned: bool,
}

/// Stream a child's stdout/stderr line by line, then reap it.
///
/// The contract that matters: the exit status comes from waiting on the
/// *process*, never from pipe EOF. See [`DRAIN_GRACE`] for why those are not
/// the same event; callers pass it, tests pass something they can wait out.
pub(crate) async fn pump_child<FO, FE>(
    child: &mut Child,
    mut on_stdout: FO,
    mut on_stderr: FE,
    cancel_rx: &mut Option<CancelRx>,
    grace: Duration,
) -> Result<PumpOutcome>
where
    FO: FnMut(&str),
    FE: FnMut(&str),
{
    let stdout = child.stdout.take().context("stdout was piped")?;
    let stderr = child.stderr.take().context("stderr was piped")?;
    let mut out = BufReader::new(stdout);
    let mut err = BufReader::new(stderr);
    let mut out_buf = Vec::new();
    let mut err_buf = Vec::new();
    let mut out_done = false;
    let mut err_done = false;
    let mut status: Option<ExitStatus> = None;
    let mut cancelled = false;

    // Phase 1 — the child is alive, so there is no deadline. A slow child is
    // not a stuck one, and the point of streaming is that a long build keeps
    // reporting. We leave on whichever lands first: both pipes at EOF (the
    // clean case), the process exiting (the case that used to hang here), or
    // cancellation.
    while !(out_done && err_done) {
        tokio::select! {
            line = read_decoded_line(&mut out, &mut out_buf), if !out_done => match line {
                Ok(Some(l)) => on_stdout(&l),
                Ok(None) => out_done = true,
                Err(e) => {
                    tracing::warn!("stdout read error: {e}");
                    out_done = true;
                }
            },
            line = read_decoded_line(&mut err, &mut err_buf), if !err_done => match line {
                Ok(Some(l)) => on_stderr(&l),
                Ok(None) => err_done = true,
                Err(e) => {
                    tracing::warn!("stderr read error: {e}");
                    err_done = true;
                }
            },
            reaped = child.wait() => {
                status = Some(reaped.context("waiting for child to exit")?);
                break;
            }
            _ = recv_cancel(cancel_rx) => {
                cancelled = true;
                break;
            }
        }
    }

    // Kill outside the loop: `child.wait()` above holds the mutable borrow for
    // as long as the select is in scope.
    if cancelled {
        tracing::warn!("cancellation received — killing child");
        let _ = child.start_kill();
    }

    // Phase 2 — bounded. Whatever the child already wrote is still worth
    // keeping, so we keep reading; we just stop caring once a descendant is the
    // only thing still holding the pipe open. Cancelling does not rescue us
    // either: `start_kill` kills the child, not the grandchild with the handle.
    let mut abandoned = false;
    if !(out_done && err_done) {
        let drain = async {
            while !(out_done && err_done) {
                tokio::select! {
                    line = read_decoded_line(&mut out, &mut out_buf), if !out_done => match line {
                        Ok(Some(l)) => on_stdout(&l),
                        _ => out_done = true,
                    },
                    line = read_decoded_line(&mut err, &mut err_buf), if !err_done => match line {
                        Ok(Some(l)) => on_stderr(&l),
                        _ => err_done = true,
                    },
                }
            }
        };
        abandoned = timeout(grace, drain).await.is_err();
    }

    let status = match status {
        Some(s) => s,
        // Both pipes reached EOF while the child stayed alive. Reads are done,
        // but the process is still the authoritative terminal condition — and
        // it may never exit on its own, so this wait stays cancellable. A bare
        // `child.wait()` here would strand the caller's cancel channel exactly
        // when it is the only way out.
        None => tokio::select! {
            reaped = child.wait() => reaped.context("waiting for child to exit")?,
            _ = recv_cancel(cancel_rx) => {
                tracing::warn!("cancellation received after EOF — killing child");
                cancelled = true;
                let _ = child.start_kill();
                child.wait().await.context("waiting for killed child to exit")?
            }
        },
    };
    Ok(PumpOutcome {
        exit_code: status.code(),
        killed: cancelled,
        abandoned,
    })
}

/// Spawns install.ps1 / install.sh with the given args and streams output.
///
/// `hermes_home_override` propagates to the child as $HERMES_HOME so the
/// install script writes to the same directory the installer is reading from.
pub async fn run_script(
    script_path: &Path,
    args: &[String],
    sink: StreamSink,
    hermes_home_override: Option<&str>,
    cancel_rx: &mut Option<CancelRx>,
) -> Result<ScriptResult> {
    let mut cmd = build_command(script_path, args);

    // The installer can be launched from a .app bundle that is later replaced
    // during self-update. Pin child scripts to a stable directory so bash/zsh
    // never starts from a deleted cwd and emits getcwd/job-working-directory
    // errors at the end of an otherwise successful install.
    if let Some(cwd) = stable_script_cwd(script_path, hermes_home_override) {
        cmd.current_dir(cwd);
    }

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
        .with_context(|| format!("spawning {} via {}", script_path.display(), interpreter_label()))?;

    // Byte-oriented readers + [`decode_console_bytes`]: do NOT use
    // `BufReader::lines()`, which requires valid UTF-8 and hides localized
    // PowerShell errors on non-English Windows (#67193). [`pump_child`] owns
    // that, plus the rule that the exit status comes from the process and not
    // from pipe EOF.
    let mut combined_stdout = String::new();
    let mut combined_stderr = String::new();

    let outcome = pump_child(
        &mut child,
        |l| {
            (sink.on_stdout_line)(l);
            combined_stdout.push_str(l);
            combined_stdout.push('\n');
        },
        |l| {
            (sink.on_stderr_line)(l);
            combined_stderr.push_str(l);
            combined_stderr.push('\n');
        },
        cancel_rx,
        DRAIN_GRACE,
    )
    .await
    .context("streaming install script output")?;

    if outcome.abandoned {
        let note = format!(
            "install script exited but a surviving descendant still holds its \
             stdout/stderr; gave up on the last {}s of output (#90455)",
            DRAIN_GRACE.as_secs()
        );
        tracing::warn!("{note}");
        (sink.on_stderr_line)(&note);
        combined_stderr.push_str(&note);
        combined_stderr.push('\n');
    }

    Ok(ScriptResult {
        stdout: combined_stdout,
        stderr: combined_stderr,
        exit_code: outcome.exit_code,
        killed: outcome.killed,
    })
}

fn stable_script_cwd<'a>(script_path: &'a Path, hermes_home_override: Option<&'a str>) -> Option<&'a Path> {
    if let Some(home) = hermes_home_override {
        let path = Path::new(home);
        if path.is_dir() {
            return Some(path);
        }
    }
    script_path.parent().filter(|p| p.is_dir())
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
    // over `pwsh.exe` (7+, may not be present). Resolve it by absolute path —
    // see `windows_powershell_exe`.
    let mut cmd = Command::new(windows_powershell_exe());
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

/// Canonical PowerShell 5.1 location under a Windows root (`%SystemRoot%`).
/// Kept separate (and test-visible) so the path layout is unit-tested on any
/// host, not just Windows.
#[cfg(any(target_os = "windows", test))]
fn powershell_under_root(root: &Path) -> std::path::PathBuf {
    root.join("System32")
        .join("WindowsPowerShell")
        .join("v1.0")
        .join("powershell.exe")
}

/// Resolves the PowerShell interpreter to spawn.
///
/// `Command::new("powershell.exe")` trusts PATH to contain
/// `%SystemRoot%\System32\WindowsPowerShell\v1.0`. On machines whose PATH was
/// trimmed or truncated (Windows silently drops entries once the variable grows
/// past its length limit), that lookup fails and the spawn dies with
/// "program not found" before install.ps1 ever runs — the installer then stalls
/// at "0 of 0 steps". Resolve by absolute path first, then fall back to PATH
/// (powershell 5.1, then pwsh 7), then a bare name as a last resort.
#[cfg(target_os = "windows")]
fn windows_powershell_exe() -> std::path::PathBuf {
    for var in ["SystemRoot", "windir"] {
        if let Ok(root) = std::env::var(var) {
            let candidate = powershell_under_root(Path::new(&root));
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    for exe in ["powershell.exe", "pwsh.exe"] {
        if let Ok(found) = which::which(exe) {
            return found;
        }
    }

    std::path::PathBuf::from("powershell.exe")
}

/// Human-readable interpreter name for spawn-failure context. On Windows this
/// is the resolved PowerShell path so a missing/odd interpreter is obvious in
/// the log (the old message only printed the script path, which read as if the
/// .ps1 itself was missing).
#[cfg(target_os = "windows")]
fn interpreter_label() -> String {
    windows_powershell_exe().display().to_string()
}

#[cfg(not(target_os = "windows"))]
fn interpreter_label() -> String {
    "bash".to_string()
}

/// Parses the LAST line of stdout that looks like a JSON object matching
/// the install.ps1 stage-result contract: `{ok: bool, stage: string, ...}`.
///
/// Mirrors `parseStageResult` from bootstrap-runner.ts. install.ps1 may
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

    #[test]
    fn stable_script_cwd_prefers_existing_hermes_home() {
        let script = Path::new("/tmp/install.sh");
        let cwd = stable_script_cwd(script, Some("/"));
        assert_eq!(cwd, Some(Path::new("/")));
    }

    #[test]
    fn powershell_under_root_uses_system32_v1_layout() {
        let resolved = powershell_under_root(Path::new("C:\\Windows"));
        let normalized = resolved.to_string_lossy().replace('\\', "/");
        assert!(
            normalized.ends_with("System32/WindowsPowerShell/v1.0/powershell.exe"),
            "unexpected powershell path: {normalized}"
        );
    }

    #[test]
    fn decode_console_bytes_keeps_valid_utf8() {
        assert_eq!(decode_console_bytes("café — ok".as_bytes()), "café — ok");
    }

    #[test]
    fn decode_console_bytes_preserves_cp1252_portuguese_error() {
        // "Não foi fornecido o terminador..." as Windows PowerShell 5.1 emits
        // under CP1252 (0xE3 = ã). BufReader::lines() previously failed here
        // with "stream did not contain valid UTF-8" and the UI only showed "No".
        let bytes: &[u8] = b"N\xE3o foi fornecido o terminador";
        assert_eq!(decode_console_bytes(bytes), "Não foi fornecido o terminador");
    }

    #[test]
    fn decode_console_bytes_maps_cp1252_only_punctuation() {
        // 0x91/0x92 are curly quotes in Windows-1252, but C1 controls under
        // Latin-1 (`b as char`). This locks the real CP1252 fallback.
        let bytes: &[u8] = b"say \x91hi\x92";
        assert_eq!(decode_console_bytes(bytes), "say \u{2018}hi\u{2019}");
        assert_ne!(
            decode_console_bytes(bytes),
            bytes.iter().map(|&b| b as char).collect::<String>(),
            "Latin-1 byte mapping must not be used for the 0x80..=0x9F range"
        );
    }

    #[tokio::test]
    async fn read_decoded_line_survives_non_utf8_and_crlf() {
        let data: &[u8] = b"N\xE3o erro\r\nnext\n";
        let mut reader = BufReader::new(data);
        let mut buf = Vec::new();
        assert_eq!(
            read_decoded_line(&mut reader, &mut buf)
                .await
                .unwrap()
                .as_deref(),
            Some("Não erro")
        );
        assert_eq!(
            read_decoded_line(&mut reader, &mut buf)
                .await
                .unwrap()
                .as_deref(),
            Some("next")
        );
        assert!(read_decoded_line(&mut reader, &mut buf)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn read_decoded_line_preserves_partial_line_across_cancellation() {
        use std::time::Duration;
        use tokio::io::AsyncWriteExt;

        let (mut tx, rx) = tokio::io::duplex(64);
        let mut reader = BufReader::new(rx);
        let mut buf = Vec::new();

        tx.write_all(b"partial").await.unwrap();
        // Poll once, then cancel (drop) the future -- exactly what
        // tokio::select! does in run_script when the other stream produces
        // a line first. The consumed bytes must survive in `buf`.
        let _ = tokio::time::timeout(
            Duration::from_millis(0),
            read_decoded_line(&mut reader, &mut buf),
        )
        .await;

        tx.write_all(b" line\n").await.unwrap();
        let line = read_decoded_line(&mut reader, &mut buf).await.unwrap();
        assert_eq!(line.as_deref(), Some("partial line"));
    }

    #[tokio::test]
    async fn read_decoded_line_emits_unterminated_final_line_at_eof() {
        let data: &[u8] = b"no trailing newline";
        let mut reader = BufReader::new(data);
        let mut buf = Vec::new();
        assert_eq!(
            read_decoded_line(&mut reader, &mut buf)
                .await
                .unwrap()
                .as_deref(),
            Some("no trailing newline")
        );
        assert!(read_decoded_line(&mut reader, &mut buf)
            .await
            .unwrap()
            .is_none());
    }

    /// Spawn `sh -c <script>` with both pipes redirected, the way run_script and
    /// run_streamed do.
    #[cfg(unix)]
    fn sh(script: &str) -> Child {
        Command::new("/bin/sh")
            .arg("-c")
            .arg(script)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn /bin/sh")
    }

    #[cfg(unix)]
    async fn pump_collect(child: &mut Child, grace: Duration) -> (PumpOutcome, Vec<String>) {
        let mut lines = Vec::new();
        let outcome = pump_child(
            child,
            |l| lines.push(l.to_string()),
            |_| {},
            &mut None,
            grace,
        )
        .await
        .expect("pump");
        (outcome, lines)
    }

    /// The #90455 deadlock: a child exits, but a descendant it spawned still
    /// holds the inherited write end of the pipe, so EOF never comes. The pump
    /// must return on the *process* exiting and abandon the drain.
    ///
    /// The Windows half of this contract lives in `-SelfTestPipeDrain`
    /// (scripts/desktop-update/windows.ps1) and runs on the Windows CI lane; the
    /// pump logic under test here is OS-agnostic, only the fixture is not.
    #[cfg(unix)]
    #[tokio::test]
    async fn pump_child_returns_when_a_grandchild_still_holds_the_pipe() {
        // `sleep` inherits stdout and outlives the shell by design. Nothing
        // redirects it -- redirecting is what would close the handle and stop
        // the bug from reproducing at all.
        let mut child = sh("sleep 30 & echo hello; exit 7");
        let started = std::time::Instant::now();
        let (outcome, lines) = pump_collect(&mut child, Duration::from_millis(300)).await;
        let elapsed = started.elapsed();

        assert!(outcome.abandoned, "drain should have been abandoned");
        assert_eq!(
            outcome.exit_code,
            Some(7),
            "exit code must survive an abandoned drain"
        );
        assert_eq!(
            lines,
            vec!["hello"],
            "output written before the pipe was stranded must survive"
        );
        assert!(!outcome.killed);
        // Far below the grandchild's 30s: a pass cannot be it exiting on its own.
        assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
    }

    /// The other cliff: a child that leaks nothing must not pay the grace. This
    /// is what fails if the pump ever waits on the deadline unconditionally
    /// instead of only when a pipe outlives its process.
    #[cfg(unix)]
    #[tokio::test]
    async fn pump_child_does_not_pay_the_grace_when_pipes_close_cleanly() {
        let mut child = sh("echo one; echo two >&2; echo three; exit 3");
        let started = std::time::Instant::now();
        let (outcome, lines) = pump_collect(&mut child, Duration::from_secs(30)).await;
        let elapsed = started.elapsed();

        assert!(!outcome.abandoned);
        assert_eq!(outcome.exit_code, Some(3));
        assert_eq!(lines, vec!["one", "three"]);
        assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
    }

    /// A chatty child must stream at pipe speed, not at one buffer per tick.
    /// The PowerShell port of this pump regressed exactly here: idling after
    /// every chunk it *did* read metered the drain and backpressured the running
    /// child. `hermes update` is this shape -- the Electron build alone is
    /// megabytes.
    #[cfg(unix)]
    #[tokio::test]
    async fn pump_child_streams_a_flood_without_metering_it() {
        let mut child = sh("i=0; while [ $i -lt 20000 ]; do echo line$i; i=$((i+1)); done; exit 0");
        let started = std::time::Instant::now();
        let (outcome, lines) = pump_collect(&mut child, Duration::from_secs(30)).await;
        let elapsed = started.elapsed();

        assert_eq!(outcome.exit_code, Some(0));
        assert!(!outcome.abandoned);
        assert_eq!(lines.len(), 20000, "every line must survive");
        assert_eq!(lines.last().unwrap(), "line19999");
        // Loose on purpose: this catches per-chunk sleeping (which would put
        // this in the tens of seconds), not small scheduler variance.
        assert!(elapsed < Duration::from_secs(20), "took {elapsed:?}");
    }

    /// Cancelling kills the child, but not a grandchild holding the pipe --
    /// `start_kill` only reaches the child. The bounded drain is what actually
    /// lets a cancel return, so cancellation and the grace are one mechanism.
    #[cfg(unix)]
    #[tokio::test]
    async fn pump_child_cancellation_returns_even_with_the_pipe_stranded() {
        let mut child = sh("sleep 30 & echo working; sleep 30");
        let (tx, rx) = mpsc::channel(1);
        let mut cancel = Some(rx);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let _ = tx.send(()).await;
        });

        let started = std::time::Instant::now();
        let mut lines = Vec::new();
        let outcome = pump_child(
            &mut child,
            |l| lines.push(l.to_string()),
            |_| {},
            &mut cancel,
            Duration::from_millis(300),
        )
        .await
        .expect("pump");

        assert!(outcome.killed, "cancellation should report killed");
        assert!(outcome.abandoned, "the grandchild still holds the pipe");
        assert_eq!(lines, vec!["working"]);
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "took {:?}",
            started.elapsed()
        );
    }

    /// The reverse topology of the test above: the pipes die and the *process*
    /// outlives them. Phase 1 leaves on EOF with no exit status, so the final
    /// wait is the only thing left holding the turn -- and it has to stay
    /// cancellable. A bare `child.wait()` there strands the cancel channel
    /// against a child that may never exit on its own.
    #[cfg(unix)]
    #[tokio::test]
    async fn pump_child_cancellation_returns_after_both_pipes_reach_eof() {
        // Closes fd 1 and 2, then lingers: both reads hit EOF immediately while
        // the process stays alive far past any plausible test duration.
        let mut child = sh("echo bye; exec 1>&- 2>&-; sleep 30");
        let (tx, rx) = mpsc::channel(1);
        let mut cancel = Some(rx);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let _ = tx.send(()).await;
        });

        let started = std::time::Instant::now();
        let mut lines = Vec::new();
        let outcome = pump_child(
            &mut child,
            |l| lines.push(l.to_string()),
            |_| {},
            &mut cancel,
            Duration::from_millis(300),
        )
        .await
        .expect("pump");
        let elapsed = started.elapsed();

        assert!(outcome.killed, "cancellation should report killed");
        assert!(
            !outcome.abandoned,
            "both pipes reached EOF, so nothing was abandoned"
        );
        assert_eq!(lines, vec!["bye"], "output written before EOF must survive");
        // Far below the child's 30s: a pass cannot be it exiting on its own.
        assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
    }
}
