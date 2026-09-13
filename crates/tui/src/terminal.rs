use anyhow::Result;
use crossterm::{
    event::{DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, SetTitle,
    },
};
use std::io::{stdout, BufWriter, Stdout, Write};

/// Restores the terminal on drop, including panic unwind.
pub struct TerminalGuard;

impl TerminalGuard {
    pub fn enter() -> Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = execute!(
            stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        ) {
            let _ = disable_raw_mode();
            return Err(error.into());
        }
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            stdout(),
            LeaveAlternateScreen,
            DisableBracketedPaste,
            DisableMouseCapture,
            SetTitle("")
        );
        let _ = execute!(stdout(), crossterm::cursor::Show);
    }
}

pub type Backend = ratatui::backend::CrosstermBackend<BufWriter<Stdout>>;

pub fn new_terminal() -> Result<ratatui::Terminal<Backend>> {
    let writer = BufWriter::with_capacity(64 * 1024, stdout());
    Ok(ratatui::Terminal::new(
        ratatui::backend::CrosstermBackend::new(writer),
    )?)
}

pub fn flush_terminal(terminal: &mut ratatui::Terminal<Backend>) {
    let _ = terminal.flush();
}

pub fn resolve_editor() -> Vec<String> {
    let explicit = std::env::var("VISUAL")
        .ok()
        .or_else(|| std::env::var("EDITOR").ok())
        .unwrap_or_default();
    let explicit = explicit.trim();
    if !explicit.is_empty() {
        return explicit.split_whitespace().map(str::to_string).collect();
    }
    if cfg!(windows) {
        vec!["notepad.exe".into()]
    } else {
        vec!["vi".into()]
    }
}

/// Leave the alt-screen, run `$VISUAL`/`$EDITOR`, restore the TUI.
pub fn edit_in_external_editor(
    terminal: &mut ratatui::Terminal<Backend>,
    initial: &str,
) -> Result<Option<String>> {
    flush_terminal(terminal);
    disable_raw_mode()?;
    execute!(
        stdout(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture,
        crossterm::cursor::Show
    )?;

    let edit_result = (|| -> Result<Option<String>> {
        let mut temp = tempfile::Builder::new()
            .prefix("hermes-edit-")
            .suffix(".md")
            .tempfile()?;
        temp.write_all(initial.as_bytes())?;
        temp.flush()?;
        let argv = resolve_editor();
        let mut cmd = std::process::Command::new(&argv[0]);
        if argv.len() > 1 {
            cmd.args(&argv[1..]);
        }
        let status = cmd.arg(temp.path()).status()?;
        if status.success() {
            Ok(Some(std::fs::read_to_string(temp.path())?))
        } else {
            Ok(None)
        }
    })();

    let restore_result = (|| -> Result<()> {
        enable_raw_mode()?;
        execute!(
            stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )?;
        terminal.clear()?;
        Ok(())
    })();
    restore_result?;
    edit_result
}

pub fn request_attention(summary: &str) {
    use std::io::Write;
    let mut out = stdout();
    let _ = write!(out, "\x07");
    let safe: String = summary
        .chars()
        .filter(|c| *c != '\x1b' && *c != '\x07' && !c.is_control())
        .take(80)
        .collect();
    if !safe.is_empty() {
        let _ = write!(out, "\x1b]9;{safe}\x07");
    }
    let _ = out.flush();
}
