use anyhow::{bail, Context, Result};
use clap::Parser;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Native TUI client for Hermes Agent.
///
/// Spawns `python -m tui_gateway.entry` the same way the official Ink TUI
/// (`ui-tui`) does, then speaks newline-delimited JSON-RPC over stdio.
#[derive(Parser, Debug)]
#[command(
    name = "hermes-tui-native",
    version,
    about = "Native Rust TUI client for Hermes Agent (opt-in; Ink remains hermes --tui)"
)]
pub struct Cli {
    /// Python interpreter that can import `tui_gateway`
    #[arg(long, env = "HERMES_PYTHON")]
    pub python: Option<PathBuf>,

    /// Hermes profile home (sessions, memory, config.yaml)
    #[arg(long, env = "HERMES_HOME")]
    pub hermes_home: Option<PathBuf>,

    /// Hermes source root — directory that contains `tui_gateway/`
    #[arg(long, env = "HERMES_PYTHON_SRC_ROOT")]
    pub src_root: Option<PathBuf>,

    /// Working directory attached to the new session
    #[arg(long, env = "HERMES_CWD")]
    pub cwd: Option<PathBuf>,

    /// Session title
    #[arg(long, env = "HERMES_TUI_TITLE", default_value = "Hermes TUI")]
    pub title: String,

    /// Resume an existing session id (`HERMES_TUI_RESUME`, same as Ink)
    #[arg(long, env = "HERMES_TUI_RESUME")]
    pub resume: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub python: PathBuf,
    pub src_root: PathBuf,
    pub hermes_home: PathBuf,
    pub cwd: PathBuf,
    pub title: String,
    pub resume: Option<String>,
    pub rpc_timeout: Duration,
    pub startup_timeout: Duration,
}

impl LaunchConfig {
    pub fn from_cli(cli: Cli) -> Result<Self> {
        let src_root = find_src_root(cli.src_root.clone(), cli.python.as_deref())
            .with_context(|| missing_gateway_hint(cli.src_root.as_deref()))?;

        let python = resolve_python(cli.python, &src_root);
        let hermes_home = cli.hermes_home.unwrap_or_else(default_hermes_home);
        let cwd = cli
            .cwd
            .or_else(|| env::current_dir().ok())
            .unwrap_or_else(|| src_root.clone());

        Ok(Self {
            python,
            src_root,
            hermes_home,
            cwd,
            title: cli.title,
            resume: cli.resume.filter(|s| !s.is_empty()),
            rpc_timeout: env_duration_ms("HERMES_TUI_RPC_TIMEOUT_MS", 120_000, 30_000),
            startup_timeout: env_duration_ms("HERMES_TUI_STARTUP_TIMEOUT_MS", 15_000, 5_000),
        })
    }
}

pub fn has_gateway(root: &Path) -> bool {
    root.join("tui_gateway").join("entry.py").is_file()
}

/// Error text when `tui_gateway/entry.py` is missing. Names the path that was
/// actually tried so a README placeholder is not copied back into the shell.
pub fn missing_gateway_hint(tried: Option<&Path>) -> String {
    let mut lines = vec!["Could not find Hermes source root (tui_gateway/entry.py).".into()];
    if let Some(p) = tried {
        let shown = p.display().to_string();
        if looks_like_docs_placeholder(&shown) {
            lines.push(format!(
                "HERMES_PYTHON_SRC_ROOT={shown} is a documentation placeholder, not a real directory."
            ));
        } else {
            lines.push(format!("looked under {shown}"));
        }
    }
    lines.push(
        "Point HERMES_PYTHON_SRC_ROOT at the hermes-agent checkout that contains tui_gateway/entry.py \
(assignment without export is invisible to this binary), or run from that repo."
            .into(),
    );
    lines.join("\n")
}

fn looks_like_docs_placeholder(path: &str) -> bool {
    let p = path.trim();
    p.contains("/path/to/") || p == "/path/to/hermes-agent" || p.ends_with("/path/to/hermes-agent")
}

fn gateway_under(path: &Path) -> Option<PathBuf> {
    if has_gateway(path) {
        return Some(path.to_path_buf());
    }
    path.ancestors()
        .find(|anc| has_gateway(anc))
        .map(|anc| anc.to_path_buf())
}

pub fn find_src_root(explicit: Option<PathBuf>, python: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        if let Some(found) = gateway_under(&p) {
            return Some(found);
        }
    }

    // Exported HERMES_PYTHON is enough — walk up from .venv/bin/python.
    if let Some(p) = python {
        if let Some(found) = gateway_under(p) {
            return Some(found);
        }
    }

    if let Ok(cwd) = env::current_dir() {
        if let Some(found) = gateway_under(&cwd) {
            return Some(found);
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(found) = gateway_under(&exe) {
            return Some(found);
        }
    }

    None
}

pub fn resolve_python(explicit: Option<PathBuf>, src_root: &Path) -> PathBuf {
    if let Some(p) = explicit {
        return p;
    }
    if let Ok(p) = env::var("PYTHON") {
        let p = p.trim();
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        for rel in ["bin/python", "bin/python3", "Scripts/python.exe"] {
            let cand = Path::new(&venv).join(rel);
            if cand.is_file() {
                return cand;
            }
        }
    }
    for rel in [
        ".venv/bin/python",
        ".venv/bin/python3",
        ".venv/Scripts/python.exe",
        "venv/bin/python",
        "venv/bin/python3",
        "venv/Scripts/python.exe",
    ] {
        let cand = src_root.join(rel);
        if cand.is_file() {
            return cand;
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

pub fn default_hermes_home() -> PathBuf {
    if let Ok(h) = env::var("HERMES_HOME") {
        let h = h.trim();
        if !h.is_empty() {
            return PathBuf::from(h);
        }
    }
    let home = env::var("HOME")
        .ok()
        .or_else(|| env::var("USERPROFILE").ok());
    match home {
        Some(h) => PathBuf::from(h).join(".hermes"),
        None => PathBuf::from(".hermes"),
    }
}

fn env_duration_ms(key: &str, default_ms: u64, min_ms: u64) -> Duration {
    let ms = env::var(key)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default_ms)
        .max(min_ms);
    Duration::from_millis(ms)
}

pub fn init_tracing(hermes_home: &Path) -> Result<()> {
    let log_spec = env::var("HERMES_TUI_LOG")
        .ok()
        .or_else(|| env::var("RUST_LOG").ok());
    if log_spec.is_none() {
        return Ok(());
    }

    let log_dir = hermes_home.join("logs");
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("creating log dir {}", log_dir.display()))?;
    let log_path = log_dir.join("tui.log");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening {}", log_path.display()))?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(file)
        .with_ansi(false)
        .init();
    Ok(())
}

pub fn validate_python_gateway(python: &Path, src_root: &Path) -> Result<()> {
    if !has_gateway(src_root) {
        bail!("no tui_gateway/entry.py under {}", src_root.display());
    }
    if python.is_absolute() && !python.exists() {
        bail!("python interpreter not found: {}", python.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::fs;

    #[test]
    fn cli_identifies_native_tui() {
        let cmd = Cli::command();
        assert_eq!(cmd.get_name(), "hermes-tui-native");
        let about = cmd.get_about().map(|s| s.to_string()).unwrap_or_default();
        assert!(
            about.contains("Native Rust TUI") && about.contains("Ink"),
            "{about}"
        );
        let long = cmd
            .get_long_about()
            .map(|s| s.to_string())
            .unwrap_or_default();
        assert!(
            long.contains("tui_gateway") && long.contains("JSON-RPC"),
            "{long}"
        );
    }

    #[test]
    fn missing_gateway_hint_flags_readme_placeholder() {
        let hint = missing_gateway_hint(Some(Path::new("/path/to/hermes-agent")));
        assert!(hint.contains("placeholder"), "{hint}");
        assert!(!hint.contains("export HERMES_PYTHON_SRC_ROOT=/path/to/hermes-agent\n"));
        let real = missing_gateway_hint(Some(Path::new("/var/tmp/hermes-src")));
        assert!(real.contains("looked under"), "{real}");
        assert!(!real.contains("placeholder"), "{real}");
    }

    #[test]
    fn has_gateway_requires_entry_py() {
        let tmp = tempfile_dir("gw");
        assert!(!has_gateway(&tmp));
        fs::create_dir_all(tmp.join("tui_gateway")).unwrap();
        fs::write(tmp.join("tui_gateway/entry.py"), "#").unwrap();
        assert!(has_gateway(&tmp));
    }

    #[test]
    fn resolve_python_prefers_explicit() {
        let root = PathBuf::from("/tmp/hermes-src");
        let py = PathBuf::from("/custom/python");
        assert_eq!(resolve_python(Some(py.clone()), &root), py);
    }

    #[test]
    fn find_src_root_walks_up_from_venv_python() {
        let tmp = tempfile_dir("src-from-py");
        fs::create_dir_all(tmp.join("tui_gateway")).unwrap();
        fs::write(tmp.join("tui_gateway/entry.py"), "#").unwrap();
        let py = tmp.join(".venv/bin/python");
        fs::create_dir_all(py.parent().unwrap()).unwrap();
        fs::write(&py, "").unwrap();
        let found = find_src_root(None, Some(&py)).expect("src root from python");
        assert_eq!(found, tmp);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn env_duration_respects_minimum() {
        // function uses env; just check default path via a missing key
        let d = env_duration_ms("HERMES_TUI_MISSING_KEY_XYZ", 120_000, 30_000);
        assert_eq!(d, Duration::from_millis(120_000));
    }

    fn tempfile_dir(tag: &str) -> PathBuf {
        let p = env::temp_dir().join(format!("hermes-tui-test-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }
}
