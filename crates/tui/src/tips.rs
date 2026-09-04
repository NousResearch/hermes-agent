//! Rotating “did you know?” copy. Sourced from official Hermes CLI tips
//! (`hermes_cli/tips.py`), slash-command docs, and this TUI’s shortcuts.

pub const ROTATE_SECS: u64 = 40;

pub const TIPS: &[&str] = &[
    "/compress flushes memories and summarizes a long thread when the window fills.",
    "/clear hides the local transcript. Press u within a few seconds to put it back.",
    "Shift+Tab cycles plan → ask → yolo. Plan mode denies writes. /yolo still jumps straight to bypass.",
    "Type !git status to run a shell command. Output is attached to the next prompt you send.",
    "Ctrl+P opens the command palette. /mcp lists MCP servers; r reloads them.",
    "/init writes AGENTS.md if the repo doesn't have one. /export copies the transcript as Markdown.",
    "/fork (or /branch) copies this thread into a sibling session. /undo drops the last exchange.",
    "/editor opens $VISUAL or $EDITOR. /focus hides everything except the current turn.",
    "/tools and /plugins list what this session loaded. /browser shows CDP status.",
    "Enter while Hermes is working queues the next prompt instead of interrupting.",
    "Type @ then Tab to attach a file. @diff / @staged inject the git patch. Images get a preview above the composer.",
    "A long paste becomes [[ head .. N lines .. tail ]]. Click the brackets for the full text.",
    "@file:main.py:10-50 injects only those lines. Paste a .png path to attach it as [[ Image ]].",
    "Click the footer model chip or Ctrl+O to switch models mid-session.",
    "Ctrl+O, then Enter on paste key, saves an API key without leaving the picker.",
    "/context maps what is eating the window. Enter in that overlay compresses.",
    "Keep this session if you can — a new one drops the prompt cache and costs more.",
    "Skills live in ~/.hermes/skills. /skills browses what this profile has loaded.",
    "Memory persists across sessions. Skills are reusable procedures the agent can load.",
    "/background opens the task list. Type a prompt and enter — or /bg /btw <prompt>.",
    "/branch forks the current session so you can try another direction without losing it.",
    "/rollback lists filesystem checkpoints. Enter twice restores, d shows the diff.",
    "Settings belong in config.yaml. .env is for keys and tokens only.",
    "/goal names what this session is for so later turns stay on the same job.",
    "Live subagents and background processes list under the composer. Click a row for the work rail.",
    "Ctrl+D splits a live git diff against HEAD. Click the branch chip to switch.",
    "Ctrl+E opens a file tree on the right with a per-file diff. o opens, r restores, u undoes.",
    "/sessions resumes an older thread. Ctrl+N starts a fresh one.",
    "/model sonnet (or any id) switches without opening the picker.",
    "hermes -c resumes the most recent CLI session. hermes -c \"title\" resumes by name.",
    "hermes doctor --fix diagnoses config and dependency issues.",
    "Type /help for keys. Click × on this bar to hide it — /tips brings it back.",
    "Esc quits. If the composer still has a draft, press Esc again to confirm.",
    "HERMES_TUI_REDUCED_MOTION=1 freezes the scan, shimmer, and spinner.",
    "Approvals: y once, a always, n deny. YOLO is the session-wide bypass.",
    "Paste a local image path in the composer — /open launches the latest one.",
];

pub const COUNT: usize = TIPS.len();

pub fn start_index() -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (secs as usize / ROTATE_SECS as usize) % COUNT.max(1)
}

pub fn persist_path(hermes_home: &std::path::Path) -> std::path::PathBuf {
    hermes_home.join("tui-tips")
}

pub fn load_open(hermes_home: &std::path::Path) -> bool {
    let raw = std::fs::read_to_string(persist_path(hermes_home)).unwrap_or_default();
    let v = raw.trim().to_ascii_lowercase();
    !matches!(v.as_str(), "0" | "off" | "false" | "hidden")
}

pub fn save_open(hermes_home: &std::path::Path, open: bool) {
    let _ = std::fs::create_dir_all(hermes_home);
    let _ = std::fs::write(persist_path(hermes_home), if open { "on" } else { "off" });
}

pub fn ellipsize(s: &str, max: usize) -> String {
    use unicode_width::UnicodeWidthStr;
    if max == 0 {
        return String::new();
    }
    if s.width() <= max {
        return s.to_string();
    }
    if max <= 1 {
        return "…".into();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch)
            .unwrap_or(1)
            .max(1);
        if w + cw + 1 > max {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_nonempty_and_short() {
        const {
            assert!(COUNT >= 12);
        }
        for t in TIPS {
            assert!(!t.is_empty());
            assert!(t.chars().count() < 140, "tip too long for one row: {t}");
        }
    }

    #[test]
    fn ellipsize_fits() {
        assert_eq!(ellipsize("hello", 10), "hello");
        let e = ellipsize("abcdefghijklmnopqrstuvwxyz", 8);
        assert!(e.ends_with('…'));
        assert!(unicode_width::UnicodeWidthStr::width(e.as_str()) <= 8);
    }

    #[test]
    fn start_index_in_range() {
        assert!(start_index() < COUNT);
    }
}
