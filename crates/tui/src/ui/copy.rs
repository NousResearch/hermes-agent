//! Ink TUI wordings (`ui-tui/src/content/*`, `lib/text.ts` tool trail).

use std::sync::OnceLock;

use super::stream::{first_path, json_str, run_command};

/// `ui-tui/src/content/placeholders.ts`
pub const PLACEHOLDERS: &[&str] = &[
    "Ask me anything…",
    "Try \"explain this codebase\"",
    "Try \"write a test for…\"",
    "Try \"refactor the auth module\"",
    "Try \"/help\" for commands",
    "Try \"fix the lint errors\"",
    "Try \"how does the config loader work?\"",
];

/// Native interrupt key, Ink sentence (`Ctrl+C to interrupt…`).
pub const BUSY_PLACEHOLDER: &str = "Esc to interrupt…";

pub fn idle_placeholder() -> &'static str {
    static PICK: OnceLock<&'static str> = OnceLock::new();
    PICK.get_or_init(|| {
        let i =
            (std::process::id() as usize).saturating_add(PLACEHOLDERS.len()) % PLACEHOLDERS.len();
        PLACEHOLDERS[i]
    })
}

/// `toolTrailLabel`: `read_file` → `Read File`.
pub fn tool_trail_label(name: &str) -> String {
    let label = name
        .split(['_', '/', ':'])
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            match c.next() {
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if label.is_empty() {
        name.to_string()
    } else {
        label
    }
}

/// `formatToolCall`: `Read File("app.rs")` or `Read File`.
pub fn format_tool_call(name: &str, preview: &str) -> String {
    let label = tool_trail_label(name);
    let preview = preview.trim();
    if preview.is_empty() {
        label
    } else {
        format!("{label}(\"{}\")", crate::tips::ellipsize(preview, 64))
    }
}

pub fn tool_preview(name: &str, content: &str) -> String {
    let n = name.to_ascii_lowercase();
    if n.contains("terminal")
        || n.contains("bash")
        || n.contains("shell")
        || n.contains("exec")
        || n == "run"
        || n.contains("command")
    {
        return run_command(content);
    }
    if let Some(url) = json_str(content, &["url", "uri", "href", "link"]) {
        return url;
    }
    if let Some(p) = first_path(content) {
        return p;
    }
    if let Some(q) = json_str(
        content,
        &["pattern", "query", "glob", "q", "command", "cmd"],
    ) {
        return q;
    }
    if let Some(s) = json_str(content, &["name", "skill", "text", "prompt"]) {
        return s;
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trail_label_title_cases_underscores() {
        assert_eq!(tool_trail_label("read_file"), "Read File");
        assert_eq!(tool_trail_label("skill_view"), "Skill View");
        assert_eq!(tool_trail_label("terminal"), "Terminal");
    }

    #[test]
    fn format_matches_ink() {
        assert_eq!(
            format_tool_call("read_file", "app.rs"),
            "Read File(\"app.rs\")"
        );
        assert_eq!(format_tool_call("todo_write", ""), "Todo Write");
    }

    #[test]
    fn idle_is_ink_list() {
        assert!(PLACEHOLDERS.contains(&idle_placeholder()));
        assert!(PLACEHOLDERS[0].starts_with("Ask me"));
    }
}
