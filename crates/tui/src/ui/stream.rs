use chrono::{DateTime, Local};
use unicode_width::UnicodeWidthStr;

use crate::state::{ChatMessage, MessageRole, TaskItem, TaskStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Search,
    Read,
    Edit,
    Run,
    Todo,
    Other,
}

pub fn clock(ts: DateTime<Local>) -> String {
    ts.format("%I:%M %p")
        .to_string()
        .trim_start_matches('0')
        .to_string()
}

pub fn tool_icon(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("todo") {
        "☑"
    } else if n.contains("skill") {
        "✦"
    } else if n.contains("replace")
        || n.contains("edit")
        || n.contains("write")
        || n.contains("patch")
    {
        "✎"
    } else if n.contains("search") || n.contains("grep") || n.contains("glob") || n.contains("find")
    {
        "⌕"
    } else if n.contains("read") || n.contains("view") || n == "cat" {
        "▣"
    } else if n.contains("terminal")
        || n.contains("bash")
        || n.contains("shell")
        || n.contains("exec")
        || n == "run"
    {
        "❯"
    } else if n.contains("web") || n.contains("fetch") || n.contains("browser") {
        "◎"
    } else if n.contains("git") {
        "⌥"
    } else {
        match tool_kind(name) {
            ToolKind::Read | ToolKind::Search => "◇",
            ToolKind::Edit | ToolKind::Run => "◆",
            ToolKind::Todo => "☑",
            ToolKind::Other => "●",
        }
    }
}

pub fn tool_kind(name: &str) -> ToolKind {
    let n = name.to_ascii_lowercase();
    if n.contains("todo") {
        ToolKind::Todo
    } else if n.contains("replace")
        || n.contains("edit")
        || n.contains("write")
        || n.contains("patch")
    {
        ToolKind::Edit
    } else if n.contains("search") || n.contains("grep") || n.contains("glob") || n.contains("find")
    {
        ToolKind::Search
    } else if n.contains("bash")
        || n.contains("shell")
        || n.contains("terminal")
        || n.contains("exec")
        || n == "run"
        || n.contains("command")
    {
        ToolKind::Run
    } else if n.contains("read_file")
        || n == "read"
        || n == "cat"
        || n.contains("view_file")
        || (n.contains("read") && !n.contains("thread") && !n.contains("already"))
    {
        ToolKind::Read
    } else {
        ToolKind::Other
    }
}

pub fn is_running(status: &str) -> bool {
    status.contains("running")
}

pub fn cluster_label(reads: usize, searches: usize) -> String {
    let mut parts = Vec::new();
    if searches > 0 {
        parts.push(format!(
            "Searched {searches} {}",
            if searches == 1 { "pattern" } else { "patterns" }
        ));
    }
    if reads > 0 {
        parts.push(format!(
            "Read {reads} {}",
            if reads == 1 { "file" } else { "files" }
        ));
    }
    parts.join(", ")
}

pub fn tool_headline(name: &str, content: &str, status: &str) -> String {
    let running = is_running(status);
    let failed = status.starts_with("failed");
    let n = name.to_ascii_lowercase();
    let mut line = match tool_kind(name) {
        ToolKind::Edit => {
            let file = first_path(content).unwrap_or_else(|| short_name(name));
            let (plus, minus) = diff_stats(content);
            if plus > 0 || minus > 0 {
                format!("Edit {file} +{plus}/-{minus}")
            } else {
                format!("Edit {file}")
            }
        }
        ToolKind::Run => {
            let cmd = run_command(content);
            if cmd.is_empty() {
                format!("Run {}", short_name(name))
            } else {
                format!("Run {}", crate::tips::ellipsize(&cmd, 42))
            }
        }
        ToolKind::Read => {
            let file = first_path(content).unwrap_or_else(|| short_name(name));
            format!("Read {file}")
        }
        ToolKind::Search => {
            let q = json_str(content, &["pattern", "query", "glob", "q"])
                .or_else(|| Some(one_line(content, 40)).filter(|s| !s.is_empty()));
            match q {
                Some(q) => format!("Searched {}", crate::tips::ellipsize(&q, 40)),
                None => format!("Search {}", short_name(name)),
            }
        }
        ToolKind::Todo => {
            let n = todos_from_content(content).len();
            if n == 0 {
                "Todo".into()
            } else {
                format!("Todo  {n}")
            }
        }
        ToolKind::Other => other_headline(&n, content),
    };
    if let Some(dur) = status_duration(status) {
        line.push_str("  ·  ");
        line.push_str(dur);
    }
    line.chars()
        .take(if running || failed { 72 } else { 80 })
        .collect()
}

fn other_headline(name: &str, content: &str) -> String {
    if name.contains("skill") {
        let skill = json_str(content, &["name", "skill", "path", "file"])
            .map(|p| basename(&p))
            .unwrap_or_else(|| short_name(name));
        return format!("Skill {skill}");
    }
    if name.contains("web")
        || name.contains("extract")
        || name.contains("fetch")
        || name.contains("browser")
    {
        let verb = if name.contains("extract") {
            "Extract"
        } else if name.contains("search") {
            "Search"
        } else if name.contains("browser") {
            "Browse"
        } else {
            "Fetch"
        };
        if let Some(url) = json_str(content, &["url", "uri", "href", "link"]) {
            return format!("{verb} {}", crate::tips::ellipsize(&hostish(&url), 42));
        }
        return verb.to_string();
    }
    let extra = json_str(
        content,
        &["path", "file", "query", "pattern", "url", "target", "name"],
    )
    .map(|s| crate::tips::ellipsize(&basename_or(&s), 40))
    .or_else(|| Some(one_line(content, 40)).filter(|s| !s.is_empty()));
    match extra {
        Some(extra) => format!("{}  {extra}", short_name(name)),
        None => short_name(name),
    }
}

fn status_duration(status: &str) -> Option<&str> {
    let (_, dur) = status.split_once(" · ")?;
    if dur.is_empty() || dur.contains("running") {
        None
    } else {
        Some(dur)
    }
}

pub fn first_path(content: &str) -> Option<String> {
    if let Some(p) = json_str(
        content,
        &["path", "file", "filename", "target_file", "file_path"],
    ) {
        return Some(basename(&p));
    }
    content.split_whitespace().find_map(|tok| {
        let clean = clean_token(tok);
        if looks_like_path(&clean) {
            Some(basename(&clean))
        } else {
            None
        }
    })
}

pub fn json_str(content: &str, keys: &[&str]) -> Option<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(s) = json_str_value(&v, keys) {
            return Some(s);
        }
    }
    recover_json_string(trimmed, keys)
}

fn json_str_value(v: &serde_json::Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(s) = v.get(*key).and_then(|x| x.as_str()) {
            let s = s.trim();
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    if let Some(args) = v.get("args") {
        if let Some(s) = json_str_value(args, keys) {
            return Some(s);
        }
    }
    None
}

fn recover_json_string(raw: &str, keys: &[&str]) -> Option<String> {
    for key in keys {
        let needle = format!("\"{key}\"");
        let mut from = 0usize;
        while let Some(rel) = raw[from..].find(&needle) {
            let i = from + rel + needle.len();
            let after = raw.get(i..).unwrap_or("");
            let Some(colon) = after.find(':') else {
                from = i;
                continue;
            };
            let rest = after[colon + 1..].trim_start();
            if !rest.starts_with('"') {
                from = i;
                continue;
            }
            let mut out = String::new();
            let mut chars = rest[1..].chars();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    if let Some(n) = chars.next() {
                        out.push(n);
                    }
                    continue;
                }
                if c == '"' {
                    break;
                }
                out.push(c);
            }
            let out = clean_token(&out);
            if !out.is_empty() {
                return Some(out);
            }
            from = i;
        }
    }
    None
}

fn clean_token(tok: &str) -> String {
    tok.trim()
        .trim_matches(|c| {
            matches!(
                c,
                '"' | '\'' | '`' | ',' | '(' | ')' | '[' | ']' | '{' | '}'
            )
        })
        .trim_end_matches(['"', ',', '}'])
        .trim()
        .to_string()
}

fn hostish(url: &str) -> String {
    let rest = url
        .split("://")
        .nth(1)
        .unwrap_or(url)
        .trim_start_matches("www.");
    rest.split('/').next().unwrap_or(url).to_string()
}

fn basename_or(s: &str) -> String {
    if looks_like_path(s) || s.contains("://") {
        basename(s)
    } else {
        s.to_string()
    }
}

pub fn edit_patch(content: &str, output: &str) -> String {
    let from_out = output.trim();
    if looks_like_patch(from_out) {
        return from_out.to_string();
    }
    let from_content = content.trim();
    if looks_like_patch(from_content) {
        return from_content.to_string();
    }
    let path = json_str(
        content,
        &["path", "file", "filename", "target_file", "file_path"],
    )
    .unwrap_or_else(|| "file".into());
    let old = json_str(content, &["old_string", "old_str", "old"]);
    let new = json_str(content, &["new_string", "new_str", "new"]);
    match (old, new) {
        (Some(old), Some(new)) => {
            let mut out = format!("--- a/{path}\n+++ b/{path}\n");
            for line in old.lines() {
                out.push('-');
                out.push_str(line);
                out.push('\n');
            }
            for line in new.lines() {
                out.push('+');
                out.push_str(line);
                out.push('\n');
            }
            out
        }
        _ => String::new(),
    }
}

fn looks_like_patch(s: &str) -> bool {
    s.lines().any(|l| {
        l.starts_with("@@")
            || l.starts_with("diff --git")
            || ((l.starts_with('+') || l.starts_with('-'))
                && !l.starts_with("+++")
                && !l.starts_with("---")
                && l.len() > 1)
    })
}

pub fn diff_stats(content: &str) -> (u32, u32) {
    let mut plus = 0u32;
    let mut minus = 0u32;
    for line in content.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if line.starts_with('+') {
            plus += 1;
        } else if line.starts_with('-') {
            minus += 1;
        }
    }
    if plus == 0 && minus == 0 {
        if let Some((p, m)) = parse_plus_minus(content) {
            return (p, m);
        }
    }
    (plus, minus)
}

/// Walk tools and collapse consecutive completed reads/searches.
pub fn next_tool_cluster(messages: &[ChatMessage], start: usize) -> Option<(usize, usize, usize)> {
    let msg = messages.get(start)?;
    let MessageRole::Tool { name, status, .. } = &msg.role else {
        return None;
    };
    if is_running(status) {
        return None;
    }
    if !matches!(tool_kind(name), ToolKind::Read | ToolKind::Search) {
        return None;
    }
    let mut reads = 0usize;
    let mut searches = 0usize;
    let mut end = start;
    for m in messages.iter().skip(start) {
        let MessageRole::Tool { name, status, .. } = &m.role else {
            break;
        };
        if is_running(status) {
            break;
        }
        match tool_kind(name) {
            ToolKind::Read => reads += 1,
            ToolKind::Search => searches += 1,
            _ => break,
        }
        end += 1;
    }
    if reads + searches >= 2 {
        Some((end, reads, searches))
    } else {
        None
    }
}

fn parse_plus_minus(content: &str) -> Option<(u32, u32)> {
    let plus_at = content.find('+')?;
    let rest = &content[plus_at + 1..];
    let p: u32 = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    let after_p = plus_at + 1 + rest.chars().take_while(|c| c.is_ascii_digit()).count();
    let tail = content.get(after_p..)?;
    let slash_minus = tail.find("/-").or_else(|| tail.find(" -"))?;
    let m_src = tail.get(slash_minus + 2..)?;
    let m: u32 = m_src
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some((p, m))
}

fn looks_like_path(tok: &str) -> bool {
    if tok.len() < 3 || tok.starts_with("http") {
        return false;
    }
    tok.contains('/') || tok.contains('\\') || tok.contains('.')
}

fn basename(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .trim()
        .to_string()
}

fn short_name(name: &str) -> String {
    name.rsplit(['/', ':']).next().unwrap_or(name).to_string()
}

pub fn todos_from_content(content: &str) -> Vec<TaskItem> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let parsed = crate::state::parse_todos(&v);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    // Truncated JSON still often has complete `"content":"..."` objects.
    recover_todos_from_partial(trimmed)
}

fn recover_todos_from_partial(raw: &str) -> Vec<TaskItem> {
    let mut out = Vec::new();
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &raw[i..];
        let Some(rel) = rest.find("\"content\"") else {
            break;
        };
        i += rel + 9;
        let after = raw.get(i..).unwrap_or("");
        let Some(colon) = after.find(':') else {
            continue;
        };
        let after = after[colon + 1..].trim_start();
        if !after.starts_with('"') {
            continue;
        }
        let mut title = String::new();
        let mut chars = after[1..].chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(n) = chars.next() {
                    title.push(n);
                }
                continue;
            }
            if c == '"' {
                break;
            }
            title.push(c);
        }
        if title.trim().is_empty() {
            continue;
        }
        let window = raw
            .get(i.saturating_sub(80)..(i + 160).min(raw.len()))
            .unwrap_or("");
        let status = if window.contains("completed") {
            crate::state::TaskStatus::Completed
        } else if window.contains("in_progress") || window.contains("in-progress") {
            crate::state::TaskStatus::InProgress
        } else if window.contains("failed") || window.contains("cancelled") {
            crate::state::TaskStatus::Failed
        } else {
            crate::state::TaskStatus::Pending
        };
        out.push(crate::state::TaskItem {
            id: format!("todo-{}", out.len()),
            title: title.trim().to_string(),
            status,
        });
    }
    out
}

pub fn task_mark(status: &TaskStatus) -> &'static str {
    match status {
        TaskStatus::InProgress => "▶",
        TaskStatus::Completed => "✓",
        TaskStatus::Failed => "×",
        TaskStatus::Pending => "○",
    }
}

/// Split glued CoT (`**Step***Next****Other**`) into readable beats.
pub fn thought_beats(raw: &str) -> Vec<String> {
    let normalized = raw
        .replace("****", "\n")
        .replace("***", "\n")
        .replace("**", "\n");
    let mut beats: Vec<String> = normalized
        .lines()
        .map(|l| l.trim().trim_matches('*').trim())
        .filter(|l| l.len() > 1)
        .map(|l| l.to_string())
        .collect();
    if beats.len() <= 1 {
        beats = raw
            .lines()
            .map(|l| l.trim().trim_matches('*').trim())
            .filter(|l| l.len() > 1)
            .map(|l| l.to_string())
            .collect();
    }
    beats
}

pub fn cluster_files(messages: &[ChatMessage], start: usize, end: usize) -> String {
    let mut names = Vec::new();
    for m in messages.get(start..end).unwrap_or(&[]) {
        if let Some(p) = first_path(&m.content) {
            if !names.iter().any(|n| n == &p) {
                names.push(p);
            }
        }
    }
    const SHOW: usize = 4;
    let extra = names.len().saturating_sub(SHOW);
    names.truncate(SHOW);
    let mut s = names.join("  ");
    if extra > 0 {
        s.push_str(&format!("  +{extra}"));
    }
    s
}

pub fn run_command(content: &str) -> String {
    let head = content.lines().next().unwrap_or(content).trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(head) {
        for key in ["command", "cmd", "script", "code"] {
            if let Some(s) = v.get(key).and_then(|x| x.as_str()) {
                let s = s.trim();
                if !s.is_empty() {
                    return s.to_string();
                }
            }
        }
        for key in ["argv", "args"] {
            if let Some(arr) = v.get(key).and_then(|a| a.as_array()) {
                let parts: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
                if !parts.is_empty() {
                    return parts.join(" ");
                }
            }
        }
    }
    if !head.starts_with('{') && !head.starts_with('[') {
        return head.to_string();
    }
    String::new()
}

/// Split a shell chain (`&&`, `||`, `;`) into visible steps. Pipes stay one step.
pub fn run_steps(command: &str) -> Vec<String> {
    let cmd = command.trim();
    if cmd.is_empty() {
        return Vec::new();
    }
    let mut steps = Vec::new();
    let mut buf = String::new();
    let mut quote: Option<char> = None;
    let mut chars = cmd.chars().peekable();
    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            buf.push(c);
            if c == '\\' {
                if let Some(n) = chars.next() {
                    buf.push(n);
                }
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                buf.push(c);
            }
            ';' => {
                let step = buf.trim();
                if !step.is_empty() {
                    steps.push(step.to_string());
                }
                buf.clear();
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                let step = buf.trim();
                if !step.is_empty() {
                    steps.push(step.to_string());
                }
                buf.clear();
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                let step = buf.trim();
                if !step.is_empty() {
                    steps.push(step.to_string());
                }
                buf.clear();
            }
            _ => buf.push(c),
        }
    }
    let step = buf.trim();
    if !step.is_empty() {
        steps.push(step.to_string());
    }
    steps
}

pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            match chars.peek() {
                Some('[') => {
                    chars.next();
                    for n in chars.by_ref() {
                        if n.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next();
                    for n in chars.by_ref() {
                        if n == '\u{7}' || n == '\u{1b}' {
                            break;
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        if c != '\r' {
            out.push(c);
        }
    }
    out
}

pub fn run_output_style(line: &str) -> ratatui::style::Color {
    let t = line.trim_start();
    if t.starts_with('✓') || t.starts_with("[OK]") {
        crate::ui::theme::Theme::accent_green()
    } else if t.starts_with('✗') || t.starts_with("[FAIL]") {
        crate::ui::theme::Theme::accent_red()
    } else if t.starts_with('⚠') || t.starts_with("[WARN]") {
        crate::ui::theme::Theme::accent_yellow()
    } else if t.starts_with('◆') || t.starts_with("┌") || t.starts_with("│") || t.starts_with("└")
    {
        crate::ui::theme::Theme::brand_gold()
    } else if t.starts_with('→') {
        crate::ui::theme::Theme::text_muted()
    } else {
        crate::ui::theme::Theme::text_secondary()
    }
}

fn one_line(s: &str, max: usize) -> String {
    let t = s.lines().next().unwrap_or("").trim();
    if t.starts_with('{') || t.starts_with('[') {
        return String::new();
    }
    if t.width() <= max {
        return t.to_string();
    }
    let mut out = String::new();
    for c in t.chars() {
        if out.width() + 1 >= max {
            break;
        }
        out.push(c);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tool(name: &str, status: &str) -> ChatMessage {
        ChatMessage {
            id: "x".into(),
            role: MessageRole::Tool {
                name: name.into(),
                status: status.into(),
                tool_id: None,
            },
            content: String::new(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        }
    }

    #[test]
    fn clock_drops_leading_zero() {
        assert_eq!("09:36 PM".trim_start_matches('0'), "9:36 PM");
        let s = clock(Local::now());
        assert!(s.contains(':'));
        assert!(s.contains("AM") || s.contains("PM"));
    }

    #[test]
    fn kinds() {
        assert_eq!(tool_kind("read_file"), ToolKind::Read);
        assert_eq!(tool_kind("grep"), ToolKind::Search);
        assert_eq!(tool_kind("search_replace"), ToolKind::Edit);
        assert_eq!(tool_kind("terminal"), ToolKind::Run);
        assert_eq!(tool_kind("todo_write"), ToolKind::Todo);
        assert_eq!(tool_icon("todo"), "☑");
        assert_eq!(tool_icon("skill_view"), "✦");
        assert_eq!(tool_icon("search_files"), "⌕");
        assert_eq!(tool_icon("read_file"), "▣");
        assert_eq!(tool_icon("terminal"), "❯");
        assert_eq!(
            run_command(r#"{"command":"cargo test --offline"}"#),
            "cargo test --offline"
        );
        assert_eq!(run_command(r#"{"argv":["ls","-la"]}"#), "ls -la");
        assert_eq!(tool_icon("search_replace"), "✎");
        assert_eq!(tool_kind("skill_view"), ToolKind::Other);
        assert_eq!(
            tool_headline("skill_view", r#"{"path":"docs/SKILL.md"}"#, "completed"),
            "Skill SKILL.md"
        );
        assert_eq!(
            first_path(r#"{"path":"/tmp/docs/SKILL.md""#).as_deref(),
            Some("SKILL.md")
        );
        assert_eq!(
            tool_headline(
                "web_extract",
                r#"{"url":"https://www.example.com/x"}"#,
                "completed · 1.2s"
            ),
            "Extract example.com  ·  1.2s"
        );
    }

    #[test]
    fn run_steps_split_shell_chains() {
        let steps = run_steps("hermes doctor; printf '\\n--- native help'");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0], "hermes doctor");
        assert!(steps[1].contains("printf"));
        assert_eq!(
            run_steps("pwd && git status --short --branch"),
            vec!["pwd".to_string(), "git status --short --branch".to_string()]
        );
        assert_eq!(run_steps("cmd | rg foo").len(), 1);
        assert_eq!(strip_ansi("\u{1b}[32m✓\u{1b}[0m ok"), "✓ ok");
    }

    #[test]
    fn cluster_two_reads() {
        let msgs = vec![
            tool("read_file", "completed"),
            tool("read_file", "completed"),
        ];
        let (end, reads, searches) = next_tool_cluster(&msgs, 0).expect("cluster");
        assert_eq!((end, reads, searches), (2, 2, 0));
        assert_eq!(cluster_label(2, 1), "Searched 1 pattern, Read 2 files");
    }

    #[test]
    fn no_cluster_single_or_running() {
        let one = vec![tool("read_file", "completed")];
        assert!(next_tool_cluster(&one, 0).is_none());
        let run = vec![
            tool("read_file", "running..."),
            tool("read_file", "completed"),
        ];
        assert!(next_tool_cluster(&run, 0).is_none());
    }

    #[test]
    fn edit_line_uses_path_and_diff() {
        let line = tool_headline(
            "search_replace",
            "src/ui/queue.rs\n+foo\n+bar\n-old",
            "completed",
        );
        assert_eq!(line, "Edit queue.rs +2/-1");
        let patch = edit_patch(
            r#"{"path":"src/ui/footer.rs","old_string":"a","new_string":"b"}"#,
            "",
        );
        assert!(patch.contains("--- a/src/ui/footer.rs"));
        assert!(patch.contains("-a"));
        assert!(patch.contains("+b"));
    }

    #[test]
    fn thought_beats_split_glued_bold() {
        let raw = "**Planning repository inspection steps***Confirming explicit maybe path usage****Considering skill documentation usage**";
        let beats = thought_beats(raw);
        assert_eq!(beats.len(), 3);
        assert_eq!(beats[0], "Planning repository inspection steps");
        assert_eq!(beats[1], "Confirming explicit maybe path usage");
        assert_eq!(beats[2], "Considering skill documentation usage");
    }

    #[test]
    fn todos_from_json_blob() {
        let raw = r#"{"todos":[{"content":"Inspect repository","status":"in_progress"},{"content":"Ship","status":"pending"}]}"#;
        let todos = todos_from_content(raw);
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0].title, "Inspect repository");
        assert_eq!(todos[0].status, TaskStatus::InProgress);
        assert_eq!(tool_headline("todo_write", raw, "completed"), "Todo  2");
        let nested = r#"{"todos":[{"content":"Inspect repository","status":"in_progress"},{"content":"Ship","status":"pending"}]}"#;
        let todos = todos_from_content(nested);
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0].title, "Inspect repository");
        let chopped = &nested[..80];
        let recovered = todos_from_content(chopped);
        assert!(
            recovered.iter().any(|t| t.title.contains("Inspect")),
            "{recovered:?}"
        );
    }

    #[test]
    fn parse_inline_diff_counts() {
        assert_eq!(parse_plus_minus("+112/-0").unwrap(), (112, 0));
        assert_eq!(
            first_path(r#"{"path":"src/app.rs"}"#).as_deref(),
            Some("app.rs")
        );
    }
}
