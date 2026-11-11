//! Local composer helpers: `!` shell, plan-mode wrap, `/init`, `/export`.

use std::path::{Path, PathBuf};

use crate::state::ChatMessage;

pub const PLAN_PREFIX: &str = concat!(
    "[Plan mode] Stay read-only. Do not edit files, run mutating shell ",
    "commands, or change git state. Inspect the repo, write a concrete plan, ",
    "and wait for approval before any write."
);

pub fn bang_command(text: &str) -> Option<&str> {
    let t = text.trim();
    let rest = t.strip_prefix('!')?;
    if rest.starts_with('[') {
        return None;
    }
    let cmd = rest.trim();
    if cmd.is_empty() {
        None
    } else {
        Some(cmd)
    }
}

pub fn wrap_prompt(plan: bool, shell_context: &str, body: &str) -> String {
    let mut out = String::new();
    if !shell_context.is_empty() {
        out.push_str(shell_context.trim());
        out.push_str("\n\n");
    }
    if plan {
        out.push_str(PLAN_PREFIX);
        out.push('\n');
    }
    out.push_str(body);
    out
}

pub fn format_shell_context(cmd: &str, stdout: &str, stderr: &str, code: i64) -> String {
    let mut out = format!("[shell] $ {cmd}\n");
    let body = [stdout, stderr]
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if body.is_empty() {
        out.push_str(&format!("(exit {code})"));
    } else {
        out.push_str(&body);
        if code != 0 {
            out.push_str(&format!("\n(exit {code})"));
        }
    }
    out
}

pub fn agents_md_path(cwd: &str) -> PathBuf {
    Path::new(cwd).join("AGENTS.md")
}

pub fn init_agents_md(cwd: &str) -> Result<String, String> {
    let path = agents_md_path(cwd);
    if path.exists() {
        return Err(format!("{} already exists", path.display()));
    }
    std::fs::write(&path, AGENTS_TEMPLATE).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

pub fn history_text(messages: &[ChatMessage], preview: usize) -> Option<String> {
    use crate::state::MessageRole;
    let preview = preview.max(40);
    let mut out = String::new();
    let mut n = 0usize;
    for msg in messages {
        let tag = match &msg.role {
            MessageRole::User => "you",
            MessageRole::Assistant => "hermes",
            _ => continue,
        };
        let body = msg.content.trim();
        if body.is_empty() {
            continue;
        }
        n += 1;
        let clipped = if body.chars().count() > preview {
            format!("{}…", body.chars().take(preview).collect::<String>())
        } else {
            body.to_string()
        };
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!("#{n} {tag}\n{clipped}"));
    }
    if n == 0 {
        None
    } else {
        Some(out)
    }
}

pub fn recap(messages: &[ChatMessage]) -> Option<String> {
    use crate::state::MessageRole;
    let user = messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::User && !m.content.trim().is_empty())?;
    let asst = messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::Assistant && !m.content.trim().is_empty());
    let clip = |s: &str| {
        let t = s.trim().replace('\n', " ");
        if t.chars().count() > 120 {
            format!("{}…", t.chars().take(119).collect::<String>())
        } else {
            t
        }
    };
    let mut out = format!("recap · you: {}", clip(&user.content));
    if let Some(a) = asst {
        out.push_str(&format!("\nrecap · hermes: {}", clip(&a.content)));
    }
    Some(out)
}

pub fn transcript_markdown(messages: &[ChatMessage]) -> String {
    use crate::state::MessageRole;
    let mut out = String::from("# Hermes session\n\n");
    for msg in messages {
        let (tag, body) = match &msg.role {
            MessageRole::User => ("user", msg.content.as_str()),
            MessageRole::Assistant => ("assistant", msg.content.as_str()),
            MessageRole::System => ("system", msg.content.as_str()),
            MessageRole::Reasoning => ("thinking", msg.content.as_str()),
            MessageRole::Tool { name, status, .. } => {
                out.push_str(&format!("### tool `{name}` ({status})\n\n"));
                if !msg.output.is_empty() {
                    out.push_str("```\n");
                    out.push_str(msg.output.trim());
                    out.push_str("\n```\n\n");
                } else if !msg.content.is_empty() {
                    out.push_str(&msg.content);
                    out.push_str("\n\n");
                }
                continue;
            }
            MessageRole::ImagePreview { path } => {
                out.push_str(&format!("![image]({path})\n\n"));
                continue;
            }
            MessageRole::Compaction => ("compact", msg.content.as_str()),
        };
        if body.trim().is_empty() {
            continue;
        }
        out.push_str(&format!("### {tag}\n\n{body}\n\n"));
    }
    out
}

const AGENTS_TEMPLATE: &str = "\
# AGENTS.md

Project instructions for coding agents in this repository.

## Build

```
# how to build
```

## Test

```
# how to test
```

## Conventions

- Prefer the smallest change that solves the request.
- Do not invent a second backend. Match existing patterns.

## Do not

- Commit secrets
- Rewrite unrelated files
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ChatMessage, MessageRole};
    use chrono::Local;

    #[test]
    fn bang_strips_prefix() {
        assert_eq!(bang_command("!ls -la"), Some("ls -la"));
        assert_eq!(bang_command("  !git status  "), Some("git status"));
        assert_eq!(bang_command("ls"), None);
        assert_eq!(bang_command("!"), None);
        assert_eq!(bang_command("![img]"), None);
    }

    #[test]
    fn wrap_includes_plan_and_shell() {
        let t = wrap_prompt(true, "[shell] $ ls\nhi", "what next");
        assert!(t.starts_with("[shell]"));
        assert!(t.contains(PLAN_PREFIX));
        assert!(t.ends_with("what next"));
        let plain = wrap_prompt(false, "", "hi");
        assert_eq!(plain, "hi");
    }

    #[test]
    fn shell_context_formats_exit() {
        let t = format_shell_context("false", "", "", 1);
        assert!(t.contains("exit 1"));
        let ok = format_shell_context("echo hi", "hi\n", "", 0);
        assert!(ok.contains("hi"));
        assert!(!ok.contains("exit"));
    }

    #[test]
    fn init_refuses_existing() {
        let dir = std::env::temp_dir().join(format!("hermes-init-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = dir.to_string_lossy().to_string();
        assert!(init_agents_md(&cwd).is_ok());
        assert!(agents_md_path(&cwd).exists());
        assert!(init_agents_md(&cwd).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_lists_user_and_assistant() {
        let msgs = vec![
            ChatMessage {
                id: "1".into(),
                role: MessageRole::User,
                content: "hello".into(),
                output: String::new(),
                timestamp: Local::now(),
                is_streaming: false,
            },
            ChatMessage {
                id: "2".into(),
                role: MessageRole::System,
                content: "skip".into(),
                output: String::new(),
                timestamp: Local::now(),
                is_streaming: false,
            },
            ChatMessage {
                id: "3".into(),
                role: MessageRole::Assistant,
                content: "hi there".into(),
                output: String::new(),
                timestamp: Local::now(),
                is_streaming: false,
            },
        ];
        let h = history_text(&msgs, 80).expect("history");
        assert!(h.contains("#1 you"));
        assert!(h.contains("hello"));
        assert!(h.contains("#2 hermes"));
        assert!(!h.contains("skip"));
        assert!(history_text(&[], 80).is_none());
    }

    #[test]
    fn recap_uses_last_exchange() {
        let msgs = vec![
            ChatMessage {
                id: "1".into(),
                role: MessageRole::User,
                content: "first".into(),
                output: String::new(),
                timestamp: Local::now(),
                is_streaming: false,
            },
            ChatMessage {
                id: "2".into(),
                role: MessageRole::Assistant,
                content: "ok".into(),
                output: String::new(),
                timestamp: Local::now(),
                is_streaming: false,
            },
            ChatMessage {
                id: "3".into(),
                role: MessageRole::User,
                content: "ship the tui".into(),
                output: String::new(),
                timestamp: Local::now(),
                is_streaming: false,
            },
        ];
        let r = recap(&msgs).expect("recap");
        assert!(r.contains("ship the tui"));
        assert!(!r.contains("first"));
        assert!(recap(&[]).is_none());
    }

    #[test]
    fn transcript_skips_empty() {
        let msgs = vec![ChatMessage {
            id: "1".into(),
            role: MessageRole::User,
            content: "hello".into(),
            output: String::new(),
            timestamp: Local::now(),
            is_streaming: false,
        }];
        let md = transcript_markdown(&msgs);
        assert!(md.contains("### user"));
        assert!(md.contains("hello"));
    }
}
