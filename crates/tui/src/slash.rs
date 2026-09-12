//! Slash-menu scoring, ported from the official Ink TUI
//! (`ui-tui/src/app/slash/fuzzyScore.ts`) / grok-cli slash-menu.
//!
//! Lower score wins. `None` means no match. Description matches live at
//! offset +3 and must not auto-execute a command (completion only).

use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashKind {
    Local,
    Command,
    Skill,
}

#[derive(Debug, Clone, Copy)]
pub struct SlashCommand {
    pub name: &'static str,
    pub args_hint: &'static str,
    pub description: &'static str,
    #[allow(dead_code)]
    pub local: bool,
}

#[derive(Debug, Clone)]
pub struct SlashEntry {
    pub name: String,
    pub args_hint: String,
    pub description: String,
    pub kind: SlashKind,
}

impl SlashCommand {
    #[allow(clippy::wrong_self_convention)]
    pub fn to_entry(&self) -> SlashEntry {
        SlashEntry {
            name: self.name.to_string(),
            args_hint: self.args_hint.to_string(),
            description: self.description.to_string(),
            kind: SlashKind::Local,
        }
    }
}

/// Client-owned commands plus the ones this TUI surfaces in the popup.
/// Gateway catalog (`commands.catalog`) can extend this at runtime later.
pub const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/help",
        args_hint: "",
        description: "Show commands and keyboard shortcuts…",
        local: true,
    },
    SlashCommand {
        name: "/copy",
        args_hint: "[n]",
        description: "Copy latest (or nth) assistant response",
        local: true,
    },
    SlashCommand {
        name: "/open",
        args_hint: "",
        description: "Open latest image or the workspace",
        local: true,
    },
    SlashCommand {
        name: "/tasks",
        args_hint: "",
        description: "Open tasks & goal overlay…",
        local: true,
    },
    SlashCommand {
        name: "/model",
        args_hint: "[name|disconnect <slug>]",
        description: "Open model switcher… disconnect <slug> drops credentials",
        local: true,
    },
    SlashCommand {
        name: "/branches",
        args_hint: "",
        description: "Switch git branch… (Ctrl+B or click the footer chip)",
        local: true,
    },
    SlashCommand {
        name: "/fork",
        args_hint: "[name]",
        description: "Fork this session into a sibling thread (alias /branch)",
        local: true,
    },
    SlashCommand {
        name: "/branch",
        args_hint: "[name]",
        description: "Alias of /fork — session.branch, not git",
        local: true,
    },
    SlashCommand {
        name: "/undo",
        args_hint: "",
        description: "Undo the last user exchange (session.undo)",
        local: true,
    },
    SlashCommand {
        name: "/save",
        args_hint: "",
        description: "Save the transcript via session.save",
        local: true,
    },
    SlashCommand {
        name: "/editor",
        args_hint: "",
        description: "Compose in $VISUAL / $EDITOR then send",
        local: true,
    },
    SlashCommand {
        name: "/prompt",
        args_hint: "",
        description: "Alias of /editor",
        local: true,
    },
    SlashCommand {
        name: "/focus",
        args_hint: "[on|off]",
        description: "Quiet transcript — last prompt and this turn only",
        local: true,
    },
    SlashCommand {
        name: "/tools",
        args_hint: "[enable|disable …]",
        description: "Toolsets overlay · space toggles · enable/disable via tools.configure",
        local: true,
    },
    SlashCommand {
        name: "/plugins",
        args_hint: "[enable|disable <key>]",
        description: "Plugins hub · space toggles via plugins.manage",
        local: true,
    },
    SlashCommand {
        name: "/browser",
        args_hint: "",
        description: "Browser CDP status",
        local: true,
    },
    SlashCommand {
        name: "/sandbox",
        args_hint: "",
        description: "Show the session terminal backend (docker/ssh/local…)",
        local: true,
    },
    SlashCommand {
        name: "/retry",
        args_hint: "",
        description: "Resend the last user message",
        local: true,
    },
    SlashCommand {
        name: "/details",
        args_hint: "",
        description: "Expand or collapse every tool card",
        local: true,
    },
    SlashCommand {
        name: "/history",
        args_hint: "[chars|stored]",
        description: "Peek live turns, or stored session.history",
        local: true,
    },
    SlashCommand {
        name: "/status",
        args_hint: "",
        description: "Live session status from the gateway",
        local: true,
    },
    SlashCommand {
        name: "/title",
        args_hint: "[name]",
        description: "Show or set the session title",
        local: true,
    },
    SlashCommand {
        name: "/cd",
        args_hint: "[path]",
        description: "Change session cwd (session.cwd.set)",
        local: true,
    },
    SlashCommand {
        name: "/pwd",
        args_hint: "",
        description: "Show session working directory",
        local: true,
    },
    SlashCommand {
        name: "/steer",
        args_hint: "<text>",
        description: "Inject after the next tool call, or queue if idle",
        local: true,
    },
    SlashCommand {
        name: "/redirect",
        args_hint: "<text>",
        description: "Correct the live model turn (session.redirect)",
        local: true,
    },
    SlashCommand {
        name: "/workspace",
        args_hint: "<cwd>",
        description: "Re-home this session's stored workspace",
        local: true,
    },
    SlashCommand {
        name: "/projects",
        args_hint: "[name|id|scan]",
        description: "Projects overlay · enter drills · s /projects scan discovers repos",
        local: true,
    },
    SlashCommand {
        name: "/cli",
        args_hint: "<argv…>",
        description: "Run hermes CLI via cli.exec (non-interactive)",
        local: true,
    },
    SlashCommand {
        name: "/queue",
        args_hint: "[text]",
        description: "Inspect the prompt queue, or enqueue a message",
        local: true,
    },
    SlashCommand {
        name: "/recap",
        args_hint: "",
        description: "Summarize the last user / assistant exchange",
        local: true,
    },
    SlashCommand {
        name: "/fast",
        args_hint: "[on|off|status]",
        description: "Toggle fast / priority service tier",
        local: true,
    },
    SlashCommand {
        name: "/reasoning",
        args_hint: "[effort|show|hide]",
        description: "Inspect or set reasoning effort",
        local: true,
    },
    SlashCommand {
        name: "/busy",
        args_hint: "[queue|steer|interrupt]",
        description: "What Enter does while the agent is running",
        local: true,
    },
    SlashCommand {
        name: "/verbose",
        args_hint: "[cycle|on|off]",
        description: "Verbose tool output (config.set verbose)",
        local: true,
    },
    SlashCommand {
        name: "/personality",
        args_hint: "[name]",
        description: "Show or set session personality",
        local: true,
    },
    SlashCommand {
        name: "/reload",
        args_hint: "",
        description: "Re-read ~/.hermes/.env into the gateway",
        local: true,
    },
    SlashCommand {
        name: "/reload-mcp",
        args_hint: "",
        description: "Reload MCP servers in the live session (reload.mcp)",
        local: true,
    },
    SlashCommand {
        name: "/reload-skills",
        args_hint: "",
        description: "Reload installed skills from disk",
        local: true,
    },
    SlashCommand {
        name: "/battery",
        args_hint: "",
        description: "Host battery status",
        local: true,
    },
    SlashCommand {
        name: "/image",
        args_hint: "<path>|detach",
        description: "Attach an image, or detach the last one",
        local: true,
    },
    SlashCommand {
        name: "/paste",
        args_hint: "",
        description: "Attach a clipboard image (clipboard.paste)",
        local: true,
    },
    SlashCommand {
        name: "/credits",
        args_hint: "",
        description: "Token usage, Nous credits, usage.bars dollars",
        local: true,
    },
    SlashCommand {
        name: "/mem",
        args_hint: "",
        description: "This process RSS/virtual size (native, not Ink V8 heap)",
        local: true,
    },
    SlashCommand {
        name: "/density",
        args_hint: "[on|off]",
        description: "Compact display preference (config.set density)",
        local: true,
    },
    SlashCommand {
        name: "/mouse",
        args_hint: "",
        description: "Toggle mouse capture (tmux-friendly)",
        local: true,
    },
    SlashCommand {
        name: "/cron",
        args_hint: "",
        description: "Cron jobs — enter peek  p pause  r resume  x remove",
        local: true,
    },
    SlashCommand {
        name: "/setup",
        args_hint: "",
        description: "Provider / runtime onboarding check",
        local: true,
    },
    SlashCommand {
        name: "/config",
        args_hint: "",
        description: "Show masked gateway config",
        local: true,
    },
    SlashCommand {
        name: "/facts",
        args_hint: "",
        description: "Project facts for this cwd",
        local: true,
    },
    SlashCommand {
        name: "/verify",
        args_hint: "",
        description: "Verification evidence for this session (alias /review)",
        local: true,
    },
    SlashCommand {
        name: "/review",
        args_hint: "",
        description: "Alias of /verify",
        local: true,
    },
    SlashCommand {
        name: "/replay",
        args_hint: "[list|N|last|load <path>|save]",
        description: "Spawn-tree overlay · enter loads a snapshot (live x/p/s off)",
        local: true,
    },
    SlashCommand {
        name: "/replay-diff",
        args_hint: "<a> <b>",
        description: "Diff two spawn trees by /replay index or path",
        local: true,
    },
    SlashCommand {
        name: "/hide",
        args_hint: "<session>",
        description: "Hide a stored session from the recents list",
        local: true,
    },
    SlashCommand {
        name: "/unhide",
        args_hint: "<session>",
        description: "Unhide a stored session (session.set_hidden)",
        local: true,
    },
    SlashCommand {
        name: "/react",
        args_hint: "[emoji|clear]",
        description: "React to the last assistant turn (message.react)",
        local: true,
    },
    SlashCommand {
        name: "/imagine",
        args_hint: "[prompt]",
        description: "Generate an image (image.generate) · no arg probes backend",
        local: true,
    },
    SlashCommand {
        name: "/insights",
        args_hint: "",
        description: "Session count and messages over 30 days",
        local: true,
    },
    SlashCommand {
        name: "/indicator",
        args_hint: "[style]",
        description: "Status indicator style (config.set indicator)",
        local: true,
    },
    SlashCommand {
        name: "/statusbar",
        args_hint: "[on|off|top|bottom]",
        description: "Status bar preference",
        local: true,
    },
    SlashCommand {
        name: "/redraw",
        args_hint: "",
        description: "Force a full TUI repaint",
        local: true,
    },
    SlashCommand {
        name: "/file",
        args_hint: "<path>",
        description: "Attach a non-image file (@file ref)",
        local: true,
    },
    SlashCommand {
        name: "/pdf",
        args_hint: "<path>",
        description: "Attach a PDF as page images (needs pdftoppm)",
        local: true,
    },
    SlashCommand {
        name: "/delete",
        args_hint: "[session]",
        description: "Delete a stored session (not the live one)",
        local: true,
    },
    SlashCommand {
        name: "/logs",
        args_hint: "",
        description: "Tail hermes-tui log file",
        local: true,
    },
    SlashCommand {
        name: "/fortune",
        args_hint: "[daily]",
        description: "A short tip as a fortune",
        local: true,
    },
    SlashCommand {
        name: "/theme-info",
        args_hint: "",
        description: "Show the active palette id",
        local: true,
    },
    SlashCommand {
        name: "/update",
        args_hint: "",
        description: "How to update Hermes Agent",
        local: true,
    },
    SlashCommand {
        name: "/vim",
        args_hint: "",
        description: "Opt-in composer vim · Esc in normal leaves vim (chat Esc stays interrupt)",
        local: true,
    },
    SlashCommand {
        name: "/motion",
        args_hint: "[on|off]",
        description:
            "Toggle chrome motion · gold wash, shimmer, spinners (env: HERMES_TUI_REDUCED_MOTION)",
        local: true,
    },
    SlashCommand {
        name: "/commit",
        args_hint: "",
        description: "Draft a commit message from git diff (llm.oneshot, does not commit)",
        local: true,
    },
    SlashCommand {
        name: "/skills",
        args_hint: "[inspect|install|search <name>]",
        description: "Skills overlay · enter inspect  i install",
        local: true,
    },
    SlashCommand {
        name: "/handoff",
        args_hint: "<telegram|discord|…>",
        description: "Queue this session to a gateway home channel",
        local: true,
    },
    SlashCommand {
        name: "/profiles",
        args_hint: "[new <slug>|clone <slug> [from]]",
        description: "Profiles overlay · /profiles new|clone via gateway",
        local: true,
    },
    SlashCommand {
        name: "/bots",
        args_hint: "",
        description: "Alias of /profiles",
        local: true,
    },
    SlashCommand {
        name: "/agents",
        args_hint: "[pause|resume|status]",
        description: "Subagent tree · pause/resume spawn · x stop  /stop kills all",
        local: true,
    },
    SlashCommand {
        name: "/processes",
        args_hint: "",
        description: "Work sidebar — processes + git diff-check (Ctrl+W)",
        local: true,
    },
    SlashCommand {
        name: "/work",
        args_hint: "",
        description: "Alias of /processes — work rail",
        local: true,
    },
    SlashCommand {
        name: "/stop",
        args_hint: "",
        description: "Kill all background processes (process.stop)",
        local: true,
    },
    SlashCommand {
        name: "/memory",
        args_hint: "",
        description: "Learned skills and memories…",
        local: true,
    },
    SlashCommand {
        name: "/journey",
        args_hint: "",
        description: "Alias of /memory",
        local: true,
    },
    SlashCommand {
        name: "/sessions",
        args_hint: "",
        description: "List and resume sessions…",
        local: true,
    },
    SlashCommand {
        name: "/compress",
        args_hint: "",
        description: "Compress the context window…",
        local: false,
    },
    SlashCommand {
        name: "/yolo",
        args_hint: "",
        description: "Toggle approval bypass (Shift+Tab cycles plan → ask → yolo)",
        local: true,
    },
    SlashCommand {
        name: "/plan",
        args_hint: "",
        description: "Enter plan mode — read-only, writes denied until you cycle out",
        local: true,
    },
    SlashCommand {
        name: "/mcp",
        args_hint: "[add|remove|test|key|login <name>]",
        description: "MCP hub — a add  t test  k key  o oauth  x remove  r reload",
        local: true,
    },
    SlashCommand {
        name: "/init",
        args_hint: "",
        description: "Write AGENTS.md in the workspace if missing",
        local: true,
    },
    SlashCommand {
        name: "/export",
        args_hint: "",
        description: "Copy the session transcript as Markdown",
        local: true,
    },
    SlashCommand {
        name: "/palette",
        args_hint: "",
        description: "Command palette (Ctrl+P)",
        local: true,
    },
    SlashCommand {
        name: "/goal",
        args_hint: "<task>",
        description: "Set or inspect the session goal",
        local: true,
    },
    SlashCommand {
        name: "/context",
        args_hint: "",
        description: "Context map… window usage, categories, compress",
        local: true,
    },
    SlashCommand {
        name: "/usage",
        args_hint: "",
        description: "Alias of /context",
        local: true,
    },
    SlashCommand {
        name: "/theme",
        args_hint: "[name]",
        description: "Switch TUI theme… palette or /theme. grok black, Omarchy, Dracula…",
        local: true,
    },
    SlashCommand {
        name: "/skin",
        args_hint: "[name]",
        description: "Alias of /theme",
        local: true,
    },
    SlashCommand {
        name: "/diff",
        args_hint: "",
        description: "Git diff split vs HEAD (Ctrl+D) — live working-tree view",
        local: true,
    },
    SlashCommand {
        name: "/files",
        args_hint: "",
        description: "File explorer + per-file diff… o open  r restore  u undo",
        local: true,
    },
    SlashCommand {
        name: "/explorer",
        args_hint: "",
        description: "Alias of /files",
        local: true,
    },
    SlashCommand {
        name: "/overview",
        args_hint: "",
        description: "Agent overview… now / queue / tasks / subagents (Ctrl+T)",
        local: true,
    },
    SlashCommand {
        name: "/trace",
        args_hint: "",
        description: "Toggle overview rail (Ctrl+G). a opens /agents",
        local: true,
    },
    SlashCommand {
        name: "/tips",
        args_hint: "",
        description: "Show or hide the did-you-know bar (click × to close, click bar for next)",
        local: true,
    },
    SlashCommand {
        name: "/thinking",
        args_hint: "",
        description: "Toggle reasoning block expansion (Tab)",
        local: true,
    },
    SlashCommand {
        name: "/background",
        args_hint: "<prompt>",
        description: "Background tasks… type a prompt, enter launch, enter a row to peek",
        local: true,
    },
    SlashCommand {
        name: "/bg",
        args_hint: "<prompt>",
        description: "Alias of /background",
        local: true,
    },
    SlashCommand {
        name: "/btw",
        args_hint: "<prompt>",
        description: "Alias of /background",
        local: true,
    },
    SlashCommand {
        name: "/rollback",
        args_hint: "",
        description: "Filesystem checkpoints… enter restore  d diff",
        local: true,
    },
    SlashCommand {
        name: "/clear",
        args_hint: "",
        description: "Clear the local transcript view… u undo",
        local: true,
    },
    SlashCommand {
        name: "/doctor",
        args_hint: "",
        description: "Run environment diagnostics…",
        local: false,
    },
    SlashCommand {
        name: "/exit",
        args_hint: "",
        description: "Quit — twice if a draft is unsaved",
        local: true,
    },
];

pub fn normalize_slash_search_query(query: &str) -> String {
    query.trim().trim_start_matches('/').to_ascii_lowercase()
}

pub fn tokenize_search_text(value: &str) -> Vec<String> {
    let normalized = value.to_ascii_lowercase();
    let mut out = vec![normalized.clone()];
    for token in normalized.split(|c: char| !c.is_ascii_alphanumeric()) {
        if !token.is_empty() {
            out.push(token.to_string());
        }
    }
    out
}

fn score_fields(fields: &[String], query: &str, offset: u32) -> Option<u32> {
    for field in fields {
        if field == query || format!("/{field}") == query {
            return Some(offset);
        }
    }
    for field in fields {
        if field.starts_with(query) || format!("/{field}").starts_with(query) {
            return Some(offset + 1);
        }
    }
    for field in fields {
        if field.contains(query) {
            return Some(offset + 2);
        }
    }
    None
}

pub fn parse_slash(cmd: &str) -> (String, String) {
    let rest = cmd.trim().trim_start_matches('/');
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg = parts.next().unwrap_or("").trim().to_string();
    (name, arg)
}

pub fn looks_like_slash_command(text: &str) -> bool {
    let t = text.trim_end();
    t.starts_with('/') && !t[1..].contains('/') && !t.contains('\n')
}

pub fn score_entry(name: &str, description: &str, query: &str) -> Option<u32> {
    let normalized = normalize_slash_search_query(query);
    if normalized.is_empty() {
        return Some(100);
    }
    let id = name.trim_start_matches('/');
    let command_fields = tokenize_search_text(id);
    let description_fields = tokenize_search_text(description);
    let name_score = score_fields(&command_fields, &normalized, 0);
    let desc_score = score_fields(&description_fields, &normalized, 3);
    match (name_score, desc_score) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

pub fn local_entries() -> Vec<SlashEntry> {
    SLASH_COMMANDS.iter().map(SlashCommand::to_entry).collect()
}

pub fn parse_catalog(v: &Value) -> Vec<SlashEntry> {
    let skill_keys = v.get("skills").and_then(|s| s.as_object());
    let mut extra = Vec::new();
    let Some(pairs) = v.get("pairs").and_then(|p| p.as_array()) else {
        return extra;
    };
    for pair in pairs {
        let name = pair.get(0).and_then(|x| x.as_str()).unwrap_or("").trim();
        let desc = pair.get(1).and_then(|x| x.as_str()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let name = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{name}")
        };
        let is_skill = skill_keys
            .is_some_and(|s| s.contains_key(&name) || s.contains_key(name.trim_start_matches('/')));
        extra.push(SlashEntry {
            name,
            args_hint: String::new(),
            description: if is_skill {
                if desc.is_empty() {
                    "skill".into()
                } else {
                    format!("skill · {desc}")
                }
            } else {
                desc.to_string()
            },
            kind: if is_skill {
                SlashKind::Skill
            } else {
                SlashKind::Command
            },
        });
    }
    extra
}

pub fn parse_complete_slash(v: &Value) -> (Vec<SlashEntry>, usize) {
    let replace_from = v.get("replace_from").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
    let items = v
        .get("items")
        .and_then(|a| a.as_array())
        .into_iter()
        .flatten()
        .filter_map(|it| {
            let raw = it.get("text").and_then(|s| s.as_str())?.trim();
            if raw.is_empty() {
                return None;
            }
            let name = raw.to_string();
            let kind = match it.get("kind").and_then(|s| s.as_str()) {
                Some("skill") => SlashKind::Skill,
                Some("local") => SlashKind::Local,
                _ => SlashKind::Command,
            };
            Some(SlashEntry {
                name,
                args_hint: String::new(),
                description: it
                    .get("meta")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string(),
                kind,
            })
        })
        .collect();
    (items, replace_from)
}

pub fn slash_arg_stage(query: &str, replace_from: usize) -> bool {
    query.contains(' ') || replace_from > 1
}

pub fn merge_entries(extra: Vec<SlashEntry>) -> Vec<SlashEntry> {
    let mut out = local_entries();
    let mut seen: HashSet<String> = out.iter().map(|e| e.name.to_ascii_lowercase()).collect();
    for e in extra {
        if seen.insert(e.name.to_ascii_lowercase()) {
            out.push(e);
        }
    }
    out
}

/// Typed payload from `slash.exec` / `command.dispatch` (Ink `parseCommandDispatch`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandDispatch {
    Exec {
        output: String,
    },
    Plugin {
        output: String,
    },
    Alias {
        target: String,
    },
    Skill {
        name: String,
        message: String,
        display: String,
    },
    Send {
        message: String,
        notice: String,
        display: String,
    },
    Prefill {
        message: String,
        notice: String,
    },
}

fn json_str(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

pub fn parse_command_dispatch(v: &Value) -> Option<CommandDispatch> {
    let ty = v.get("type")?.as_str()?;
    match ty {
        "exec" => Some(CommandDispatch::Exec {
            output: json_str(v, "output"),
        }),
        "plugin" => Some(CommandDispatch::Plugin {
            output: json_str(v, "output"),
        }),
        "alias" => {
            let target = v.get("target")?.as_str()?.to_string();
            Some(CommandDispatch::Alias { target })
        }
        "skill" => {
            let name = v.get("name")?.as_str()?.to_string();
            Some(CommandDispatch::Skill {
                name,
                message: json_str(v, "message"),
                display: json_str(v, "display"),
            })
        }
        "send" => {
            let message = v.get("message")?.as_str()?.to_string();
            Some(CommandDispatch::Send {
                message,
                notice: json_str(v, "notice"),
                display: json_str(v, "display"),
            })
        }
        "prefill" => {
            let message = v.get("message")?.as_str()?.to_string();
            Some(CommandDispatch::Prefill {
                message,
                notice: json_str(v, "notice"),
            })
        }
        _ => None,
    }
}

/// `command.dispatch` saying the name is unknown — keep the original slash.exec error.
pub fn is_dispatch_routing_noise(err: &str) -> bool {
    let t = err.to_ascii_lowercase();
    t.contains("not a quick/plugin/skill command")
        || t.contains("not a quick/plugin/bundle/skill command")
}

pub fn rank_entries<'a>(query: &str, entries: &'a [SlashEntry]) -> Vec<&'a SlashEntry> {
    let mut scored: Vec<(&SlashEntry, u32)> = entries
        .iter()
        .filter_map(|e| score_entry(&e.name, &e.description, query).map(|s| (e, s)))
        .collect();
    scored.sort_by_key(|(_, s)| *s);
    scored.into_iter().map(|(e, _)| e).collect()
}

pub fn executable_entries<'a>(query: &str, entries: &'a [SlashEntry]) -> Vec<&'a SlashEntry> {
    let normalized = normalize_slash_search_query(query);
    if normalized.is_empty() {
        return entries.iter().collect();
    }
    let mut scored: Vec<(&SlashEntry, u32)> = entries
        .iter()
        .filter_map(|e| {
            let score = score_entry(&e.name, &e.description, query)?;
            if score < 3 {
                Some((e, score))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by_key(|(_, s)| *s);
    scored.into_iter().map(|(e, _)| e).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_beats_prefix() {
        assert_eq!(
            score_entry("/help", "Show commands and keyboard shortcuts", "help"),
            Some(0)
        );
        assert_eq!(
            score_entry("/help", "Show commands and keyboard shortcuts", "hel"),
            Some(1)
        );
    }

    #[test]
    fn description_match_is_not_executable() {
        // /copy description mentions "assistant"; that is a completion-only hit.
        let catalog = local_entries();
        let exec = executable_entries("assistant", &catalog);
        assert!(exec.is_empty());
        let ranked = rank_entries("assistant", &catalog);
        assert!(ranked.iter().any(|c| c.name == "/copy"));
    }

    #[test]
    fn exact_name_executes() {
        let catalog = local_entries();
        let matches = executable_entries("/clear", &catalog);
        assert_eq!(matches[0].name, "/clear");
    }

    #[test]
    fn parse_splits_arg() {
        let (n, a) = parse_slash("/goal ship the tui");
        assert_eq!(n, "goal");
        assert_eq!(a, "ship the tui");
    }

    #[test]
    fn looks_like_slash_rejects_paths() {
        assert!(looks_like_slash_command("/help"));
        assert!(!looks_like_slash_command("/usr/bin"));
        assert!(!looks_like_slash_command("look at /help"));
    }

    #[test]
    fn catalog_merges_skills_and_keeps_local() {
        let extra = parse_catalog(&serde_json::json!({
            "pairs": [
                ["/help", "gateway help"],
                ["/doctor", "Run diagnostics"],
                ["/x-content", "Write for X"],
                ["skills", "Installed skills overlay"]
            ],
            "skills": {
                "/x-content": { "usage": 3, "origin": "local" }
            }
        }));
        assert_eq!(extra.len(), 4);
        let skill = extra.iter().find(|e| e.name == "/x-content").unwrap();
        assert_eq!(skill.kind, SlashKind::Skill);
        assert!(skill.description.starts_with("skill"));
        let merged = merge_entries(extra);
        let help = merged.iter().find(|e| e.name == "/help").unwrap();
        assert_eq!(help.kind, SlashKind::Local);
        assert!(merged.iter().any(|e| e.name == "/x-content"));
        let ranked = rank_entries("x-con", &merged);
        assert_eq!(ranked[0].name, "/x-content");
        let exec = executable_entries("assistant", &merged);
        assert!(exec.is_empty());
        let (items, from) = parse_complete_slash(&serde_json::json!({
            "items": [
                { "text": "concise", "meta": "short", "kind": "command" },
                { "text": "/x-content", "meta": "write", "kind": "skill" }
            ],
            "replace_from": 13
        }));
        assert_eq!(from, 13);
        assert!(slash_arg_stage("/personality ", from));
        assert_eq!(items[0].name, "concise");
        assert_eq!(items[1].kind, SlashKind::Skill);
        let send = parse_command_dispatch(&serde_json::json!({
            "type": "send",
            "message": "kickoff",
            "notice": "⊙ Goal set",
            "display": "/goal kickoff"
        }))
        .unwrap();
        assert!(matches!(
            send,
            CommandDispatch::Send { message, notice, .. }
                if message == "kickoff" && notice.contains("Goal")
        ));
        let skill = parse_command_dispatch(&serde_json::json!({
            "type": "skill",
            "name": "x-content",
            "message": "SKILL.md body",
            "display": "/x-content"
        }))
        .unwrap();
        assert!(matches!(skill, CommandDispatch::Skill { name, .. } if name == "x-content"));
        assert!(parse_command_dispatch(&serde_json::json!({ "output": "plain" })).is_none());
        assert!(is_dispatch_routing_noise(
            "not a quick/plugin/bundle/skill command: foo"
        ));
    }

    #[test]
    fn leftover_gateway_verbs_are_local() {
        let catalog = local_entries();
        for name in [
            "/replay-diff",
            "/hide",
            "/unhide",
            "/react",
            "/imagine",
            "/reload-mcp",
            "/redirect",
            "/workspace",
            "/projects",
            "/cli",
            "/vim",
            "/motion",
            "/commit",
            "/handoff",
            "/mem",
        ] {
            assert!(
                catalog
                    .iter()
                    .any(|c| c.name == name && c.kind == SlashKind::Local),
                "{name}"
            );
        }
    }

    #[test]
    fn background_aliases_are_local() {
        let catalog = local_entries();
        for name in ["/stop", "/processes", "/work"] {
            assert!(
                catalog
                    .iter()
                    .any(|c| c.name == name && c.kind == SlashKind::Local),
                "{name}"
            );
        }
        for name in ["/background", "/bg", "/btw"] {
            let entry = catalog.iter().find(|c| c.name == name).expect(name);
            assert_eq!(entry.kind, SlashKind::Local);
        }
        for cmd in ["/background research", "/bg research", "/btw research"] {
            let (n, a) = parse_slash(cmd);
            assert!(matches!(n.as_str(), "background" | "bg" | "btw"));
            assert_eq!(a, "research");
        }
    }
}
