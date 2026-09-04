//! Local SKILL.md preview for the skills overlay.
//! Gateway `skills.manage list` returns names grouped by category only.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::slash::{SlashEntry, SlashKind};
use crate::state::SkillCard;

const MAX_FILES: usize = 400;
const MAX_DEPTH: usize = 8;
const MAX_PREVIEW_CHARS: usize = 900;
const MAX_PREVIEW_LINES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDoc {
    pub name: String,
    pub description: String,
    pub preview: String,
}

/// A `commands/<name>.md` file next to SKILL.md (Cursor/Grok slash files).
#[derive(Debug, Clone)]
pub struct NestedSkillCmd {
    pub slash: String,
    pub skill_slash: String,
    pub description: String,
    pub body: String,
}

pub fn nested_skill_commands(hermes_home: &Path, cwd: &str) -> Vec<NestedSkillCmd> {
    let mut out = Vec::new();
    let mut roots = vec![hermes_home.join("skills")];
    if !cwd.is_empty() {
        roots.push(Path::new(cwd).join(".hermes").join("skills"));
        roots.push(Path::new(cwd).join("skills"));
    }
    for root in roots {
        if root.is_dir() {
            scan_nested_cmds(&root, &mut out, 0);
        }
    }
    out
}

pub fn nested_slash_entries(hermes_home: &Path, cwd: &str) -> Vec<SlashEntry> {
    nested_skill_commands(hermes_home, cwd)
        .into_iter()
        .map(|c| SlashEntry {
            name: c.slash,
            args_hint: String::new(),
            description: if c.description.is_empty() {
                format!("skill · {}", c.skill_slash)
            } else {
                format!("skill · {}", c.description)
            },
            kind: SlashKind::Skill,
        })
        .collect()
}

/// Rewrite `/devgod-audit target` into `/devgod <command body>\n\ntarget`.
pub fn expand_nested_slash(
    slash: &str,
    arg: &str,
    hermes_home: &Path,
    cwd: &str,
) -> Option<String> {
    let want = slash.trim().to_ascii_lowercase();
    let cmd = nested_skill_commands(hermes_home, cwd)
        .into_iter()
        .find(|c| c.slash.eq_ignore_ascii_case(&want))?;
    let mut line = format!("{} {}", cmd.skill_slash, cmd.body.trim());
    let arg = arg.trim();
    if !arg.is_empty() {
        line.push_str("\n\n");
        line.push_str(arg);
    }
    Some(line)
}

pub fn enrich_skill_cards(cards: &mut [SkillCard], hermes_home: &Path, cwd: &str) {
    let mut docs = HashMap::new();
    let mut roots = vec![hermes_home.join("skills")];
    if !cwd.is_empty() {
        roots.push(Path::new(cwd).join(".hermes").join("skills"));
        roots.push(Path::new(cwd).join("skills"));
    }
    for root in roots {
        if root.is_dir() {
            scan_into(&root, &mut docs, 0);
        }
    }
    for card in cards {
        let key = card.name.to_ascii_lowercase();
        if let Some(doc) = docs.get(&key) {
            if card.description.is_empty() {
                card.description = doc.description.clone();
            }
            if card.preview.is_empty() {
                card.preview = doc.preview.clone();
            }
        }
    }
}

pub fn parse_skill_md(raw: &str, fallback_name: &str) -> SkillDoc {
    let (name, description, body) = split_frontmatter(raw, fallback_name);
    let description = if description.is_empty() {
        first_prose_line(body)
    } else {
        description
    };
    SkillDoc {
        name,
        description,
        preview: preview_body(body),
    }
}

fn split_frontmatter<'a>(raw: &'a str, fallback_name: &str) -> (String, String, &'a str) {
    let trimmed = raw.trim_start_matches('\u{feff}');
    let Some(rest) = trimmed.strip_prefix("---") else {
        return (fallback_name.to_string(), String::new(), trimmed);
    };
    let rest = rest.trim_start_matches(['\r', '\n']);
    let Some(end) = rest.find("\n---") else {
        return (fallback_name.to_string(), String::new(), trimmed);
    };
    let fm = &rest[..end];
    let body = rest[end + 4..].trim_start_matches(['\r', '\n']);
    let mut name = fallback_name.to_string();
    let mut description = String::new();
    let mut lines = fm.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim_end();
        if let Some(v) = line.strip_prefix("name:") {
            let v = unquote(v.trim());
            if !v.is_empty() {
                name = v;
            }
            continue;
        }
        if let Some(v) = line.strip_prefix("description:") {
            let v = v.trim();
            if v == "|" || v == ">" || v == "|-" || v == ">-" {
                let mut block = String::new();
                while let Some(next) = lines.peek() {
                    if next.starts_with(' ') || next.starts_with('\t') || next.is_empty() {
                        let Some(taken) = lines.next() else {
                            break;
                        };
                        if !block.is_empty() {
                            block.push(' ');
                        }
                        block.push_str(taken.trim());
                    } else {
                        break;
                    }
                }
                description = block;
            } else {
                description = unquote(v);
            }
        }
    }
    (name, description, body)
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn first_prose_line(body: &str) -> String {
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("---") {
            continue;
        }
        return line.to_string();
    }
    String::new()
}

fn preview_body(body: &str) -> String {
    let mut out = String::new();
    let mut n = 0usize;
    for line in body.lines() {
        if n >= MAX_PREVIEW_LINES || out.len() >= MAX_PREVIEW_CHARS {
            break;
        }
        if out.is_empty() && line.trim().is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        n += 1;
    }
    if out.len() > MAX_PREVIEW_CHARS {
        out.truncate(MAX_PREVIEW_CHARS);
        out.push('…');
    }
    out
}

fn scan_into(dir: &Path, docs: &mut HashMap<String, SkillDoc>, depth: usize) {
    if depth > MAX_DEPTH || docs.len() >= MAX_FILES {
        return;
    }
    let skill_md = dir.join("SKILL.md");
    if skill_md.is_file() {
        if let Ok(raw) = fs::read_to_string(&skill_md) {
            let fallback = dir.file_name().and_then(|s| s.to_str()).unwrap_or("skill");
            let doc = parse_skill_md(&raw, fallback);
            docs.entry(doc.name.to_ascii_lowercase())
                .or_insert(doc.clone());
            docs.entry(fallback.to_ascii_lowercase()).or_insert(doc);
        }
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if docs.len() >= MAX_FILES {
            return;
        }
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "node_modules" || name == "target" {
            continue;
        }
        scan_into(&path, docs, depth + 1);
    }
}

fn scan_nested_cmds(dir: &Path, out: &mut Vec<NestedSkillCmd>, depth: usize) {
    if depth > MAX_DEPTH || out.len() >= MAX_FILES {
        return;
    }
    let skill_md = dir.join("SKILL.md");
    if skill_md.is_file() {
        let fallback = dir.file_name().and_then(|s| s.to_str()).unwrap_or("skill");
        let parent = if let Ok(raw) = fs::read_to_string(&skill_md) {
            parse_skill_md(&raw, fallback).name
        } else {
            fallback.to_string()
        };
        let skill_slash = format!("/{}", parent.to_ascii_lowercase().replace([' ', '_'], "-"));
        let cmds = dir.join("commands");
        if cmds.is_dir() {
            if let Ok(entries) = fs::read_dir(&cmds) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("md") {
                        continue;
                    }
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if stem.is_empty() {
                        continue;
                    }
                    let slash = format!("/{stem}");
                    if slash.eq_ignore_ascii_case(&skill_slash) {
                        continue;
                    }
                    let Ok(raw) = fs::read_to_string(&path) else {
                        continue;
                    };
                    let doc = parse_skill_md(&raw, &stem);
                    out.push(NestedSkillCmd {
                        slash,
                        skill_slash: skill_slash.clone(),
                        description: doc.description,
                        body: preview_body_full(&raw),
                    });
                }
            }
        }
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "node_modules" || name == "target" {
            continue;
        }
        scan_nested_cmds(&path, out, depth + 1);
    }
}

fn preview_body_full(raw: &str) -> String {
    let (_, _, body) = split_frontmatter(raw, "");
    body.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_frontmatter_and_preview() {
        let raw = "---\nname: gstack\ndescription: Deploy ritual and browser QA\n---\n\n# gstack\n\nUse this when shipping.\n";
        let doc = parse_skill_md(raw, "gstack");
        assert_eq!(doc.name, "gstack");
        assert_eq!(doc.description, "Deploy ritual and browser QA");
        assert!(doc.preview.contains("Use this when shipping"));
    }

    #[test]
    fn multiline_description() {
        let raw = "---\nname: x\ndescription: |\n  First line of the skill.\n  Second line.\n---\n# Title\nbody\n";
        let doc = parse_skill_md(raw, "x");
        assert!(doc.description.contains("First line"));
        assert!(doc.description.contains("Second line"));
    }

    #[test]
    fn enrich_reads_skill_md_from_disk() {
        let root = std::env::temp_dir().join(format!(
            "hermes-tui-skills-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let skill = root.join("skills").join("agentkey");
        fs::create_dir_all(&skill).unwrap();
        let mut f = fs::File::create(skill.join("SKILL.md")).unwrap();
        writeln!(
            f,
            "---\nname: agentkey\ndescription: Local key agent\n---\n\n# agentkey\n\nPreview body here.\n"
        )
        .unwrap();
        let mut cards = vec![SkillCard {
            name: "agentkey".into(),
            category: "general".into(),
            description: String::new(),
            preview: String::new(),
        }];
        enrich_skill_cards(&mut cards, &root, "");
        assert_eq!(cards[0].description, "Local key agent");
        assert!(cards[0].preview.contains("Preview body here"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn nested_commands_expand_to_parent_skill() {
        let root = std::env::temp_dir().join(format!(
            "hermes-tui-skcmd-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let skill = root.join("skills").join("devgod");
        fs::create_dir_all(skill.join("commands")).unwrap();
        fs::write(
            skill.join("SKILL.md"),
            "---\nname: devgod\ndescription: Engineering OS\n---\n\n# devgod\n",
        )
        .unwrap();
        fs::write(
            skill.join("commands").join("devgod-audit.md"),
            "---\ndescription: Audit only, no edits\n---\n\n# /devgod-audit\n\nMode: audit.\n",
        )
        .unwrap();
        let cmds = nested_skill_commands(&root, "");
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].slash, "/devgod-audit");
        assert_eq!(cmds[0].skill_slash, "/devgod");
        assert!(cmds[0].description.contains("Audit"));
        let line = expand_nested_slash("/devgod-audit", "crates/tui", &root, "").unwrap();
        assert!(line.starts_with("/devgod "));
        assert!(line.contains("Mode: audit"));
        assert!(line.contains("crates/tui"));
        let _ = fs::remove_dir_all(&root);
    }
}
