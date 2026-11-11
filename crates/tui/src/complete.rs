//! Composer path / `@ref` completion. The gateway expands `@file` / `@diff`
//! on `prompt.submit`; this crate only completes the token and attaches images.

use serde_json::Value;
use std::path::{Path, PathBuf};

pub const IMAGE_EXTS: &[&str] = &[".png", ".jpg", ".jpeg", ".webp", ".gif"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteItem {
    pub text: String,
    pub display: String,
    pub meta: String,
}

impl CompleteItem {
    pub fn keep_open(&self) -> bool {
        self.text.ends_with('/') || self.text.ends_with(':')
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathTrigger {
    pub replace_from: usize,
    pub word: String,
}

/// Last `@ref` or path-ish token at the caret (end of input).
pub fn path_trigger(input: &str) -> Option<PathTrigger> {
    if input.is_empty() {
        return None;
    }
    if crate::slash::looks_like_slash_command(input) && !input.contains(' ') {
        return None;
    }
    let start = last_token_start(input);
    let word = &input[start..];
    if word.is_empty() {
        return None;
    }
    if word.starts_with('@') {
        return Some(PathTrigger {
            replace_from: start,
            word: word.to_string(),
        });
    }
    if is_path_word(word) {
        return Some(PathTrigger {
            replace_from: start,
            word: word.to_string(),
        });
    }
    None
}

fn last_token_start(input: &str) -> usize {
    // Keep backtick-quoted `@file:`values` intact.
    if let Some(at) = input.rfind('@') {
        let prefix = &input[..at];
        if prefix.is_empty() || prefix.ends_with(char::is_whitespace) {
            return at;
        }
    }
    input
        .rfind(|c: char| c.is_whitespace())
        .map(|i| i + 1)
        .unwrap_or(0)
}

fn is_path_word(word: &str) -> bool {
    word.starts_with("./")
        || word.starts_with("../")
        || word.starts_with("~/")
        || word.starts_with('/')
        || word.contains('/')
}

pub fn apply_fill(input: &str, replace_from: usize, item: &CompleteItem) -> String {
    let mut out = input[..replace_from.min(input.len())].to_string();
    out.push_str(&item.text);
    if !item.keep_open() && !out.ends_with(' ') {
        out.push(' ');
    }
    out
}

pub fn parse_items(v: &Value) -> Vec<CompleteItem> {
    v.get("items")
        .and_then(|a| a.as_array())
        .into_iter()
        .flatten()
        .filter_map(|it| {
            let text = it.get("text").and_then(|s| s.as_str())?.to_string();
            if text.is_empty() {
                return None;
            }
            let display = it
                .get("display")
                .and_then(|s| s.as_str())
                .unwrap_or(&text)
                .to_string();
            let meta = it
                .get("meta")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            Some(CompleteItem {
                text,
                display,
                meta,
            })
        })
        .collect()
}

/// Offline / RPC-fail listing so `@` still does something.
pub fn local_items(word: &str, cwd: &str) -> Vec<CompleteItem> {
    if word == "@" || word == "@d" || word == "@di" || word == "@dif" {
        return hint_items()
            .into_iter()
            .filter(|i| i.text.starts_with(word) || word == "@")
            .collect();
    }
    if word == "@" {
        return hint_items();
    }
    let is_context = word.starts_with('@');
    let (tag, path_part) = split_at_path(word);
    let Some(path_part) = path_part else {
        return hint_items()
            .into_iter()
            .filter(|i| i.text.starts_with(word))
            .collect();
    };
    list_dir_items(cwd, tag, path_part, is_context)
}

fn hint_items() -> Vec<CompleteItem> {
    vec![
        item("@diff", "@diff", "git diff"),
        item("@staged", "@staged", "staged diff"),
        item("@file:", "@file:", "attach file"),
        item("@folder:", "@folder:", "attach folder"),
        item("@url:", "@url:", "fetch url"),
        item("@git:", "@git:", "git log"),
    ]
}

fn item(text: &str, display: &str, meta: &str) -> CompleteItem {
    CompleteItem {
        text: text.into(),
        display: display.into(),
        meta: meta.into(),
    }
}

fn split_at_path(word: &str) -> (&str, Option<&str>) {
    if !word.starts_with('@') {
        return ("file", Some(word));
    }
    let rest = &word[1..];
    if rest.is_empty() {
        return ("", None);
    }
    if let Some((tag, tail)) = rest.split_once(':') {
        (tag, Some(tail))
    } else {
        (rest, None)
    }
}

fn list_dir_items(cwd: &str, tag: &str, path_part: &str, is_context: bool) -> Vec<CompleteItem> {
    let root = Path::new(cwd);
    let (search, prefix) = if path_part.ends_with('/') || path_part.is_empty() {
        (root.join(path_part), String::new())
    } else {
        let p = Path::new(path_part);
        (
            root.join(p.parent().unwrap_or(Path::new(""))),
            p.file_name()
                .map(|s| s.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default(),
        )
    };
    let Ok(rd) = std::fs::read_dir(&search) else {
        return Vec::new();
    };
    let want_dir = tag == "folder";
    let mut out = Vec::new();
    let mut rows: Vec<(String, bool)> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                return None;
            }
            if !prefix.is_empty() && !name.to_ascii_lowercase().starts_with(&prefix) {
                return None;
            }
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            Some((name, is_dir))
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, is_dir) in rows {
        if tag == "file" && is_dir {
            // still offer dirs so the user can walk
        }
        if want_dir && !is_dir {
            continue;
        }
        let suffix = if is_dir { "/" } else { "" };
        let rel = if path_part.ends_with('/') {
            format!("{path_part}{name}{suffix}")
        } else if let Some((dir, _)) = path_part.rsplit_once('/') {
            format!("{dir}/{name}{suffix}")
        } else {
            format!("{name}{suffix}")
        };
        let text = if is_context {
            let kind = if tag.is_empty() {
                if is_dir {
                    "folder"
                } else {
                    "file"
                }
            } else {
                tag
            };
            format!("@{kind}:{rel}")
        } else {
            rel.clone()
        };
        out.push(CompleteItem {
            text,
            display: format!("{name}{suffix}"),
            meta: if is_dir { "dir".into() } else { String::new() },
        });
        if out.len() >= 24 {
            break;
        }
    }
    out
}

pub fn is_image_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    IMAGE_EXTS.iter().any(|ext| lower.ends_with(ext))
}

pub fn looks_like_dropped_image(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() || t.contains('\n') {
        return false;
    }
    is_image_path(t)
}

/// Ink `looksLikeDroppedPath`: a single-line paste that is a file URI or path,
/// not a slash command or http(s) URL.
pub fn looks_like_dropped_path(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() || t.contains('\n') {
        return false;
    }
    if t.starts_with("http://") || t.starts_with("https://") {
        return false;
    }
    if t.starts_with("file://")
        || t.starts_with("~/")
        || t.starts_with("./")
        || t.starts_with("../")
    {
        return true;
    }
    if let Some(rest) = t.strip_prefix('/') {
        return rest.contains('/') || rest.contains('.');
    }
    is_image_path(t)
}

/// Resolve `@file:` / `@folder:` / bare image paths in composer text.
pub fn image_refs_in(text: &str, cwd: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|c| c == '\'' || c == '"' || c == '(' || c == ')');
        let path = if let Some(rest) = token.strip_prefix("@file:") {
            unquote_ref(rest)
        } else if is_image_path(token) {
            token.to_string()
        } else {
            continue;
        };
        let p = resolve_in_cwd(cwd, &path);
        if is_image_path(&p.to_string_lossy()) && p.is_file() && !out.iter().any(|e| e == &p) {
            out.push(p);
        }
    }
    out
}

fn unquote_ref(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('`') && s.ends_with('`') && s.len() >= 2)
        || (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

pub fn resolve_in_cwd(cwd: &str, path: &str) -> PathBuf {
    let p = PathBuf::from(path.trim());
    if p.is_absolute() {
        p
    } else if let Some(rest) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join(rest)
    } else {
        Path::new(cwd).join(p)
    }
}

pub fn next_image_index(text: &str) -> usize {
    let mut max = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 10 < bytes.len() {
        if let Some(rest) = text[i..].strip_prefix("[[ Image ") {
            if let Some(end) = rest.find(" ]]") {
                if let Ok(n) = rest[..end].trim().parse::<usize>() {
                    max = max.max(n);
                }
                i += 10 + end;
                continue;
            }
        }
        i += 1;
    }
    max + 1
}

pub fn image_token(index: usize) -> String {
    format!("[[ Image {index} ]]")
}

/// Swap image paths / `@file:` images for `[[ Image N ]]` so the gateway
/// does not dump binary into the prompt. `image.attach` carries the pixels.
pub fn rewrite_image_tokens(text: &str, cwd: &str) -> (String, Vec<PathBuf>) {
    let refs = image_refs_in(text, cwd);
    if refs.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let mut out = text.to_string();
    let mut attached = Vec::new();
    let mut idx = next_image_index(&out);
    for path in refs {
        let as_str = path.to_string_lossy().to_string();
        let rel = path
            .strip_prefix(cwd)
            .ok()
            .map(|p| p.to_string_lossy().to_string());
        let mut replaced = false;
        for candidate in [
            format!("@file:{as_str}"),
            rel.as_ref()
                .map(|r| format!("@file:{r}"))
                .unwrap_or_default(),
            as_str.clone(),
            rel.clone().unwrap_or_default(),
        ] {
            if candidate.is_empty() {
                continue;
            }
            if out.contains(&candidate) {
                let tok = image_token(idx);
                out = out.replacen(&candidate, &tok, 1);
                replaced = true;
                break;
            }
        }
        if replaced {
            attached.push(path);
            idx += 1;
        }
    }
    (out, attached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_opens_trigger() {
        let t = path_trigger("@").unwrap();
        assert_eq!(t.word, "@");
        assert_eq!(t.replace_from, 0);
        let t = path_trigger("see @file:src/").unwrap();
        assert_eq!(t.word, "@file:src/");
        assert!(t.replace_from > 0);
    }

    #[test]
    fn slash_command_skips_path() {
        assert!(path_trigger("/help").is_none());
        assert!(path_trigger("read /usr/local").is_some());
    }

    #[test]
    fn fill_keeps_dir_open() {
        let filled = apply_fill("@file:s", 0, &item("@file:src/", "src/", "dir"));
        assert_eq!(filled, "@file:src/");
        let filled = apply_fill("@file:s", 0, &item("@file:src/a.rs", "a.rs", ""));
        assert_eq!(filled, "@file:src/a.rs ");
    }

    #[test]
    fn parse_gateway_items() {
        let v = serde_json::json!({
            "items": [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": ""}
            ]
        });
        let items = parse_items(&v);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "@diff");
    }

    #[test]
    fn image_index_and_rewrite() {
        let dir = std::env::temp_dir().join(format!("ht-img-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("shot.png");
        std::fs::write(&png, b"not-a-real-png").unwrap();
        let cwd = dir.to_string_lossy().to_string();
        let (out, paths) = rewrite_image_tokens("look at @file:shot.png please", &cwd);
        assert_eq!(paths.len(), 1);
        assert!(out.contains("[[ Image 1 ]]"));
        assert!(!out.contains("@file:shot.png"));
        assert_eq!(next_image_index(&out), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn local_at_hints() {
        let items = local_items("@", ".");
        assert!(items.iter().any(|i| i.text == "@file:"));
        assert!(items.iter().any(|i| i.text == "@diff"));
    }

    #[test]
    fn local_file_listing() {
        let dir = std::env::temp_dir().join(format!("ht-comp-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("alpha.rs"), b"x").unwrap();
        let items = local_items("@file:al", &dir.to_string_lossy());
        assert!(
            items.iter().any(|i| i.text.contains("alpha.rs")),
            "{items:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dropped_image_heuristic() {
        assert!(looks_like_dropped_image("/tmp/a.png"));
        assert!(!looks_like_dropped_image("hello.png extra"));
        assert!(!looks_like_dropped_image("notes.md"));
        assert!(looks_like_dropped_path("/usr/bin/test"));
        assert!(looks_like_dropped_path("file:///tmp/a.png"));
        assert!(!looks_like_dropped_path("/help"));
        assert!(!looks_like_dropped_path("https://example.com/a.png"));
        assert!(!looks_like_dropped_path("hello world"));
    }
}
