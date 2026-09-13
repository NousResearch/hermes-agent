//! Flattened workspace tree for the files pane.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

const MAX_DEPTH: usize = 5;
const MAX_ROWS: usize = 280;

const SKIP: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".DS_Store",
    ".idea",
    ".direnv",
    ".turbo",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRow {
    pub rel: String,
    pub name: String,
    pub is_dir: bool,
    pub depth: usize,
    pub git: char,
    pub expanded: bool,
}

pub fn skip_name(name: &str) -> bool {
    SKIP.contains(&name)
}

pub fn parse_porcelain(text: &str) -> HashMap<String, char> {
    let mut out = HashMap::new();
    for line in text.lines() {
        if line.len() < 4 {
            continue;
        }
        let code = &line[..2];
        let mut path = line[3..].trim();
        if let Some((_, right)) = path.split_once(" -> ") {
            path = right;
        }
        let mark = if code.contains('?') {
            '?'
        } else if code.contains('D') {
            'D'
        } else if code.contains('A') {
            'A'
        } else if code.contains('M') || code.contains('U') {
            'M'
        } else {
            ' '
        };
        if mark != ' ' {
            out.insert(path.replace('\\', "/"), mark);
        }
    }
    out
}

pub fn git_mark(status: &HashMap<String, char>, rel: &str) -> char {
    if let Some(c) = status.get(rel) {
        return *c;
    }
    for (path, mark) in status {
        if path.starts_with(rel) && path.get(rel.len()..).is_some_and(|s| s.starts_with('/')) {
            return *mark;
        }
    }
    ' '
}

pub fn visible_rows(
    root: &Path,
    expanded: &HashSet<String>,
    status: &HashMap<String, char>,
) -> Vec<FileRow> {
    let mut out = Vec::new();
    walk(root, "", 0, expanded, status, &mut out);
    out
}

fn walk(
    root: &Path,
    rel: &str,
    depth: usize,
    expanded: &HashSet<String>,
    status: &HashMap<String, char>,
    out: &mut Vec<FileRow>,
) {
    if depth > MAX_DEPTH || out.len() >= MAX_ROWS {
        return;
    }
    let dir = if rel.is_empty() {
        root.to_path_buf()
    } else {
        root.join(rel)
    };
    let Ok(rd) = fs::read_dir(&dir) else {
        return;
    };
    let mut kids: Vec<(String, bool)> = Vec::new();
    for ent in rd.flatten() {
        if out.len() >= MAX_ROWS {
            break;
        }
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') && name != ".gitignore" && name != ".env.example" {
            if skip_name(&name) {
                continue;
            }
            if name != ".github" && name != ".hermes" {
                continue;
            }
        }
        if skip_name(&name) {
            continue;
        }
        let is_dir = ent.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if ent.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            continue;
        }
        kids.push((name.to_string(), is_dir));
    }
    kids.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase()))
    });
    for (name, is_dir) in kids {
        if out.len() >= MAX_ROWS {
            return;
        }
        let child_rel = if rel.is_empty() {
            name.clone()
        } else {
            format!("{rel}/{name}")
        };
        let open = is_dir && expanded.contains(&child_rel);
        out.push(FileRow {
            git: git_mark(status, &child_rel),
            rel: child_rel.clone(),
            name,
            is_dir,
            depth,
            expanded: open,
        });
        if open {
            walk(root, &child_rel, depth + 1, expanded, status, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn porcelain_marks_modified_and_untracked() {
        let map = parse_porcelain(" M src/app.rs\n?? scratch.txt\nA  new.rs\n");
        assert_eq!(map.get("src/app.rs"), Some(&'M'));
        assert_eq!(map.get("scratch.txt"), Some(&'?'));
        assert_eq!(map.get("new.rs"), Some(&'A'));
    }

    #[test]
    fn git_mark_bubbles_to_parent() {
        let mut map = HashMap::new();
        map.insert("src/app.rs".into(), 'M');
        assert_eq!(git_mark(&map, "src"), 'M');
        assert_eq!(git_mark(&map, "src/app.rs"), 'M');
        assert_eq!(git_mark(&map, "README.md"), ' ');
    }

    #[test]
    fn skip_build_dirs() {
        assert!(skip_name("target"));
        assert!(skip_name("node_modules"));
        assert!(!skip_name("src"));
    }
}
