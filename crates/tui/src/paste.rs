//! Collapse long pastes into Grok-style `[[ … ]]` chips.

pub const LINE_MIN: usize = 5;
pub const CHAR_MIN: usize = 2000;

#[derive(Debug, Clone)]
pub struct PasteChip {
    pub label: String,
    pub body: String,
    pub path: Option<String>,
}

pub fn should_collapse(text: &str) -> bool {
    if text.trim().is_empty() {
        return false;
    }
    text.lines().count() >= LINE_MIN || text.chars().count() >= CHAR_MIN
}

pub fn fmt_k(n: usize) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Grok: `[[ head.. [N lines] .. tail ]]`
pub fn token_label(text: &str) -> String {
    let lines = text.lines().count().max(1);
    let k = fmt_k(lines);
    let preview = edge_preview(text);
    if preview.is_empty() {
        return format!("[[ [{k} lines] ]]");
    }
    if let Some((head, tail)) = preview.split_once(".. ") {
        format!(
            "[[ {}.. [{k} lines] .. {} ]]",
            head.trim_end(),
            tail.trim_start()
        )
    } else {
        format!("[[ {preview} [{k} lines] ]]")
    }
}

fn edge_preview(s: &str) -> String {
    let one = s
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("]]", "] ]");
    if one.is_empty() {
        return String::new();
    }
    const HEAD: usize = 16;
    const TAIL: usize = 28;
    if one.chars().count() <= HEAD + TAIL + 4 {
        return one;
    }
    let head: String = one.chars().take(HEAD).collect();
    let tail: String = one
        .chars()
        .rev()
        .take(TAIL)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{}.. {}", head.trim_end(), tail.trim_start())
}

pub fn unique_label(base: &str, existing: &[PasteChip]) -> String {
    if !existing.iter().any(|c| c.label == base) {
        return base.to_string();
    }
    for n in 2..99 {
        let candidate = format!("{}·{n}", base.trim_end_matches(" ]]")) + " ]]";
        if !existing.iter().any(|c| c.label == candidate) {
            return candidate;
        }
    }
    format!("{base}·x")
}

/// Spans of `[[ … ]]` in `text` as byte ranges.
pub fn token_spans(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            if let Some(rel) = text[i + 2..].find("]]") {
                let end = i + 2 + rel + 2;
                if !text[i..end].contains('\n') {
                    out.push((i, end));
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

pub fn token_at(text: &str, byte_off: usize) -> Option<&str> {
    token_spans(text)
        .into_iter()
        .find(|(a, b)| byte_off >= *a && byte_off < *b)
        .map(|(a, b)| &text[a..b])
}

pub fn byte_offset(lines: &[impl AsRef<str>], row: usize, col: usize) -> usize {
    let mut n = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let s = line.as_ref();
        if i < row {
            n += s.len() + 1;
        } else {
            n += s.chars().take(col).map(|c| c.len_utf8()).sum::<usize>();
            break;
        }
    }
    n
}

pub fn expand(text: &str, chips: &[PasteChip]) -> String {
    let mut out = text.to_string();
    for chip in chips {
        if out.contains(&chip.label) {
            out = out.replace(&chip.label, &chip.body);
        }
    }
    out
}

pub fn prune(chips: &mut Vec<PasteChip>, text: &str) {
    chips.retain(|c| text.contains(&c.label));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_threshold() {
        assert!(!should_collapse("hi\nthere"));
        assert!(should_collapse("a\nb\nc\nd\ne\nf"));
        assert!(should_collapse(&"x".repeat(2000)));
    }

    #[test]
    fn grok_label_has_brackets_and_count() {
        let body = "fn main() {\n    println!(\"hi\");\n}\nfn extra() {}\nfn more() {}\n";
        let label = token_label(body);
        assert!(label.starts_with("[[ "));
        assert!(label.ends_with(" ]]"));
        assert!(label.contains("lines"));
    }

    #[test]
    fn token_at_hits_chip() {
        let t = "see [[ hello [3 lines] ]] please";
        let start = t.find("[[").unwrap();
        assert_eq!(token_at(t, start + 3), Some("[[ hello [3 lines] ]]"));
        assert!(token_at(t, 0).is_none());
    }

    #[test]
    fn expand_puts_body_back() {
        let chips = vec![PasteChip {
            label: "[[ x [2 lines] ]]".into(),
            body: "one\ntwo".into(),
            path: None,
        }];
        assert_eq!(
            expand("before [[ x [2 lines] ]] after", &chips),
            "before one\ntwo after"
        );
    }

    #[test]
    fn unique_label_suffixes() {
        let existing = vec![PasteChip {
            label: "[[ a [1 lines] ]]".into(),
            body: "a".into(),
            path: None,
        }];
        let u = unique_label("[[ a [1 lines] ]]", &existing);
        assert_ne!(u, existing[0].label);
        assert!(u.contains("[["));
    }
}
