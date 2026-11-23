//! Official strict cumulative threat-pattern data, evaluated natively.
use fancy_regex::{Regex, RegexBuilder};
use serde::Deserialize;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct Pattern {
    pattern: String,
    id: String,
}
#[derive(Deserialize)]
struct Data {
    patterns: Vec<Pattern>,
    negated_forget: String,
    emoji_tag_sequences: Vec<String>,
}
struct Scanner {
    patterns: Vec<(Regex, String)>,
    negated_forget: Regex,
    emoji_tag_sequences: Vec<String>,
}
static SCANNER: OnceLock<Result<Scanner, String>> = OnceLock::new();

fn scanner() -> Result<&'static Scanner, &'static str> {
    SCANNER
        .get_or_init(|| {
            let data: Data = serde_json::from_str(include_str!("threat_patterns.json"))
                .map_err(|_| "Invalid static memory threat data".to_owned())?;
            Ok(Scanner {
                patterns: data
                    .patterns
                    .into_iter()
                    .map(|p| {
                        compile(&p.pattern)
                            .map(|regex| (regex, p.id.clone()))
                            .map_err(|error| {
                                format!(
                                    "Could not compile official memory pattern {}: {error}",
                                    p.id
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                negated_forget: compile(&data.negated_forget).map_err(|error| {
                    format!("Could not compile memory negation pattern: {error}")
                })?,
                emoji_tag_sequences: data
                    .emoji_tag_sequences
                    .into_iter()
                    .map(|flag| {
                        format!(
                            "\u{1f3f4}{}\u{e007f}",
                            flag.chars()
                                .map(|c| char::from_u32(0xe0000 + u32::from(c)).expect("ASCII tag"))
                                .collect::<String>()
                        )
                    })
                    .collect(),
            })
        })
        .as_ref()
        .map_err(String::as_str)
}

pub(super) fn findings(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let Ok(scanner) = scanner() else {
        return vec!["memory_scanner_unavailable".into()];
    };
    let raw = scanner
        .emoji_tag_sequences
        .iter()
        .fold(content.to_owned(), |text, flag| text.replace(flag, " "));
    let mut result = Vec::new();
    for c in raw.chars() {
        if matches!(u32::from(c),0x200b..=0x200d|0x2060|0x2062..=0x2064|0xfeff|0x202a..=0x202e|0x2066..=0x2069|0xe0000..=0xe007f)
        {
            let id = format!("invisible_unicode_U+{:04X}", u32::from(c));
            if !result.contains(&id) {
                result.push(id);
            }
        }
    }
    let folded = content.nfkc().collect::<String>();
    let normal = collapse_whitespace(&folded);
    let mut masked = folded.clone();
    // Keep offsets stable; only blank the six ASCII letters in negated forget.
    let spans = scanner
        .negated_forget
        .captures_iter(&folded)
        .filter_map(|found| found.ok())
        .filter_map(|captures| {
            captures
                .get(0)
                .map(|matched| (matched.end() - 6, matched.end()))
        })
        .collect::<Vec<_>>();
    for (start, end) in spans.into_iter().rev() {
        masked.replace_range(start..end, "      ");
    }
    let masked = collapse_whitespace(&masked);
    for (regex, id) in &scanner.patterns {
        let text = if id == "forget_rules" {
            &masked
        } else {
            &normal
        };
        // A bounded native backtracking error fails closed for prompt memory.
        match regex.is_match(text) {
            Ok(false) => {}
            Ok(true) => result.push(id.clone()),
            Err(_) => result.push("memory_scan_limit".into()),
        }
    }
    result
}

pub(super) fn first_error(content: &str) -> Option<String> {
    let findings = findings(content);
    let id = findings.first()?;
    Some(if let Some(code) = id.strip_prefix("invisible_unicode_") {
        format!(
            "Blocked: content contains invisible unicode character {code} (possible injection)."
        )
    } else {
        format!(
            "Blocked: content matches threat pattern '{id}'. Content is injected into the system prompt and must not contain injection or exfiltration payloads."
        )
    })
}

fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut run = String::new();
    for c in text.chars() {
        if c.is_whitespace() {
            run.push(c);
            continue;
        }
        flush(&mut result, &mut run);
        result.push(c);
    }
    flush(&mut result, &mut run);
    result
}
fn flush(result: &mut String, run: &mut String) {
    if run.chars().count() > 1 {
        result.push(if run.contains('\n') { '\n' } else { ' ' });
    } else {
        result.push_str(run);
    }
    run.clear();
}

// Keep Unicode matching, but execute repeated word predicates in the bounded
// VM. Expanding nested Unicode word quantifiers into a DFA exhausts its NFA
// allocation bound; the equivalent one-character lookahead avoids expansion.
fn compile(pattern: &str) -> Result<Regex, String> {
    let pattern = pattern.replace(r"[^\W_]", r"(?:(?=[^\W_]).)");
    let mut rewritten = String::new();
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                if next == 'w' && !in_class {
                    rewritten.push_str(r"(?:(?=\w).)");
                } else {
                    rewritten.push('\\');
                    rewritten.push(next);
                }
            } else {
                rewritten.push(c);
            }
        } else {
            if c == '[' {
                in_class = true;
            } else if c == ']' {
                in_class = false;
            }
            rewritten.push(c);
        }
    }
    RegexBuilder::new(&format!("(?i){rewritten}"))
        .delegate_size_limit(32 * 1024 * 1024)
        .backtrack_limit(1_000_000)
        .build()
        .map_err(|error| error.to_string())
}

#[cfg(test)]
pub(super) fn initialization_error() -> Option<&'static str> {
    scanner().err()
}
