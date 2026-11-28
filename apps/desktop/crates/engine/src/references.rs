//! `@` references in a prompt (agent/context_references.py): `@file:path[:a-b]`,
//! `@folder:path`, `@diff`, `@staged`, `@git:N` and `@url:https://…`. Each is
//! expanded once when the message is sent and frozen on the turn
//! (`Turn::context`), so later rounds resend identical bytes.
use std::{fmt::Write as _, path::{Path, PathBuf}, process::Command, sync::LazyLock, time::Duration};

const MAX_REFS: usize = 16;
const MAX_FILE_BYTES: u64 = 400 * 1024;
const MAX_URL_BYTES: usize = 400 * 1024;
const SENSITIVE: &[&str] = &[".ssh", ".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile", ".netrc", ".pgpass", ".npmrc", ".pypirc", ".hermes/.env", ".hermes/auth.json", ".aws", ".gnupg", ".config/gh"];

static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(
    r#"(?:^|[^\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>(?:`[^`\n]+`|"[^"\n]+"|'[^'\n]+')(?::\d+(?:-\d+)?)?|\S+))"#).unwrap());

#[derive(Debug, Clone, PartialEq)]
pub struct Reference { pub raw: String, pub kind: String, pub value: String }

pub fn parse(prompt: &str) -> Vec<Reference> {
    PATTERN.captures_iter(prompt).take(MAX_REFS).map(|c| {
        let (kind, value) = match c.name("simple") { Some(s) => (s.as_str().to_owned(), String::new()), None => (c["kind"].to_owned(), trim_value(&c["value"])) };
        let raw = if value.is_empty() { format!("@{kind}") } else { format!("@{kind}:{value}") };
        Reference { raw, kind, value }
    }).collect()
}
fn trim_value(value: &str) -> String {
    let mut v = value.trim_end_matches([',', '.', ';', '!', '?']).to_owned();
    for (close, open) in [(')', '('), (']', '['), ('}', '{')] {
        if v.ends_with(close) && v.matches(close).count() > v.matches(open).count() { v.pop(); }
    }
    v
}
/// `path[:start[-end]]`, the path possibly quoted.
fn file_value(value: &str) -> (String, Option<usize>, Option<usize>) {
    let quoted = ['`', '"', '\''].iter().find_map(|q| value.strip_prefix(*q).and_then(|rest| rest.find(*q).map(|end| (rest[..end].to_owned(), &rest[end + 1..]))));
    let (path, tail) = match quoted { Some((p, t)) => (p, t.to_owned()), None => match value.rsplit_once(':') {
        Some((p, r)) if !r.is_empty() && r.split('-').all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())) => (p.to_owned(), format!(":{r}")),
        _ => (value.to_owned(), String::new()) } };
    let mut range = tail.trim_start_matches(':').split('-').filter_map(|n| n.parse().ok());
    (path, range.next(), range.next())
}

fn resolve(workspace: &Path, target: &str) -> Result<PathBuf, String> {
    let expanded = match target.strip_prefix("~/") { Some(rest) => directories::BaseDirs::new().map(|d| d.home_dir().join(rest)).ok_or("no home directory")?, None => PathBuf::from(target) };
    let path = if expanded.is_absolute() { expanded } else { workspace.join(expanded) };
    let path = path.canonicalize().map_err(|_| "not found".to_owned())?;
    if let Some(home) = directories::BaseDirs::new().map(|d| d.home_dir().to_path_buf()) {
        if SENSITIVE.iter().any(|s| path.starts_with(home.join(s))) { return Err("refused: that path holds credentials".into()); }
    }
    Ok(path)
}

fn tokens(text: &str) -> usize { text.len().div_ceil(4) }
fn fence(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or_default() {
        "rs" => "rust", "py" => "python", "ts" | "tsx" => "typescript", "js" | "jsx" => "javascript", "json" => "json", "md" => "markdown",
        "toml" => "toml", "yaml" | "yml" => "yaml", "sh" => "bash", "html" => "html", "css" => "css", "go" => "go", "swift" => "swift", _ => "",
    }
}

fn expand_file(r: &Reference, workspace: &Path) -> Result<String, String> {
    let (target, start, end) = file_value(&r.value);
    let path = resolve(workspace, &target)?;
    if !path.is_file() { return Err("path is not a file".into()); }
    let size = path.metadata().map_err(|e| e.to_string())?.len();
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if bytes.iter().take(8192).any(|b| *b == 0) { return Ok(format!("📄 {} (binary file, {size} bytes — not inlined; use the file tools)", r.raw)); }
    if size > MAX_FILE_BYTES && start.is_none() { return Ok(format!("📄 {} (~{} tokens — too large to inline; read it with the file tools)", r.raw, size / 4)); }
    let text = String::from_utf8_lossy(&bytes);
    let text = text.trim_start_matches('\u{feff}');
    let body = match start {
        Some(a) => text.lines().enumerate().skip(a.saturating_sub(1)).take(end.unwrap_or(a).saturating_sub(a) + 1).map(|(i, l)| format!("{:>5} {l}", i + 1)).collect::<Vec<_>>().join("\n"),
        None => text.to_owned(),
    };
    Ok(format!("📄 {} ({} tokens)\n```{}\n{body}\n```", r.raw, tokens(&body), fence(&path)))
}

fn expand_folder(r: &Reference, workspace: &Path) -> Result<String, String> {
    let path = resolve(workspace, &file_value(&r.value).0)?;
    if !path.is_dir() { return Err("path is not a folder".into()); }
    let mut lines = vec![format!("{}/", path.strip_prefix(workspace).unwrap_or(&path).display())];
    let mut count = 0;
    fn walk(dir: &Path, depth: usize, lines: &mut Vec<String>, count: &mut usize) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = read.flatten().filter(|e| { let n = e.file_name(); let n = n.to_string_lossy(); !n.starts_with('.') && !matches!(n.as_ref(), "node_modules" | "target" | "__pycache__" | "dist" | "build") }).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if *count >= 200 { return; }
            *count += 1;
            let indent = "  ".repeat(depth);
            let kind = entry.file_type();
            if kind.as_ref().is_ok_and(|k| k.is_dir()) {
                lines.push(format!("{indent}- {}/", entry.file_name().to_string_lossy()));
                if depth < 2 { walk(&entry.path(), depth + 1, lines, count); }
            } else {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                lines.push(format!("{indent}- {} ({size} bytes)", entry.file_name().to_string_lossy()));
            }
        }
    }
    walk(&path, 0, &mut lines, &mut count);
    if count >= 200 { lines.push("- ...".into()); }
    let listing = lines.join("\n");
    Ok(format!("📁 {} ({} tokens)\n{listing}", r.raw, tokens(&listing)))
}

fn expand_git(r: &Reference, workspace: &Path) -> Result<String, String> {
    let args: Vec<String> = match r.kind.as_str() {
        "diff" => vec!["diff".into()],
        "staged" => vec!["diff".into(), "--staged".into()],
        _ => vec!["log".into(), format!("-{}", r.value.parse::<u32>().unwrap_or(1).clamp(1, 10)), "-p".into()],
    };
    // Repository config must never run code: no hooks, fsmonitor, external diff or textconv.
    let mut command = Command::new("git");
    command.current_dir(workspace).env("GIT_TERMINAL_PROMPT", "0").env("GIT_CONFIG_NOSYSTEM", "1")
        .args(["--no-pager", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "diff.external=", "-c", "core.pager=cat"]).args(&args);
    if args[0] == "diff" || args[0] == "log" { command.args(["--no-ext-diff", "--no-textconv"]); }
    let output = command.output().map_err(|_| "git is not available".to_owned())?;
    if !output.status.success() { return Err(String::from_utf8_lossy(&output.stderr).trim().chars().take(300).collect()); }
    let mut text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if text.is_empty() { text = "(no output)".into(); }
    if text.len() > MAX_FILE_BYTES as usize { let mut cut = MAX_FILE_BYTES as usize; while !text.is_char_boundary(cut) { cut -= 1; } text.truncate(cut); text.push_str("\n… (truncated)"); }
    Ok(format!("🧾 git {} ({} tokens)\n```diff\n{text}\n```", args.join(" "), tokens(&text)))
}

async fn expand_url(r: &Reference) -> Result<String, String> {
    let url = reqwest::Url::parse(&r.value).map_err(|_| "invalid URL".to_owned())?;
    if !matches!(url.scheme(), "http" | "https") || !url.username().is_empty() || url.password().is_some() { return Err("only plain http(s) links are fetched".into()); }
    let host = url.host_str().unwrap_or_default().to_owned();
    // No private, loopback or link-local destinations (SSRF).
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs = tokio::net::lookup_host((host.as_str(), port)).await.map_err(|_| "could not resolve host".to_owned())?;
    for addr in addrs {
        let private = match addr.ip() {
            std::net::IpAddr::V4(ip) => ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified() || ip.is_broadcast() || ip.octets()[0] == 100 && (ip.octets()[1] & 0xc0) == 64,
            std::net::IpAddr::V6(ip) => ip.is_loopback() || ip.is_unspecified() || (ip.segments()[0] & 0xfe00) == 0xfc00 || (ip.segments()[0] & 0xffc0) == 0xfe80,
        };
        if private { return Err("refused: that link points at a private network address".into()); }
    }
    let client = reqwest::Client::builder().timeout(Duration::from_secs(20)).redirect(reqwest::redirect::Policy::limited(3))
        .user_agent(concat!("HermesRust/", env!("CARGO_PKG_VERSION"))).build().map_err(|e| e.to_string())?;
    let mut response = client.get(url).send().await.map_err(|e| format!("fetch failed: {e}"))?;
    if !response.status().is_success() { return Err(format!("HTTP {}", response.status())); }
    let html = response.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("html"));
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        bytes.extend_from_slice(&chunk);
        if bytes.len() > MAX_URL_BYTES * 4 { break; }
    }
    let raw = String::from_utf8_lossy(&bytes);
    let mut text = if html { html_text(&raw) } else { raw.into_owned() };
    if text.trim().is_empty() { return Err("no content extracted".into()); }
    if text.len() > MAX_URL_BYTES { let mut cut = MAX_URL_BYTES; while !text.is_char_boundary(cut) { cut -= 1; } text.truncate(cut); text.push_str("\n… (truncated)"); }
    Ok(format!("🌐 {} ({} tokens)\n{}", r.raw, tokens(&text), text.trim()))
}

/// Readable text of an HTML page: scripts/styles dropped, block tags as line breaks.
fn html_text(html: &str) -> String {
    static DROP: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"(?is)<(script|style|noscript|svg|head)\b.*?</(script|style|noscript|svg|head)>").unwrap());
    static BLOCK: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr|/section|/article)\b[^>]*>").unwrap());
    static TAG: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"<[^>]*>").unwrap());
    static BLANK: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"\n\s*\n\s*\n+").unwrap());
    let text = DROP.replace_all(html, "");
    let text = BLOCK.replace_all(&text, "\n");
    let text = TAG.replace_all(&text, "");
    let text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'");
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    BLANK.replace_all(&lines.join("\n"), "\n\n").into_owned()
}

/// The context block appended to the prompt for the model, or "" when it has no references.
pub async fn expand(prompt: &str, workspace: &Path) -> String {
    let refs = parse(prompt);
    if refs.is_empty() { return String::new(); }
    let (mut blocks, mut warnings) = (vec![], vec![]);
    for r in &refs {
        let result = match r.kind.as_str() {
            "file" => expand_file(r, workspace),
            "folder" => expand_folder(r, workspace),
            "diff" | "staged" | "git" => expand_git(r, workspace),
            "url" => expand_url(r).await,
            _ => Err("unsupported reference type".into()),
        };
        match result { Ok(block) => blocks.push(block), Err(e) => warnings.push(format!("{}: {e}", r.raw)) }
    }
    let mut out = String::new();
    if !warnings.is_empty() { let _ = write!(out, "\n\n--- Context Warnings ---\n{}", warnings.iter().map(|w| format!("- {w}")).collect::<Vec<_>>().join("\n")); }
    if !blocks.is_empty() { let _ = write!(out, "\n\n--- Attached Context ---\n\n{}", blocks.join("\n\n")); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_the_hermes_reference_grammar() {
        let refs = parse("look at @file:src/main.rs:10-20, @folder:docs and @diff; also @git:3 @url:https://example.com/a. not me@file:x");
        let raw: Vec<_> = refs.iter().map(|r| r.raw.as_str()).collect();
        assert_eq!(raw, ["@file:src/main.rs:10-20", "@folder:docs", "@diff", "@git:3", "@url:https://example.com/a"]);
        assert_eq!(file_value("src/main.rs:10-20"), ("src/main.rs".into(), Some(10), Some(20)));
        assert_eq!(file_value("\"my file.rs\":5"), ("my file.rs".into(), Some(5), None));
        assert_eq!(parse("@file:\"a b.txt\"")[0].value, "\"a b.txt\"");
        assert_eq!(parse("(see @file:x.rs)")[0].value, "x.rs");
    }
    #[tokio::test]
    async fn expands_files_folders_and_ranges_inside_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "one\ntwo\nthree\nfour\n").unwrap();
        std::fs::write(dir.path().join("blob.bin"), [0u8, 1, 2]).unwrap();
        let out = expand("@file:src/a.rs:2-3 @folder:src @file:blob.bin @file:missing.rs", dir.path()).await;
        assert!(out.contains("--- Attached Context ---"));
        assert!(out.contains("    2 two\n    3 three") && !out.contains("four"), "{out}");
        assert!(out.contains("```rust"));
        assert!(out.contains("- a.rs (19 bytes)"));
        assert!(out.contains("binary file"));
        assert!(out.contains("--- Context Warnings ---\n- @file:missing.rs: not found"));
        assert_eq!(expand("plain text", dir.path()).await, "");
        assert!(html_text("<html><head><title>x</title></head><body><script>bad()</script><p>Hi&amp;bye</p></body></html>").contains("Hi&bye"));
        assert!(expand_url(&Reference { raw: "@url:http://127.0.0.1/".into(), kind: "url".into(), value: "http://127.0.0.1/".into() }).await.unwrap_err().contains("private"));
    }
}
