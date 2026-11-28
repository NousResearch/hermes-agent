//! Port of official v0.23.3 (caafe5389) `app/artifacts/artifact-utils.ts`.
//! Native adapters supply saved engine turns and explicit imported archive rows.
use hermes_engine::{Entry, History};
use hermes_library::{HermesArchive, Library, SessionSummary};
use percent_encoding::percent_decode_str;
use regex::Regex;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::OnceLock,
};
use url::Url;

pub const RECENT_SESSIONS: usize = 30;
const SAFE_LOAD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 50_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Image,
    File,
    Link,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Filter {
    #[default]
    All,
    Images,
    Files,
    Links,
}
impl Filter {
    pub const ALL: [Self; 4] = [Self::All, Self::Images, Self::Files, Self::Links];
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Images => "Images",
            Self::Files => "Files",
            Self::Links => "Links",
        }
    }
    pub fn accepts(self, kind: Kind) -> bool {
        matches!(
            (self, kind),
            (Self::All, _)
                | (Self::Images, Kind::Image)
                | (Self::Files, Kind::File)
                | (Self::Links, Kind::Link)
        )
    }
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub id: String,
    pub kind: Kind,
    pub value: String,
    pub label: String,
    pub session_id: String,
    pub session_title: String,
    pub timestamp: f64,
    /// Owner-pinned cwd; independent of the currently selected chat/project.
    pub base_dir: PathBuf,
}
impl Artifact {
    pub fn matches(&self, filter: Filter, query: &str) -> bool {
        let query = query.trim().to_lowercase();
        filter.accepts(self.kind)
            && (query.is_empty()
                || [&self.label, &self.value, &self.session_title]
                    .iter()
                    .any(|field| field.to_lowercase().contains(&query)))
    }
}

#[derive(Debug, Default)]
pub struct Index {
    pub artifacts: Vec<Artifact>,
    pub failures: Vec<(String, String)>,
    pub sessions: usize,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: Value,
    pub text: Option<String>,
    pub context: Option<String>,
    pub tool_name: String,
    pub timestamp: Option<f64>,
    pub base_dir: PathBuf,
}

struct Patterns {
    image: Regex,
    link: Regex,
    url: Regex,
    path: Regex,
    windows: Regex,
    image_ext: Regex,
    file_ext: Regex,
    producer: Regex,
    strong_key: Regex,
    producer_key: Regex,
    screenshot: Regex,
    wheel: Regex,
    package: Regex,
    media_anchor: Regex,
    media_bare: Regex,
}
fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let re = |s| Regex::new(s).expect("upstream artifact expression must compile");
        Patterns {
            image: re(r"!\[([^\]]*)\]\(([^)\s]+)\)"),
            link: re(r"\[([^\]]+)\]\(([^)\s]+)\)"),
            url: re(r#"https?://[^\s<>"')`]+"#),
            path: re(r#"(?i)(^|[\s("'`])((?:/|~[\\/]|\.\.?[\\/]|\\\\)[^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)"#),
            windows: re(r#"(?i)(^|[\s("'`])([A-Za-z]:[\\/][^\s"'`<>]+(?:\.[a-z0-9]{1,8})?)"#),
            image_ext: re(r"(?i)\.(?:png|jpe?g|gif|webp|svg|bmp)(?:\?.*)?$"),
            file_ext: re(r"(?i)\.(?:png|jpe?g|gif|webp|svg|bmp|pdf|txt|json|md|csv|xlsx?|docx?|pptx?|html|zip|tar|gz|avi|flac|m4a|mkv|mp3|ogg|opus|wav|webm|mp4|mov)(?:\?.*)?$"),
            producer: re(r"(?i)(?:^|_)(?:creat(?:e|ion)|download|export|generat(?:e|ion)|render|save|speech|tts|write)(?:_|$)"),
            strong_key: re(r"(?i)^(?:artifact_(?:file|image|path|url)|files?_(?:created|modified|written)|generated_(?:file|image|path|url)|media_tag|output_(?:file|path|url)|result_(?:file|path|url)|saved_to|screenshot_path)$"),
            producer_key: re(r"(?i)^(?:artifact(?:s|_(?:file|image|path|url))?|attachment(?:s|_(?:file|image|path|url))?|download(?:s|_(?:file|path|url))?|(?:audio|image|video)(?:_(?:file|path|url))?|file_path|local_path|media(?:_(?:file|path|url))?|path)$"),
            screenshot: re(r"(?i)Screenshot path:\s*([^\r\n<>]+)"),
            wheel: re(r"\.whl(?:\.metadata)?$"),
            package: re(r"/packages/.+\.(?:tar\.gz|tar\.bz2|zip|egg)(?:\.metadata)?$"),
            // Rust regex has no lookahead. Candidate end boundaries are checked
            // explicitly below, preserving the original whitespace-path rule.
            media_anchor: re(r"^(?:~/|/|[A-Za-z]:[/\\])\S+?(?:[^\S\n]+\S+?)*?\.(?:geojson|tiff|jpeg|webp|docx|xlsx|pptx|html|flac|webm|json|yaml|opus|png|jpg|gif|bmp|svg|mp4|mov|avi|mkv|3gp|mp3|m2a|wav|ogg|m4a|pdf|doc|odt|rtf|txt|md|epub|xls|ods|csv|tsv|xml|yml|kmz|kml|gpx|ppt|odp|key|zip|tar|gz|tgz|bz2|xz|7z|rar|apk|ipa|htm)"),
            media_bare: re(r#"^[^\s`"]+"#),
        }
    })
}

pub fn is_file_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    value.to_ascii_lowercase().starts_with("file:")
        || value.starts_with('/')
        || value.starts_with("~/")
        || value.starts_with("~\\")
        || value.starts_with("./")
        || value.starts_with(".\\")
        || value.starts_with("../")
        || value.starts_with("..\\")
        || value.starts_with("\\\\")
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'/' | b'\\'))
}
fn web(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("http://")
}
fn decode_media(value: &str) -> String {
    value
        .strip_prefix("#media:")
        .and_then(|encoded| {
            // decodeURIComponent fails the whole href on malformed percent escapes.
            let bytes = encoded.as_bytes();
            if bytes.iter().enumerate().any(|(i, b)| {
                *b == b'%'
                    && (i + 2 >= bytes.len()
                        || !bytes[i + 1].is_ascii_hexdigit()
                        || !bytes[i + 2].is_ascii_hexdigit())
            }) {
                return None;
            }
            percent_decode_str(encoded)
                .decode_utf8()
                .ok()
                .map(|text| text.into_owned())
        })
        .unwrap_or_else(|| value.to_owned())
}
fn normalize(value: &str) -> String {
    let mut value = value.trim();
    for _ in 0..3 {
        if value.len() >= 2
            && matches!(value.as_bytes()[0], b'"' | b'\'' | b'`')
            && value.as_bytes().first() == value.as_bytes().last()
        {
            value = value[1..value.len() - 1].trim();
        } else {
            break;
        }
    }
    value
        .trim_end_matches(['`', '*'])
        .trim_end_matches([')', ',', '.', ';'])
        .to_owned()
}
fn package_download(value: &str) -> bool {
    let path = Url::parse(value)
        .ok()
        .filter(|url| matches!(url.scheme(), "http" | "https" | "file"))
        .map(|url| {
            percent_decode_str(url.path())
                .decode_utf8_lossy()
                .to_lowercase()
        })
        .unwrap_or_else(|| value.replace('\\', "/").to_lowercase());
    [
        "/.cache/pip/",
        "/library/caches/pip/",
        "/appdata/local/pip/cache/",
    ]
    .iter()
    .any(|needle| path.contains(needle))
        || patterns().wheel.is_match(&path)
        || (web(value) && patterns().package.is_match(&path))
}
fn looks_like(value: &str, explicit: bool) -> bool {
    if !explicit && package_download(value) {
        return false;
    }
    if web(value) || value.starts_with("data:image/") {
        return true;
    }
    is_file_path(value) && (explicit || patterns().file_ext.is_match(value))
}
fn kind(value: &str) -> Kind {
    if value.starts_with("data:image/") || patterns().image_ext.is_match(value) {
        Kind::Image
    } else if is_file_path(value) {
        Kind::File
    } else {
        Kind::Link
    }
}
fn label(value: &str) -> String {
    Url::parse(value)
        .ok()
        .and_then(|url| {
            url.path_segments()
                .and_then(|mut parts| parts.rfind(|part| !part.is_empty()).map(str::to_owned))
        })
        .or_else(|| {
            value
                .split(['/', '\\'])
                .rfind(|part| !part.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| value.to_owned())
}
pub fn normalize_timestamp(value: Option<f64>) -> Option<f64> {
    value
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| {
            if value < 10_000_000_000.0 {
                value * 1000.0
            } else {
                value
            }
        })
}
fn summary_time(session: &SessionSummary) -> f64 {
    chrono::DateTime::parse_from_rfc3339(&session.updated_at)
        .or_else(|_| chrono::DateTime::parse_from_rfc3339(&session.created_at))
        .map(|time| time.timestamp_millis() as f64)
        .unwrap_or(0.0)
}

fn media_values(text: &str, push: &mut impl FnMut(String, bool)) {
    let mut remainder = text;
    while let Some(at) = remainder.find("MEDIA:") {
        let body = remainder[at + 6..].trim_start();
        let mut captured = None;
        if let Some(quote) = body.chars().next().filter(|c| ['`', '"', '\''].contains(c))
            && let Some(end) = body[1..].find(quote)
        {
            captured = Some((body[1..end + 1].to_owned(), end + 2));
        }
        if captured.is_none() {
            for matched in patterns().media_anchor.find_iter(body) {
                let tail = &body[matched.end()..];
                if tail.is_empty()
                    || tail.starts_with("MEDIA:")
                    || tail
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_whitespace() || "`\"'*_,;:)]}".contains(c))
                {
                    captured = Some((matched.as_str().to_owned(), matched.end()));
                    break;
                }
            }
        }
        if captured.is_none() {
            captured = patterns().media_bare.find(body).map(|m| {
                (
                    m.as_str()
                        .trim_end_matches(['`', '"', '\'', '*', '_'])
                        .to_owned(),
                    m.end(),
                )
            });
        }
        let Some((mut value, consumed)) = captured else {
            remainder = &remainder[at + 6..];
            continue;
        };
        while value.ends_with(['.', ',', ';', ':', '!', '?']) {
            let next = &value[..value.len() - 1];
            if !plausible(next) {
                break;
            }
            value.pop();
        }
        if plausible(&value) {
            push(value, true);
        }
        remainder = &body[consumed.min(body.len())..];
    }
}
fn plausible(value: &str) -> bool {
    value.contains(['/', '\\'])
        || value
            .char_indices()
            .any(|(i, c)| c == '.' && value[i + 1..].chars().next().is_some_and(|c| c != '.'))
}

fn scan_text(text: &str, push: &mut impl FnMut(String, bool)) {
    let p = patterns();
    media_values(text, push);
    for capture in p.image.captures_iter(text) {
        push(capture[2].to_owned(), false);
    }
    for capture in p.link.captures_iter(text) {
        let start = capture.get(0).unwrap().start();
        if start > 0 && text.as_bytes()[start - 1] == b'!' {
            continue;
        }
        let value = decode_media(&capture[2]);
        if looks_like(&value, false) {
            push(value, false);
        }
    }
    for matched in p.url.find_iter(text) {
        if looks_like(matched.as_str(), false) {
            push(matched.as_str().to_owned(), false);
        }
    }
    for re in [&p.path, &p.windows] {
        for capture in re.captures_iter(text) {
            push(capture[2].to_owned(), false);
        }
    }
}
fn strings(value: &Value, keys: &mut Vec<String>, f: &mut impl FnMut(&str, &[String])) {
    match value {
        Value::String(value) => f(value, keys),
        Value::Array(values) => {
            for value in values {
                strings(value, keys, f);
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                keys.push(key.clone());
                strings(value, keys, f);
                keys.pop();
            }
        }
        _ => {}
    }
}
fn untrusted_payload(text: &str) -> Option<&str> {
    let text = text.trim();
    if !text.starts_with("<untrusted_tool_result") {
        return None;
    }
    let begin = text.find('>')? + 1;
    let end = text.rfind("</untrusted_tool_result>")?;
    let body = text.get(begin..end)?.trim();
    Some(
        body.split_once("\n\n")
            .map_or(body, |(_, payload)| payload)
            .trim(),
    )
}
fn scan_message(message: &Message, push: &mut impl FnMut(String, bool)) {
    let text = message
        .content
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .or(message
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty()))
        .or(message
            .context
            .as_deref()
            .filter(|text| !text.trim().is_empty()))
        .unwrap_or("");
    if message.role == "assistant" {
        scan_text(text, push);
        return;
    }
    if message.role != "tool" {
        return;
    }
    let name = message.tool_name.trim().to_lowercase();
    let producer = patterns().producer.is_match(&name) || name.starts_with("bfl_flux3_");
    let terminal = name == "terminal";
    if producer || terminal {
        scan_text(text, push);
    }
    if name == "browser_vision" {
        for capture in patterns().screenshot.captures_iter(text) {
            push(capture[1].to_owned(), false);
        }
    }
    let mut payloads = vec![];
    for candidate in [Some(text), untrusted_payload(text)].into_iter().flatten() {
        if let Ok(value) = serde_json::from_str::<Value>(candidate) {
            payloads.push(value);
        }
    }
    if message.content.is_object() || message.content.is_array() {
        let structured = if message.content.get("_multimodal") == Some(&Value::Bool(true)) {
            message.content.get("meta").cloned().unwrap_or(Value::Null)
        } else {
            message.content.clone()
        };
        payloads.push(structured);
    }
    for payload in payloads {
        strings(&payload, &mut vec![], &mut |value, keys| {
            let shell = terminal
                && keys
                    .iter()
                    .any(|key| matches!(key.to_lowercase().as_str(), "output" | "stdout" | "path"));
            if shell {
                scan_text(value, push);
                return;
            }
            if !keys.iter().any(|key| {
                patterns().strong_key.is_match(key)
                    || (producer && patterns().producer_key.is_match(key))
            }) {
                return;
            }
            media_values(value, push);
            let normalized = normalize(&decode_media(value));
            if looks_like(&normalized, false) {
                push(normalized, false);
            }
        });
    }
}

pub fn collect(
    session: &SessionSummary,
    messages: impl IntoIterator<Item = Message>,
) -> Vec<Artifact> {
    let mut found = vec![];
    let mut seen = HashSet::new();
    for message in messages {
        let timestamp =
            normalize_timestamp(message.timestamp).unwrap_or_else(|| summary_time(session));
        scan_message(&message, &mut |candidate, explicit| {
            let value = normalize(&decode_media(&candidate));
            if !looks_like(&value, explicit) || !seen.insert(value.clone()) {
                return;
            }
            found.push(Artifact {
                id: format!("{}:{value}", session.id),
                kind: kind(&value),
                label: label(&value),
                value,
                session_id: session.id.clone(),
                session_title: if session.title.trim().is_empty() {
                    "Untitled session".to_owned()
                } else {
                    session.title.clone()
                },
                timestamp,
                base_dir: message.base_dir.clone(),
            });
        });
    }
    found
}

fn native_messages(history: &History, base_dir: &Path) -> Vec<Message> {
    history
        .turns
        .iter()
        .flat_map(|turn| {
            turn.entries.iter().map(|entry| {
                let (role, content, tool_name) = match entry {
                    Entry::Assistant { text, .. } => {
                        ("assistant", Value::String(text.clone()), String::new())
                    }
                    Entry::Tool { name, result, .. } => {
                        // The native engine reports workspace-relative write paths.
                        // Anchor that actual output so the upstream artifact classifier
                        // sees the same file reference, without indexing tool inputs.
                        let mut result = result.clone();
                        if name == "write_file"
                            && let Ok(mut value) = serde_json::from_str::<Value>(&result)
                            && value.get("verified") == Some(&json!(true))
                            && value.get("bytes_written").is_some()
                            && let Some(path) = value.get("path").and_then(Value::as_str)
                            && !is_file_path(path)
                        {
                            value["path"] = Value::String(format!("./{path}"));
                            result = value.to_string();
                        }
                        ("tool", Value::String(result), name.clone())
                    }
                };
                Message {
                    role: role.to_owned(),
                    content,
                    text: None,
                    context: None,
                    tool_name,
                    timestamp: None,
                    base_dir: base_dir.to_owned(),
                }
            })
        })
        .collect()
}
fn archive_messages(archive: &HermesArchive, fallback: &Path) -> Vec<Message> {
    let cwd: HashMap<_, _> = archive
        .sessions
        .iter()
        .map(|session| {
            (
                session.id.as_str(),
                session.cwd.as_deref().unwrap_or(fallback),
            )
        })
        .collect();
    archive
        .messages
        .iter()
        .filter(|message| {
            (message.active || message.compacted)
                && !message
                    .display_metadata
                    .get("model_only")
                    .is_some_and(|value| value == true || value == 1)
        })
        .map(|message| Message {
            role: message.role.clone(),
            content: message.content.clone(),
            text: message
                .raw
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_owned),
            context: message
                .raw
                .get("context")
                .and_then(Value::as_str)
                .map(str::to_owned),
            tool_name: message.tool_name.clone().unwrap_or_default(),
            timestamp: Some(message.timestamp),
            base_dir: cwd
                .get(message.session_id.as_str())
                .copied()
                .unwrap_or(fallback)
                .to_owned(),
        })
        .collect()
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err("Session transcript is not a regular file".to_owned());
    }
    if file.metadata().map_err(|error| error.to_string())?.len() > SAFE_LOAD_BYTES {
        return Err("Session exceeds the Desktop safe-load limit".to_owned());
    }
    serde_json::from_reader(file.take(SAFE_LOAD_BYTES + 1)).map_err(|error| error.to_string())
}
pub fn load(library: &Library, live: Option<(SessionSummary, History)>) -> Result<Index, String> {
    let sessions = library
        .sessions(None)
        .map_err(|error| format!("{error:#}"))?;
    let mut index = Index {
        sessions: sessions.len().min(RECENT_SESSIONS),
        ..Default::default()
    };
    for summary in sessions.into_iter().take(RECENT_SESSIONS) {
        let result = (|| -> Result<Vec<Artifact>, String> {
            let directory = library
                .session_dir(&summary.id)
                .map_err(|error| format!("{error:#}"))?;
            let base = library
                .workspace_for_session(&summary.id)
                .map_err(|error| format!("{error:#}"))?
                .unwrap_or_else(|| directory.join("workspace"));
            let history = if let Some((_, history)) = live
                .as_ref()
                .filter(|(session, _)| session.id == summary.id)
            {
                history.clone()
            } else {
                read_json(&directory.join("conversation.json"))?
            };
            let mut messages = if directory.join("hermes-source.json").is_file() {
                archive_messages(&read_json(&directory.join("hermes-source.json"))?, &base)
            } else {
                vec![]
            };
            messages.extend(native_messages(&history, &base));
            let artifacts = collect(&summary, messages);
            if artifacts.len() > MAX_ARTIFACTS
                || artifacts
                    .iter()
                    .map(|artifact| artifact.value.len())
                    .sum::<usize>()
                    > 16 * 1024 * 1024
            {
                return Err("Session artifact index exceeds the Desktop safe-load limit".to_owned());
            }
            Ok(artifacts)
        })();
        match result {
            Ok(artifacts) if index.artifacts.len() + artifacts.len() <= MAX_ARTIFACTS => {
                index.artifacts.extend(artifacts)
            }
            Ok(_) => index.failures.push((
                summary.title,
                "Artifact count exceeds the Desktop safe-load limit".to_owned(),
            )),
            Err(error) => index.failures.push((summary.title, error)),
        }
    }
    index.artifacts.sort_by(|a, b| {
        b.timestamp
            .total_cmp(&a.timestamp)
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(index)
}

pub fn local_path(artifact: &Artifact) -> Result<PathBuf, String> {
    if web(&artifact.value) || artifact.value.starts_with("data:") {
        return Err("This artifact is not a local file".to_owned());
    }
    let value = &artifact.value;
    let path = if value.to_ascii_lowercase().starts_with("file:") {
        Url::parse(value)
            .map_err(|error| error.to_string())?
            .to_file_path()
            .map_err(|_| "This file URL is not a local path on this system".to_owned())?
    } else if value.starts_with("~/") || value.starts_with("~\\") {
        PathBuf::from(std::env::var_os("HOME").ok_or("Could not find your home directory")?)
            .join(&value[2..])
    } else {
        #[cfg(not(windows))]
        if (value.as_bytes().get(1) == Some(&b':')
            && value
                .as_bytes()
                .get(2)
                .is_some_and(|byte| matches!(byte, b'/' | b'\\')))
            || value.starts_with("\\\\")
        {
            return Err("This artifact belongs to a Windows filesystem".to_owned());
        }
        let value = Path::new(value);
        if value.is_absolute() {
            value.to_owned()
        } else {
            artifact.base_dir.join(value)
        }
    };
    Ok(path)
}

#[cfg(test)]
mod tests;
