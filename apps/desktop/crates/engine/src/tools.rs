//! Native, bounded workspace tools. All filesystem operations after construction
//! use a directory capability; model-provided paths never become ambient paths.
//!
//! The engine owns approval. It validates a call, shows its arguments and this
//! module's approval summary, and invokes only after the user grants that call.
//! Writes additionally require `Context::with_write_approval(true)`. A grant is
//! deliberately not stored in the shared cache and does not grant authority to
//! the original context.

use anyhow::{Context as _, Result, bail, ensure};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    ffi::OsStr,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const MAX_PATH_BYTES: usize = 4096;
const MAX_READ_BYTES: usize = 1_048_576;
const MAX_WRITE_BYTES: usize = 65_536;
const MAX_OUTPUT_BYTES: usize = 98_304;
const MAX_LINE_BYTES: usize = 16_384;
const MAX_SCAN_BYTES: usize = 8_388_608;
const MAX_ENTRIES: usize = 8192;
const MAX_DEPTH: usize = 16;
const MAX_RESULTS: usize = 200;
const MAX_SKILLS: usize = 128;
const MAX_SNAPSHOTS: usize = 16;
const MAX_CACHE_BYTES: usize = 8_388_608;

/// Workspace authority and the text versions actually observed by this session.
/// Clones share read observations, but approval is a value on each clone.
#[derive(Clone, Debug)]
pub struct Context {
    root: PathBuf,
    input_root: PathBuf,
    dir: Arc<Dir>,
    observations: Arc<Mutex<VecDeque<Observation>>>,
    write_approved: bool,
    checkpoints: Option<crate::checkpoint::Store>,
    redact_secrets:bool,
    read_chars:usize,
    output_bytes:usize,
    output_lines:usize,
    line_chars:usize,
}

impl Context {
    pub fn new(workspace: impl AsRef<Path>) -> Result<Self> {
        let input = workspace.as_ref();
        let input_root = if input.is_absolute() {
            input.to_owned()
        } else {
            std::env::current_dir()?.join(input)
        };
        let root = input_root
            .canonicalize()
            .context("Workspace does not exist")?;
        ensure!(root.is_dir(), "Workspace must be a directory");
        let dir = Dir::open_ambient_dir(&root, ambient_authority())
            .context("Could not open workspace")?;
        Ok(Self {
            root,
            input_root,
            dir: Arc::new(dir),
            observations: Arc::new(Mutex::new(VecDeque::new())),
            write_approved: false,
            checkpoints: None, redact_secrets:false, read_chars:100_000,output_bytes:MAX_OUTPUT_BYTES, output_lines:2000, line_chars:MAX_LINE_BYTES,
        })
    }

    pub(crate) fn configure(&mut self,settings:&crate::settings::ProfileSettings){
        self.redact_secrets=settings.redact_secrets;
        self.read_chars=settings.file_read_max_chars;
        self.output_bytes=settings.tool_output_bytes.min(MAX_OUTPUT_BYTES);
        self.output_lines=settings.tool_output_lines;
        self.line_chars=settings.tool_output_line_length;
    }
    pub fn with_checkpoints(&self, profile_root: &Path, enabled: bool, max_snapshots: usize) -> Result<Self> {
        let mut next = self.clone();
        next.checkpoints = if enabled { Some(crate::checkpoint::Store::open(profile_root, max_snapshots)?) } else { None };
        Ok(next)
    }
    pub fn workspace_root(&self) -> &Path {
        &self.root
    }

    /// Use on a short-lived clone for the specific call approved by the engine.
    /// Passing false also revokes an approval on an already approved clone.
    pub fn with_write_approval(&self, approved: bool) -> Self {
        let mut context = self.clone();
        context.write_approved = approved;
        context
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Approval {
    pub summary: String,
    pub mutates_workspace: bool,
}

/// OpenAI-compatible schemas. Names and the common file/skill arguments follow
/// Hermes tools; this desktop subset explicitly documents its bounded behavior.
pub fn tool_specs() -> Vec<Value> {
    vec![
        spec(
            "get_current_time",
            "Read the system clock and return the current UTC time as an ISO 8601 timestamp.",
            json!({}),
            &[],
        ),
        spec(
            "list_directory",
            "List one workspace directory without following symbolic links. Returns up to 200 sorted entries and a truncation flag. Paths are relative to the selected workspace.",
            json!({
                "path":{"type":"string","description":"Workspace directory, default '.'","default":"."},
                "limit":{"type":"integer","minimum":1,"maximum":200,"default":100}
            }),
            &[],
        ),
        spec(
            "read_file",
            "Read a UTF-8 text file inside the workspace with line numbers and pagination. Files are capped at 1 MiB and each response is bounded. Read every page before replacing an existing file; changed files must be read again.",
            json!({
                "path":{"type":"string","description":"Workspace-relative path, or an absolute path inside the workspace"},
                "offset":{"type":"integer","minimum":1,"default":1,"description":"First line, 1-indexed"},
                "limit":{"type":"integer","minimum":1,"maximum":2000,"default":2000,"description":"Maximum lines; next_offset continues a truncated response"}
            }),
            &["path"],
        ),
        spec(
            "search_files",
            "Search workspace UTF-8 file contents for a literal string, or find filenames using '*' and '?' wildcards. This native subset uses literal content matching, not regular expressions. Traversal, scanned bytes and results are bounded; generated/dependency directories and symlinks are skipped.",
            json!({
                "pattern":{"type":"string","minLength":1,"maxLength":512},
                "target":{"type":"string","enum":["content","files"],"default":"content"},
                "path":{"type":"string","default":".","description":"Workspace directory or file"},
                "file_glob":{"type":"string","description":"Optional '*'/'?' filename filter"},
                "limit":{"type":"integer","minimum":1,"maximum":200,"default":50},
                "offset":{"type":"integer","minimum":0,"maximum":10000,"default":0},
                "case_sensitive":{"type":"boolean","default":true}
            }),
            &["pattern"],
        ),
        spec(
            "write_file",
            "After explicit approval, create or atomically replace a workspace UTF-8 file with its complete content (maximum 64 KiB). Creates missing parent directories. An existing file must first be fully read in this session and must still match that read; otherwise it is left untouched. Symlinks are refused.",
            json!({
                "path":{"type":"string"},
                "content":{"type":"string","maxLength":65536,"description":"Complete replacement UTF-8 text, not a patch"}
            }),
            &["path", "content"],
        ),
        spec(
            "skills_list",
            "Discover workspace SKILL.md files and return their names, descriptions, paths and unique IDs. Searches the workspace, including skills/, .agents/skills/, .codex/skills/ and .hermes/skills/, with bounded traversal. Use skill_view to read instructions.",
            json!({
                "category":{"type":"string","description":"Optional category or directory-segment filter"}
            }),
            &[],
        ),
        spec(
            "skill_view",
            "Read a discovered skill's SKILL.md, or a text reference within that skill directory. Use a unique ID from skills_list when names are ambiguous. This tool reads files; it does not execute scripts.",
            json!({
                "name":{"type":"string","description":"Skill name or unique ID from skills_list"},
                "file_path":{"type":"string","description":"Optional relative file inside the skill directory, e.g. references/api.md"},
                "offset":{"type":"integer","minimum":1,"default":1},
                "limit":{"type":"integer","minimum":1,"maximum":2000,"default":2000}
            }),
            &["name"],
        ),
    ]
}

fn spec(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"type":"function","function":{
        "name":name,"description":description,
        "parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":false}
    }})
}

/// Validate structure, bounds and workspace paths before asking for approval.
/// This reads metadata only; it never executes a tool or reads file content.
pub fn validate(name: &str, args: &Value, context: &Context) -> Result<Approval> {
    let operation = parse(name, args, context)?;
    let (summary, mutates_workspace) = match &operation {
        Operation::Clock => ("Read the current system clock in UTC.".to_owned(), false),
        Operation::List { path, limit } => {
            check_path(context, path, false)?;
            ensure!(
                context.dir.metadata(path)?.is_dir(),
                "Path is not a directory"
            );
            (
                format!(
                    "List up to {limit} entries in workspace directory {}.",
                    display(path)
                ),
                false,
            )
        }
        Operation::Read {
            path,
            offset,
            limit,
        } => {
            check_regular_file(context, path)?;
            (
                format!(
                    "Read up to {limit} lines from workspace file {}, starting at line {offset}.",
                    display(path)
                ),
                false,
            )
        }
        Operation::Search { args, path } => {
            check_path(context, path, false)?;
            (
                format!(
                    "Search workspace {} for {:?}; return up to {} matches.",
                    display(path),
                    args.pattern,
                    args.limit
                ),
                false,
            )
        }
        Operation::Write { path, content } => {
            check_path(context, path, true)?;
            if let Ok(metadata) = context.dir.symlink_metadata(path) {
                ensure!(metadata.is_file(), "Destination is not a regular file");
            }
            (
                format!(
                    "Write {} UTF-8 bytes to workspace file {}. Existing content will be replaced only if it still matches a complete session read.",
                    content.len(),
                    display(path)
                ),
                true,
            )
        }
        Operation::Skills { category } => (
            match category {
                Some(category) => {
                    format!("Discover workspace SKILL.md files in category {category:?}.")
                }
                None => "Discover workspace SKILL.md files and their descriptions.".to_owned(),
            },
            false,
        ),
        Operation::Skill {
            name, file_path, ..
        } => (
            format!(
                "Read skill {name:?}{} from the workspace.",
                file_path
                    .as_ref()
                    .map(|p| format!(", reference {}", display(p)))
                    .unwrap_or_default()
            ),
            false,
        ),
    };
    Ok(Approval {
        summary,
        mutates_workspace,
    })
}

/// Invoke a structurally valid tool after the engine obtains call-specific user
/// approval. Errors are returned for the engine to serialize as {"error": ...}.
/// Successful values never contain an `error` property.
pub fn invoke(name: &str, args: &Value, context: &Context) -> Result<Value> {
    // Repeat validation after the approval wait: filesystem paths may change.
    validate(name, args, context)?;
    match parse(name, args, context)? {
        Operation::Clock => {
            Ok(json!({"utc":chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)}))
        }
        Operation::List { path, limit } => list_directory(context, &path, limit),
        Operation::Read {
            path,
            offset,
            limit,
        } => read_file(context, &path, offset, limit),
        Operation::Search { args, path } => search_files(context, &path, &args),
        Operation::Write { path, content } => {
            ensure!(
                context.write_approved,
                "write_file requires explicit approval for this call"
            );
            write_file(context, &path, &content)
        }
        Operation::Skills { category } => {
            let (skills, walk) = discover_skills(context)?;
            let skills: Vec<_> = skills
                .into_iter()
                .filter(|s| {
                    category.as_ref().is_none_or(|category| {
                        s.category.eq_ignore_ascii_case(category)
                            || s.path.components().any(|part| {
                                part.as_os_str()
                                    .to_string_lossy()
                                    .eq_ignore_ascii_case(category)
                            })
                    })
                })
                .map(|s| s.value())
                .collect();
            Ok(
                json!({"skills":skills,"truncated":walk.truncated,"scanned_entries":walk.scanned,"skipped_entries":walk.skipped}),
            )
        }
        Operation::Skill {
            name,
            file_path,
            offset,
            limit,
        } => skill_view(context, &name, file_path.as_deref(), offset, limit),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyArgs {}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListArgs {
    #[serde(default = "dot")]
    path: String,
    #[serde(default = "hundred")]
    limit: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    path: String,
    #[serde(default = "one")]
    offset: usize,
    #[serde(default = "two_thousand")]
    limit: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    pattern: String,
    #[serde(default = "content_target")]
    target: String,
    #[serde(default = "dot")]
    path: String,
    file_glob: Option<String>,
    #[serde(default = "fifty")]
    limit: usize,
    #[serde(default)]
    offset: usize,
    #[serde(default = "yes")]
    case_sensitive: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArgs {
    path: String,
    content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillsArgs {
    category: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillArgs {
    name: String,
    file_path: Option<String>,
    #[serde(default = "one")]
    offset: usize,
    #[serde(default = "two_thousand")]
    limit: usize,
}

fn dot() -> String {
    ".".into()
}
fn content_target() -> String {
    "content".into()
}
fn one() -> usize {
    1
}
fn fifty() -> usize {
    50
}
fn hundred() -> usize {
    100
}
fn two_thousand() -> usize {
    2000
}
fn yes() -> bool {
    true
}

enum Operation {
    Clock,
    List {
        path: PathBuf,
        limit: usize,
    },
    Read {
        path: PathBuf,
        offset: usize,
        limit: usize,
    },
    Search {
        path: PathBuf,
        args: SearchArgs,
    },
    Write {
        path: PathBuf,
        content: String,
    },
    Skills {
        category: Option<String>,
    },
    Skill {
        name: String,
        file_path: Option<PathBuf>,
        offset: usize,
        limit: usize,
    },
}

fn decode<T: DeserializeOwned>(args: &Value) -> Result<T> {
    ensure!(args.is_object(), "Tool arguments must be a JSON object");
    serde_json::from_value(args.clone()).context("Invalid tool arguments")
}

fn parse(name: &str, args: &Value, context: &Context) -> Result<Operation> {
    match name {
        "get_current_time" => {
            let _: EmptyArgs = decode(args)?;
            Ok(Operation::Clock)
        }
        "list_directory" | "list_files" => {
            let args: ListArgs = decode(args)?;
            ensure!(
                (1..=MAX_RESULTS).contains(&args.limit),
                "limit must be between 1 and {MAX_RESULTS}"
            );
            Ok(Operation::List {
                path: relative_path(context, &args.path)?,
                limit: args.limit,
            })
        }
        "read_file" => {
            let args: ReadArgs = decode(args)?;
            check_pagination(args.offset, args.limit)?;
            Ok(Operation::Read {
                path: relative_path(context, &args.path)?,
                offset: args.offset,
                limit: args.limit,
            })
        }
        "search_files" => {
            let args: SearchArgs = decode(args)?;
            ensure!(
                !args.pattern.is_empty()
                    && args.pattern.len() <= 512
                    && !args.pattern.contains(['\n', '\r', '\0']),
                "pattern must be a nonempty single-line string of at most 512 bytes"
            );
            ensure!(
                matches!(args.target.as_str(), "content" | "files"),
                "target must be content or files"
            );
            ensure!(
                (1..=MAX_RESULTS).contains(&args.limit),
                "limit must be between 1 and {MAX_RESULTS}"
            );
            ensure!(args.offset <= 10_000, "offset must be at most 10000");
            if let Some(glob) = &args.file_glob {
                ensure!(
                    !glob.is_empty() && glob.len() <= 512 && !glob.contains(['\n', '\r', '\0']),
                    "file_glob must contain 1 to 512 bytes"
                );
            }
            let path = relative_path(context, &args.path)?;
            Ok(Operation::Search { path, args })
        }
        "write_file" => {
            let args: WriteArgs = decode(args)?;
            ensure!(
                args.content.len() <= MAX_WRITE_BYTES,
                "content exceeds the 64 KiB write limit"
            );
            ensure!(
                !args.content.contains('\0'),
                "Only UTF-8 text without NUL bytes can be written"
            );
            let path = relative_path(context, &args.path)?;
            ensure!(path != Path::new("."), "Destination must name a file");
            Ok(Operation::Write {
                path,
                content: args.content,
            })
        }
        "skills_list" | "list_skills" => {
            let args: SkillsArgs = decode(args)?;
            if let Some(category) = &args.category {
                ensure!(
                    !category.is_empty()
                        && category.len() <= 128
                        && !category.chars().any(char::is_control),
                    "category must contain 1 to 128 printable bytes"
                );
            }
            Ok(Operation::Skills {
                category: args.category,
            })
        }
        "skill_view" | "read_skill" => {
            let args: SkillArgs = decode(args)?;
            ensure!(
                !args.name.trim().is_empty()
                    && args.name.len() <= MAX_PATH_BYTES
                    && !args.name.chars().any(char::is_control),
                "name must be a nonempty skill name or ID"
            );
            // IDs are paths but declared skill names can contain spaces/colons.
            ensure!(
                !Path::new(&args.name).is_absolute()
                    && !Path::new(&args.name)
                        .components()
                        .any(|c| matches!(c, Component::ParentDir)),
                "Skill names must refer to the workspace"
            );
            check_pagination(args.offset, args.limit)?;
            let file_path = args.file_path.map(|path| local_path(&path)).transpose()?;
            Ok(Operation::Skill {
                name: args.name,
                file_path,
                offset: args.offset,
                limit: args.limit,
            })
        }
        _ => bail!("Unknown tool: {name}"),
    }
}

fn check_pagination(offset: usize, limit: usize) -> Result<()> {
    ensure!(
        (1..=MAX_READ_BYTES + 1).contains(&offset),
        "offset must be a 1-indexed line number no greater than {}",
        MAX_READ_BYTES + 1
    );
    ensure!(
        (1..=2000).contains(&limit),
        "limit must be between 1 and 2000"
    );
    Ok(())
}

fn relative_path(context: &Context, raw: &str) -> Result<PathBuf> {
    ensure!(
        !raw.is_empty() && raw.len() <= MAX_PATH_BYTES && !raw.chars().any(char::is_control),
        "path must contain 1 to {MAX_PATH_BYTES} printable bytes"
    );
    let input = Path::new(raw);
    let relative = if input.is_absolute() {
        input
            .strip_prefix(&context.root)
            .or_else(|_| input.strip_prefix(&context.input_root))
            .context("Path is outside the selected workspace")?
    } else {
        input
    };
    local_path(&relative.to_string_lossy())
}

fn local_path(raw: &str) -> Result<PathBuf> {
    ensure!(
        !raw.is_empty() && raw.len() <= MAX_PATH_BYTES && !raw.chars().any(char::is_control),
        "Invalid relative path"
    );
    let mut path = PathBuf::new();
    for component in Path::new(raw).components() {
        match component {
            Component::Normal(part) => path.push(part),
            Component::CurDir => {}
            _ => bail!("Parent traversal and absolute paths are not allowed here"),
        }
    }
    if path.as_os_str().is_empty() {
        path.push(".");
    }
    Ok(path)
}

fn display(path: &Path) -> String {
    #[cfg(windows)]
    {
        path.to_string_lossy().replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        path.to_string_lossy().into_owned()
    }
}

fn check_path(context: &Context, path: &Path, allow_missing: bool) -> Result<()> {
    let mut prefix = PathBuf::new();
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        if component == Component::CurDir {
            continue;
        }
        ensure!(
            matches!(component, Component::Normal(_)),
            "Invalid workspace path"
        );
        prefix.push(component);
        match context.dir.symlink_metadata(&prefix) {
            Ok(metadata) => {
                ensure!(
                    !metadata.file_type().is_symlink(),
                    "Symbolic links are not followed: {}",
                    display(&prefix)
                );
                if components.peek().is_some() {
                    ensure!(
                        metadata.is_dir(),
                        "Parent is not a directory: {}",
                        display(&prefix)
                    );
                }
            }
            Err(error) if allow_missing && error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("Cannot access workspace path {}", display(&prefix)));
            }
        }
    }
    Ok(())
}

fn check_regular_file(context: &Context, path: &Path) -> Result<()> {
    check_path(context, path, false)?;
    let metadata = context.dir.symlink_metadata(path)?;
    ensure!(
        metadata.is_file(),
        "Path is not a regular file: {}",
        display(path)
    );
    ensure!(
        metadata.len() <= MAX_READ_BYTES as u64,
        "File exceeds the 1 MiB read limit"
    );
    Ok(())
}

fn read_text(context: &Context, path: &Path) -> Result<String> {
    check_regular_file(context, path)?;
    let file = context
        .dir
        .open(path)
        .with_context(|| format!("Could not read {}", display(path)))?;
    ensure!(
        file.metadata()?.is_file(),
        "Path changed; expected a regular file"
    );
    let mut bytes = Vec::new();
    file.take((MAX_READ_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_READ_BYTES,
        "File exceeds the 1 MiB read limit"
    );
    ensure!(
        !bytes.contains(&0),
        "Binary files are not supported; file contains a NUL byte"
    );
    String::from_utf8(bytes).context("File is not UTF-8 text")
}

#[derive(Debug)]
struct Observation {
    path: PathBuf,
    bytes: Vec<u8>,
    covered: Vec<(usize, usize)>,
}

fn observe(context: &Context, path: &Path, bytes: &[u8], start: usize, end: usize) -> Result<()> {
    let mut cache = context
        .observations
        .lock()
        .map_err(|_| anyhow::anyhow!("Read observation cache is unavailable"))?;
    let mut observation = cache
        .iter()
        .position(|item| item.path == path)
        .and_then(|index| cache.remove(index))
        .filter(|item| item.bytes == bytes)
        .unwrap_or_else(|| Observation {
            path: path.to_owned(),
            bytes: bytes.to_vec(),
            covered: Vec::new(),
        });
    observation.covered.push((start, end));
    observation.covered.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in observation.covered {
        if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    observation.covered = merged;
    cache.push_back(observation);
    while cache.len() > MAX_SNAPSHOTS
        || cache.iter().map(|item| item.bytes.len()).sum::<usize>() > MAX_CACHE_BYTES
    {
        cache.pop_front();
    }
    Ok(())
}

fn observed_version(context: &Context, path: &Path) -> Result<Vec<u8>> {
    let cache = context
        .observations
        .lock()
        .map_err(|_| anyhow::anyhow!("Read observation cache is unavailable"))?;
    let observation = cache.iter().find(|item| item.path == path)
        .context("Existing file has not been read in this session. Use read_file, merge your changes, then retry")?;
    ensure!(
        observation
            .covered
            .iter()
            .any(|&(start, end)| start == 0 && end >= observation.bytes.len()),
        "Existing file has only been partially read. Read every page before replacing it"
    );
    Ok(observation.bytes.clone())
}

fn read_file(context: &Context, path: &Path, offset: usize, limit: usize) -> Result<Value> {
    let text = read_text(context, path)?;
    let total_lines = text.split_inclusive('\n').count();
    let mut output = String::new();
    let mut cursor = 0;
    let mut start = text.len();
    let mut end = text.len();
    let mut returned = 0;
    let mut characters=0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if index + 1 < offset {
            cursor += line.len();
            continue;
        }
        if returned == 0 {
            start = cursor;
            end = cursor;
        }
        if returned >= limit.min(context.output_lines) {
            break;
        }
        let plain = line.strip_suffix('\n').unwrap_or(line);
        let plain = plain.strip_suffix('\r').unwrap_or(plain);
        let numbered = format!("{}|{}\n", index + 1, plain);
        let count=numbered.chars().count();
        ensure!(count<=context.line_chars,"Line {} exceeds the configured tool output line length",index+1);
        if output.len() + numbered.len() > context.output_bytes || characters+count>context.read_chars {
            ensure!(
                returned > 0,
                "One line exceeds the bounded response size; split this file into shorter lines"
            );
            break;
        }
        output.push_str(&numbered);
        characters+=count;
        cursor += line.len();
        end = cursor;
        returned += 1;
    }
    // The empty file is fully observed, but a request beyond a nonempty EOF is not.
    if text.is_empty() {
        start = 0;
        end = 0;
    }
    if returned > 0 || text.is_empty() {
        observe(context, path, text.as_bytes(), start, end)?;
    }
    let next_offset = if offset.saturating_sub(1) + returned < total_lines {
        Some(offset + returned)
    } else {
        None
    };
    Ok(json!({
        "path":display(path),"content":output,"offset":offset,"lines_returned":returned,
        "total_lines":total_lines,"next_offset":next_offset,"truncated":next_offset.is_some(),"bytes":text.len()
    }))
}

struct DirectoryEntries {
    entries: Vec<(String, &'static str, u64)>,
    skipped: usize,
    truncated: bool,
}

fn entries(context: &Context, path: &Path, budget: usize) -> Result<DirectoryEntries> {
    check_path(context, path, false)?;
    let directory = context.dir.open_dir(path)?;
    let mut result = DirectoryEntries {
        entries: Vec::new(),
        skipped: 0,
        truncated: false,
    };
    for (index, entry) in directory.entries()?.enumerate() {
        if index >= budget {
            result.truncated = true;
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                result.skipped += 1;
                continue;
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str().filter(|s| !s.chars().any(char::is_control)) else {
            result.skipped += 1;
            continue;
        };
        let metadata = match directory.symlink_metadata(name) {
            Ok(metadata) => metadata,
            Err(_) => {
                result.skipped += 1;
                continue;
            }
        };
        let kind = if metadata.file_type().is_symlink() {
            "symlink"
        } else if metadata.is_dir() {
            "directory"
        } else if metadata.is_file() {
            "file"
        } else {
            "other"
        };
        result.entries.push((name.to_owned(), kind, metadata.len()));
    }
    result.entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

fn list_directory(context: &Context, path: &Path, limit: usize) -> Result<Value> {
    let result = entries(context, path, MAX_ENTRIES)?;
    let mut truncated = result.truncated || result.entries.len() > limit;
    let mut entries = Vec::new();
    let mut bytes = 0;
    for (name, kind, size) in result.entries.into_iter().take(limit) {
        let entry = json!({"path":display(&path.join(&name)),"name":name,"kind":kind,"bytes":if kind == "file" { Some(size) } else { None }});
        let entry_bytes = serde_json::to_vec(&entry)?.len();
        if bytes + entry_bytes > MAX_OUTPUT_BYTES {
            truncated = true;
            break;
        }
        bytes += entry_bytes;
        entries.push(entry);
    }
    Ok(
        json!({"path":display(path),"entries":entries,"truncated":truncated,"skipped_entries":result.skipped}),
    )
}

#[derive(Default)]
struct Walk {
    files: Vec<PathBuf>,
    scanned: usize,
    skipped: usize,
    truncated: bool,
}

fn skipped_directory(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".hg"
            | ".svn"
            | "node_modules"
            | "target"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".cache"
            | ".next"
            | "dist"
            | "build"
    )
}

fn walk(context: &Context, start: &Path) -> Result<Walk> {
    check_path(context, start, false)?;
    let metadata = context.dir.symlink_metadata(start)?;
    if metadata.is_file() {
        return Ok(Walk {
            files: vec![start.to_owned()],
            scanned: 1,
            ..Walk::default()
        });
    }
    ensure!(
        metadata.is_dir(),
        "Search path is not a regular file or directory"
    );
    let mut result = Walk::default();
    let mut pending = vec![(start.to_owned(), 0_usize)];
    while let Some((path, depth)) = pending.pop() {
        if result.scanned >= MAX_ENTRIES {
            result.truncated = true;
            break;
        }
        let listing = match entries(context, &path, MAX_ENTRIES - result.scanned) {
            Ok(listing) => listing,
            Err(_) if path != start => {
                result.skipped += 1;
                continue;
            }
            Err(error) => return Err(error),
        };
        result.scanned += listing.entries.len() + listing.skipped;
        result.skipped += listing.skipped;
        result.truncated |= listing.truncated;
        let mut directories = Vec::new();
        for (name, kind, _) in listing.entries {
            let child = path.join(&name);
            match kind {
                "file" => result.files.push(child),
                "directory" if skipped_directory(&name) => result.skipped += 1,
                "directory" if depth < MAX_DEPTH => directories.push((child, depth + 1)),
                "directory" => {
                    result.skipped += 1;
                    result.truncated = true;
                }
                _ => result.skipped += 1,
            }
        }
        pending.extend(directories.into_iter().rev());
    }
    result.files.sort();
    Ok(result)
}

fn glob_matches(pattern: &str, text: &str) -> bool {
    let pattern: Vec<_> = pattern.chars().collect();
    let text: Vec<_> = text.chars().collect();
    let (mut p, mut t, mut star, mut retry) = (0, 0, None, 0);
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some(p);
            p += 1;
            retry = t;
        } else if let Some(previous_star) = star {
            retry += 1;
            t = retry;
            p = previous_star + 1;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

fn name_matches(pattern: &str, path: &Path, case_sensitive: bool) -> bool {
    let name = if pattern.contains('/') {
        display(path)
    } else {
        path.file_name()
            .unwrap_or_else(|| OsStr::new(""))
            .to_string_lossy()
            .into_owned()
    };
    if case_sensitive {
        glob_matches(pattern, &name)
    } else {
        glob_matches(&pattern.to_lowercase(), &name.to_lowercase())
    }
}

fn truncate_text(text: &str, bytes: usize) -> (&str, bool) {
    if text.len() <= bytes {
        return (text, false);
    }
    let mut boundary = bytes;
    while !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    (&text[..boundary], true)
}

fn search_files(context: &Context, path: &Path, args: &SearchArgs) -> Result<Value> {
    let mut walked = walk(context, path)?;
    let mut matches = Vec::new();
    let mut matched = 0;
    let mut scanned_files = 0;
    let mut scanned_bytes = 0;
    let mut output_bytes = 0;
    let needle = if args.case_sensitive {
        args.pattern.clone()
    } else {
        args.pattern.to_lowercase()
    };
    'files: for file in &walked.files {
        if args
            .file_glob
            .as_ref()
            .is_some_and(|glob| !name_matches(glob, file, args.case_sensitive))
        {
            continue;
        }
        if args.target == "files" {
            if !name_matches(&args.pattern, file, args.case_sensitive) {
                continue;
            }
            matched += 1;
            if matched <= args.offset {
                continue;
            }
            let result = json!({"path":display(file)});
            let result_bytes = serde_json::to_vec(&result)?.len();
            if matches.len() >= args.limit || output_bytes + result_bytes > MAX_OUTPUT_BYTES {
                walked.truncated = true;
                break;
            }
            output_bytes += result_bytes;
            matches.push(result);
            continue;
        }
        let metadata = match context.dir.symlink_metadata(file) {
            Ok(metadata) => metadata,
            Err(_) => {
                walked.skipped += 1;
                continue;
            }
        };
        if metadata.len() > MAX_READ_BYTES as u64 {
            walked.skipped += 1;
            continue;
        }
        if scanned_bytes + metadata.len() as usize > MAX_SCAN_BYTES {
            walked.truncated = true;
            break;
        }
        let text = match read_text(context, file) {
            Ok(text) => text,
            Err(_) => {
                walked.skipped += 1;
                continue;
            }
        };
        if scanned_bytes + text.len() > MAX_SCAN_BYTES {
            walked.truncated = true;
            break;
        }
        scanned_files += 1;
        scanned_bytes += text.len();
        for (index, line) in text.lines().enumerate() {
            let found = if args.case_sensitive {
                line.contains(&needle)
            } else {
                line.to_lowercase().contains(&needle)
            };
            if !found {
                continue;
            }
            matched += 1;
            if matched <= args.offset {
                continue;
            }
            let (line, line_truncated) = truncate_text(line, MAX_LINE_BYTES);
            if matches.len() >= args.limit
                || output_bytes + line.len() + display(file).len() + 100 > MAX_OUTPUT_BYTES
            {
                walked.truncated = true;
                break 'files;
            }
            output_bytes += line.len() + display(file).len() + 100;
            matches.push(json!({"path":display(file),"line":index + 1,"text":line,"line_truncated":line_truncated}));
        }
    }
    Ok(json!({
        "matches":matches,"target":args.target,"offset":args.offset,"truncated":walked.truncated,
        "scanned_entries":walked.scanned,"scanned_files":scanned_files,"scanned_bytes":scanned_bytes,
        "skipped_entries":walked.skipped
    }))
}

fn ensure_parent(context: &Context, path: &Path) -> Result<Dir> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut prefix = PathBuf::new();
    for part in parent.components() {
        if part == Component::CurDir {
            continue;
        }
        prefix.push(part);
        match context.dir.symlink_metadata(&prefix) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink() && metadata.is_dir(),
                "Parent must be a workspace directory without symbolic links"
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                context
                    .dir
                    .create_dir(&prefix)
                    .with_context(|| format!("Could not create {}", display(&prefix)))?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    context
        .dir
        .open_dir(if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        })
        .context("Could not open destination directory")
}

static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

struct TempWrite<'a> {
    directory: &'a Dir,
    name: String,
}
impl Drop for TempWrite<'_> {
    fn drop(&mut self) {
        let _ = self.directory.remove_file(&self.name);
    }
}

fn write_file(context: &Context, path: &Path, content: &str) -> Result<Value> {
    check_path(context, path, true)?;
    let existing = match context.dir.symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "Destination is not a regular file"
            );
            let observed = observed_version(context, path)?;
            if context.redact_secrets && crate::privacy::contains_secret(std::str::from_utf8(&observed)?){
                anyhow::bail!("The existing file contains masked credentials. Edit it manually to preserve its original values.");
            }
            ensure!(
                read_text(context, path)?.as_bytes() == observed,
                "File changed since it was read. File was left untouched; read it again and merge your changes"
            );
            Some((observed, metadata.permissions()))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let checkpoint = if let Some(store) = &context.checkpoints {
        let before = existing.as_ref().map(|(bytes, _)| std::str::from_utf8(bytes)).transpose()?;
        Some(store.capture(&context.root, path, before, Some(content))?)
    } else { None };
    let directory = ensure_parent(context, path)?;
    let destination = path.file_name().context("Destination must name a file")?;
    let mut temporary = None;
    for _ in 0..16 {
        let name = format!(
            ".hermes-write-{}-{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        );
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        match directory.open_with(&name, &options) {
            Ok(file) => {
                temporary = Some((
                    TempWrite {
                        directory: &directory,
                        name,
                    },
                    file,
                ));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context("Could not create temporary workspace file"),
        }
    }
    let (temporary, mut file) = temporary.context("Could not allocate temporary workspace file")?;
    file.write_all(content.as_bytes())?;
    if let Some((_, permissions)) = &existing {
        file.set_permissions(permissions.clone())?;
    }
    file.sync_all()?;
    drop(file);
    // Check again immediately before replacement, after preparing/flushing data.
    // External editors do not participate in a common lock, so this is an
    // optimistic concurrency check rather than an OS-wide editor transaction.
    if let Some((observed, _)) = &existing {
        check_path(context, path, false)?;
        ensure!(
            read_text(context, path)?.as_bytes() == observed,
            "File changed while preparing the write. File was left untouched; read it again and merge your changes"
        );
        directory
            .rename(&temporary.name, &directory, destination)
            .context("Could not replace workspace file")?;
    } else {
        // hard_link creates the final name exclusively: a file created during
        // the approval wait or save must never be silently overwritten.
        directory
            .hard_link(&temporary.name, &directory, destination)
            .context("Destination appeared while creating the file; it was left untouched")?;
    }
    drop(temporary);
    let actual = read_text(context, path)?;
    ensure!(
        actual == content,
        "File was modified by another application immediately after saving"
    );
    observe(context, path, content.as_bytes(), 0, content.len())?;
    let mut result = json!({"path":display(path),"bytes_written":content.len(),"created":existing.is_none(),"verified":true});
    if let (Some(store), Some(token)) = (&context.checkpoints, checkpoint) {
        let committed = store.commit(&token).is_ok();
        result["checkpoint_id"] = json!(token.id);
        result["checkpoint_ready"] = json!(committed);
    }
    Ok(result)
}

struct Skill {
    name: String,
    description: String,
    category: String,
    path: PathBuf,
    id: String,
}
impl Skill {
    fn value(&self) -> Value {
        json!({"name":self.name,"description":self.description,"category":self.category,"path":display(&self.path),"id":self.id})
    }
}

fn frontmatter_field(text: &str, field: &str) -> Option<String> {
    let mut lines = text.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let mut value: Option<String> = None;
    let mut folded = false;
    for line in lines.take(100) {
        if line.trim() == "---" {
            break;
        }
        if folded {
            if line.starts_with([' ', '\t']) {
                let item = value.as_mut()?;
                if !item.is_empty() {
                    item.push(' ');
                }
                item.push_str(line.trim());
                continue;
            }
            break;
        }
        if let Some(rest) = line
            .strip_prefix(field)
            .and_then(|rest| rest.strip_prefix(':'))
        {
            let rest = rest.trim();
            folded = matches!(rest, ">" | "|" | ">-" | "|-");
            value = Some(if folded {
                String::new()
            } else {
                rest.trim_matches(['\'', '"']).to_owned()
            });
            if !folded {
                break;
            }
        }
    }
    value
        .map(|value| truncate_text(&value, 1024).0.to_owned())
        .filter(|value| !value.is_empty())
}

fn discover_skills(context: &Context) -> Result<(Vec<Skill>, Walk)> {
    let mut walked = walk(context, Path::new("."))?;
    let mut skills = Vec::new();
    let mut bytes = 0;
    for path in &walked.files {
        if path.file_name() != Some(OsStr::new("SKILL.md")) {
            continue;
        }
        if skills.len() >= MAX_SKILLS {
            walked.truncated = true;
            break;
        }
        let text = match read_text(context, path) {
            Ok(text) => text,
            Err(_) => {
                walked.skipped += 1;
                continue;
            }
        };
        bytes += text.len();
        if bytes > MAX_SCAN_BYTES {
            walked.truncated = true;
            break;
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let id = if parent.as_os_str().is_empty() {
            ".".into()
        } else {
            display(parent).trim_start_matches("./").to_owned()
        };
        let fallback = parent
            .file_name()
            .or_else(|| context.root.file_name())
            .unwrap_or_else(|| OsStr::new("workspace"))
            .to_string_lossy()
            .into_owned();
        let name = frontmatter_field(&text, "name").unwrap_or(fallback);
        let description = frontmatter_field(&text, "description").unwrap_or_else(|| {
            text.lines()
                .find(|line| {
                    !line.trim().is_empty() && !line.starts_with('#') && line.trim() != "---"
                })
                .map(|line| truncate_text(line.trim(), 240).0.to_owned())
                .unwrap_or_default()
        });
        let category = frontmatter_field(&text, "category").unwrap_or_else(|| {
            parent
                .parent()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        skills.push(Skill {
            name,
            description,
            category,
            path: path.to_owned(),
            id,
        });
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id)));
    Ok((skills, walked))
}

fn skill_view(
    context: &Context,
    name: &str,
    file_path: Option<&Path>,
    offset: usize,
    limit: usize,
) -> Result<Value> {
    let (skills, _) = discover_skills(context)?;
    let exact = skills
        .iter()
        .find(|skill| skill.id == name || display(&skill.path).trim_start_matches("./") == name);
    let skill = if let Some(skill) = exact {
        skill
    } else {
        let mut matches = skills.iter().filter(|skill| skill.name == name);
        let skill = matches.next().context(
            "Skill not found in the workspace. Use skills_list to discover available skill IDs",
        )?;
        ensure!(
            matches.next().is_none(),
            "Several skills have this name. Use a unique ID from skills_list"
        );
        skill
    };
    let parent = skill.path.parent().unwrap_or_else(|| Path::new("."));
    let path = match file_path {
        Some(file) => parent.join(file),
        None => skill.path.clone(),
    };
    let mut result = read_file(context, &path, offset, limit)?;
    result["name"] = json!(skill.name);
    result["id"] = json!(skill.id);
    if file_path.is_none() {
        let linked = walk(context, parent)?;
        let files: Vec<_> = linked
            .files
            .iter()
            .filter(|path| **path != skill.path)
            .take(128)
            .filter_map(|path| path.strip_prefix(parent).ok().map(display))
            .collect();
        result["linked_files"] = json!(files);
        result["linked_files_truncated"] = json!(linked.truncated || linked.files.len() > 129);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn workspace() -> (TempDir, Context) {
        let directory = tempfile::tempdir().unwrap();
        let context = Context::new(directory.path()).unwrap();
        (directory, context)
    }

    fn read(context: &Context, path: &str) -> Value {
        invoke("read_file", &json!({"path":path}), context).unwrap()
    }

    #[test]
    fn registry_has_hermes_names_and_closed_json_schemas() {
        let specs = tool_specs();
        assert_eq!(specs.len(), 7);
        for spec in &specs {
            assert_eq!(spec["type"], "function");
            assert_eq!(spec["function"]["parameters"]["type"], "object");
            assert_eq!(
                spec["function"]["parameters"]["additionalProperties"],
                false
            );
        }
        for name in [
            "get_current_time",
            "read_file",
            "search_files",
            "write_file",
            "skills_list",
            "skill_view",
        ] {
            assert!(specs.iter().any(|spec| spec["function"]["name"] == name));
        }
    }

    #[test]
    fn clock_and_validation_do_not_need_write_authority() {
        let (_directory, context) = workspace();
        let result = invoke("get_current_time", &json!({}), &context).unwrap();
        assert!(chrono::DateTime::parse_from_rfc3339(result["utc"].as_str().unwrap()).is_ok());
        assert!(result["utc"].as_str().unwrap().ends_with('Z'));
        assert!(
            !validate("get_current_time", &json!({}), &context)
                .unwrap()
                .mutates_workspace
        );
        assert!(validate("get_current_time", &json!({"command":"anything"}), &context).is_err());
        assert!(validate("terminal", &json!({"command":"pwd"}), &context).is_err());
        assert!(validate("read_file", &json!({"path":"../outside"}), &context).is_err());
        assert!(validate("read_file", &json!({"path":"/etc/passwd"}), &context).is_err());
        assert!(validate("read_file", &json!({"path":"file","offset":0}), &context).is_err());
        assert!(
            validate(
                "search_files",
                &json!({"pattern":"x","limit":201}),
                &context
            )
            .is_err()
        );
        assert!(
            validate(
                "write_file",
                &json!({"path":"x","content":"ok","approved":true}),
                &context
            )
            .is_err()
        );
    }

    #[test]
    fn writes_require_explicit_ephemeral_approval() {
        let (directory, context) = workspace();
        let arguments = json!({"path":"notes/new.txt","content":"hello\n"});
        let approval = validate("write_file", &arguments, &context).unwrap();
        assert!(approval.mutates_workspace);
        assert!(approval.summary.contains("6 UTF-8 bytes"));
        assert!(
            invoke("write_file", &arguments, &context)
                .unwrap_err()
                .to_string()
                .contains("explicit approval")
        );
        assert!(!directory.path().join("notes").exists());
        let approved = context.with_write_approval(true);
        assert_eq!(
            invoke("write_file", &arguments, &approved).unwrap()["created"],
            true
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("notes/new.txt")).unwrap(),
            "hello\n"
        );
        assert!(invoke("write_file", &arguments, &context).is_err());
        assert!(
            invoke(
                "write_file",
                &arguments,
                &approved.with_write_approval(false)
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read_dir(directory.path().join("notes"))
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn overwrite_requires_complete_current_read_and_preserves_external_edits() {
        let (directory, context) = workspace();
        let path = directory.path().join("note.txt");
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let approved = context.with_write_approval(true);
        let arguments = json!({"path":"note.txt","content":"replacement\n"});
        assert!(
            invoke("write_file", &arguments, &approved)
                .unwrap_err()
                .to_string()
                .contains("has not been read")
        );
        let first = invoke("read_file", &json!({"path":"note.txt","limit":1}), &context).unwrap();
        assert_eq!(first["next_offset"], 2);
        assert!(
            invoke("write_file", &arguments, &approved)
                .unwrap_err()
                .to_string()
                .contains("partially read")
        );
        invoke(
            "read_file",
            &json!({"path":"note.txt","offset":2}),
            &context,
        )
        .unwrap();
        std::fs::write(&path, "user edit\n").unwrap();
        assert!(
            invoke("write_file", &arguments, &approved)
                .unwrap_err()
                .to_string()
                .contains("changed since")
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "user edit\n");
        read(&context, "note.txt");
        assert_eq!(
            invoke("write_file", &arguments, &approved).unwrap()["verified"],
            true
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement\n");
    }

    #[test]
    fn rereading_a_changed_page_invalidates_previous_page_coverage() {
        let (directory, context) = workspace();
        let path = directory.path().join("note.txt");
        std::fs::write(&path, "one\ntwo\n").unwrap();
        invoke("read_file", &json!({"path":"note.txt","limit":1}), &context).unwrap();
        std::fs::write(&path, "new one\ntwo\n").unwrap();
        invoke(
            "read_file",
            &json!({"path":"note.txt","offset":2}),
            &context,
        )
        .unwrap();
        let error = invoke(
            "write_file",
            &json!({"path":"note.txt","content":"x"}),
            &context.with_write_approval(true),
        )
        .unwrap_err();
        assert!(error.to_string().contains("partially read"));
        assert_eq!(std::fs::read_to_string(path).unwrap(), "new one\ntwo\n");
    }

    #[test]
    fn read_pagination_utf8_binary_and_size_bounds() {
        let (directory, context) = workspace();
        std::fs::write(directory.path().join("text.txt"), "alpha\r\nβeta\nlast").unwrap();
        let result = invoke(
            "read_file",
            &json!({"path":"text.txt","offset":2,"limit":1}),
            &context,
        )
        .unwrap();
        assert_eq!(result["content"], "2|βeta\n");
        assert_eq!(result["total_lines"], 3);
        assert_eq!(result["next_offset"], 3);
        std::fs::write(directory.path().join("empty"), "").unwrap();
        assert_eq!(read(&context, "empty")["total_lines"], 0);
        std::fs::write(directory.path().join("binary"), [0_u8, 1, 2]).unwrap();
        assert!(invoke("read_file", &json!({"path":"binary"}), &context).is_err());
        std::fs::write(
            directory.path().join("huge"),
            vec![b'x'; MAX_READ_BYTES + 1],
        )
        .unwrap();
        assert!(invoke("read_file", &json!({"path":"huge"}), &context).is_err());
        assert!(
            validate(
                "write_file",
                &json!({"path":"x","content":"x".repeat(MAX_WRITE_BYTES + 1)}),
                &context
            )
            .is_err()
        );
        let absolute = context.workspace_root().join("text.txt");
        assert!(invoke("read_file", &json!({"path":absolute}), &context).is_ok());
    }

    #[test]
    fn list_and_literal_content_or_glob_search_are_bounded() {
        let (directory, context) = workspace();
        std::fs::create_dir(directory.path().join("src")).unwrap();
        std::fs::create_dir(directory.path().join("node_modules")).unwrap();
        std::fs::write(
            directory.path().join("src/a.rs"),
            "Needle [literal]\nsecond needle\n",
        )
        .unwrap();
        std::fs::write(directory.path().join("src/b.txt"), "Needle\n").unwrap();
        std::fs::write(directory.path().join("node_modules/ignored.rs"), "Needle\n").unwrap();
        let listing = invoke("list_directory", &json!({"path":"src","limit":1}), &context).unwrap();
        assert_eq!(listing["entries"][0]["name"], "a.rs");
        assert_eq!(listing["truncated"], true);
        let literal = invoke(
            "search_files",
            &json!({"pattern":"[literal]","file_glob":"*.rs"}),
            &context,
        )
        .unwrap();
        assert_eq!(literal["matches"].as_array().unwrap().len(), 1);
        assert_eq!(literal["matches"][0]["line"], 1);
        let limited = invoke(
            "search_files",
            &json!({"pattern":"needle","case_sensitive":false,"limit":1}),
            &context,
        )
        .unwrap();
        assert_eq!(limited["matches"].as_array().unwrap().len(), 1);
        assert_eq!(limited["truncated"], true);
        let files = invoke(
            "search_files",
            &json!({"pattern":"*.rs","target":"files"}),
            &context,
        )
        .unwrap();
        assert_eq!(files["matches"].as_array().unwrap().len(), 1);
        assert!(
            files["matches"][0]["path"]
                .as_str()
                .unwrap()
                .ends_with("src/a.rs")
        );
    }

    #[test]
    fn discovers_skills_and_reads_scoped_references_without_executing_them() {
        let (directory, context) = workspace();
        let root = directory.path().join(".agents/skills/writing");
        std::fs::create_dir_all(root.join("references")).unwrap();
        std::fs::write(root.join("SKILL.md"), "---\nname: writing\ndescription: >\n  Draft clear prose\n  with examples.\ncategory: communication\n---\n# Writing\nRead the reference.\n").unwrap();
        std::fs::write(root.join("references/example.md"), "reference text\n").unwrap();
        let listing = invoke(
            "skills_list",
            &json!({"category":"communication"}),
            &context,
        )
        .unwrap();
        assert_eq!(listing["skills"].as_array().unwrap().len(), 1);
        assert_eq!(
            listing["skills"][0]["description"],
            "Draft clear prose with examples."
        );
        let result = invoke("skill_view", &json!({"name":"writing"}), &context).unwrap();
        assert!(result["content"].as_str().unwrap().contains("# Writing"));
        assert_eq!(result["linked_files"][0], "references/example.md");
        let reference = invoke(
            "skill_view",
            &json!({"name":"writing","file_path":"references/example.md"}),
            &context,
        )
        .unwrap();
        assert_eq!(reference["content"], "1|reference text\n");
        assert!(
            validate(
                "skill_view",
                &json!({"name":"writing","file_path":"../outside.txt"}),
                &context
            )
            .is_err()
        );
        assert!(validate("skill_view", &json!({"name":"../outside"}), &context).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escapes_are_rejected_for_read_write_search_and_skills() {
        use std::os::unix::fs::symlink;
        let (directory, context) = workspace();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), "external\n").unwrap();
        std::fs::write(
            outside.path().join("SKILL.md"),
            "---\nname: external\n---\n",
        )
        .unwrap();
        symlink(outside.path(), directory.path().join("escape")).unwrap();
        symlink(
            outside.path().join("secret"),
            directory.path().join("alias"),
        )
        .unwrap();
        for path in ["escape/secret", "alias"] {
            assert!(invoke("read_file", &json!({"path":path}), &context).is_err());
            assert!(
                invoke(
                    "write_file",
                    &json!({"path":path,"content":"overwritten"}),
                    &context.with_write_approval(true)
                )
                .is_err()
            );
        }
        assert!(
            invoke(
                "write_file",
                &json!({"path":"escape/new","content":"bad"}),
                &context.with_write_approval(true)
            )
            .is_err()
        );
        assert!(
            invoke(
                "search_files",
                &json!({"pattern":"external","path":"escape"}),
                &context
            )
            .is_err()
        );
        assert!(
            invoke("search_files", &json!({"pattern":"external"}), &context).unwrap()["matches"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            invoke("skills_list", &json!({}), &context).unwrap()["skills"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret")).unwrap(),
            "external\n"
        );
        assert!(!outside.path().join("new").exists());
    }

    #[cfg(unix)]
    #[test]
    fn paths_changed_to_symlinks_after_validation_are_rejected() {
        use std::os::unix::fs::symlink;
        let (directory, context) = workspace();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("nested")).unwrap();
        let args = json!({"path":"nested/new","content":"bad"});
        validate("write_file", &args, &context).unwrap();
        std::fs::remove_dir(directory.path().join("nested")).unwrap();
        symlink(outside.path(), directory.path().join("nested")).unwrap();
        assert!(invoke("write_file", &args, &context.with_write_approval(true)).is_err());
        assert!(!outside.path().join("new").exists());
    }
}
