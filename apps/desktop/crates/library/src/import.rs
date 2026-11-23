//! Hermes SQLite compatibility, based on upstream
//! `persistence/sqlite/_schema_ddl.py` and `_store/messages.py`.
//!
//! Original databases are opened only with `std::fs::File` for reading. SQLite
//! receives a private, stable copy of the database and optional WAL, so it cannot
//! migrate the source or create/update a source `-shm` file.
use super::*;
use hermes_engine::{Entry, Status, ToolCall, Turn};
use rusqlite::{Connection, OpenFlags, types::ValueRef};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    time::{Duration, SystemTime},
};
use tempfile::TempDir;

const MAX_MESSAGES: usize = 20_000;
const MAX_TRANSCRIPT_BYTES: usize = 64 * 1024 * 1024;
const MAX_SNAPSHOT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_SESSIONS: usize = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermesSession {
    pub id: String,
    pub title: String,
    pub model: Option<String>,
    pub source: String,
    pub cwd: Option<PathBuf>,
    pub started_at: f64,
    pub last_activity_at: f64,
    pub message_count: u64,
    pub archived: bool,
    pub hidden: bool,
    pub parent_session_id: Option<String>,
    pub end_reason: Option<String>,
    /// Original selected column values. Listing reads metadata only;
    /// `read_session` retains every original column in its archive.
    pub raw: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermesMessage {
    pub id: i64,
    pub session_id: String,
    pub role: String,
    /// Decodes the upstream `\0json:` sentinel without interpreting ordinary strings.
    pub content: Value,
    pub tool_call_id: Option<String>,
    pub tool_calls: Value,
    pub tool_name: Option<String>,
    pub timestamp: f64,
    pub active: bool,
    pub compacted: bool,
    pub display_metadata: Value,
    /// Every original column, including inactive messages and reasoning fields.
    /// SQLite BLOB values use an explicit `$sqlite_blob_hex` wrapper.
    pub raw: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermesArchive {
    pub version: u32,
    pub database: PathBuf,
    pub upstream_session_id: String,
    pub imported_at: String,
    pub schema_version: Option<i64>,
    pub sessions: Vec<HermesSession>,
    pub messages: Vec<HermesMessage>,
}

#[derive(Debug, Clone)]
pub struct ImportReport {
    pub session: SessionSummary,
    pub imported_messages: usize,
    pub archive_path: PathBuf,
    pub already_imported: bool,
    pub warnings: Vec<String>,
}

/// A manual, immutable snapshot of one user-selected Hermes `state.db`.
///
/// Opening this handle neither discovers nor automatically imports conversations.
/// Keep the handle for the listing/picker and import the selected rows explicitly.
pub struct HermesHistory {
    database: PathBuf,
    connection: Connection,
    _snapshot: TempDir,
    session_columns: HashSet<String>,
}

impl HermesHistory {
    pub fn open(path: &Path) -> Result<Self> {
        let database = fs::canonicalize(path).context("Choose an existing Hermes state.db file")?;
        ensure!(
            database.is_file(),
            "The Hermes database path must be a file"
        );
        let snapshot = stable_snapshot(&database)?;
        let connection = Connection::open_with_flags(
            snapshot.path().join("state.db"),
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("Could not open the read-only Hermes database snapshot")?;
        connection.busy_timeout(Duration::from_secs(2))?;
        connection.set_limit(
            rusqlite::limits::Limit::SQLITE_LIMIT_LENGTH,
            MAX_TRANSCRIPT_BYTES as i32,
        )?;
        connection.execute_batch("PRAGMA query_only = ON; PRAGMA trusted_schema = OFF;")?;
        let session_columns = table_columns(&connection, "sessions")?;
        let message_columns = table_columns(&connection, "messages")?;
        for column in ["id", "started_at", "source"] {
            ensure!(
                session_columns.contains(column),
                "Unsupported Hermes schema: sessions.{column} is missing"
            );
        }
        for column in ["id", "session_id", "role", "content", "timestamp"] {
            ensure!(
                message_columns.contains(column),
                "Unsupported Hermes schema: messages.{column} is missing"
            );
        }
        Ok(Self {
            database,
            connection,
            _snapshot: snapshot,
            session_columns,
        })
    }

    pub fn database(&self) -> &Path {
        &self.database
    }

    /// All physical session rows, newest first. Compression segments are combined
    /// when imported; archived/hidden rows remain labeled for the caller's picker.
    pub fn sessions(&self) -> Result<Vec<HermesSession>> {
        // Keep the picker a metadata read. In particular, system prompts,
        // model_config and other potentially large private payloads are only
        // copied after the user explicitly selects a conversation to import.
        let fields: Vec<_> = [
            "id",
            "title",
            "model",
            "source",
            "cwd",
            "started_at",
            "ended_at",
            "last_activity_at",
            "message_count",
            "archived",
            "hidden",
            "parent_session_id",
            "end_reason",
        ]
        .into_iter()
        .filter(|field| self.session_columns.contains(*field))
        .collect();
        let mut statement = self.connection.prepare(&format!(
            "SELECT {} FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?1",
            fields.join(", ")
        ))?;
        let names = statement
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut rows = statement.query([MAX_SESSIONS as i64 + 1])?;
        let mut sessions = Vec::new();
        let mut bytes = 0;
        while let Some(row) = rows.next()? {
            ensure!(
                sessions.len() < MAX_SESSIONS,
                "This database contains more than {MAX_SESSIONS} sessions; use a smaller Hermes export"
            );
            let raw = row_map(row, &names)?;
            bytes += serde_json::to_vec(&raw)?.len();
            ensure!(
                bytes <= MAX_TRANSCRIPT_BYTES,
                "This history has more than 64 MiB of session metadata; use a smaller Hermes backup"
            );
            sessions.push(parse_session(raw)?);
        }
        Ok(sessions)
    }

    /// Reads the selected conversation and its compression lineage into a
    /// lossless archive. No provider connection or tool execution takes place.
    pub fn read_session(&self, id: &str) -> Result<HermesArchive> {
        ensure!(
            !id.is_empty() && id.len() <= 4096,
            "Invalid Hermes session identifier"
        );
        let sessions = self.lineage(id)?;
        let mut messages = vec![];
        let mut bytes = 0;
        for session in &sessions {
            let mut statement = self
                .connection
                .prepare("SELECT * FROM messages WHERE session_id = ?1 ORDER BY id ASC LIMIT ?2")?;
            let names = statement
                .column_names()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let mut rows =
                statement.query(rusqlite::params![session.id, MAX_MESSAGES as i64 + 1])?;
            while let Some(row) = rows.next()? {
                ensure!(
                    messages.len() < MAX_MESSAGES,
                    "This conversation exceeds the {MAX_MESSAGES} message import limit; use a smaller Hermes export"
                );
                let raw = row_map(row, &names)?;
                bytes += serde_json::to_vec(&raw)?.len();
                ensure!(
                    bytes <= MAX_TRANSCRIPT_BYTES,
                    "This conversation exceeds the 64 MiB import limit; use a smaller Hermes export"
                );
                messages.push(parse_message(raw)?);
            }
        }
        let schema_version = self
            .connection
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |row| {
                row.get(0)
            })
            .ok();
        Ok(HermesArchive {
            version: 1,
            database: self.database.clone(),
            upstream_session_id: id.to_owned(),
            imported_at: now(),
            schema_version,
            sessions,
            messages,
        })
    }

    fn session(&self, id: &str) -> Result<Option<HermesSession>> {
        let mut statement = self
            .connection
            .prepare("SELECT * FROM sessions WHERE id = ?1")?;
        let names = statement
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut rows = statement.query([id])?;
        rows.next()?
            .map(|row| row_map(row, &names).and_then(parse_session))
            .transpose()
    }

    fn lineage(&self, id: &str) -> Result<Vec<HermesSession>> {
        let selected = self
            .session(id)?
            .context("The selected Hermes conversation no longer exists")?;
        if !self.session_columns.contains("parent_session_id")
            || !self.session_columns.contains("end_reason")
        {
            return Ok(vec![selected]);
        }
        let mut chain = vec![selected];
        let mut seen = HashSet::from([id.to_owned()]);
        loop {
            let child = chain.last().unwrap();
            if explicit_fork(child) {
                break;
            }
            let Some(parent_id) = child.parent_session_id.as_deref() else {
                break;
            };
            let Some(parent) = self.session(parent_id)? else {
                break;
            };
            if parent.end_reason.as_deref() != Some("compression") {
                break;
            }
            ensure!(
                chain.len() < 1000 && seen.insert(parent.id.clone()),
                "Invalid or excessively deep Hermes compression lineage"
            );
            chain.push(parent);
        }
        chain.reverse();
        // Continue from the selected segment. Match upstream's preferred child:
        // compression-ended child, then open child, then newest closed child.
        while chain.last().unwrap().end_reason.as_deref() == Some("compression") {
            let parent_id = &chain.last().unwrap().id;
            let mut statement = self
                .connection
                .prepare("SELECT * FROM sessions WHERE parent_session_id = ?1")?;
            let names = statement
                .column_names()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let mut rows = statement.query([parent_id])?;
            let mut children = vec![];
            while let Some(row) = rows.next()? {
                ensure!(
                    children.len() < MAX_SESSIONS,
                    "This session has too many compression children to import safely"
                );
                let child = parse_session(row_map(row, &names)?)?;
                if !explicit_fork(&child) {
                    children.push(child);
                }
            }
            children.sort_by(|a, b| {
                child_rank(a)
                    .cmp(&child_rank(b))
                    .then_with(|| b.last_activity_at.total_cmp(&a.last_activity_at))
                    .then_with(|| b.started_at.total_cmp(&a.started_at))
                    .then_with(|| b.id.cmp(&a.id))
            });
            let Some(child) = children.into_iter().next() else {
                break;
            };
            ensure!(
                chain.len() < 1000 && seen.insert(child.id.clone()),
                "Invalid or excessively deep Hermes compression lineage"
            );
            chain.push(child);
        }
        Ok(chain)
    }
}

impl Library {
    /// Explicitly copy one selected upstream conversation into native storage.
    /// Source cwd is provenance only; a workspace is used only when selected by
    /// the caller. Reimporting a segment from the same database is idempotent.
    pub fn import_hermes(
        &self,
        source: &HermesHistory,
        id: &str,
        project_id: Option<&str>,
    ) -> Result<ImportReport> {
        let archive = source.read_session(id)?;
        let (history, imported_messages, warnings) = convert_history(&archive);
        let title = archive
            .sessions
            .iter()
            .rev()
            .find(|session| !session.title.is_empty())
            .map(|session| session.title.as_str());
        self.with_catalog(true, |catalog| {
            if let Some(id) = project_id { ensure_project(catalog, id)?; }
            let upstream_ids: HashSet<_> = archive.sessions.iter().map(|session| session.id.as_str()).collect();
            if let Some(existing) = catalog.sessions.iter().find(|session| matches!(&session.source,
                SessionSource::Hermes { database, session_id, .. } if database == &source.database && upstream_ids.contains(session_id.as_str()))) {
                return Ok(ImportReport { session: self.summary(existing), imported_messages: 0, archive_path: self.checked_directory(&existing.id)?.join("hermes-source.json"), already_imported: true, warnings: vec!["This Hermes conversation has already been imported. The existing native copy was retained.".to_owned()] });
            }
            let mut record = self.new_record(project_id, title, SessionSource::Hermes { database: source.database.clone(), session_id: id.to_owned(), imported_at: archive.imported_at.clone() });
            if record.title == "New chat" && let Some(title) = title_from_history(&history) {
                record.title = title;
            }
            let directory = self.create_directory(&record.id)?;
            let archive_path = directory.join("hermes-source.json");
            let save = (|| -> Result<()> {
                atomic_json(&archive_path, &archive)?;
                atomic_json(&directory.join("conversation.json"), &history)
            })();
            if let Err(error) = save {
                // Only this freshly-created private directory is removed.
                let _ = fs::remove_file(&archive_path);
                let _ = fs::remove_file(directory.join("conversation.json"));
                let _ = fs::remove_dir(&directory);
                return Err(error).context("Could not save the imported conversation");
            }
            catalog.active_session = Some(record.id.clone());
            catalog.selected_project = record.project_id.clone();
            let report = ImportReport { session: self.summary(&record), imported_messages, archive_path, already_imported: false, warnings };
            catalog.sessions.push(record);
            Ok(report)
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Stamp {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    inode: u64,
}

fn stamp(path: &Path) -> Result<Option<Stamp>> {
    match fs::metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_file(),
                "Expected a regular Hermes database or WAL file"
            );
            #[cfg(unix)]
            use std::os::unix::fs::MetadataExt;
            Ok(Some(Stamp {
                len: metadata.len(),
                modified: metadata.modified()?,
                #[cfg(unix)]
                inode: metadata.ino(),
            }))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    name.into()
}

fn stable_snapshot(database: &Path) -> Result<TempDir> {
    let wal = sidecar(database, "-wal");
    let journal = sidecar(database, "-journal");
    for _ in 0..3 {
        let before = (stamp(database)?, stamp(&wal)?, stamp(&journal)?);
        ensure!(before.0.is_some(), "The Hermes database disappeared");
        // A nonzero journal header means uncommitted main-file pages may need
        // recovery. Never copy those into a supposedly clean snapshot.
        if let Some(state) = &before.2
            && state.len > 0
        {
            let mut header = [0u8; 8];
            let count = File::open(&journal)?.read(&mut header)?;
            ensure!(
                count < 8 || header == [0; 8],
                "Hermes has an active rollback journal. Close Hermes cleanly and retry the import"
            );
        }
        let bytes = before
            .0
            .as_ref()
            .map_or(0, |stamp| stamp.len)
            .saturating_add(before.1.as_ref().map_or(0, |stamp| stamp.len));
        ensure!(
            bytes <= MAX_SNAPSHOT_BYTES,
            "This Hermes database and WAL exceed the 4 GiB snapshot limit; use a smaller backup"
        );
        let snapshot = tempfile::Builder::new()
            .prefix("hermes-readonly-import-")
            .tempdir()?;
        // File::open is always read-only. Do not use SQLite's source connection:
        // even SQLITE_OPEN_READ_ONLY may update/create WAL shared-memory files.
        copy_readonly(database, &snapshot.path().join("state.db"))?;
        if before.1.is_some()
            && let Err(error) = copy_readonly(&wal, &snapshot.path().join("state.db-wal"))
        {
            if stamp(&wal)?.is_none() {
                continue;
            }
            return Err(error);
        }
        if before != (stamp(database)?, stamp(&wal)?, stamp(&journal)?) {
            continue;
        }
        let mut header = [0u8; 16];
        File::open(snapshot.path().join("state.db"))?
            .read_exact(&mut header)
            .context("This file is not a SQLite database")?;
        ensure!(
            &header == b"SQLite format 3\0",
            "This file is not a SQLite database"
        );
        return Ok(snapshot);
    }
    bail!(
        "Hermes history changed during the snapshot. Finish the active turn or close Hermes and retry"
    )
}

fn copy_readonly(source: &Path, destination: &Path) -> Result<()> {
    let mut source =
        File::open(source).context("Could not read the selected Hermes history file")?;
    let mut destination = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination)?;
    std::io::copy(&mut source, &mut destination)?;
    Ok(())
}

fn table_columns(connection: &Connection, table: &str) -> Result<HashSet<String>> {
    let (kind, sql): (String, String) = connection
        .query_row(
            "SELECT type, sql FROM sqlite_schema WHERE name = ?1",
            [table],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .with_context(|| {
            format!("This is not a supported Hermes history database: {table} is missing")
        })?;
    ensure!(
        kind == "table" && !sql.to_ascii_uppercase().contains("CREATE VIRTUAL TABLE"),
        "The Hermes {table} object must be a regular SQLite table"
    );
    let storage_kind: String = connection.query_row(
        "SELECT type FROM pragma_table_list WHERE schema = 'main' AND name = ?1",
        [table],
        |row| row.get(0),
    )?;
    ensure!(
        storage_kind == "table",
        "The Hermes {table} object must not be a virtual or shadow table"
    );
    // `table` is one of two hard-coded names, never caller-controlled SQL.
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    Ok(statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<_>>()?)
}

fn row_map(row: &rusqlite::Row<'_>, names: &[String]) -> Result<BTreeMap<String, Value>> {
    names.iter().enumerate().map(|(index, name)| {
        let value = match row.get_ref(index)? {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(value) => Value::from(value),
            ValueRef::Real(value) => { ensure!(value.is_finite(), "The Hermes database contains a non-finite numeric value"); Value::from(value) },
            ValueRef::Text(value) => Value::String(std::str::from_utf8(value).context("The Hermes database contains invalid UTF-8 text")?.to_owned()),
            ValueRef::Blob(value) => json!({"$sqlite_blob_hex": value.iter().map(|byte| format!("{byte:02x}")).collect::<String>()}),
        };
        Ok((name.clone(), value))
    }).collect()
}

fn text(raw: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    raw.get(key).and_then(Value::as_str).map(str::to_owned)
}
fn number(raw: &BTreeMap<String, Value>, key: &str) -> f64 {
    raw.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}
fn flag(raw: &BTreeMap<String, Value>, key: &str, default: bool) -> bool {
    raw.get(key)
        .and_then(Value::as_i64)
        .map_or(default, |value| value != 0)
}
fn parse_session(raw: BTreeMap<String, Value>) -> Result<HermesSession> {
    let id = text(&raw, "id").context("A Hermes session has no text identifier")?;
    let started_at = number(&raw, "started_at");
    Ok(HermesSession {
        id,
        title: text(&raw, "title").unwrap_or_default(),
        model: text(&raw, "model"),
        source: text(&raw, "source").unwrap_or_default(),
        cwd: text(&raw, "cwd")
            .filter(|text| !text.is_empty())
            .map(PathBuf::from),
        started_at,
        last_activity_at: number(&raw, "last_activity_at")
            .max(number(&raw, "ended_at"))
            .max(started_at),
        message_count: raw
            .get("message_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        archived: flag(&raw, "archived", false),
        hidden: flag(&raw, "hidden", false),
        parent_session_id: text(&raw, "parent_session_id"),
        end_reason: text(&raw, "end_reason"),
        raw,
    })
}

fn decode_json_field(value: &Value) -> Value {
    value
        .as_str()
        .and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or_else(|| value.clone())
}

fn decode_content(value: &Value) -> Value {
    value
        .as_str()
        .and_then(|text| text.strip_prefix("\0json:"))
        .and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or_else(|| value.clone())
}

fn parse_message(raw: BTreeMap<String, Value>) -> Result<HermesMessage> {
    Ok(HermesMessage {
        id: raw
            .get("id")
            .and_then(Value::as_i64)
            .context("A Hermes message has no integer identifier")?,
        session_id: text(&raw, "session_id")
            .context("A Hermes message has no session identifier")?,
        role: text(&raw, "role").context("A Hermes message has no role")?,
        content: decode_content(raw.get("content").unwrap_or(&Value::Null)),
        tool_call_id: text(&raw, "tool_call_id"),
        tool_calls: decode_json_field(raw.get("tool_calls").unwrap_or(&Value::Null)),
        tool_name: text(&raw, "tool_name"),
        timestamp: number(&raw, "timestamp"),
        active: flag(&raw, "active", true),
        compacted: flag(&raw, "compacted", false),
        display_metadata: decode_json_field(raw.get("display_metadata").unwrap_or(&Value::Null)),
        raw,
    })
}

fn explicit_fork(session: &HermesSession) -> bool {
    let config = decode_json_field(session.raw.get("model_config").unwrap_or(&Value::Null));
    session.source == "tool"
        || ["_branched_from", "_delegate_from", "_reset_from"]
            .iter()
            .any(|key| config.get(key).is_some_and(|value| !value.is_null()))
}

fn child_rank(session: &HermesSession) -> u8 {
    if session.end_reason.as_deref() == Some("compression") {
        0
    } else if session.raw.get("ended_at").is_none_or(Value::is_null) {
        1
    } else {
        2
    }
}

fn visible_messages(archive: &HermesArchive) -> Vec<&HermesMessage> {
    let mut representatives: HashMap<String, (usize, &HermesMessage)> = HashMap::new();
    for (index, message) in archive.messages.iter().enumerate() {
        if !(message.active || message.compacted)
            || message
                .display_metadata
                .get("model_only")
                .is_some_and(|value| value == true || value == 1)
        {
            continue;
        }
        let identity = if let Some(value) = message
            .raw
            .get("display_identity")
            .filter(|value| !value.is_null())
        {
            format!("display:{value}")
        } else if let Some(uid) = message
            .raw
            .get("message_uid")
            .and_then(Value::as_str)
            .filter(|uid| !uid.is_empty())
        {
            format!("uid:{uid}")
        } else {
            format!(
                "legacy:{}",
                json!([
                    message.role,
                    message.content,
                    message.timestamp,
                    message.tool_call_id,
                    message.tool_calls
                ])
            )
        };
        match representatives.get_mut(&identity) {
            Some((_, current)) if (message.active, message.id) > (current.active, current.id) => {
                *current = message
            }
            None => {
                representatives.insert(identity, (index, message));
            }
            _ => {}
        }
    }
    let mut messages: Vec<_> = representatives.into_values().collect();
    messages.sort_by_key(|(index, _)| *index);
    messages.into_iter().map(|(_, message)| message).collect()
}

fn content_text(content: &Value, warnings: &mut HashSet<String>) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts.iter().filter_map(|part| {
            if let Some(text) = part.get("text").and_then(Value::as_str) { Some(text.to_owned()) }
            else { warnings.insert("Image/audio references and structured messages are preserved in hermes-source.json. The native conversation imports text parts; linked attachment files stay at their original paths.".to_owned()); None }
        }).collect::<Vec<_>>().join("\n"),
        _ => {
            warnings.insert("Structured content is preserved in hermes-source.json and shown as JSON in the native conversation.".to_owned());
            content.to_string()
        },
    }
}

fn convert_history(archive: &HermesArchive) -> (History, usize, Vec<String>) {
    let mut history = History::default();
    let mut warnings = HashSet::new();
    let messages = visible_messages(archive);
    let count = messages.len();
    for message in messages {
        let model = archive
            .sessions
            .iter()
            .find(|session| session.id == message.session_id)
            .and_then(|session| session.model.clone())
            .unwrap_or_else(|| "imported-hermes".to_owned());
        let text = content_text(&message.content, &mut warnings);
        if message.role == "user" {
            if text.trim().is_empty() {
                warnings.insert("An upstream user message has no text; its complete original content is retained in hermes-source.json.".to_owned());
            }
            history.turns.push(Turn { context: String::new(),
                attachments: vec![],
                prompt: text,
                model,
                entries: vec![],
                status: Status::Complete,
            });
            continue;
        }
        if !matches!(message.role.as_str(), "assistant" | "tool" | "function") {
            warnings.insert("System and non-chat rows are preserved in hermes-source.json and are not installed as native runtime instructions.".to_owned());
            continue;
        }
        if history.turns.is_empty() {
            history.turns.push(Turn { context: String::new(),
                attachments: vec![],
                prompt: "[Imported Hermes conversation begins here]".to_owned(),
                model,
                entries: vec![],
                status: Status::Complete,
            });
            warnings.insert("The upstream transcript begins without a user message; a labeled opening turn was added.".to_owned());
        }
        let turn = history.turns.last_mut().unwrap();
        if message.role == "assistant" {
            let mut calls = vec![];
            if let Some(raw_calls) = message.tool_calls.as_array() {
                for (index, call) in raw_calls.iter().enumerate() {
                    let function = call.get("function").unwrap_or(call);
                    let Some(name) = function.get("name").and_then(Value::as_str) else {
                        warnings.insert("Malformed historical tool calls are retained only in hermes-source.json.".to_owned());
                        continue;
                    };
                    let arguments = match function.get("arguments") {
                        Some(Value::String(text)) => text.clone(),
                        Some(value) => value.to_string(),
                        None => "{}".to_owned(),
                    };
                    calls.push(ToolCall {
                        id: call
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("imported-{}-{index}", message.id)),
                        name: name.to_owned(),
                        arguments,
                    });
                }
            } else if !message.tool_calls.is_null() {
                warnings.insert(
                    "Malformed historical tool calls are retained only in hermes-source.json."
                        .to_owned(),
                );
            }
            turn.entries.push(Entry::Assistant { text, calls });
        } else {
            warnings.insert("Historical tool results are recorded with approval=false because Hermes history does not prove native user approval. Import never executes tools.".to_owned());
            let id = message
                .tool_call_id
                .clone()
                .unwrap_or_else(|| format!("imported-tool-{}", message.id));
            let name = message
                .tool_name
                .clone()
                .or_else(|| {
                    turn.entries.iter().find_map(|entry| match entry {
                        Entry::Assistant { calls, .. } => calls
                            .iter()
                            .find(|call| call.id == id)
                            .map(|call| call.name.clone()),
                        _ => None,
                    })
                })
                .unwrap_or_else(|| "imported_tool".to_owned());
            turn.entries.push(Entry::Tool {
                id,
                name,
                approved: false,
                result: text,
            });
        }
    }
    for turn in &mut history.turns {
        if turn.entries.is_empty()
            || matches!(turn.entries.last(), Some(Entry::Assistant { calls, .. }) if !calls.is_empty())
            || matches!(turn.entries.last(), Some(Entry::Tool { .. }))
        {
            turn.status = Status::Interrupted;
        }
    }
    warnings.insert("Upstream provider settings and workspace execution permission are not activated. Select a workspace explicitly before continuing.".to_owned());
    let mut warnings: Vec<_> = warnings.into_iter().collect();
    warnings.sort();
    (history, count, warnings)
}
