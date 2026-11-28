//! Durable native conversation catalog and explicit, read-only Hermes history import.
//!
//! The catalog never holds an engine conversation lock. Callers open the directory
//! returned by [`Library::session_dir`] with `hermes_engine::Session` and call
//! [`Library::touch_session`] after a turn to refresh its title and activity time.
mod bots;
pub mod webhooks;
pub mod profiles;
mod import;
mod lifecycle;
mod native_backup;

pub use import::{HermesArchive, HermesHistory, HermesMessage, HermesSession, ImportReport};
pub use bots::{Bot, Group, BOT_CHAT_TITLE};
pub use lifecycle::{ArchiveFilter, SessionQuery};
pub use native_backup::{NativeBackupPlan, NativeBackupSummary, NativeRestoreReport, inspect_native_backup};

use anyhow::{Context, Result, bail, ensure};
use chrono::{SecondsFormat, Utc};
use fs2::FileExt;
use hermes_engine::History;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Project {
    pub id: String,
    pub name: String,
    /// An existing, canonical workspace directory selected by the user.
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionSource {
    Native,
    /// A copy of the pre-catalog Rust conversation, preserving its original file.
    LegacyRust,
    Hermes {
        database: PathBuf,
        session_id: String,
        imported_at: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub project_id: Option<String>,
    pub directory: PathBuf,
    pub created_at: String,
    pub updated_at: String,
    /// Time of the last completed turn. `None` for a chat that has no messages.
    #[serde(default)]
    pub last_active: Option<String>,
    /// The bot whose canonical "Bot Chat" this is. Bot chats stay out of the Sessions list.
    #[serde(default)]
    pub bot: Option<String>,
    pub source: SessionSource,
    #[serde(default)]
    pub archived: bool,
    #[serde(default)]
    pub pinned: bool,
}

impl SessionSummary {
    /// The desktop sidebar's recency key: last message time, falling back to
    /// creation time (`max(last_active, started_at)` in the original app).
    /// Renaming, pinning and archiving do not move a session.
    pub fn recency(&self) -> &str {
        match &self.last_active {
            Some(active) if active.as_str() > self.created_at.as_str() => active,
            _ => &self.created_at,
        }
    }
    /// A native chat nobody has sent a message in yet.
    pub fn is_empty_draft(&self) -> bool {
        self.last_active.is_none() && matches!(self.source, SessionSource::Native)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionRecord {
    id: String,
    title: String,
    project_id: Option<String>,
    created_at: String,
    updated_at: String,
    #[serde(default)]
    last_active: Option<String>,
    #[serde(default)]
    bot: Option<String>,
    source: SessionSource,
    #[serde(default)]
    archived: bool,
    #[serde(default)]
    pinned: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Catalog {
    version: u32,
    projects: Vec<Project>,
    sessions: Vec<SessionRecord>,
    selected_project: Option<String>,
    active_session: Option<String>,
    #[serde(default)]
    pinned_order: Vec<String>,
    #[serde(default)]
    pending_deletion: Vec<String>,
    /// Set once `last_active` has been backfilled for pre-existing sessions.
    #[serde(default)]
    activity_migrated: bool,
}

impl Default for Catalog {
    fn default() -> Self {
        Self {
            version: 1,
            projects: vec![],
            sessions: vec![],
            selected_project: None,
            active_session: None,
            pinned_order: vec![], pending_deletion:vec![], activity_migrated: true,
        }
    }
}

/// A lightweight handle; clones share an atomically saved, briefly locked catalog.
#[derive(Debug, Clone)]
pub struct Library {
    root: PathBuf,
}

impl Library {
    pub fn open_default() -> Result<Self> {
        Self::open(hermes_engine::Session::default_dir()?)
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(root.as_ref())
            .context("Could not create the native library directory")?;
        let root = fs::canonicalize(root.as_ref())
            .context("Could not resolve the native library directory")?;
        let library = Self { root };
        let sessions = library.root.join("sessions");
        if sessions.exists() {
            ensure!(
                !fs::symlink_metadata(&sessions)?.file_type().is_symlink(),
                "The library sessions directory must not be a symbolic link"
            );
        }
        fs::create_dir_all(&sessions)?;
        library.with_catalog(true, |catalog| {
            // Upgrade the original one-conversation Rust build by copying its
            // transcript. The old conversation and lock file remain untouched.
            if catalog.sessions.is_empty() && !library.root.join("library.json").exists() {
                let old = library.root.join("conversation.json");
                if old.is_file() {
                    let history: History = serde_json::from_reader(File::open(old)?).context(
                        "Could not read the original Rust conversation for catalog migration",
                    )?;
                    let mut record = library.new_record(
                        None,
                        title_from_history(&history).as_deref(),
                        SessionSource::LegacyRust,
                    );
                    let directory = library.create_directory(&record.id)?;
                    if let Err(error) = atomic_json(&directory.join("conversation.json"), &history)
                    {
                        let _ = fs::remove_dir(&directory);
                        return Err(error);
                    }
                    record.updated_at = record.created_at.clone();
                    catalog.active_session = Some(record.id.clone());
                    catalog.sessions.push(record);
                }
            }
            if !catalog.activity_migrated {
                for record in &mut catalog.sessions {
                    if record.last_active.is_some() { continue; }
                    let has_turns = match record.source {
                        SessionSource::Native => File::open(library.root.join("sessions").join(&record.id).join("conversation.json")).ok()
                            .and_then(|file| serde_json::from_reader::<_, History>(file).ok())
                            .is_some_and(|history| !history.turns.is_empty()),
                        _ => true,
                    };
                    if has_turns { record.last_active = Some(record.updated_at.clone()); }
                }
                catalog.activity_migrated = true;
            }
            Ok(())
        })?;
        library.finish_deletions()?;
        Ok(library)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `None` lists every project; `Some(id)` filters to that workspace.
    pub fn sessions(&self, project_id: Option<&str>) -> Result<Vec<SessionSummary>> {
        self.list_sessions(&SessionQuery {
            project_id: project_id.map(str::to_owned),
            ..Default::default()
        })
    }

    pub fn create_session(
        &self,
        project_id: Option<&str>,
        title: Option<&str>,
    ) -> Result<SessionSummary> {
        let default_project = if project_id.is_none(){
            hermes_engine::settings::ProfileSettings::load(&self.root)?.default_directory
                .map(|path|self.add_project(&path)).transpose()?
        }else{None};
        let project_id=project_id.or_else(||default_project.as_ref().map(|p|p.id.as_str()));
        self.with_catalog(true, |catalog| {
            if let Some(id) = project_id {
                ensure_project(catalog, id)?;
            }
            if title.is_none() && let Some(existing) = catalog.sessions.iter().find(|record| {
                record.last_active.is_none() && record.bot.is_none() && matches!(record.source, SessionSource::Native) && !record.archived && !record.pinned
                    && record.title == "New chat" && record.project_id.as_deref() == project_id
                    && self.root.join("sessions").join(&record.id).join("conversation.json").is_file()
            }) {
                let summary = self.summary(existing);
                catalog.active_session = Some(existing.id.clone());
                catalog.selected_project = existing.project_id.clone();
                return Ok(summary);
            }
            let record = self.new_record(project_id, title, SessionSource::Native);
            let directory = self.create_directory(&record.id)?;
            if let Err(error) =
                atomic_json(&directory.join("conversation.json"), &History::default())
            {
                let _ = fs::remove_dir(&directory);
                return Err(error);
            }
            catalog.active_session = Some(record.id.clone());
            catalog.selected_project = record.project_id.clone();
            let summary = self.summary(&record);
            catalog.sessions.push(record);
            Ok(summary)
        })
    }

    pub fn rename_session(&self, id: &str, title: &str) -> Result<SessionSummary> {
        let title = title.trim();
        ensure!(!title.is_empty(), "A conversation title cannot be empty");
        ensure!(
            title.chars().count() <= 200,
            "A conversation title must contain at most 200 characters"
        );
        self.with_catalog(true, |catalog| {
            let session = find_session_mut(catalog, id)?;
            session.title = title.to_owned();
            session.updated_at = now();
            Ok(self.summary(session))
        })
    }

    pub fn touch_session(&self, id: &str) -> Result<SessionSummary> {
        self.with_catalog(true, |catalog| {
            let session = find_session_mut(catalog, id)?;
            let path = self.checked_directory(id)?.join("conversation.json");
            let history: History = serde_json::from_reader(File::open(path)?)
                .context("Could not refresh the conversation title")?;
            if session.title == "New chat" && let Some(title) = title_from_history(&history) {
                session.title = title;
            }
            let stamp = now();
            if !history.turns.is_empty() { session.last_active = Some(stamp.clone()); }
            session.updated_at = stamp;
            Ok(self.summary(session))
        })
    }

    pub fn session(&self, id: &str) -> Result<SessionSummary> {
        self.with_catalog(
            false,
            |catalog| Ok(self.summary(find_session(catalog, id)?)),
        )
    }

    pub fn session_dir(&self, id: &str) -> Result<PathBuf> {
        self.with_catalog(false, |catalog| {
            find_session(catalog, id)?;
            self.checked_directory(id)
        })
    }

    pub fn workspace_for_session(&self, id: &str) -> Result<Option<PathBuf>> {
        self.with_catalog(false, |catalog| {
            let session = find_session(catalog, id)?;
            session
                .project_id
                .as_deref()
                .map(|id| ensure_project(catalog, id).map(|project| project.path.clone()))
                .transpose()
        })
    }

    pub fn active_session(&self) -> Result<Option<SessionSummary>> {
        self.with_catalog(false, |catalog| {
            catalog
                .active_session
                .as_deref()
                .map(|id| find_session(catalog, id).map(|session| self.summary(session)))
                .transpose()
        })
    }

    pub fn select_session(&self, id: &str) -> Result<()> {
        self.with_catalog(true, |catalog| {
            let project_id = find_session(catalog, id)?.project_id.clone();
            self.checked_directory(id)?;
            catalog.active_session = Some(id.to_owned());
            catalog.selected_project = project_id;
            Ok(())
        })
    }

    pub fn projects(&self) -> Result<Vec<Project>> {
        self.with_catalog(false, |catalog| Ok(catalog.projects.clone()))
    }

    pub fn add_project(&self, path: &Path) -> Result<Project> {
        let path = fs::canonicalize(path).context("Choose an existing workspace folder")?;
        ensure!(path.is_dir(), "The workspace must be an existing folder");
        self.with_catalog(true, |catalog| {
            if let Some(project) = catalog.projects.iter().find(|project| project.path == path) {
                return Ok(project.clone());
            }
            let project = Project {
                id: new_id("project"),
                name: path
                    .file_name()
                    .unwrap_or(path.as_os_str())
                    .to_string_lossy()
                    .into_owned(),
                path,
            };
            catalog.projects.push(project.clone());
            Ok(project)
        })
    }

    pub fn select_project(&self, id: Option<&str>) -> Result<()> {
        self.with_catalog(true, |catalog| {
            if let Some(id) = id {
                ensure_project(catalog, id)?;
            }
            catalog.selected_project = id.map(str::to_owned);
            catalog.active_session = catalog
                .sessions
                .iter()
                .filter(|session| !session.archived && session.project_id.as_deref() == id)
                .max_by(|a, b| a.updated_at.cmp(&b.updated_at))
                .map(|session| session.id.clone());
            Ok(())
        })
    }

    pub fn selected_project(&self) -> Result<Option<Project>> {
        self.with_catalog(false, |catalog| {
            catalog
                .selected_project
                .as_deref()
                .map(|id| ensure_project(catalog, id).cloned())
                .transpose()
        })
    }

    fn new_record(
        &self,
        project_id: Option<&str>,
        title: Option<&str>,
        source: SessionSource,
    ) -> SessionRecord {
        let timestamp = now();
        let title = title
            .map(normalize_title)
            .filter(|title| !title.is_empty())
            .unwrap_or_else(|| "New chat".to_owned());
        SessionRecord {
            id: new_id("chat"),
            title,
            project_id: project_id.map(str::to_owned),
            created_at: timestamp.clone(),
            last_active: (!matches!(source, SessionSource::Native)).then(|| timestamp.clone()),
            bot: None,
            updated_at: timestamp,
            source,
            archived: false,
            pinned: false,
        }
    }

    fn create_directory(&self, id: &str) -> Result<PathBuf> {
        let directory = self.root.join("sessions").join(id);
        fs::create_dir(&directory).context("Could not create the conversation directory")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
            File::open(self.root.join("sessions"))?.sync_all()?;
        }
        Ok(directory)
    }

    fn checked_directory(&self, id: &str) -> Result<PathBuf> {
        ensure!(safe_id(id), "Invalid native conversation identifier");
        let directory = self.root.join("sessions").join(id);
        ensure!(
            !fs::symlink_metadata(&directory)
                .context("The conversation directory is missing")?
                .file_type()
                .is_symlink(),
            "The conversation directory must not be a symbolic link"
        );
        ensure!(
            directory.is_dir(),
            "The conversation path is not a directory"
        );
        Ok(directory)
    }

    fn summary(&self, record: &SessionRecord) -> SessionSummary {
        SessionSummary {
            id: record.id.clone(),
            title: record.title.clone(),
            project_id: record.project_id.clone(),
            directory: self.root.join("sessions").join(&record.id),
            created_at: record.created_at.clone(),
            updated_at: record.updated_at.clone(),
            last_active: record.last_active.clone(),
            bot: record.bot.clone(),
            source: record.source.clone(),
            archived: record.archived,
            pinned: record.pinned,
        }
    }

    fn with_catalog<T>(&self, save: bool, f: impl FnOnce(&mut Catalog) -> Result<T>) -> Result<T> {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("library.lock"))?;
        lock.lock_exclusive()
            .context("Could not lock the native conversation catalog")?;
        let path = self.root.join("library.json");
        let mut catalog: Catalog = match File::open(&path) {
            Ok(file) => serde_json::from_reader(file)
                .context("Could not read the native conversation catalog")?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Catalog::default(),
            Err(error) => {
                return Err(error).context("Could not open the native conversation catalog");
            }
        };
        validate_catalog(&catalog)?;
        for session in &catalog.sessions {
            self.checked_directory(&session.id)?;
        }
        let result = f(&mut catalog)?;
        if save {
            validate_catalog(&catalog)?;
            atomic_json(&path, &catalog)
                .context("Could not save the native conversation catalog")?;
        }
        Ok(result)
    }
}

fn validate_catalog(catalog: &Catalog) -> Result<()> {
    ensure!(
        catalog.version == 1,
        "Unsupported native catalog format {}",
        catalog.version
    );
    let mut ids = HashSet::new();
    for project in &catalog.projects {
        ensure!(
            safe_id(&project.id) && ids.insert(&project.id),
            "Invalid or duplicate workspace identifier in catalog"
        );
        ensure!(
            project.path.is_absolute(),
            "Workspace paths in the catalog must be absolute"
        );
    }
    ids.clear();
    for session in &catalog.sessions {
        ensure!(
            safe_id(&session.id) && ids.insert(&session.id),
            "Invalid or duplicate conversation identifier in catalog"
        );
        if let Some(id) = session.project_id.as_deref() {
            ensure_project(catalog, id)?;
        }
    }
    if let Some(id) = catalog.selected_project.as_deref() {
        ensure_project(catalog, id)?;
    }
    if let Some(id) = catalog.active_session.as_deref() {
        find_session(catalog, id)?;
    }
    let mut pinned_ids = HashSet::new();
    for id in &catalog.pinned_order {
        ensure!(
            pinned_ids.insert(id) && find_session(catalog, id)?.pinned,
            "Invalid or duplicate pin order in the conversation catalog"
        );
    }
    Ok(())
}

fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}
fn ensure_project<'a>(catalog: &'a Catalog, id: &str) -> Result<&'a Project> {
    catalog
        .projects
        .iter()
        .find(|project| project.id == id)
        .context("Workspace does not exist in this library")
}
fn find_session<'a>(catalog: &'a Catalog, id: &str) -> Result<&'a SessionRecord> {
    catalog
        .sessions
        .iter()
        .find(|session| session.id == id)
        .context("Conversation does not exist in this library")
}
fn find_session_mut<'a>(catalog: &'a mut Catalog, id: &str) -> Result<&'a mut SessionRecord> {
    catalog
        .sessions
        .iter_mut()
        .find(|session| session.id == id)
        .context("Conversation does not exist in this library")
}
fn normalize_title(title: &str) -> String {
    title
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(200)
        .collect()
}
fn title_from_history(history: &History) -> Option<String> {
    history
        .turns
        .iter()
        .map(|turn| normalize_title(&turn.prompt))
        .find(|title| !title.is_empty())
        .map(|title| title.chars().take(80).collect())
}
fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true)
}
fn new_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{prefix}-{nanos:x}-{:x}-{:x}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("Missing parent directory")?;
    let mut file = NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(path)
        .with_context(|| format!("Could not replace {}", path.display()))?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests;
