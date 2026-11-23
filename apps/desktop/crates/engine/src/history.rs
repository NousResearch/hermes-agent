use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{fs::{self, File, OpenOptions}, io::Write, path::{Path, PathBuf}};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Entry {
    Assistant { text: String, calls: Vec<ToolCall> },
    Tool { id: String, name: String, approved: bool, result: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", content = "reason", rename_all = "snake_case")]
pub enum Status { Running, Complete, Interrupted, Failed(String) }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Turn {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<crate::Attachment>,
    pub prompt: String,
    /// What the prompt's @-references expanded to when it was sent (agent/context_references.py
    /// "--- Attached Context ---"). Frozen with the turn so later rounds resend identical bytes.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub context: String,
    pub model: String,
    pub entries: Vec<Entry>,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct History {
    version: u32,
    pub turns: Vec<Turn>,
}

impl Default for History {
    fn default() -> Self { Self { version: 1, turns: Vec::new() } }
}

pub struct Store {
    dir: PathBuf,
    // Hold an exclusive process lock for the lifetime of this conversation.
    _lock: File,
}

impl Store {
    pub fn default_dir() -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("HERMES_RUST_DATA_DIR") { return Ok(path.into()); }
        let dirs = ProjectDirs::from("org", "HermesRust", "Hermes Rust Desktop")
            .context("Could not locate the application data directory")?;
        Ok(dirs.data_local_dir().to_path_buf())
    }

    pub fn open(dir: &Path) -> Result<(Self, History)> {
        fs::create_dir_all(dir).context("Could not create the conversation directory")?;
        let lock = OpenOptions::new().create(true).truncate(false).read(true).write(true)
            .open(dir.join("session.lock")).context("Could not open the conversation lock")?;
        lock.try_lock_exclusive().context("This conversation is already open in another window or cannot be locked")?;
        let path = dir.join("conversation.json");
        let mut history: History = match File::open(&path) {
            Ok(file) => serde_json::from_reader(file).context("Could not read the saved conversation")?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => History::default(),
            Err(err) => return Err(err).context("Could not open the saved conversation"),
        };
        if history.version != 1 { bail!("Unsupported conversation format {}", history.version); }
        let store = Self { dir: dir.to_path_buf(), _lock: lock };
        let mut recovered = false;
        for turn in &mut history.turns {
            if turn.status == Status::Running {
                turn.status = Status::Interrupted;
                recovered = true;
            }
        }
        if recovered { store.save(&history)?; }
        Ok((store, history))
    }

    pub fn save(&self, history: &History) -> Result<()> {
        let mut file = NamedTempFile::new_in(&self.dir)?;
        serde_json::to_writer_pretty(&mut file, history)?;
        file.write_all(b"\n")?;
        file.as_file().sync_all()?;
        file.persist(self.dir.join("conversation.json")).context("Could not replace the saved conversation")?;
        #[cfg(unix)]
        File::open(&self.dir)?.sync_all()?;
        Ok(())
    }
}
