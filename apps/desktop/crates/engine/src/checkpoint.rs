//! Opt-in bounded checkpoints for approved native file writes and safe manual
//! rollback. Stock 0284ce17 uses a shadow Git tree; this native subset snapshots
//! the exact affected UTF-8 file, never touching the project's Git repository.
//! Restores preserve later edits, require an explicit confirmation, and create
//! a pre-rollback checkpoint so a restore can itself be undone.
use anyhow::{Context, Result, bail, ensure};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use chrono::Utc;
use fs2::FileExt;
use ring::digest::{SHA256, digest};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use tempfile::NamedTempFile;
const MAX_FILE: usize = 1_048_576;
const MAX_STORE: usize = 64 * 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(1);
fn identifier() -> String {
    format!(
        "{:016x}{:08x}",
        Utc::now().timestamp_nanos_opt().unwrap_or_default() as u64,
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
fn stamp() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
fn hash(bytes: &[u8]) -> String {
    digest(&SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
    max: usize,
}
#[derive(Serialize, Deserialize, Default)]
struct Document {
    #[serde(default)]
    records: Vec<Record>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Record {
    id: String,
    created_at: String,
    label: String,
    workspace: PathBuf,
    path: PathBuf,
    before: Option<String>,
    after_hash: Option<String>,
    after_bytes: usize,
    mode: Option<u32>,
    root_identity: Option<(u64, u64)>,
    ready: bool,
    restored_at: Option<String>,
}
#[derive(Clone, Debug)]
pub struct Summary {
    pub id: String,
    pub created_at: String,
    pub label: String,
    pub workspace: PathBuf,
    pub path: PathBuf,
    pub before_bytes: usize,
    pub after_bytes: usize,
    pub ready: bool,
    pub restored: bool,
}
#[derive(Clone, Debug)]
pub struct Token {
    pub id: String,
}
#[derive(Clone, Debug)]
pub struct RestorePlan {
    pub id: String,
    pub workspace: PathBuf,
    pub path: PathBuf,
    pub description: String,
    pub preview: String,
    fingerprint: String,
}
impl Store {
    pub fn open(profile_root: &Path, max_snapshots: usize) -> Result<Self> {
        ensure!(
            (1..=100).contains(&max_snapshots),
            "Checkpoint retention must be between 1 and 100"
        );
        let root = profile_root.canonicalize()?.join("checkpoints");
        if let Ok(metadata) = fs::symlink_metadata(&root) {
            ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "Checkpoint directory must not be a symbolic link"
            );
        }
        fs::create_dir_all(&root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self {
            root,
            max: max_snapshots,
        })
    }
    fn read(&self) -> Result<Document> {
        let path = self.root.join("native-files.json");
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Document::default());
            }
            Err(error) => return Err(error.into()),
        };
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "Checkpoint index must be a regular file"
        );
        ensure!(
            metadata.len() <= MAX_STORE as u64,
            "Checkpoint index exceeds its limit"
        );
        let mut bytes = Vec::new();
        File::open(path)?
            .take((MAX_STORE + 1) as u64)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= MAX_STORE,
            "Checkpoint index exceeds its limit"
        );
        serde_json::from_slice(&bytes)
            .context("Checkpoint index is unreadable; it was left unchanged")
    }
    fn transaction<T>(&self, operation: impl FnOnce(&mut Document) -> Result<T>) -> Result<T> {
        let path = self.root.join("native-files.lock");
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "Checkpoint lock must not be a symbolic link"
            );
        }
        let mut options = fs::OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let lock = options.open(path)?;
        lock.lock_exclusive()?;
        let mut document = self.read()?;
        let value = operation(&mut document)?;
        let bytes = serde_json::to_vec(&document)?;
        ensure!(
            bytes.len() <= MAX_STORE,
            "Checkpoint storage is full; the file was left untouched"
        );
        let mut temp = NamedTempFile::new_in(&self.root)?;
        temp.write_all(&bytes)?;
        temp.as_file().sync_all()?;
        temp.persist(self.root.join("native-files.json"))?;
        Ok(value)
    }
    pub fn capture(
        &self,
        workspace: &Path,
        path: &Path,
        before: Option<&str>,
        after: Option<&str>,
    ) -> Result<Token> {
        self.capture_labeled(workspace, path, before, after, "before write_file")
    }
    fn capture_labeled(
        &self,
        workspace: &Path,
        path: &Path,
        before: Option<&str>,
        after: Option<&str>,
        label: &str,
    ) -> Result<Token> {
        ensure!(
            before.is_some() || after.is_some(),
            "A checkpoint needs a file change"
        );
        ensure!(
            before.is_none_or(|text| text.len() <= MAX_FILE)
                && after.is_none_or(|text| text.len() <= MAX_FILE),
            "File exceeds the 1 MiB checkpoint limit"
        );
        let workspace = workspace.canonicalize()?;
        let path = relative(path)?;
        ensure!(
            !workspace.join(&path).starts_with(&self.root),
            "The checkpoint store cannot checkpoint itself"
        );
        let directory = Dir::open_ambient_dir(&workspace, ambient_authority())?;
        let actual = read_current(&directory, &path)?;
        ensure!(
            actual.as_deref() == before,
            "File changed before the checkpoint was captured; it was left untouched"
        );
        #[cfg(unix)]
        let mode = {
            use cap_std::fs::MetadataExt;
            directory
                .symlink_metadata(&path)
                .ok()
                .map(|metadata| metadata.mode())
        };
        #[cfg(not(unix))]
        let mode = None;
        let record = Record {
            id: identifier(),
            created_at: stamp(),
            label: label.into(),
            workspace,
            path,
            before: before.map(str::to_owned),
            after_hash: after.map(|text| hash(text.as_bytes())),
            after_bytes: after.map_or(0, str::len),
            mode,
            root_identity: identity(&directory)?,
            ready: false,
            restored_at: None,
        };
        let token = Token {
            id: record.id.clone(),
        };
        self.transaction(|document| {
            document.records.push(record);
            while document.records.len() > self.max {
                let index = document
                    .records
                    .iter()
                    .position(|record| record.ready || record.restored_at.is_some())
                    .context("Checkpoint retention is full of pending writes")?;
                document.records.remove(index);
            }
            Ok(())
        })?;
        Ok(token)
    }
    pub fn commit(&self, token: &Token) -> Result<()> {
        self.transaction(|document| {
            let record = document
                .records
                .iter_mut()
                .find(|record| record.id == token.id)
                .context("Checkpoint no longer exists")?;
            let directory = open_workspace(record)?;
            verify_after(record, &directory)?;
            record.ready = true;
            Ok(())
        })
    }
    pub fn list(&self) -> Result<Vec<Summary>> {
        let mut records = self.read()?.records;
        records.reverse();
        Ok(records
            .into_iter()
            .map(|record| {
                let recovered = record.ready
                    || open_workspace(&record)
                        .is_ok_and(|directory| verify_after(&record, &directory).is_ok());
                Summary {
                    id: record.id,
                    created_at: record.created_at,
                    label: record.label,
                    workspace: record.workspace,
                    path: record.path,
                    before_bytes: record.before.as_ref().map_or(0, String::len),
                    after_bytes: record.after_bytes,
                    ready: recovered,
                    restored: record.restored_at.is_some(),
                }
            })
            .collect())
    }
    pub fn prepare_restore(&self, id: &str) -> Result<RestorePlan> {
        let record = self
            .read()?
            .records
            .into_iter()
            .find(|record| record.id == id)
            .context("Checkpoint not found")?;
        ensure!(
            record.restored_at.is_none(),
            "This checkpoint was already restored"
        );
        let directory = open_workspace(&record)?;
        verify_after(&record, &directory)?;
        let description = if let Some(before) = &record.before {
            format!(
                "Restore {} to its previous {} bytes. Current contents will be replaced only if they still match the approved write.",
                record.path.display(),
                before.len()
            )
        } else {
            format!(
                "Remove {} created by this approved write. It will be removed only if its contents are still unchanged.",
                record.path.display()
            )
        };
        let mut preview: String = record
            .before
            .as_deref()
            .map(|text| text.chars().take(16000).collect())
            .unwrap_or_else(|| "This file did not exist before the write.".into());
        if record
            .before
            .as_ref()
            .is_some_and(|text| text.chars().count() > 16000)
        {
            preview.push_str("\n\n[Preview truncated; restore uses the complete saved file.]");
        }
        let fingerprint = hash(&serde_json::to_vec(&record)?);
        Ok(RestorePlan {
            id: record.id,
            workspace: record.workspace,
            path: record.path,
            description,
            preview,
            fingerprint,
        })
    }
    pub fn restore(&self, plan: &RestorePlan, confirmed: bool) -> Result<Summary> {
        ensure!(
            confirmed,
            "Explicit confirmation is required before restoring files"
        );
        let record = self
            .read()?
            .records
            .into_iter()
            .find(|record| record.id == plan.id)
            .context("Checkpoint not found")?;
        ensure!(
            hash(&serde_json::to_vec(&record)?) == plan.fingerprint,
            "Checkpoint changed since the preview; review it again"
        );
        ensure!(
            record.restored_at.is_none(),
            "This checkpoint was already restored"
        );
        let directory = open_workspace(&record)?;
        verify_after(&record, &directory)?;
        let current = read_current(&directory, &record.path)?;
        let undo = self.capture_labeled(
            &record.workspace,
            &record.path,
            current.as_deref(),
            record.before.as_deref(),
            "before rollback",
        )?;
        verify_after(&record, &directory)?;
        match &record.before {
            Some(before) => atomic_write(
                &directory,
                &record.path,
                before,
                record.mode,
                record.after_hash.as_deref(),
            )?,
            None => {
                let parent = open_parent(&directory, &record.path)?;
                let name = record.path.file_name().context("Invalid checkpoint file")?;
                verify_current(&parent, Path::new(name), record.after_hash.as_deref())?;
                parent.remove_file(name)?;
            }
        }
        let restored = read_current(&directory, &record.path)?;
        ensure!(
            restored == record.before,
            "File changed during rollback; review the workspace before continuing"
        );
        self.commit(&undo)?;
        self.transaction(|document| {
            if let Some(original) = document
                .records
                .iter_mut()
                .find(|item| item.id == record.id)
            {
                original.restored_at = Some(stamp());
            }
            Ok(())
        })?;
        Ok(Summary {
            id: record.id,
            created_at: record.created_at,
            label: record.label,
            workspace: record.workspace,
            path: record.path,
            before_bytes: record.before.as_ref().map_or(0, String::len),
            after_bytes: record.after_bytes,
            ready: true,
            restored: true,
        })
    }
}
fn relative(path: &Path) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for part in path.components() {
        match part {
            Component::Normal(part) => result.push(part),
            Component::CurDir => {}
            _ => bail!("Checkpoint path is outside its workspace"),
        }
    }
    ensure!(
        !result.as_os_str().is_empty() && result.as_os_str().len() <= 4096,
        "Checkpoint must name a workspace file"
    );
    Ok(result)
}
fn check_links(directory: &Dir, path: &Path) -> Result<()> {
    let mut prefix = PathBuf::new();
    for part in relative(path)?.components() {
        prefix.push(part);
        match directory.symlink_metadata(&prefix) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink(),
                "Checkpoint restore does not follow symbolic links"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}
fn read_current(directory: &Dir, path: &Path) -> Result<Option<String>> {
    check_links(directory, path)?;
    let metadata = match directory.symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_FILE as u64,
        "Current path is not a bounded regular file"
    );
    let file = directory.open(path)?;
    ensure!(file.metadata()?.is_file(), "Current path changed");
    let mut bytes = Vec::new();
    file.take((MAX_FILE + 1) as u64).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_FILE,
        "Current file exceeds the checkpoint limit"
    );
    Ok(Some(
        String::from_utf8(bytes).context("Current file is not UTF-8 text")?,
    ))
}
fn identity(directory: &Dir) -> Result<Option<(u64, u64)>> {
    #[cfg(unix)]
    {
        use cap_std::fs::MetadataExt;
        let metadata = directory.metadata(".")?;
        Ok(Some((metadata.dev(), metadata.ino())))
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        Ok(None)
    }
}
fn open_workspace(record: &Record) -> Result<Dir> {
    ensure!(
        record.workspace.is_absolute(),
        "Invalid checkpoint workspace"
    );
    ensure!(
        record
            .before
            .as_ref()
            .is_none_or(|text| text.len() <= MAX_FILE)
            && record.after_bytes <= MAX_FILE,
        "Checkpoint file exceeds the 1 MiB restore limit"
    );
    ensure!(
        record.after_hash.as_ref().is_none_or(
            |hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        ),
        "Invalid checkpoint content fingerprint"
    );
    let directory = Dir::open_ambient_dir(&record.workspace, ambient_authority())
        .context("Checkpoint workspace is unavailable")?;
    ensure!(
        record.root_identity.is_none() || identity(&directory)? == record.root_identity,
        "The original workspace was replaced; refusing rollback into a different directory"
    );
    check_links(&directory, &record.path)?;
    Ok(directory)
}
fn verify_after(record: &Record, directory: &Dir) -> Result<()> {
    verify_current(directory, &record.path, record.after_hash.as_deref())
}
fn verify_current(directory: &Dir, path: &Path, expected_hash: Option<&str>) -> Result<()> {
    let current = read_current(directory, path)?;
    let actual_hash = current.as_ref().map(|text| hash(text.as_bytes()));
    ensure!(
        actual_hash.as_deref() == expected_hash,
        "File changed after Hermes wrote it. Later edits were left untouched"
    );
    Ok(())
}
fn open_parent(directory: &Dir, path: &Path) -> Result<Dir> {
    check_links(directory, path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(directory.open_dir(if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    })?)
}
fn atomic_write(
    directory: &Dir,
    path: &Path,
    content: &str,
    mode: Option<u32>,
    expected_hash: Option<&str>,
) -> Result<()> {
    let parent = open_parent(directory, path)?;
    let destination = path.file_name().context("Invalid checkpoint file")?;
    let name = format!(".hermes-rollback-{}.tmp", identifier());
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = parent.open_with(&name, &options)?;
    let result = (|| -> Result<()> {
        file.write_all(content.as_bytes())?;
        #[cfg(unix)]
        if let Some(mode) = mode {
            use cap_std::fs::PermissionsExt;
            file.set_permissions(cap_std::fs::Permissions::from_mode(mode))?;
        }
        #[cfg(not(unix))]
        let _ = mode;
        file.sync_all()?;
        drop(file);
        verify_current(&parent, Path::new(destination), expected_hash)?;
        if expected_hash.is_some() {
            parent.rename(&name, &parent, destination)?;
        } else {
            parent.hard_link(&name, &parent, destination)?;
        }
        Ok(())
    })();
    let _ = parent.remove_file(&name);
    result
}
