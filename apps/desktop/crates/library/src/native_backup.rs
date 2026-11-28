//! Versioned native profile snapshots. This is deliberately a distinct format
//! from stock whole-Hermes-home backups and Hermes Cloud transfer archives.
//! Export reads only named native files. Restore adds new, detached sessions and
//! inert recovery copies; it never replaces settings, workspaces, jobs or keys.
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use ring::digest::{SHA256, digest};
use std::{collections::HashMap, io::Read};

const FORMAT: &str = "hermes.native.profile-backup";
const MAX_ARCHIVE: u64 = 192 * 1024 * 1024;
const MAX_FILE: u64 = 32 * 1024 * 1024;
const MAX_CONTENT: u64 = 128 * 1024 * 1024;
const MAX_SESSIONS: usize = 10_000;
const MAX_FILES: usize = 50_000;
const PROFILE_FILES: &[&str] = &[
    "agent-settings.json",
    "desktop-preferences.json",
    "statusbar-preferences.json",
    "desktop-keybinds.json",
    "project-discovery.json",
    "providers/model-options.json",
    "memories/MEMORY.md",
    "memories/USER.md",
    "cron/jobs.json",
];
const SESSION_FILES: &[&str] = &[
    "conversation.json",
    "hermes-source.json",
    "draft.txt",
    "draft-attachments.json",
    "queued-prompts.json",
];

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Archive {
    format: String,
    version: u32,
    created_at: String,
    projects: Vec<Project>,
    #[serde(default)]
    pinned_order: Vec<String>,
    sessions: Vec<Snapshot>,
    profile: Vec<Blob>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    record: SessionRecord,
    files: Vec<Blob>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Blob {
    path: String,
    bytes: u64,
    sha256: String,
    base64: String,
}

#[derive(Debug, Clone)]
pub struct NativeBackupSummary {
    pub created_at: String,
    pub sessions: usize,
    pub projects: usize,
    pub files: usize,
    pub content_bytes: u64,
}
/// Validated data held between inspection and the explicit import click. The
/// source path is never re-opened after inspection, so a changed source file
/// cannot replace the archive that the user reviewed.
pub struct NativeBackupPlan {
    archive: Archive,
    summary: NativeBackupSummary,
    digest: String,
}
impl NativeBackupPlan {
    pub fn summary(&self) -> &NativeBackupSummary {
        &self.summary
    }
}
#[derive(Debug, Clone)]
pub struct NativeRestoreReport {
    pub sessions: usize,
    pub recovery_directory: PathBuf,
}

impl Library {
    /// An export destination must be a new file. Even an explicit save-panel
    /// selection never silently overwrites an existing file.
    pub fn export_native_backup(&self, destination: &Path) -> Result<NativeBackupSummary> {
        ensure!(
            !destination.exists(),
            "Choose a new backup filename; existing files are retained"
        );
        let mut content = 0u64;
        let archive = self.with_catalog(false, |catalog| {
            ensure!(
                catalog.sessions.len() <= MAX_SESSIONS,
                "Too many sessions for a native backup"
            );
            let mut sessions = Vec::new();
            for record in &catalog.sessions {
                let directory = self.checked_directory(&record.id)?;
                let mut files = Vec::new();
                for path in SESSION_FILES {
                    if let Some(blob) = capture(&directory, path, &mut content)? {
                        files.push(blob);
                    }
                }
                ensure!(
                    files.iter().any(|blob| blob.path == "conversation.json"),
                    "Session {} has no saved conversation",
                    record.id
                );
                let attachments = directory.join("attachments");
                let attachment_metadata = match fs::symlink_metadata(&attachments) {
                    Ok(metadata) => Some(metadata),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                };
                if let Some(metadata) = attachment_metadata {
                    ensure!(
                        metadata.file_type().is_dir(),
                        "Attachment snapshots must be a regular directory"
                    );
                    let mut names = Vec::new();
                    for entry in fs::read_dir(&attachments)? {
                        let entry = entry?;
                        let name = entry
                            .file_name()
                            .into_string()
                            .map_err(|_| anyhow::anyhow!("Invalid attachment filename"))?;
                        ensure!(
                            hex_digest(&name),
                            "Unsupported attachment snapshot filename"
                        );
                        names.push(name);
                    }
                    names.sort();
                    for name in names {
                        files.push(
                            capture(&directory, &format!("attachments/{name}"), &mut content)?
                                .context("An attachment disappeared during backup")?,
                        );
                    }
                }
                sessions.push(Snapshot {
                    record: record.clone(),
                    files,
                });
            }
            let mut profile = Vec::new();
            for path in PROFILE_FILES {
                if let Some(blob) = capture(&self.root, path, &mut content)? {
                    profile.push(blob);
                }
            }
            Ok(Archive {
                format: FORMAT.into(),
                version: 1,
                created_at: now(),
                projects: catalog.projects.clone(),
                pinned_order: catalog.pinned_order.clone(),
                sessions,
                profile,
            })
        })?;
        let summary = validate(&archive)?;
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .context("Choose a full backup destination path")?;
        ensure!(
            parent.is_dir(),
            "The backup destination folder does not exist"
        );
        let mut temporary = NamedTempFile::new_in(parent)?;
        serde_json::to_writer(temporary.as_file_mut(), &archive)?;
        ensure!(
            temporary.as_file().metadata()?.len() <= MAX_ARCHIVE,
            "Native archive exceeds the 192 MiB limit"
        );
        temporary.as_file_mut().write_all(b"\n")?;
        temporary.as_file().sync_all()?;
        temporary
            .persist_noclobber(destination)
            .context("Could not save a new native backup file")?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(summary)
    }

    /// Add all sessions as new IDs with empty private workspaces. Original
    /// project paths are recovery metadata only and are never opened or linked.
    /// Queued prompts and settings remain inert under the recovery directory.
    pub fn import_native_backup(&self, plan: &NativeBackupPlan) -> Result<NativeRestoreReport> {
        validate(&plan.archive)?;
        let import_id = new_id("backup");
        let recovery_root = self.root.join("backup-imports");
        private_directory(&recovery_root)?;
        let recovery = recovery_root.join(&import_id);
        fs::create_dir(&recovery)?;
        private_directory(&recovery)?;
        let mut snapshot = NamedTempFile::new_in(&recovery)?;
        serde_json::to_writer(
            snapshot.as_file_mut(),
            &serde_json::json!({
                "format":FORMAT,"version":1,"source_sha256":plan.digest,"created_at":plan.archive.created_at,
                "imported_at":now(),"projects":plan.archive.projects,
                "scope":"New detached sessions; settings, memory and queued work are recovery copies only. No credentials or workspace files."
            }),
        )?;
        snapshot.as_file().sync_all()?;
        snapshot.persist_noclobber(recovery.join("manifest.json"))?;
        let profile = recovery.join("profile");
        private_directory(&profile)?;
        for blob in &plan.archive.profile {
            write_blob(&profile, blob, &blob.path)?;
        }
        let sessions = self.with_catalog(true, |catalog| {
            let mut imported = 0;
            let mut pins = HashMap::new();
            for saved in &plan.archive.sessions {
                let mut record = saved.record.clone();
                let original_id = record.id.clone();
                record.id = new_id("chat");
                record.project_id = None;
                let directory = self.create_directory(&record.id)?;
                for blob in &saved.files {
                    if matches!(blob.path.as_str(), "queued-prompts.json") {
                        let queue = recovery.join("sessions").join(&record.id);
                        private_directory(&queue)?;
                        write_blob(&queue, blob, &blob.path)?;
                    } else {
                        write_blob(&directory, blob, &blob.path)?;
                    }
                }
                // Preserve lineage without treating the old path as an active
                // workspace or using it as an executable/open-file target.
                atomic_json(
                    &directory.join("backup-source.json"),
                    &serde_json::json!({
                        "archive_sha256":plan.digest,"original_session_id":original_id,
                        "original_project_id":saved.record.project_id,"imported_at":now()
                    }),
                )?;
                if record.pinned {
                    pins.insert(saved.record.id.clone(), record.id.clone());
                }
                catalog.sessions.push(record);
                imported += 1;
            }
            for original in plan.archive.pinned_order.iter().chain(
                plan.archive
                    .sessions
                    .iter()
                    .filter(|session| session.record.pinned)
                    .map(|session| &session.record.id),
            ) {
                if let Some(id) = pins.remove(original) {
                    catalog.pinned_order.push(id);
                }
            }
            // No active-session or selected-project mutation is made here.
            Ok(imported)
        })?;
        atomic_json(
            &recovery.join("complete.json"),
            &serde_json::json!({"sessions":sessions,"completed_at":now()}),
        )?;
        Ok(NativeRestoreReport {
            sessions,
            recovery_directory: recovery,
        })
    }
}

pub fn inspect_native_backup(path: &Path) -> Result<NativeBackupPlan> {
    let bytes = bounded_file(path, MAX_ARCHIVE)?;
    let archive: Archive = serde_json::from_slice(&bytes)
        .context("Choose a version 1 native .hermes-native.json backup")?;
    let summary = validate(&archive)?;
    Ok(NativeBackupPlan {
        archive,
        summary,
        digest: sha256(&bytes),
    })
}

fn capture(root: &Path, path: &str, total: &mut u64) -> Result<Option<Blob>> {
    let full = checked_file(root, path)?;
    let bytes = match fs::symlink_metadata(&full) {
        Ok(_) => bounded_file(&full, MAX_FILE)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    *total = total
        .checked_add(bytes.len() as u64)
        .context("Backup size overflow")?;
    ensure!(
        *total <= MAX_CONTENT,
        "Native backup exceeds the 128 MiB content limit"
    );
    Ok(Some(Blob {
        path: path.into(),
        bytes: bytes.len() as u64,
        sha256: sha256(&bytes),
        base64: STANDARD.encode(bytes),
    }))
}
fn bounded_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).context("Could not inspect backup input")?;
    ensure!(
        metadata.file_type().is_file() && metadata.len() <= limit,
        "Backup input must be a regular file within the native size limit"
    );
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "Backup input grew beyond the native size limit"
    );
    Ok(bytes)
}
fn checked_file(root: &Path, path: &str) -> Result<PathBuf> {
    // All filenames first pass the fixed schema allowlists. Check every
    // intermediate component too: a symlink to a workspace is never traversed.
    ensure!(safe_relative(path), "Unsafe native backup path");
    let mut current = root.to_path_buf();
    let components: Vec<_> = Path::new(path).components().collect();
    for component in &components[..components.len() - 1] {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) => ensure!(
                metadata.file_type().is_dir(),
                "Native backup input directories must not be symbolic links"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(root.join(path));
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(root.join(path))
}
fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && Path::new(path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}
fn hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}
fn sha256(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut value = String::with_capacity(64);
    for byte in digest(&SHA256, bytes).as_ref() {
        write!(value, "{byte:02x}").unwrap();
    }
    value
}
fn decode(blob: &Blob) -> Result<Vec<u8>> {
    ensure!(
        blob.bytes <= MAX_FILE && blob.base64.len() as u64 <= MAX_FILE.div_ceil(3) * 4,
        "Backup file exceeds the native size limit"
    );
    let bytes = STANDARD
        .decode(&blob.base64)
        .context("Invalid backup file encoding")?;
    ensure!(
        bytes.len() as u64 == blob.bytes && sha256(&bytes) == blob.sha256,
        "Backup file size or checksum does not match"
    );
    Ok(bytes)
}
fn validate(archive: &Archive) -> Result<NativeBackupSummary> {
    ensure!(
        archive.format == FORMAT && archive.version == 1,
        "Unsupported native backup format or version"
    );
    ensure!(
        chrono::DateTime::parse_from_rfc3339(&archive.created_at).is_ok(),
        "Invalid backup creation time"
    );
    ensure!(
        archive.sessions.len() <= MAX_SESSIONS && archive.projects.len() <= MAX_SESSIONS,
        "Native backup has too many sessions or projects"
    );
    let mut projects = HashSet::new();
    for project in &archive.projects {
        ensure!(
            safe_id(&project.id) && projects.insert(&project.id) && project.path.is_absolute(),
            "Invalid backup project metadata"
        );
    }
    let mut ids = HashSet::new();
    let mut total = 0u64;
    let mut file_count = 0usize;
    for saved in &archive.sessions {
        let record = &saved.record;
        ensure!(
            safe_id(&record.id)
                && ids.insert(&record.id)
                && !record.title.trim().is_empty()
                && record.title.chars().count() <= 200,
            "Invalid backup conversation metadata"
        );
        ensure!(
            record
                .project_id
                .as_ref()
                .is_none_or(|id| projects.contains(id)),
            "Backup session references an unknown project"
        );
        ensure!(
            chrono::DateTime::parse_from_rfc3339(&record.created_at).is_ok()
                && chrono::DateTime::parse_from_rfc3339(&record.updated_at).is_ok(),
            "Invalid backup conversation time"
        );
        let mut files = HashMap::new();
        for blob in &saved.files {
            ensure!(
                SESSION_FILES.contains(&blob.path.as_str())
                    || blob
                        .path
                        .strip_prefix("attachments/")
                        .is_some_and(hex_digest),
                "Unsupported backup conversation filename"
            );
            ensure!(
                files.insert(blob.path.as_str(), decode(blob)?).is_none(),
                "Duplicate file in backup conversation"
            );
            total = total
                .checked_add(blob.bytes)
                .context("Backup size overflow")?;
            file_count += 1;
        }
        let transcript: serde_json::Value = serde_json::from_slice(
            files
                .get("conversation.json")
                .context("Backup conversation has no transcript")?,
        )
        .context("Invalid native transcript in backup")?;
        ensure!(
            transcript
                .get("version")
                .and_then(serde_json::Value::as_u64)
                == Some(1),
            "Unsupported backup transcript version"
        );
        let history: History =
            serde_json::from_value(transcript).context("Invalid native transcript in backup")?;
        for turn in &history.turns {
            hermes_engine::Attachment::validate_set(&turn.attachments)?;
            for attachment in &turn.attachments {
                let value = serde_json::to_value(attachment)?;
                let key = value
                    .get("digest")
                    .and_then(serde_json::Value::as_str)
                    .context("Invalid backup attachment reference")?;
                let blob = files
                    .get(format!("attachments/{key}").as_str())
                    .context("Backup is missing an attachment snapshot")?;
                ensure!(
                    sha256(blob) == key
                        && value.get("bytes").and_then(serde_json::Value::as_u64)
                            == Some(blob.len() as u64),
                    "Backup attachment snapshot has changed"
                );
            }
        }
    }
    let mut pinned = HashSet::new();
    for id in &archive.pinned_order {
        ensure!(
            pinned.insert(id)
                && archive
                    .sessions
                    .iter()
                    .any(|session| &session.record.id == id && session.record.pinned),
            "Invalid backup pin order"
        );
    }
    let mut names = HashSet::new();
    for blob in &archive.profile {
        ensure!(
            PROFILE_FILES.contains(&blob.path.as_str()) && names.insert(&blob.path),
            "Unsupported or duplicate native profile filename"
        );
        decode(blob)?;
        total = total
            .checked_add(blob.bytes)
            .context("Backup size overflow")?;
        file_count += 1;
    }
    ensure!(
        total <= MAX_CONTENT && file_count <= MAX_FILES,
        "Native backup exceeds its content or file count limits"
    );
    Ok(NativeBackupSummary {
        created_at: archive.created_at.clone(),
        sessions: archive.sessions.len(),
        projects: archive.projects.len(),
        files: file_count,
        content_bytes: total,
    })
}
fn private_directory(path: &Path) -> Result<()> {
    if path.exists() {
        ensure!(
            fs::symlink_metadata(path)?.file_type().is_dir(),
            "Backup recovery directories must be regular directories"
        );
    }
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
fn write_blob(root: &Path, blob: &Blob, target: &str) -> Result<()> {
    ensure!(safe_relative(target), "Unsafe backup recovery path");
    let path = root.join(target);
    let parent = path.parent().context("Backup file has no folder")?;
    private_directory(parent)?;
    let mut temp = NamedTempFile::new_in(parent)?;
    temp.write_all(&decode(blob)?)?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(&path)
        .context("Could not add a new backup recovery file")?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}
