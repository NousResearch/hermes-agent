use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tempfile::NamedTempFile;

// Existing externally edited memories may exceed the model budget. Preserve
// them; this separate transport bound rejects exceptional files without edits.
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

pub(super) fn private_dir(path: &Path) -> Result<PathBuf> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "Memory storage must be a real directory"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path).context("Could not create profile memory storage")?
        }
        Err(error) => return Err(error).context("Could not inspect profile memory storage"),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    fs::canonicalize(path).context("Could not locate profile memory storage")
}

pub(super) fn lock(dir: &Path, name: &str) -> Result<File> {
    let metadata = fs::symlink_metadata(dir)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "Profile memory directory was replaced"
    );
    let path = dir.join(format!("{name}.lock"));
    checked_path(&path)?;
    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&path)
        .context("Could not open profile memory lock")?;
    verify(&path, &file)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => return Ok(file),
            Err(error) if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() => {
                ensure!(
                    Instant::now() < deadline,
                    "Profile memory storage is busy; retry shortly"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error).context("Could not lock profile memory storage"),
        }
    }
}

pub(super) fn read(dir: &Path, name: &str) -> Result<String> {
    let path = dir.join(name);
    if !checked_path(&path)? {
        return Ok(String::new());
    }
    let file = File::open(&path)
        .context("Memory file exists but could not be read; nothing was changed")?;
    verify(&path, &file)?;
    ensure!(
        file.metadata()?.len() <= MAX_FILE_BYTES,
        "Existing memory file exceeds the native read bound; nothing was changed"
    );
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_FILE_BYTES,
        "Existing memory file exceeds the native read bound; nothing was changed"
    );
    let text = String::from_utf8(bytes)
        .context("Memory file contains invalid UTF-8; nothing was changed")?;
    Ok(text
        .strip_prefix('\u{feff}')
        .unwrap_or(&text)
        .replace("\r\n", "\n")
        .replace('\r', "\n"))
}

pub(super) fn write(dir: &Path, name: &str, content: &str) -> Result<()> {
    let path = dir.join(name);
    checked_path(&path)?;
    let mut file =
        NamedTempFile::new_in(dir).context("Could not create private memory temporary file")?;
    file.write_all(content.as_bytes())?;
    file.as_file().sync_all()?;
    file.persist(&path)
        .context("Could not atomically save profile memory")?;
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    Ok(())
}

pub(super) fn backup(dir: &Path, name: &str, raw: &str) -> Result<String> {
    let timestamp = chrono::Utc::now().timestamp();
    for index in 0..1024 {
        let suffix = if index == 0 {
            String::new()
        } else {
            format!(".{index}")
        };
        let path = dir.join(format!("{name}.bak.{timestamp}{suffix}"));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(mut file) => {
                file.write_all(raw.as_bytes())?;
                file.sync_all()?;
                return Ok(path.to_string_lossy().into_owned());
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .context("Could not snapshot memory drift; existing memory was preserved");
            }
        }
    }
    anyhow::bail!("Could not allocate a unique memory drift backup; existing memory was preserved")
}

fn checked_path(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "Memory path must be a regular file, not a link"
            );
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("Could not inspect profile memory file"),
    }
}
fn verify(path: &Path, file: &File) -> Result<()> {
    let current = fs::symlink_metadata(path)?;
    ensure!(
        current.is_file()
            && !current.file_type().is_symlink()
            && same_file(&current, &file.metadata()?),
        "Memory path changed while opening it"
    );
    Ok(())
}
#[cfg(unix)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}
#[cfg(not(unix))]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.is_file()
        && right.is_file()
        && left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
}
