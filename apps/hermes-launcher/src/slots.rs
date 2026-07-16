//! Slot management — versioned slots + atomic flip.
//!
//! The on-disk layout (§2.2):
//!   $HERMES_HOME/
//!   ├── versions/
//!   │   ├── 1.42.0/          # unpacked bundle (immutable after verify)
//!   │   └── 1.43.0/
//!   ├── current.txt          # THE commit point: one line, the active version
//!   ├── previous.txt         # instant rollback target
//!   └── current -> versions/1.43.0   # convenience symlink (best-effort)
//!
//! The flip is a file rename-over — atomic on every platform (POSIX rename(),
//! Windows MoveFileExW(MOVEFILE_REPLACE_EXISTING)). One mechanism, no per-
//! platform commit logic to diverge.

use anyhow::{bail, Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Read the active version from `current.txt`.
/// This is the ONE reader for the current version — nothing else should
/// parse `current.txt` directly.
pub fn resolve_current(hermes_home: &Path) -> Result<Option<String>> {
    let current_txt = hermes_home.join("current.txt");
    if !current_txt.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&current_txt)
        .with_context(|| format!("cannot read {}", current_txt.display()))?;
    let version = content.trim().to_string();
    if version.is_empty() {
        Ok(None)
    } else {
        Ok(Some(version))
    }
}

/// Read the previous version from `previous.txt`.
pub fn resolve_previous(hermes_home: &Path) -> Result<Option<String>> {
    let prev_txt = hermes_home.join("previous.txt");
    if !prev_txt.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&prev_txt)
        .with_context(|| format!("cannot read {}", prev_txt.display()))?;
    let version = content.trim().to_string();
    if version.is_empty() {
        Ok(None)
    } else {
        Ok(Some(version))
    }
}

/// The slot root directory: `versions/`
pub fn versions_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("versions")
}

/// Path for a specific version's slot: `versions/<version>`
pub fn slot_path(hermes_home: &Path, version: &str) -> PathBuf {
    versions_dir(hermes_home).join(version)
}

/// Staging path for a version being downloaded/unpacked: `versions/<version>.staging`
pub fn staging_path(hermes_home: &Path, version: &str) -> PathBuf {
    versions_dir(hermes_home).join(format!("{}.staging", version))
}

/// Create the staging directory for a version.
/// Caller is responsible for unpacking the bundle into it.
pub fn stage(hermes_home: &Path, version: &str) -> Result<PathBuf> {
    let staging = staging_path(hermes_home, version);
    if staging.exists() {
        // Clean up any leftover staging from a previous interrupted attempt
        fs::remove_dir_all(&staging)
            .with_context(|| format!("cannot remove stale staging {}", staging.display()))?;
    }
    fs::create_dir_all(&staging)
        .with_context(|| format!("cannot create staging dir {}", staging.display()))?;
    Ok(staging)
}

/// Commit staging: fsync the directory, then rename to the final slot path.
/// The slot is immutable after this point.
pub fn commit_staging(hermes_home: &Path, version: &str) -> Result<PathBuf> {
    let staging = staging_path(hermes_home, version);
    let target = slot_path(hermes_home, version);

    if !staging.exists() {
        bail!("staging directory does not exist: {}", staging.display());
    }

    // If the target already exists (re-install of same version), remove it first.
    if target.exists() {
        fs::remove_dir_all(&target)
            .with_context(|| format!("cannot remove existing slot {}", target.display()))?;
    }

    // fsync the staging directory to ensure all file contents are on disk.
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let dir = fs::File::open(&staging)
            .with_context(|| format!("cannot open staging dir for fsync"))?;
        let _ = nix::unistd::fsync(dir.as_raw_fd());
    }

    // Rename staging → final slot path.
    fs::rename(&staging, &target).with_context(|| {
        format!(
            "cannot rename {} to {}",
            staging.display(),
            target.display()
        )
    })?;

    Ok(target)
}

/// THE atomic flip: replace `current.txt` with the new version string.
///
/// 1. Write `current.txt.new` with the new version
/// 2. fsync the file
/// 3. Rename over `current.txt` (atomic on every platform)
/// 4. Update `previous.txt` with the old version
/// 5. Refresh the `current` convenience symlink (best-effort, POSIX only)
///
/// Nothing load-bearing reads the symlink — `resolve_current` is the only reader.
pub fn flip(hermes_home: &Path, new_version: &str) -> Result<()> {
    let current_txt = hermes_home.join("current.txt");
    let previous_txt = hermes_home.join("previous.txt");
    let new_txt = hermes_home.join("current.txt.new");

    // Read the old current version (for previous.txt)
    let old_version = resolve_current(hermes_home).unwrap_or(None);

    // Write the new version to a temp file
    let mut file = fs::File::create(&new_txt)
        .with_context(|| format!("cannot create {}", new_txt.display()))?;
    writeln!(file, "{}", new_version)?;
    file.sync_all().context("cannot fsync current.txt.new")?;
    drop(file);

    // Atomic rename over current.txt
    fs::rename(&new_txt, &current_txt).with_context(|| format!("cannot flip current.txt"))?;

    // Update previous.txt with the old version
    if let Some(old) = old_version {
        fs::write(&previous_txt, format!("{}\n", old))
            .with_context(|| format!("cannot write {}", previous_txt.display()))?;
    }

    // Refresh the convenience symlink (best-effort, POSIX only)
    #[cfg(unix)]
    {
        let symlink = hermes_home.join("current");
        let target = slot_path(hermes_home, new_version);
        // Remove old symlink if it exists (don't fail if it doesn't)
        let _ = fs::remove_file(&symlink);
        // Create new symlink (best-effort — don't fail the flip if this fails)
        let _ = std::os::unix::fs::symlink(&target, &symlink);
    }

    Ok(())
}

/// Rollback: rewrite `current.txt` from `previous.txt`.
/// Swaps current ↔ previous.
pub fn rollback(hermes_home: &Path) -> Result<String> {
    let prev = resolve_previous(hermes_home)?
        .ok_or_else(|| anyhow::anyhow!("no previous version to roll back to"))?;

    let current = resolve_current(hermes_home).unwrap_or(None);

    // Flip to the previous version
    flip(hermes_home, &prev)?;

    // Update previous.txt to point at what was current before rollback
    if let Some(curr) = current {
        let previous_txt = hermes_home.join("previous.txt");
        fs::write(&previous_txt, format!("{}\n", curr))
            .with_context(|| format!("cannot write {}", previous_txt.display()))?;
    }

    Ok(prev)
}

/// Garbage-collect old slots, keeping the N most recent (always keeping
/// the targets of `current` and `previous`).
pub fn gc(hermes_home: &Path, keep_n: usize) -> Result<Vec<String>> {
    let versions_dir = versions_dir(hermes_home);
    if !versions_dir.exists() {
        return Ok(Vec::new());
    }

    let current = resolve_current(hermes_home).unwrap_or(None);
    let previous = resolve_previous(hermes_home).unwrap_or(None);

    // Collect all version directories (exclude .staging dirs)
    let mut slots: Vec<(String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&versions_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".staging") {
            continue;
        }
        if entry.path().is_dir() {
            slots.push((name, entry.path()));
        }
    }

    // Sort by name (version strings sort chronologically for calver)
    slots.sort_by(|a, b| a.0.cmp(&b.0));

    // Keep the last N, plus current and previous
    let to_keep: std::collections::HashSet<String> = slots
        .iter()
        .rev()
        .take(keep_n)
        .map(|(v, _)| v.clone())
        .chain(current.into_iter())
        .chain(previous.into_iter())
        .collect();

    let mut removed = Vec::new();
    for (version, path) in &slots {
        if !to_keep.contains(version) {
            if let Err(e) = fs::remove_dir_all(path) {
                eprintln!("warn: cannot remove old slot {}: {}", path.display(), e);
            } else {
                removed.push(version.clone());
            }
        }
    }

    Ok(removed)
}

/// Clean up any stale `.staging` directories.
pub fn cleanup_stale_staging(hermes_home: &Path) -> Result<Vec<String>> {
    let versions_dir = versions_dir(hermes_home);
    if !versions_dir.exists() {
        return Ok(Vec::new());
    }

    let mut removed = Vec::new();
    for entry in fs::read_dir(&versions_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".staging") && entry.path().is_dir() {
            if let Err(e) = fs::remove_dir_all(entry.path()) {
                eprintln!(
                    "warn: cannot remove staging {}: {}",
                    entry.path().display(),
                    e
                );
            } else {
                removed.push(name);
            }
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_current_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(resolve_current(tmp.path()).unwrap(), None);
    }

    #[test]
    fn test_flip_sets_current() {
        let tmp = tempfile::tempdir().unwrap();
        flip(tmp.path(), "1.0.0").unwrap();
        assert_eq!(
            resolve_current(tmp.path()).unwrap(),
            Some("1.0.0".to_string())
        );
    }

    #[test]
    fn test_flip_updates_previous() {
        let tmp = tempfile::tempdir().unwrap();
        flip(tmp.path(), "1.0.0").unwrap();
        flip(tmp.path(), "2.0.0").unwrap();
        assert_eq!(
            resolve_current(tmp.path()).unwrap(),
            Some("2.0.0".to_string())
        );
        assert_eq!(
            resolve_previous(tmp.path()).unwrap(),
            Some("1.0.0".to_string())
        );
    }

    #[test]
    fn test_rollback() {
        let tmp = tempfile::tempdir().unwrap();
        flip(tmp.path(), "1.0.0").unwrap();
        flip(tmp.path(), "2.0.0").unwrap();
        let rolled = rollback(tmp.path()).unwrap();
        assert_eq!(rolled, "1.0.0");
        assert_eq!(
            resolve_current(tmp.path()).unwrap(),
            Some("1.0.0".to_string())
        );
        assert_eq!(
            resolve_previous(tmp.path()).unwrap(),
            Some("2.0.0".to_string())
        );
    }

    #[test]
    fn test_rollback_fails_without_previous() {
        let tmp = tempfile::tempdir().unwrap();
        flip(tmp.path(), "1.0.0").unwrap();
        assert!(rollback(tmp.path()).is_err());
    }

    #[test]
    fn test_stage_creates_staging_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = stage(tmp.path(), "1.0.0").unwrap();
        assert!(staging.exists());
        assert!(staging.to_string_lossy().ends_with("1.0.0.staging"));
    }

    #[test]
    fn test_stage_cleans_leftover() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a leftover staging dir
        let staging = staging_path(tmp.path(), "1.0.0");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("junk"), "old").unwrap();
        // Stage again — should clean and recreate
        let staging = stage(tmp.path(), "1.0.0").unwrap();
        assert!(!staging.join("junk").exists());
        assert!(staging.exists());
    }

    #[test]
    fn test_commit_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = stage(tmp.path(), "1.0.0").unwrap();
        fs::write(staging.join("manifest.json"), "{}").unwrap();
        let slot = commit_staging(tmp.path(), "1.0.0").unwrap();
        assert!(slot.exists());
        assert!(slot.join("manifest.json").exists());
        // Staging dir should be gone
        assert!(!staging_path(tmp.path(), "1.0.0").exists());
    }

    #[test]
    fn test_commit_staging_fails_without_staging() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(commit_staging(tmp.path(), "1.0.0").is_err());
    }

    #[test]
    fn test_gc_keeps_current_and_previous() {
        let tmp = tempfile::tempdir().unwrap();
        // Create 3 slots
        for v in ["1.0.0", "2.0.0", "3.0.0"] {
            let staging = stage(tmp.path(), v).unwrap();
            fs::write(staging.join("manifest.json"), "{}").unwrap();
            commit_staging(tmp.path(), v).unwrap();
        }
        // Flip to 3.0.0, with 2.0.0 as previous
        flip(tmp.path(), "1.0.0").unwrap();
        flip(tmp.path(), "2.0.0").unwrap();
        flip(tmp.path(), "3.0.0").unwrap();

        // GC with keep_n=1 — should remove 1.0.0 but keep 2.0.0 (previous) and 3.0.0 (current)
        let removed = gc(tmp.path(), 1).unwrap();
        assert_eq!(removed, vec!["1.0.0".to_string()]);
        assert!(slot_path(tmp.path(), "2.0.0").exists());
        assert!(slot_path(tmp.path(), "3.0.0").exists());
    }

    #[test]
    fn test_cleanup_stale_staging() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(staging_path(tmp.path(), "1.0.0")).unwrap();
        fs::create_dir_all(staging_path(tmp.path(), "2.0.0")).unwrap();
        let removed = cleanup_stale_staging(tmp.path()).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(!staging_path(tmp.path(), "1.0.0").exists());
        assert!(!staging_path(tmp.path(), "2.0.0").exists());
    }

    #[test]
    fn test_flip_is_atomic_no_partial_state() {
        // After a successful flip, current.txt contains exactly the new version.
        // There's no intermediate state where current.txt is empty or partial.
        let tmp = tempfile::tempdir().unwrap();
        flip(tmp.path(), "1.0.0").unwrap();
        let content = fs::read_to_string(tmp.path().join("current.txt")).unwrap();
        assert_eq!(content.trim(), "1.0.0");
        // The .new file should not exist (it was renamed)
        assert!(!tmp.path().join("current.txt.new").exists());
    }
}
