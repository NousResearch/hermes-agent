//! Downloaded artifact validation and atomic cache writes.
//!
//! Install-time downloads use this module so partially-written files and
//! checksum mismatches cannot poison the bootstrap cache.

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// HTTP artifact download request.
#[derive(Debug, Clone)]
pub struct DownloadSpec<'a> {
    pub url: String,
    pub user_agent: &'a str,
    pub expected_sha256: Option<&'a str>,
}

/// Download an artifact and cache it with optional SHA-256 verification.
pub async fn download_to_cache(spec: DownloadSpec<'_>, dest_path: &Path) -> Result<()> {
    let response = reqwest::Client::new()
        .get(&spec.url)
        .header("User-Agent", spec.user_agent)
        .send()
        .await
        .with_context(|| format!("GET {}", spec.url))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download artifact: HTTP {} from {}",
            response.status(),
            spec.url
        ));
    }

    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("reading body of {}", spec.url))?;

    write_atomic_verified(dest_path, &bytes, spec.expected_sha256).await
}

/// Return the lowercase SHA-256 digest for `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Atomically write bytes to `dest_path` after optional SHA-256 verification.
pub async fn write_atomic_verified(
    dest_path: &Path,
    bytes: &[u8],
    expected_sha256: Option<&str>,
) -> Result<()> {
    if let Some(expected) = expected_sha256 {
        let actual = sha256_hex(bytes);
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(anyhow!(
                "checksum mismatch for {}: expected {}, got {}",
                dest_path.display(),
                expected,
                actual
            ));
        }
    }

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating artifact parent dir {}", parent.display())
        })?;
    }

    let tmp_path = temp_path(dest_path);
    let backup_path = backup_path(dest_path);
    remove_file_if_exists(&tmp_path).await?;
    remove_file_if_exists(&backup_path).await?;

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("creating temp file {}", tmp_path.display()))?;
    file.write_all(bytes)
        .await
        .with_context(|| format!("writing temp file {}", tmp_path.display()))?;
    file.flush().await.context("flushing temp file")?;
    drop(file);

    if dest_path.exists() {
        tokio::fs::rename(dest_path, &backup_path).await.with_context(|| {
            format!(
                "backing up {} to {}",
                dest_path.display(),
                backup_path.display()
            )
        })?;
    }

    match tokio::fs::rename(&tmp_path, dest_path).await {
        Ok(()) => {
            remove_file_if_exists(&backup_path).await?;
            Ok(())
        }
        Err(err) => {
            if backup_path.exists() {
                let _ = tokio::fs::rename(&backup_path, dest_path).await;
            }
            Err(err).with_context(|| {
                format!(
                    "renaming {} to {}",
                    tmp_path.display(),
                    dest_path.display()
                )
            })
        }
    }
}

/// Extract a ZIP archive into `dest_dir`, rejecting entries that escape it.
pub fn extract_zip_archive(archive_path: &Path, dest_dir: &Path) -> Result<Vec<PathBuf>> {
    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("opening zip archive {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading zip archive {}", archive_path.display()))?;
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("creating extract dir {}", dest_dir.display()))?;

    let mut extracted = Vec::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("reading zip entry {index}"))?;
        let enclosed = entry
            .enclosed_name()
            .ok_or_else(|| anyhow!("unsafe zip entry: {}", entry.name()))?;
        if entry.is_symlink() {
            return Err(anyhow!("unsupported symlink zip entry: {}", entry.name()));
        }
        let out_path = dest_dir.join(enclosed);

        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)
                .with_context(|| format!("creating zip dir {}", out_path.display()))?;
            continue;
        }

        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating zip parent dir {}", parent.display()))?;
        }

        let mut out_file = std::fs::File::create(&out_path)
            .with_context(|| format!("creating extracted file {}", out_path.display()))?;
        std::io::copy(&mut entry.by_ref(), &mut out_file)
            .with_context(|| format!("extracting {}", out_path.display()))?;
        extracted.push(out_path);
    }

    Ok(extracted)
}

fn temp_path(dest_path: &Path) -> PathBuf {
    dest_path.with_extension({
        let ext = dest_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("tmp");
        format!("{ext}.tmp")
    })
}

fn backup_path(dest_path: &Path) -> PathBuf {
    dest_path.with_extension({
        let ext = dest_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("bak");
        format!("{ext}.bak")
    })
}

async fn remove_file_if_exists(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("removing {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hermes-artifact-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_test_zip(path: &std::path::Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        for (name, bytes) in entries {
            zip.start_file(*name, SimpleFileOptions::default())
                .unwrap();
            zip.write_all(bytes).unwrap();
        }
        zip.finish().unwrap();
    }

    fn write_symlink_zip(path: &std::path::Path, name: &str, target: &[u8]) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let target = std::str::from_utf8(target).unwrap();
        zip.add_symlink(name, target, SimpleFileOptions::default())
            .unwrap();
        zip.finish().unwrap();
    }

    #[test]
    fn sha256_hex_matches_known_content() {
        assert_eq!(
            sha256_hex(b"hermes"),
            "8cfde6efdfc4ed5ab1f6acbbd1ba49bf31932f84d0a4c090eb41c7d151e8b180"
        );
    }

    #[tokio::test]
    async fn atomic_write_verified_replaces_destination() {
        let root = unique_tmp_dir("replace");
        let dest = root.join("artifact.txt");
        std::fs::write(&dest, b"old").unwrap();
        let expected = sha256_hex(b"new");

        write_atomic_verified(&dest, b"new", Some(&expected))
            .await
            .unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"new");
        assert!(!dest.with_extension("txt.tmp").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn atomic_write_verified_rejects_checksum_mismatch_without_destination() {
        let root = unique_tmp_dir("reject");
        let dest = root.join("artifact.txt");

        let err = write_atomic_verified(&dest, b"bad", Some(&"0".repeat(64)))
            .await
            .unwrap_err();

        assert!(err.to_string().contains("checksum mismatch"));
        assert!(!dest.exists());
        assert!(!dest.with_extension("txt.tmp").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn extract_zip_archive_writes_nested_files() {
        let root = unique_tmp_dir("zip-ok");
        let archive = root.join("repo.zip");
        let dest = root.join("out");
        write_test_zip(&archive, &[("repo-main/README.md", b"ok")]);

        let extracted = extract_zip_archive(&archive, &dest).unwrap();

        assert_eq!(extracted, vec![dest.join("repo-main").join("README.md")]);
        assert_eq!(
            std::fs::read(dest.join("repo-main").join("README.md")).unwrap(),
            b"ok"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn extract_zip_archive_rejects_parent_traversal_entries() {
        let root = unique_tmp_dir("zip-traversal");
        let archive = root.join("repo.zip");
        let dest = root.join("out");
        write_test_zip(&archive, &[("../escape.txt", b"bad")]);

        let err = extract_zip_archive(&archive, &dest).unwrap_err();

        assert!(err.to_string().contains("unsafe zip entry"));
        assert!(!root.join("escape.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn extract_zip_archive_rejects_symlink_entries() {
        let root = unique_tmp_dir("zip-symlink");
        let archive = root.join("repo.zip");
        let dest = root.join("out");
        write_symlink_zip(&archive, "repo-main/link", b"/etc/passwd");

        let err = extract_zip_archive(&archive, &dest).unwrap_err();

        assert!(err.to_string().contains("symlink"));
        assert!(!dest.join("repo-main").join("link").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
