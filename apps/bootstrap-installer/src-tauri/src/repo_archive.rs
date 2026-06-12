//! Repository ZIP archive fallback planning.
//!
//! This module owns the low-level pieces needed for a future no-Git fresh
//! install path: choosing an immutable GitHub archive URL and unpacking it into
//! the managed checkout directory without overwriting user data.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

/// Source marker written into archive-created checkouts.
pub const SOURCE_MARKER_NAME: &str = ".hermes-source.json";

/// GitHub repository archive selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoArchiveSpec {
    pub owner: String,
    pub repo: String,
    pub commit: Option<String>,
    pub branch: Option<String>,
}

impl RepoArchiveSpec {
    /// Build the GitHub ZIP archive URL, preferring immutable commit pins.
    pub fn github_zip_url(&self) -> Result<String> {
        let archive_ref = self.archive_ref()?;
        Ok(format!(
            "https://github.com/{}/{}/archive/{}.zip",
            self.owner, self.repo, archive_ref
        ))
    }

    fn archive_ref(&self) -> Result<&str> {
        if let Some(commit) = self.commit.as_deref().filter(|value| !value.trim().is_empty()) {
            return Ok(commit);
        }
        if let Some(branch) = self.branch.as_deref().filter(|value| !value.trim().is_empty()) {
            return Ok(branch);
        }
        Err(anyhow!("repo archive requires a commit or branch ref"))
    }
}

/// Download a GitHub repository archive and extract it into a fresh install root.
pub async fn download_and_extract_fresh(
    spec: &RepoArchiveSpec,
    cache_dir: &Path,
    install_root: &Path,
) -> Result<PathBuf> {
    let archive_path = archive_cache_path(cache_dir, spec)?;
    crate::artifact::download_to_cache(
        crate::artifact::DownloadSpec {
            url: spec.github_zip_url()?,
            user_agent: "hermes-setup/0.0.1",
            expected_sha256: None,
        },
        &archive_path,
    )
    .await
    .context("downloading repository archive")?;
    extract_repo_archive_to_install_root(&archive_path, install_root)?;
    Ok(archive_path)
}

/// Write the install source marker for an archive-created checkout.
pub fn write_archive_source_marker(
    install_root: &Path,
    spec: &RepoArchiveSpec,
    archive_path: &Path,
    git_initialized: bool,
) -> Result<serde_json::Value> {
    let marker = serde_json::json!({
        "schemaVersion": 1,
        "method": "github_archive",
        "owner": spec.owner,
        "repo": spec.repo,
        "ref": spec.archive_ref()?,
        "commit": spec.commit,
        "branch": spec.branch,
        "archive": archive_path,
        "gitInitialized": git_initialized,
    });
    let text = serde_json::to_string_pretty(&marker)
        .context("serializing repository source marker")?
        + "\n";
    let marker_path = install_root.join(SOURCE_MARKER_NAME);
    std::fs::write(&marker_path, text)
        .with_context(|| format!("writing repository source marker {}", marker_path.display()))?;
    Ok(marker)
}

/// Read the archive source marker when present.
pub fn read_archive_source_marker(install_root: &Path) -> Result<Option<serde_json::Value>> {
    let marker_path = install_root.join(SOURCE_MARKER_NAME);
    let text = match std::fs::read_to_string(&marker_path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err)
                .with_context(|| format!("reading repository source marker {}", marker_path.display()));
        }
    };
    serde_json::from_str(&text)
        .with_context(|| format!("parsing repository source marker {}", marker_path.display()))
        .map(Some)
}

/// Return the cache path for a repository ZIP archive.
pub fn archive_cache_path(cache_dir: &Path, spec: &RepoArchiveSpec) -> Result<PathBuf> {
    let archive_ref = sanitize_ref(spec.archive_ref()?);
    Ok(cache_dir.join(format!("{}-{}.zip", spec.repo, archive_ref)))
}

/// Extract a GitHub archive ZIP into `install_root`, stripping its single root dir.
///
/// The function intentionally refuses to write into an existing install root.
/// This keeps the first no-Git path safe while update semantics remain owned by
/// the existing Git-based scripts.
pub fn extract_repo_archive_to_install_root(archive_path: &Path, install_root: &Path) -> Result<()> {
    if install_root.exists() {
        return Err(anyhow!(
            "install root already exists: {}",
            install_root.display()
        ));
    }

    let parent = install_root.parent().ok_or_else(|| {
        anyhow!(
            "install root has no parent directory: {}",
            install_root.display()
        )
    })?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating install parent {}", parent.display()))?;

    let tmp_dir = archive_tmp_dir(install_root);
    remove_dir_if_exists(&tmp_dir)?;
    std::fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("creating archive temp dir {}", tmp_dir.display()))?;

    let result: Result<()> = (|| {
        crate::artifact::extract_zip_archive(archive_path, &tmp_dir)?;
        let archive_root = single_top_level_dir(&tmp_dir)?;
        std::fs::rename(&archive_root, install_root).with_context(|| {
            format!(
                "moving extracted repo {} to {}",
                archive_root.display(),
                install_root.display()
            )
        })?;
        Ok(())
    })();

    let cleanup = remove_dir_if_exists(&tmp_dir);
    result?;
    cleanup
}

/// Refresh an existing archive-created checkout from a GitHub repository ZIP.
///
/// This mirrors the legacy Python ZIP update contract: replace repository
/// files from the archive, but preserve runtime/user state that does not belong
/// to the source archive.
pub fn refresh_existing_checkout_from_archive(archive_path: &Path, install_root: &Path) -> Result<()> {
    if !install_root.is_dir() {
        return Err(anyhow!(
            "install root does not exist: {}",
            install_root.display()
        ));
    }

    let tmp_dir = archive_tmp_dir(install_root);
    remove_dir_if_exists(&tmp_dir)?;
    std::fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("creating archive temp dir {}", tmp_dir.display()))?;

    let result: Result<()> = (|| {
        crate::artifact::extract_zip_archive(archive_path, &tmp_dir)?;
        let archive_root = single_top_level_dir(&tmp_dir)?;
        for entry in std::fs::read_dir(&archive_root)
            .with_context(|| format!("reading archive root {}", archive_root.display()))?
        {
            let entry = entry.with_context(|| format!("reading entry under {}", archive_root.display()))?;
            let name = entry.file_name();
            if should_preserve_refresh_entry(&name) {
                continue;
            }
            let source = entry.path();
            let dest = install_root.join(&name);
            remove_path_if_exists(&dest)?;
            std::fs::rename(&source, &dest).with_context(|| {
                format!("moving refreshed repo entry {} to {}", source.display(), dest.display())
            })?;
        }
        Ok(())
    })();

    let cleanup = remove_dir_if_exists(&tmp_dir);
    result?;
    cleanup
}

fn should_preserve_refresh_entry(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some("venv" | "node_modules" | ".git" | ".env" | SOURCE_MARKER_NAME)
    )
}

fn archive_tmp_dir(install_root: &Path) -> PathBuf {
    let name = install_root
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("hermes-agent");
    install_root.with_file_name(format!("{name}.archive-tmp-{}", std::process::id()))
}

fn single_top_level_dir(tmp_dir: &Path) -> Result<PathBuf> {
    let entries = std::fs::read_dir(tmp_dir)
        .with_context(|| format!("reading archive temp dir {}", tmp_dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("reading entries under {}", tmp_dir.display()))?;
    if entries.len() != 1 {
        return Err(anyhow!(
            "repo archive must contain exactly one top-level directory, found {}",
            entries.len()
        ));
    }
    let path = entries[0].path();
    if !path.is_dir() {
        return Err(anyhow!(
            "repo archive top-level entry is not a directory: {}",
            path.display()
        ));
    }
    Ok(path)
}

fn remove_dir_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_dir_all(path) {
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

    #[test]
    fn archive_url_prefers_commit_over_branch() {
        let spec = RepoArchiveSpec {
            owner: "NousResearch".into(),
            repo: "hermes-agent".into(),
            commit: Some("02d26981d3d4ad50e142399b8476f59ad5953ff0".into()),
            branch: Some("main".into()),
        };

        assert_eq!(
            spec.github_zip_url().unwrap(),
            "https://github.com/NousResearch/hermes-agent/archive/02d26981d3d4ad50e142399b8476f59ad5953ff0.zip"
        );
    }

    #[test]
    fn archive_extract_strips_single_top_level_directory() {
        let root = std::env::temp_dir().join(format!(
            "hermes-repo-archive-strip-{}",
            std::process::id()
        ));
        let archive = root.join("repo.zip");
        let install_root = root.join("hermes-agent");
        std::fs::create_dir_all(&root).unwrap();
        write_test_zip(
            &archive,
            &[
                ("hermes-agent-main/README.md", b"ok"),
                ("hermes-agent-main/scripts/install.ps1", b"# install"),
            ],
        );

        extract_repo_archive_to_install_root(&archive, &install_root).unwrap();

        assert_eq!(std::fs::read(install_root.join("README.md")).unwrap(), b"ok");
        assert!(install_root.join("scripts").join("install.ps1").exists());
        assert!(!install_root.join("hermes-agent-main").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn archive_extract_refuses_to_overwrite_existing_install_root() {
        let root = std::env::temp_dir().join(format!(
            "hermes-repo-archive-existing-{}",
            std::process::id()
        ));
        let archive = root.join("repo.zip");
        let install_root = root.join("hermes-agent");
        std::fs::create_dir_all(&install_root).unwrap();
        std::fs::write(install_root.join("local.txt"), b"user").unwrap();
        write_test_zip(&archive, &[("hermes-agent-main/README.md", b"ok")]);

        let err = extract_repo_archive_to_install_root(&archive, &install_root).unwrap_err();

        assert!(err.to_string().contains("install root already exists"));
        assert_eq!(std::fs::read(install_root.join("local.txt")).unwrap(), b"user");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn archive_cache_path_sanitizes_refs() {
        let cache_dir = std::path::PathBuf::from("C:/cache");
        let spec = RepoArchiveSpec {
            owner: "NousResearch".into(),
            repo: "hermes-agent".into(),
            commit: None,
            branch: Some("feature/native repo".into()),
        };

        assert_eq!(
            archive_cache_path(&cache_dir, &spec).unwrap(),
            cache_dir.join("hermes-agent-feature_native_repo.zip")
        );
    }

    #[test]
    fn archive_source_marker_round_trips_install_source() {
        let root = std::env::temp_dir().join(format!(
            "hermes-repo-archive-source-{}",
            std::process::id()
        ));
        let install_root = root.join("hermes-agent");
        std::fs::create_dir_all(&install_root).unwrap();
        let spec = RepoArchiveSpec {
            owner: "NousResearch".into(),
            repo: "hermes-agent".into(),
            commit: Some("abcdef123".into()),
            branch: Some("main".into()),
        };

        let marker = write_archive_source_marker(
            &install_root,
            &spec,
            &root.join("hermes-agent-abcdef123.zip"),
            false,
        )
        .unwrap();
        let read_back = read_archive_source_marker(&install_root)
            .unwrap()
            .expect("marker should exist");

        assert_eq!(marker["schemaVersion"], 1);
        assert_eq!(read_back["method"], "github_archive");
        assert_eq!(read_back["ref"], "abcdef123");
        assert_eq!(read_back["branch"], "main");
        assert_eq!(read_back["gitInitialized"], false);
        let bytes = std::fs::read(install_root.join(SOURCE_MARKER_NAME)).unwrap();
        assert!(!bytes.starts_with(&[0xef, 0xbb, 0xbf]));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn archive_refresh_replaces_repo_files_and_preserves_runtime_state() {
        let root = std::env::temp_dir().join(format!(
            "hermes-repo-archive-refresh-{}",
            std::process::id()
        ));
        let archive = root.join("repo.zip");
        let install_root = root.join("hermes-agent");
        std::fs::create_dir_all(install_root.join("scripts")).unwrap();
        std::fs::create_dir_all(install_root.join("venv")).unwrap();
        std::fs::create_dir_all(install_root.join("node_modules")).unwrap();
        std::fs::write(install_root.join("README.md"), b"old").unwrap();
        std::fs::write(install_root.join("scripts").join("install.ps1"), b"old script").unwrap();
        std::fs::write(install_root.join("venv").join("pyvenv.cfg"), b"keep venv").unwrap();
        std::fs::write(install_root.join("node_modules").join("cache.txt"), b"keep node").unwrap();
        std::fs::write(install_root.join(".env"), b"keep env").unwrap();
        std::fs::write(install_root.join(SOURCE_MARKER_NAME), b"keep marker").unwrap();
        write_test_zip(
            &archive,
            &[
                ("hermes-agent-main/README.md", b"new"),
                ("hermes-agent-main/scripts/install.ps1", b"new script"),
                ("hermes-agent-main/venv/pyvenv.cfg", b"archive venv"),
            ],
        );

        refresh_existing_checkout_from_archive(&archive, &install_root).unwrap();

        assert_eq!(std::fs::read(install_root.join("README.md")).unwrap(), b"new");
        assert_eq!(
            std::fs::read(install_root.join("scripts").join("install.ps1")).unwrap(),
            b"new script"
        );
        assert_eq!(
            std::fs::read(install_root.join("venv").join("pyvenv.cfg")).unwrap(),
            b"keep venv"
        );
        assert_eq!(
            std::fs::read(install_root.join("node_modules").join("cache.txt")).unwrap(),
            b"keep node"
        );
        assert_eq!(std::fs::read(install_root.join(".env")).unwrap(), b"keep env");
        assert_eq!(
            std::fs::read(install_root.join(SOURCE_MARKER_NAME)).unwrap(),
            b"keep marker"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn archive_lifecycle_smoke_covers_update_repair_and_lite_uninstall() {
        let root = std::env::temp_dir().join(format!(
            "hermes-repo-archive-lifecycle-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let install_root = hermes_home.join("hermes-agent");
        let archive = root.join("fresh.zip");
        let update_archive = root.join("update.zip");
        std::fs::create_dir_all(&root).unwrap();
        write_test_zip(
            &archive,
            &[
                ("hermes-agent-main/README.md", b"fresh"),
                ("hermes-agent-main/scripts/install.sh", b"#!/bin/sh\n"),
            ],
        );
        write_test_zip(
            &update_archive,
            &[
                ("hermes-agent-main/README.md", b"updated"),
                ("hermes-agent-main/scripts/install.sh", b"#!/bin/sh\necho update\n"),
            ],
        );

        extract_repo_archive_to_install_root(&archive, &install_root).unwrap();
        let spec = RepoArchiveSpec {
            owner: "NousResearch".into(),
            repo: "hermes-agent".into(),
            commit: None,
            branch: Some("main".into()),
        };
        write_archive_source_marker(&install_root, &spec, &archive, false).unwrap();
        let source_marker_before = std::fs::read(install_root.join(SOURCE_MARKER_NAME)).unwrap();
        for runtime_root in hermes_manager::paths::managed_runtime_roots(&hermes_home) {
            std::fs::create_dir_all(&runtime_root).unwrap();
        }
        for runtime_file in hermes_manager::paths::managed_runtime_files(&hermes_home) {
            std::fs::write(runtime_file, b"managed").unwrap();
        }
        std::fs::write(hermes_home.join("config.yaml"), b"model: test\n").unwrap();
        hermes_manager::commands::install_metadata(&hermes_home).unwrap();

        refresh_existing_checkout_from_archive(&update_archive, &install_root).unwrap();

        assert_eq!(std::fs::read(install_root.join("README.md")).unwrap(), b"updated");
        assert_eq!(
            std::fs::read(install_root.join(SOURCE_MARKER_NAME)).unwrap(),
            source_marker_before
        );
        assert!(hermes_home.join("config.yaml").exists());

        let repaired = hermes_manager::commands::repair_clean(&hermes_home).unwrap();
        assert!(repaired
            .iter()
            .any(|path| path.ends_with("hermes-agent")));
        assert!(repaired
            .iter()
            .any(|path| path.ends_with("bootstrap-cache")));
        assert!(!install_root.exists());
        assert!(hermes_home.join("config.yaml").exists());

        for runtime_root in hermes_manager::paths::managed_runtime_roots(&hermes_home) {
            std::fs::create_dir_all(&runtime_root).unwrap();
        }
        for runtime_file in hermes_manager::paths::managed_runtime_files(&hermes_home) {
            std::fs::write(runtime_file, b"managed").unwrap();
        }
        hermes_manager::commands::install_metadata(&hermes_home).unwrap();
        let planned = hermes_manager::commands::uninstall_lite_plan(&hermes_home).unwrap();
        let removed = hermes_manager::commands::uninstall_lite(&hermes_home).unwrap();

        assert_eq!(removed, planned);
        assert!(hermes_home.join("config.yaml").exists());
        assert!(!install_root.exists());
        assert!(!hermes_home.join("bootstrap-cache").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}

fn remove_path_if_exists(path: &Path) -> Result<()> {
    if path.is_dir() {
        return remove_dir_if_exists(path);
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("removing {}", path.display())),
    }
}

fn sanitize_ref(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
