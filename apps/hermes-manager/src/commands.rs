//! Command implementations for the Hermes install manager.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use crate::installed_manifest::{InstalledKind, InstalledManifest};
use crate::ownership::ensure_safe_to_delete;
use crate::paths;
use crate::{ManagerError, Result};

/// Print status information for diagnostics.
pub fn doctor(hermes_home: &Path) -> Vec<String> {
    vec![
        format!("hermes_home={}", hermes_home.display()),
        format!("agent_root={}", paths::agent_root(hermes_home).display()),
        format!(
            "installed_manifest={}",
            paths::installed_manifest_path(hermes_home).display()
        ),
    ]
}

/// Create manager state and an initial installed-files manifest if missing.
pub fn install_metadata(hermes_home: &Path) -> Result<()> {
    let manifest_path = paths::installed_manifest_path(hermes_home);
    if manifest_path.exists() {
        return Ok(());
    }
    let mut manifest = InstalledManifest::new(hermes_home.to_path_buf());
    for runtime_root in paths::managed_runtime_roots(hermes_home) {
        if runtime_root == paths::agent_root(hermes_home) || runtime_root.exists() {
            manifest.add_entry(runtime_root, InstalledKind::Directory);
        }
    }
    manifest.write_atomic(&manifest_path)
}

/// Remove managed runtime paths while preserving user data.
pub fn uninstall_lite(hermes_home: &Path) -> Result<Vec<String>> {
    let manifest_path = paths::installed_manifest_path(hermes_home);
    let manifest = read_installed_manifest_or_default(hermes_home, &manifest_path)?;
    validate_manifest_home(hermes_home, &manifest)?;
    preflight_uninstall_lite_entries(hermes_home, &manifest)?;

    let mut removed = Vec::new();

    for entry in manifest.entries.iter().rev() {
        if !entry.path.exists() {
            continue;
        }
        match entry.kind {
            InstalledKind::File => {
                fs::remove_file(&entry.path).map_err(|err| ManagerError::io(&entry.path, err))?;
            }
            InstalledKind::Directory => {
                fs::remove_dir_all(&entry.path)
                    .map_err(|err| ManagerError::io(&entry.path, err))?;
            }
        }
        removed.push(entry.path.display().to_string());
    }

    Ok(removed)
}

/// Report paths that lite uninstall would remove without deleting them.
pub fn uninstall_lite_plan(hermes_home: &Path) -> Result<Vec<String>> {
    let manifest_path = paths::installed_manifest_path(hermes_home);
    let manifest = read_installed_manifest_or_default(hermes_home, &manifest_path)?;
    validate_manifest_home(hermes_home, &manifest)?;
    preflight_uninstall_lite_entries(hermes_home, &manifest)?;

    let mut planned = Vec::new();
    for entry in manifest.entries.iter().rev() {
        if !entry.path.exists() {
            continue;
        }
        planned.push(entry.path.display().to_string());
    }

    Ok(planned)
}

fn read_installed_manifest_or_default(
    hermes_home: &Path,
    manifest_path: &Path,
) -> Result<InstalledManifest> {
    match InstalledManifest::read(manifest_path) {
        Ok(manifest) => Ok(manifest),
        Err(ManagerError::Io { source, .. }) if source.kind() == ErrorKind::NotFound => {
            let mut manifest = InstalledManifest::new(hermes_home.to_path_buf());
            manifest.add_entry(paths::agent_root(hermes_home), InstalledKind::Directory);
            Ok(manifest)
        }
        Err(err) => Err(err),
    }
}

fn validate_manifest_home(hermes_home: &Path, manifest: &InstalledManifest) -> Result<()> {
    match (
        fs::canonicalize(hermes_home),
        fs::canonicalize(&manifest.hermes_home),
    ) {
        (Ok(active), Ok(recorded)) if active == recorded => Ok(()),
        (Ok(_), Ok(_)) => Err(manifest_home_mismatch_error(hermes_home, manifest)),
        _ if manifest.hermes_home == hermes_home => Ok(()),
        _ => Err(manifest_home_mismatch_error(hermes_home, manifest)),
    }
}

fn manifest_home_mismatch_error(hermes_home: &Path, manifest: &InstalledManifest) -> ManagerError {
    ManagerError::InvalidManifest(format!(
        "installed manifest hermes_home {} does not match active Hermes home {}",
        manifest.hermes_home.display(),
        hermes_home.display()
    ))
}

fn preflight_uninstall_lite_entries(
    hermes_home: &Path,
    manifest: &InstalledManifest,
) -> Result<()> {
    for entry in &manifest.entries {
        ensure_safe_to_delete(hermes_home, &entry.path)?;
        ensure_lite_uninstall_entry_allowed(hermes_home, &entry.path)?;
    }
    Ok(())
}

fn ensure_lite_uninstall_entry_allowed(hermes_home: &Path, candidate: &Path) -> Result<()> {
    if paths::managed_runtime_roots(hermes_home)
        .iter()
        .any(|root| crate::ownership::is_inside_root(root, candidate))
    {
        return Ok(());
    }

    Err(ManagerError::InvalidManifest(format!(
        "installed manifest entry is not a lite-uninstall runtime path: {}",
        candidate.display()
    )))
}

/// Remove the runtime checkout and bootstrap marker so the next launch repairs it.
pub fn repair_clean(hermes_home: &Path) -> Result<Vec<String>> {
    let mut removed = Vec::new();
    for runtime_root in paths::managed_runtime_roots(hermes_home) {
        ensure_safe_to_delete(hermes_home, &runtime_root)?;
        if runtime_root.exists() {
            fs::remove_dir_all(&runtime_root)
                .map_err(|err| ManagerError::io(&runtime_root, err))?;
            removed.push(runtime_root.display().to_string());
        }
    }

    let marker = paths::agent_root(hermes_home).join(".hermes-bootstrap-complete");
    ensure_safe_to_delete(hermes_home, &marker)?;
    if !paths::agent_root(hermes_home).exists() && marker.exists() {
        fs::remove_file(&marker).map_err(|err| ManagerError::io(&marker, err))?;
        removed.push(marker.display().to_string());
    }

    Ok(removed)
}

/// Report runtime checkout paths that repair cleanup would remove.
pub fn repair_clean_plan(hermes_home: &Path) -> Result<Vec<String>> {
    let mut planned = Vec::new();
    for runtime_root in paths::managed_runtime_roots(hermes_home) {
        ensure_safe_to_delete(hermes_home, &runtime_root)?;
        if runtime_root.exists() {
            planned.push(runtime_root.display().to_string());
        }
    }

    let marker = paths::agent_root(hermes_home).join(".hermes-bootstrap-complete");
    ensure_safe_to_delete(hermes_home, &marker)?;
    if !paths::agent_root(hermes_home).exists() && marker.exists() {
        planned.push(marker.display().to_string());
    }

    Ok(planned)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::installed_manifest::{InstalledKind, InstalledManifest};
    use crate::paths;

    #[test]
    fn install_metadata_creates_manifest() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        fs::create_dir_all(&hermes_home).expect("Hermes home should be created");

        super::install_metadata(&hermes_home).expect("install metadata should be created");

        let manifest_path = paths::installed_manifest_path(&hermes_home);
        let manifest = InstalledManifest::read(&manifest_path).expect("manifest should be read");
        assert_eq!(manifest.hermes_home, hermes_home);
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(
            manifest.entries[0].path,
            paths::agent_root(&manifest.hermes_home)
        );
        assert_eq!(manifest.entries[0].kind, InstalledKind::Directory);
    }

    #[test]
    fn install_metadata_records_existing_managed_runtime_dirs() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let bin_dir = hermes_home.join("bin");
        let node_dir = hermes_home.join("node");
        let git_dir = hermes_home.join("git");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::create_dir_all(&bin_dir).expect("bin dir should be created");
        fs::create_dir_all(&node_dir).expect("node dir should be created");
        fs::create_dir_all(&git_dir).expect("git dir should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        super::install_metadata(&hermes_home).expect("install metadata should be created");

        let manifest = InstalledManifest::read(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be read");
        let paths = manifest
            .entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec![agent_root, bin_dir, node_dir, git_dir]);
        assert!(!paths.contains(&user_config));
    }

    #[test]
    fn uninstall_lite_removes_managed_runtime_dirs_outside_agent_root() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let bin_dir = hermes_home.join("bin");
        let node_dir = hermes_home.join("node");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::create_dir_all(&bin_dir).expect("bin dir should be created");
        fs::create_dir_all(&node_dir).expect("node dir should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        let mut manifest = InstalledManifest::new(hermes_home.clone());
        manifest.add_entry(agent_root.clone(), InstalledKind::Directory);
        manifest.add_entry(bin_dir.clone(), InstalledKind::Directory);
        manifest.add_entry(node_dir.clone(), InstalledKind::Directory);
        manifest
            .write_atomic(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be written");

        let removed = super::uninstall_lite(&hermes_home).expect("runtime dirs should be removed");

        assert_eq!(
            removed,
            vec![
                node_dir.display().to_string(),
                bin_dir.display().to_string(),
                agent_root.display().to_string(),
            ]
        );
        assert!(!agent_root.exists());
        assert!(!bin_dir.exists());
        assert!(!node_dir.exists());
        assert!(user_config.exists());
    }

    #[test]
    fn uninstall_lite_removes_only_manifest_entries() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let managed_file = agent_root.join("managed.txt");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::write(&managed_file, "managed").expect("managed file should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        let mut manifest = InstalledManifest::new(hermes_home.clone());
        manifest.add_entry(managed_file.clone(), InstalledKind::File);
        manifest
            .write_atomic(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be written");

        let removed =
            super::uninstall_lite(&hermes_home).expect("manifest entries should be removed");

        assert_eq!(removed, vec![managed_file.display().to_string()]);
        assert!(!managed_file.exists());
        assert!(agent_root.exists());
        assert!(user_config.exists());
    }

    #[test]
    fn uninstall_lite_plan_reports_entries_without_removing_them() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let managed_file = agent_root.join("managed.txt");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::write(&managed_file, "managed").expect("managed file should be created");

        let mut manifest = InstalledManifest::new(hermes_home.clone());
        manifest.add_entry(managed_file.clone(), InstalledKind::File);
        manifest
            .write_atomic(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be written");

        let planned = super::uninstall_lite_plan(&hermes_home).expect("plan should be created");

        assert_eq!(planned, vec![managed_file.display().to_string()]);
        assert!(managed_file.exists());
    }

    #[test]
    fn repair_clean_plan_reports_runtime_paths_without_removing_them() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let marker = agent_root.join(".hermes-bootstrap-complete");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::write(&marker, "{}").expect("marker should be created");

        let planned = super::repair_clean_plan(&hermes_home).expect("plan should be created");

        assert_eq!(planned, vec![agent_root.display().to_string()]);
        assert!(agent_root.exists());
        assert!(marker.exists());
    }

    #[test]
    fn repair_clean_removes_managed_runtime_dirs_but_preserves_user_data() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let bin_dir = hermes_home.join("bin");
        let node_dir = hermes_home.join("node");
        let git_dir = hermes_home.join("git");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::create_dir_all(&bin_dir).expect("bin dir should be created");
        fs::create_dir_all(&node_dir).expect("node dir should be created");
        fs::create_dir_all(&git_dir).expect("git dir should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        let planned = super::repair_clean_plan(&hermes_home).expect("plan should be created");
        assert_eq!(
            planned,
            vec![
                agent_root.display().to_string(),
                bin_dir.display().to_string(),
                node_dir.display().to_string(),
                git_dir.display().to_string(),
            ]
        );

        let removed = super::repair_clean(&hermes_home).expect("runtime dirs should be removed");

        assert_eq!(removed, planned);
        assert!(!agent_root.exists());
        assert!(!bin_dir.exists());
        assert!(!node_dir.exists());
        assert!(!git_dir.exists());
        assert!(user_config.exists());
    }

    #[test]
    fn uninstall_lite_defaults_to_agent_root_when_manifest_is_missing() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let managed_file = agent_root.join("managed.txt");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::write(&managed_file, "managed").expect("managed file should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        let removed = super::uninstall_lite(&hermes_home)
            .expect("missing manifest should fall back to agent root");

        assert_eq!(removed, vec![agent_root.display().to_string()]);
        assert!(!agent_root.exists());
        assert!(user_config.exists());
    }

    #[test]
    fn uninstall_lite_rejects_config_manifest_entry() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&hermes_home).expect("Hermes home should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        let mut manifest = InstalledManifest::new(hermes_home.clone());
        manifest.add_entry(user_config.clone(), InstalledKind::File);
        manifest
            .write_atomic(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be written");

        assert!(super::uninstall_lite(&hermes_home).is_err());
        assert!(user_config.exists());
    }

    #[test]
    fn uninstall_lite_rejects_manifest_home_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let other_home = dir.path().join("other-hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let managed_file = agent_root.join("managed.txt");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::create_dir_all(&other_home).expect("other home should be created");
        fs::write(&managed_file, "managed").expect("managed file should be created");

        let mut manifest = InstalledManifest::new(other_home);
        manifest.add_entry(managed_file.clone(), InstalledKind::File);
        manifest
            .write_atomic(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be written");

        assert!(super::uninstall_lite(&hermes_home).is_err());
        assert!(managed_file.exists());
    }

    #[test]
    fn uninstall_lite_preflight_rejects_all_when_any_entry_is_disallowed() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let hermes_home = dir.path().join("hermes");
        let agent_root = paths::agent_root(&hermes_home);
        let managed_file = agent_root.join("managed.txt");
        let user_config = hermes_home.join("config.yaml");
        fs::create_dir_all(&agent_root).expect("agent root should be created");
        fs::write(&managed_file, "managed").expect("managed file should be created");
        fs::write(&user_config, "model: test").expect("user config should be created");

        let mut manifest = InstalledManifest::new(hermes_home.clone());
        manifest.add_entry(managed_file.clone(), InstalledKind::File);
        manifest.add_entry(user_config.clone(), InstalledKind::File);
        manifest
            .write_atomic(&paths::installed_manifest_path(&hermes_home))
            .expect("manifest should be written");

        assert!(super::uninstall_lite(&hermes_home).is_err());
        assert!(managed_file.exists());
        assert!(user_config.exists());
    }
}
