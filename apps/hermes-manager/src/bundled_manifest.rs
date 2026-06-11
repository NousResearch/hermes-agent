//! Bundled runtime manifest schema and validation.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ManagerError, Result};

/// Supported bundled manifest schema version.
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Manifest describing resources shipped in a release package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundledManifest {
    pub schema_version: u32,
    pub hermes_version: String,
    pub source_commit: String,
    pub resources: Vec<BundledResource>,
    #[serde(default)]
    pub embedded_resources: Vec<EmbeddedResource>,
}

/// A single bundled resource entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundledResource {
    pub kind: ResourceKind,
    pub path: String,
    pub sha256: String,
}

/// A resource embedded inside another release binary rather than staged as a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddedResource {
    pub kind: EmbeddedResourceKind,
    pub name: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// Resource types the first manager slice understands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    AgentSnapshot,
    CoreWheelhouse,
    PythonRuntime,
    Tool,
}

/// Embedded resource types the manager can validate as metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddedResourceKind {
    InstallScript,
}

impl BundledManifest {
    /// Read a bundled manifest from JSON.
    pub fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|err| ManagerError::io(path, err))?;
        let manifest: Self =
            serde_json::from_str(&text).map_err(|err| ManagerError::json(path, err))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate schema and required fields.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(ManagerError::InvalidManifest(format!(
                "schema_version {} is not supported",
                self.schema_version
            )));
        }
        if self.hermes_version.trim().is_empty() {
            return Err(ManagerError::InvalidManifest(
                "hermes_version is empty".into(),
            ));
        }
        if self.source_commit.trim().is_empty() {
            return Err(ManagerError::InvalidManifest(
                "source_commit is empty".into(),
            ));
        }
        if self.resources.is_empty() {
            return Err(ManagerError::InvalidManifest("resources is empty".into()));
        }
        for resource in &self.resources {
            if resource.path.trim().is_empty() {
                return Err(ManagerError::InvalidManifest(
                    "resource path is empty".into(),
                ));
            }
            if !is_safe_resource_path(&resource.path) {
                return Err(ManagerError::InvalidManifest(format!(
                    "resource {} has unsafe resource path",
                    resource.path
                )));
            }
            if resource.sha256.len() != 64
                || !resource.sha256.chars().all(|ch| ch.is_ascii_hexdigit())
            {
                return Err(ManagerError::InvalidManifest(format!(
                    "resource {} has invalid sha256",
                    resource.path
                )));
            }
        }
        for resource in &self.embedded_resources {
            if resource.name.trim().is_empty() {
                return Err(ManagerError::InvalidManifest(
                    "embedded resource name is empty".into(),
                ));
            }
            if resource.size_bytes == 0 {
                return Err(ManagerError::InvalidManifest(format!(
                    "embedded resource {} is empty",
                    resource.name
                )));
            }
            if resource.sha256.len() != 64
                || !resource.sha256.chars().all(|ch| ch.is_ascii_hexdigit())
            {
                return Err(ManagerError::InvalidManifest(format!(
                    "embedded resource {} has invalid sha256",
                    resource.name
                )));
            }
        }
        Ok(())
    }

    /// Verify every resource checksum relative to `root`.
    pub fn verify_resources(&self, root: &Path) -> Result<()> {
        self.validate()?;
        for resource in &self.resources {
            let path = root.join(&resource.path);
            let bytes = fs::read(&path).map_err(|err| ManagerError::io(&path, err))?;
            let digest = Sha256::digest(&bytes);
            let actual = digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            if !actual.eq_ignore_ascii_case(&resource.sha256) {
                return Err(ManagerError::InvalidManifest(format!(
                    "resource {} checksum mismatch: expected {}, got {}",
                    resource.path, resource.sha256, actual
                )));
            }
        }
        Ok(())
    }
}

fn is_safe_resource_path(path: &str) -> bool {
    let path = path.trim();
    if path.is_empty() || Path::new(path).is_absolute() || has_windows_drive_prefix(path) {
        return false;
    }

    let normalized = path.replace('\\', "/");
    !normalized.starts_with('/') && !normalized.split('/').any(|component| component == "..")
}

fn has_windows_drive_prefix(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest_with_resource_path(path: &str) -> BundledManifest {
        BundledManifest {
            schema_version: SUPPORTED_SCHEMA_VERSION,
            hermes_version: "0.16.0".into(),
            source_commit: "615ad9792".into(),
            resources: vec![BundledResource {
                kind: ResourceKind::AgentSnapshot,
                path: path.into(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            }],
            embedded_resources: Vec::new(),
        }
    }

    #[test]
    fn reads_valid_fixture() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/bundled-manifest.valid.json");
        let manifest = BundledManifest::read(&path).expect("fixture should parse");
        assert_eq!(manifest.schema_version, SUPPORTED_SCHEMA_VERSION);
        assert_eq!(manifest.resources.len(), 2);
        assert_eq!(manifest.resources[0].kind, ResourceKind::AgentSnapshot);
    }

    #[test]
    fn rejects_unsupported_schema() {
        let manifest = BundledManifest {
            schema_version: 99,
            hermes_version: "0.16.0".into(),
            source_commit: "615ad9792".into(),
            resources: vec![BundledResource {
                kind: ResourceKind::AgentSnapshot,
                path: "resources/agent".into(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            }],
            embedded_resources: Vec::new(),
        };

        let err = manifest.validate().expect_err("schema should be rejected");
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn rejects_absolute_resource_paths() {
        for unsafe_path in ["/absolute", r"C:\absolute"] {
            let manifest = valid_manifest_with_resource_path(unsafe_path);
            let err = manifest
                .validate()
                .expect_err("absolute resource path should be rejected");
            assert!(err.to_string().contains("unsafe resource path"));
        }
    }

    #[test]
    fn rejects_windows_drive_relative_resource_paths() {
        for unsafe_path in ["C:payload", r"C:..\payload"] {
            let manifest = valid_manifest_with_resource_path(unsafe_path);
            let err = manifest
                .validate()
                .expect_err("Windows drive-relative resource path should be rejected");
            assert!(err.to_string().contains("unsafe resource path"));
        }
    }

    #[test]
    fn rejects_parent_traversal_resource_paths() {
        for unsafe_path in ["../resources/agent", "resources/../agent"] {
            let manifest = valid_manifest_with_resource_path(unsafe_path);
            let err = manifest
                .validate()
                .expect_err("parent traversal resource path should be rejected");
            assert!(err.to_string().contains("unsafe resource path"));
        }
    }

    #[test]
    fn rejects_invalid_sha256_values() {
        for invalid_sha256 in [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
        ] {
            let mut manifest = valid_manifest_with_resource_path("resources/agent");
            manifest.resources[0].sha256 = invalid_sha256.into();

            let err = manifest
                .validate()
                .expect_err("invalid sha256 should be rejected");
            assert!(err.to_string().contains("invalid sha256"));
        }
    }

    #[test]
    fn verify_resources_checks_relative_file_hashes() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let resource = dir.path().join("hermes-manager.exe");
        fs::write(&resource, b"manager").expect("resource should be written");
        let manifest = BundledManifest {
            schema_version: SUPPORTED_SCHEMA_VERSION,
            hermes_version: "0.16.0".into(),
            source_commit: "615ad9792".into(),
            resources: vec![BundledResource {
                kind: ResourceKind::Tool,
                path: "hermes-manager.exe".into(),
                sha256: "6ee4a469cd4e91053847f5d3fcb61dbcc91e8f0ef10be7748da4c4a1ba382d17".into(),
            }],
            embedded_resources: Vec::new(),
        };

        manifest
            .verify_resources(dir.path())
            .expect("resource checksum should match");
    }

    #[test]
    fn verify_resources_rejects_checksum_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        fs::write(dir.path().join("hermes-manager.exe"), b"changed")
            .expect("resource should be written");
        let manifest = valid_manifest_with_resource_path("hermes-manager.exe");

        let err = manifest
            .verify_resources(dir.path())
            .expect_err("checksum mismatch should be rejected");

        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn validates_embedded_resource_metadata() {
        let mut manifest = valid_manifest_with_resource_path("resources/agent");
        manifest.embedded_resources.push(EmbeddedResource {
            kind: EmbeddedResourceKind::InstallScript,
            name: "install.ps1".into(),
            size_bytes: 14,
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        });

        manifest
            .validate()
            .expect("embedded metadata should validate");

        manifest.embedded_resources[0].sha256 = "bad".into();
        let err = manifest
            .validate()
            .expect_err("invalid embedded sha should be rejected");
        assert!(err
            .to_string()
            .contains("embedded resource install.ps1 has invalid sha256"));
    }
}
