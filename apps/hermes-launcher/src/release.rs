//! Release fetching + verification.
//!
//! Supports https:// (GitHub Releases API) and file:// (E2E fixtures).
//! Verification: Ed25519 signature on manifest.json + sha256 of every file.
//!
//! See docs/updater-world.md §2.1, §2.3.1.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The manifest.json schema (task 0.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub version: String,
    pub channel: String,
    pub git_sha: String,
    pub platform: String,
    pub min_updater_version: String,
    #[serde(default)]
    pub desktop: bool,
    pub files: HashMap<String, String>,
}

/// The manifest.json.sig schema (task 0.4 — Ed25519 via PyNaCl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub algorithm: String,
    pub pubkey: String,
    pub signature: String,
}

/// A release source — where bundles come from.
#[derive(Debug, Clone)]
pub enum ReleaseSource {
    /// GitHub Releases API (https://).
    Https { base_url: String },
    /// Local file:// directory (E2E fixtures).
    File { base_path: PathBuf },
}

impl ReleaseSource {
    /// Parse a source URL string into a ReleaseSource.
    pub fn parse(url: &str) -> Result<Self> {
        if url.starts_with("file://") {
            let path = url.strip_prefix("file://").unwrap();
            Ok(ReleaseSource::File {
                base_path: PathBuf::from(path),
            })
        } else if url.starts_with("https://") || url.starts_with("http://") {
            Ok(ReleaseSource::Https {
                base_url: url.to_string(),
            })
        } else {
            bail!(
                "unsupported source URL scheme: {} (use https:// or file://)",
                url
            )
        }
    }

    /// Resolve the URLs for a given version + platform.
    /// Returns (bundle_url, manifest_url, sig_url).
    pub fn resolve(&self, version: &str, platform: &str) -> Result<(String, String, String)> {
        let bundle_name = format!("hermes-{}-{}.tar.zst", version, platform);
        let manifest_name = "manifest.json";
        let sig_name = "manifest.json.sig";

        match self {
            ReleaseSource::File { base_path } => {
                let base = base_path.join(version);
                Ok((
                    format!("file://{}", base.join(&bundle_name).display()),
                    format!("file://{}", base.join(manifest_name).display()),
                    format!("file://{}", base.join(sig_name).display()),
                ))
            }
            ReleaseSource::Https { base_url } => {
                let base = format!("{}/{}", base_url.trim_end_matches('/'), version);
                Ok((
                    format!("{}/{}", base, bundle_name),
                    format!("{}/{}", base, manifest_name),
                    format!("{}/{}", base, sig_name),
                ))
            }
        }
    }

    /// Fetch the latest available version for a channel.
    /// For file:// sources, reads a `latest-<channel>.txt` file in the base path.
    /// For https:// sources, queries the GitHub Releases API.
    pub fn latest(&self, channel: &str) -> Result<String> {
        match self {
            ReleaseSource::File { base_path } => {
                let latest_file = base_path.join(format!("latest-{}.txt", channel));
                let content = std::fs::read_to_string(&latest_file).with_context(|| {
                    format!("cannot read latest file: {}", latest_file.display())
                })?;
                Ok(content.trim().to_string())
            }
            ReleaseSource::Https { base_url } => {
                // TODO: GitHub Releases API call. For now, construct the URL
                // and let the caller decide. This is implemented in task 1.4's
                // apply flow which has the reqwest client.
                bail!(
                    "https:// latest() not yet implemented — use file:// for E2E tests, or specify --version"
                )
            }
        }
    }

    /// Download a file from the source to a local path.
    /// For file:// sources, this is a local copy. For https://, an HTTP GET.
    pub async fn download(&self, url: &str, dest: &Path) -> Result<()> {
        if url.starts_with("file://") {
            let src = url.strip_prefix("file://").unwrap();
            std::fs::copy(src, dest)
                .with_context(|| format!("failed to copy {} to {}", src, dest.display()))?;
            Ok(())
        } else {
            // HTTP download (task 1.4's apply flow uses this with reqwest)
            let client = reqwest::Client::new();
            let resp = client
                .get(url)
                .send()
                .await
                .with_context(|| format!("HTTP GET failed: {}", url))?;
            if !resp.status().is_success() {
                bail!("HTTP {} for {}", resp.status(), url);
            }
            let bytes = resp.bytes().await.context("failed to read response body")?;
            std::fs::write(dest, &bytes)
                .with_context(|| format!("failed to write {}", dest.display()))?;
            Ok(())
        }
    }
}

/// Verify a bundle directory: signature + file hashes.
///
/// 1. Read manifest.json + manifest.json.sig
/// 2. Verify the Ed25519 signature over manifest.json bytes
/// 3. Verify every file hash in the manifest matches the actual files
/// 4. Check for extra files not in the manifest
///
/// Returns Ok(()) if everything verifies, Err with details otherwise.
pub fn verify_bundle(bundle_dir: &Path, expected_pubkey: Option<&str>) -> Result<Manifest> {
    let manifest_path = bundle_dir.join("manifest.json");
    let sig_path = bundle_dir.join("manifest.json.sig");

    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("cannot read {}", manifest_path.display()))?;
    let manifest: Manifest =
        serde_json::from_slice(&manifest_bytes).context("failed to parse manifest.json")?;

    if manifest.schema != 1 {
        bail!(
            "unsupported manifest schema: {} (expected 1)",
            manifest.schema
        );
    }

    // Verify signature if present
    if sig_path.exists() {
        let sig_bytes = std::fs::read(&sig_path)
            .with_context(|| format!("cannot read {}", sig_path.display()))?;
        let sig: Signature =
            serde_json::from_slice(&sig_bytes).context("failed to parse manifest.json.sig")?;

        let pubkey = expected_pubkey.unwrap_or(&sig.pubkey);
        verify_ed25519(&manifest_bytes, &sig.signature, pubkey)?;
    }

    // Verify file hashes
    verify_file_hashes(bundle_dir, &manifest)?;

    Ok(manifest)
}

/// Verify the Ed25519 signature over manifest bytes.
fn verify_ed25519(manifest_bytes: &[u8], signature_b64: &str, pubkey_b64: &str) -> Result<()> {
    use base64::Engine;

    let signature_bytes = base64::engine::general_purpose::STANDARD
        .decode(signature_b64)
        .context("invalid base64 in signature")?;
    let pubkey_bytes = base64::engine::general_purpose::STANDARD
        .decode(pubkey_b64)
        .context("invalid base64 in pubkey")?;

    let pubkey_arr: [u8; 32] = pubkey_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid pubkey length (expected 32 bytes)"))?;
    let verify_key = ed25519_dalek::VerifyingKey::from_bytes(&pubkey_arr)
        .context("invalid ed25519 public key")?;

    let sig_arr: [u8; 64] = signature_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid signature length (expected 64 bytes)"))?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_arr);

    use ed25519_dalek::Verifier;
    verify_key
        .verify(manifest_bytes, &signature)
        .map_err(|_| anyhow::anyhow!("signature verification failed — manifest may be tampered"))?;

    Ok(())
}

/// Verify every file hash in the manifest matches the actual files.
/// Also checks for extra files not in the manifest.
fn verify_file_hashes(bundle_dir: &Path, manifest: &Manifest) -> Result<()> {
    use sha2::{Digest, Sha256};

    let mut errors: Vec<String> = Vec::new();

    // Check every file in the manifest
    for (rel_path, expected_hash) in &manifest.files {
        let filepath = bundle_dir.join(rel_path);
        if !filepath.exists() {
            errors.push(format!("missing: {}", rel_path));
            continue;
        }
        let actual_hash = compute_sha256(&filepath)?;
        if actual_hash != *expected_hash {
            errors.push(format!(
                "tampered: {} (expected {}, got {})",
                rel_path, expected_hash, actual_hash
            ));
        }
    }

    // Check for extra files
    let manifest_files: std::collections::HashSet<&String> = manifest.files.keys().collect();
    for entry in walkdir(bundle_dir) {
        let rel = entry.strip_prefix(bundle_dir).unwrap_or(&entry);
        let rel_str = rel.to_string_lossy().to_string();
        if rel_str == "manifest.json" || rel_str == "manifest.json.sig" {
            continue;
        }
        if !manifest_files.contains(&rel_str) {
            errors.push(format!("extra file not in manifest: {}", rel_str));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        bail!("hash verification failed:\n  {}", errors.join("\n  "))
    }
}

/// Compute sha256:<hex> for a file.
fn compute_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    let mut file =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut buf = [0u8; 65536];
    use std::io::Read;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

/// Recursively walk a directory, yielding all regular file paths.
fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    walkdir_inner(dir, dir, &mut result);
    result
}

fn walkdir_inner(root: &Path, dir: &Path, result: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip .staging dirs
                if path.file_name().map(|n| n == ".staging").unwrap_or(false) {
                    continue;
                }
                walkdir_inner(root, &path, result);
            } else if path.is_file() {
                result.push(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::rngs::OsRng;
    use sha2::Sha256;
    use std::io::Write;

    fn make_bundle_fixture(dir: &Path) {
        std::fs::create_dir_all(dir.join("runtime/venv/bin")).unwrap();
        std::fs::create_dir_all(dir.join("app")).unwrap();
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("runtime/venv/bin/python"), "# fake python").unwrap();
        std::fs::write(dir.join("app/run_agent.py"), "# fake source\n").unwrap();
        std::fs::write(dir.join("bin/hermes"), "#!/bin/sh\necho hermes\n").unwrap();
    }

    fn write_manifest(dir: &Path) -> Manifest {
        let mut files = HashMap::new();
        for entry in walkdir(dir) {
            let rel = entry
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .to_string();
            if rel == "manifest.json" || rel == "manifest.json.sig" {
                continue;
            }
            let hash = compute_sha256(&entry).unwrap();
            files.insert(rel, hash);
        }
        let manifest = Manifest {
            schema: 1,
            version: "2026.07.15".to_string(),
            channel: "nightly".to_string(),
            git_sha: "a".repeat(40),
            platform: "linux-x64".to_string(),
            min_updater_version: "0.1.0".to_string(),
            desktop: false,
            files,
        };
        let manifest_path = dir.join("manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    fn sign_manifest(dir: &Path) -> (String, String) {
        let manifest_bytes = std::fs::read(dir.join("manifest.json")).unwrap();
        let mut rng = OsRng;
        let key = SigningKey::generate(&mut rng);
        let signed = key.sign(&manifest_bytes);
        let pubkey =
            base64::engine::general_purpose::STANDARD.encode(key.verifying_key().to_bytes());
        let signature = base64::engine::general_purpose::STANDARD.encode(signed.to_bytes());
        let sig = Signature {
            algorithm: "ed25519".to_string(),
            pubkey: pubkey.clone(),
            signature: signature.clone(),
        };
        std::fs::write(
            dir.join("manifest.json.sig"),
            serde_json::to_string_pretty(&sig).unwrap(),
        )
        .unwrap();
        (pubkey, signature)
    }

    #[test]
    fn test_parse_file_source() {
        let src = ReleaseSource::parse("file:///tmp/releases").unwrap();
        assert!(matches!(src, ReleaseSource::File { .. }));
    }

    #[test]
    fn test_parse_https_source() {
        let src = ReleaseSource::parse("https://github.com/.../releases").unwrap();
        assert!(matches!(src, ReleaseSource::Https { .. }));
    }

    #[test]
    fn test_parse_invalid_source() {
        assert!(ReleaseSource::parse("ftp://example.com").is_err());
    }

    #[test]
    fn test_file_source_resolve() {
        let src = ReleaseSource::parse("file:///tmp/releases").unwrap();
        let (bundle, manifest, sig) = src.resolve("2026.07.15", "linux-x64").unwrap();
        assert!(bundle.contains("hermes-2026.07.15-linux-x64.tar.zst"));
        assert!(manifest.contains("manifest.json"));
        assert!(sig.contains("manifest.json.sig"));
    }

    #[test]
    fn test_file_source_latest() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("latest-stable.txt"), "2026.07.15\n").unwrap();
        let src = ReleaseSource::File {
            base_path: tmp.path().to_path_buf(),
        };
        let version = src.latest("stable").unwrap();
        assert_eq!(version, "2026.07.15");
    }

    #[test]
    fn test_verify_clean_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        let manifest = verify_bundle(tmp.path(), None).unwrap();
        assert_eq!(manifest.schema, 1);
        assert_eq!(manifest.version, "2026.07.15");
    }

    #[test]
    fn test_verify_tampered_file() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        // Tamper with a file
        std::fs::write(tmp.path().join("app/run_agent.py"), "# TAMPERED\n").unwrap();
        let result = verify_bundle(tmp.path(), None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("tampered"));
    }

    #[test]
    fn test_verify_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        std::fs::remove_file(tmp.path().join("app/run_agent.py")).unwrap();
        let result = verify_bundle(tmp.path(), None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing"));
    }

    #[test]
    fn test_verify_extra_file() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        std::fs::write(tmp.path().join("evil.py"), "# evil").unwrap();
        let result = verify_bundle(tmp.path(), None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("extra"));
    }

    #[test]
    fn test_verify_with_signature() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        let (pubkey, _) = sign_manifest(tmp.path());
        // Should verify with the correct pubkey
        verify_bundle(tmp.path(), Some(&pubkey)).unwrap();
    }

    #[test]
    fn test_verify_tampered_manifest_signature_fails() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        let (pubkey, _) = sign_manifest(tmp.path());
        // Tamper with manifest content (but keep the old signature)
        let manifest_path = tmp.path().join("manifest.json");
        let mut manifest: Manifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest.version = "tampered".to_string();
        std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();
        // Signature verification should fail
        let result = verify_bundle(tmp.path(), Some(&pubkey));
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_wrong_pubkey_fails() {
        let tmp = tempfile::tempdir().unwrap();
        make_bundle_fixture(tmp.path());
        write_manifest(tmp.path());
        sign_manifest(tmp.path());
        // Use a different (wrong) pubkey
        let mut rng = OsRng;
        let wrong_key = SigningKey::generate(&mut rng);
        let wrong_pubkey =
            base64::engine::general_purpose::STANDARD.encode(wrong_key.verifying_key().to_bytes());
        let result = verify_bundle(tmp.path(), Some(&wrong_pubkey));
        assert!(result.is_err());
    }
}
