//! Self-restage + bootstrap hop (§2.3.1).
//!
//! The updater updates itself with a bootstrap hop: when a bundle's
//! min_updater_version exceeds the staged binary, the old updater extracts
//! the new one from the already-verified bundle and re-execs into it, once.
//!
//! See docs/updater-world.md §2.3.1.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Check if the staged updater needs to hop to a newer version.
/// Simple semver comparison: if manifest.min_updater_version > my_version, hop.
pub fn needs_hop(my_version: &str, min_updater_version: &str) -> bool {
    let mine = parse_semver(my_version);
    let required = parse_semver(min_updater_version);
    required > mine
}

/// Parse a semver string into a (major, minor, patch) tuple for comparison.
/// Non-numeric parts are treated as 0.
fn parse_semver(version: &str) -> (u32, u32, u32) {
    let parts: Vec<&str> = version.split('.').collect();
    let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    (major, minor, patch)
}

/// Perform the bootstrap hop: extract `bin/hermes` from the verified bundle
/// to a temp path, re-exec into it with the original argv + `--hopped`.
///
/// If `--hopped` is already in the argv, refuse (one-shot guard — no loops).
pub fn hop(bundle_dir: &Path, original_argv: &[String]) -> Result<()> {
    // Guard against infinite loops
    if original_argv.iter().any(|a| a == "--hopped") {
        bail!("bootstrap hop loop detected — --hopped already present in argv");
    }

    // Extract the new updater binary from the bundle
    let bundle_binary = bundle_dir.join("bin").join("hermes");
    if !bundle_binary.exists() {
        bail!(
            "cannot hop: bundle binary not found at {}",
            bundle_binary.display()
        );
    }

    // Copy to a temp path
    let temp_binary =
        std::env::temp_dir().join(format!("hermes-updater-hop-{}", std::process::id()));
    std::fs::copy(&bundle_binary, &temp_binary)
        .with_context(|| format!("cannot copy hop binary to {}", temp_binary.display()))?;

    // Make it executable (POSIX)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&temp_binary)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&temp_binary, perms)?;
    }

    // Build the new argv: same args + --hopped
    let mut new_argv: Vec<String> = original_argv.to_vec();
    new_argv.push("--hopped".to_string());

    // Re-exec into the new binary
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let mut cmd = std::process::Command::new(&temp_binary);
        cmd.args(&new_argv[1..]); // skip argv[0]
        let err = cmd.exec();
        bail!("failed to re-exec into hopped updater: {}", err);
    }

    #[cfg(not(unix))]
    {
        let mut cmd = std::process::Command::new(&temp_binary);
        cmd.args(&new_argv[1..]);
        let status = cmd.status().context("failed to spawn hopped updater")?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

/// Restage the updater binary from the current slot.
///
/// POSIX: write to `bin/.hermes-updater.new`, rename over the old path.
///   A running old instance keeps executing its unlinked inode happily.
/// Windows: rename running exe to `.old.exe`, move new into place, sweep
///   `.old.exe` best-effort now + on the next run.
pub fn self_restage(staged_path: &Path, new_binary: &Path) -> Result<()> {
    if !new_binary.exists() {
        bail!("new binary not found: {}", new_binary.display());
    }

    #[cfg(unix)]
    {
        let temp_path = staged_path.with_extension("new");
        std::fs::copy(new_binary, &temp_path)
            .with_context(|| format!("cannot copy to {}", temp_path.display()))?;

        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&temp_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&temp_path, perms)?;

        // Atomic rename over the old binary
        std::fs::rename(&temp_path, staged_path).with_context(|| {
            format!(
                "cannot rename {} over {}",
                temp_path.display(),
                staged_path.display()
            )
        })?;
    }

    #[cfg(not(unix))]
    {
        // Windows: can't overwrite a running exe, but CAN rename it.
        let old_path = staged_path.with_extension("old.exe");
        // Try to rename the running exe
        let _ = std::fs::rename(staged_path, &old_path);
        // Move the new binary into place
        std::fs::copy(new_binary, staged_path)
            .with_context(|| format!("cannot copy to {}", staged_path.display()))?;
        // Sweep .old.exe best-effort
        let _ = std::fs::remove_file(&old_path);
    }

    Ok(())
}

/// Sweep any `.old.exe` files from previous Windows restages.
pub fn sweep_old_binaries(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".old.exe") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_hop_same_version() {
        assert!(!needs_hop("0.1.0", "0.1.0"));
    }

    #[test]
    fn test_needs_hop_newer_required() {
        assert!(needs_hop("0.1.0", "0.2.0"));
        assert!(needs_hop("0.1.0", "1.0.0"));
    }

    #[test]
    fn test_needs_hop_older_required() {
        assert!(!needs_hop("0.2.0", "0.1.0"));
    }

    #[test]
    fn test_needs_hop_major_bump() {
        assert!(needs_hop("0.9.9", "1.0.0"));
    }

    #[test]
    fn test_parse_semver() {
        assert_eq!(parse_semver("1.2.3"), (1, 2, 3));
        assert_eq!(parse_semver("0.1.0"), (0, 1, 0));
        assert_eq!(parse_semver("1"), (1, 0, 0));
        assert_eq!(parse_semver("invalid"), (0, 0, 0));
    }

    #[test]
    fn test_hop_refuses_with_hopped_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let result = hop(
            tmp.path(),
            &["hermes-updater".to_string(), "--hopped".to_string()],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("loop"));
    }

    #[test]
    fn test_hop_fails_without_bundle_binary() {
        let tmp = tempfile::tempdir().unwrap();
        let result = hop(
            tmp.path(),
            &["hermes-updater".to_string(), "apply".to_string()],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_self_restage_posix() {
        let tmp = tempfile::tempdir().unwrap();
        let staged = tmp.path().join("hermes-updater");
        let new_binary = tmp.path().join("new-hermes");

        // Create the "old" staged binary
        std::fs::write(&staged, "old binary").unwrap();
        // Create the "new" binary
        std::fs::write(&new_binary, "new binary").unwrap();

        self_restage(&staged, &new_binary).unwrap();

        // The staged binary should now contain the new content
        let content = std::fs::read_to_string(&staged).unwrap();
        assert_eq!(content, "new binary");
    }

    #[test]
    fn test_self_restage_fails_without_new_binary() {
        let tmp = tempfile::tempdir().unwrap();
        let staged = tmp.path().join("hermes-updater");
        let new_binary = tmp.path().join("nonexistent");

        std::fs::write(&staged, "old binary").unwrap();
        assert!(self_restage(&staged, &new_binary).is_err());
    }

    #[test]
    fn test_sweep_old_binaries() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("hermes-updater.old.exe"), "old").unwrap();
        std::fs::write(tmp.path().join("hermes-updater"), "current").unwrap();

        sweep_old_binaries(tmp.path()).unwrap();

        assert!(!tmp.path().join("hermes-updater.old.exe").exists());
        assert!(tmp.path().join("hermes-updater").exists());
    }
}
