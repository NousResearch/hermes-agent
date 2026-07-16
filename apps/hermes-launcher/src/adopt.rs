//! Adopt verb (hop 3, Rust side) — migrate a legacy git-checkout install
//! to managed slots.
//!
//! See docs/updater-world.md §2.13 and
//! docs/plans/updater-rework/03-phase2-compat-and-adoption.md task 2.6.

use crate::release::{self, ReleaseSource};
use crate::slots;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Perform the adoption: download a bundle, create a slot, flip, re-point
/// the PATH symlink. The checkout is left completely untouched.
pub fn adopt(
    hermes_home: &Path,
    from_checkout: &Path,
    source: Option<&str>,
    undo: bool,
) -> Result<()> {
    if undo {
        return adopt_undo(hermes_home);
    }

    // 1. Read the checkout's git SHA (for choosing the matching bundle)
    let git_sha = read_checkout_sha(from_checkout)?;
    println!(
        "==> Adopting from checkout: {} ({})",
        from_checkout.display(),
        &git_sha[..8]
    );

    // 2. Determine the release source
    let source_url = source.unwrap_or("https://github.com/NousResearch/hermes-agent/releases");
    let release_source = ReleaseSource::parse(source_url)?;

    // 3. Find the latest version for the stable channel
    let version = match &release_source {
        ReleaseSource::File { base_path } => {
            let latest_file = base_path.join("latest-stable.txt");
            if latest_file.exists() {
                std::fs::read_to_string(&latest_file)?.trim().to_string()
            } else {
                bail!("no latest-stable.txt found in file:// source — specify --version")
            }
        }
        ReleaseSource::Https { .. } => {
            // TODO: GitHub Releases API call. For now, require a version.
            bail!("https:// source not yet implemented for adopt — use file:// for testing")
        }
    };
    println!("==> Target version: {}", version);

    // 4. Download the bundle
    let platform = detect_platform()?;
    let (bundle_url, manifest_url, sig_url) = release_source.resolve(&version, &platform)?;

    let staging = slots::stage(hermes_home, &version)?;
    println!("==> Staging to: {}", staging.display());

    // Download bundle archive
    let archive_path = staging.join(format!("hermes-{}-{}.tar.zst", version, platform));
    println!("==> Downloading bundle...");
    // For file:// sources, download is a local copy
    let bundle_url_stripped = bundle_url.strip_prefix("file://").unwrap_or(&bundle_url);
    if Path::new(bundle_url_stripped).exists() {
        std::fs::copy(bundle_url_stripped, &archive_path)
            .context("failed to copy bundle archive")?;
    } else {
        // TODO: HTTP download (task 1.3's download() is async)
        bail!("HTTP download not yet implemented — use file:// for testing");
    }

    // Unpack the bundle into staging
    println!("==> Unpacking bundle...");
    unpack_bundle(&archive_path, &staging)?;

    // 5. Verify the bundle
    println!("==> Verifying bundle...");
    let manifest = release::verify_bundle(&staging, None)?;
    println!("    Manifest verified: {} files", manifest.files.len());

    // 6. Commit staging → slot
    println!("==> Committing slot...");
    let slot = slots::commit_staging(hermes_home, &version)?;
    println!("    Slot at: {}", slot.display());

    // 7. Flip current.txt
    println!("==> Flipping...");
    slots::flip(hermes_home, &version)?;
    println!("    current.txt → {}", version);

    // 8. Re-point the PATH symlink
    let launcher = hermes_home.join("bin").join("hermes");
    let link_dir = find_command_link_dir()?;
    let symlink_path = link_dir.join("hermes");

    // Record the old target for undo
    let pre_adopt_path = hermes_home.join(".pre-adopt-target");
    if symlink_path.exists() || symlink_path.is_symlink() {
        if let Ok(target) = std::fs::read_link(&symlink_path) {
            std::fs::write(&pre_adopt_path, target.to_string_lossy().as_bytes())
                .context("cannot write .pre-adopt-target")?;
        }
    }

    // Re-point the symlink
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&symlink_path);
        std::os::unix::fs::symlink(&launcher, &symlink_path).with_context(|| {
            format!(
                "cannot symlink {} → {}",
                symlink_path.display(),
                launcher.display()
            )
        })?;
    }

    println!(
        "==> Symlink: {} → {}",
        symlink_path.display(),
        launcher.display()
    );

    // 9. Verify the checkout is untouched
    let new_sha = read_checkout_sha(from_checkout)?;
    if new_sha != git_sha {
        bail!(
            "CHECKOUT WAS MODIFIED! Expected {}, got {}. The checkout should be untouched.",
            git_sha,
            new_sha
        );
    }
    println!("==> Checkout untouched (SHA unchanged)");

    println!();
    println!("✓ Adoption complete!");
    println!("  Version:  {}", version);
    println!("  Slot:    {}", slot.display());
    println!("  Symlink: {}", symlink_path.display());
    println!();
    println!("  Undo with: hermes-updater adopt --undo");

    Ok(())
}

/// Undo a previous adoption: re-point the symlink at the old target.
fn adopt_undo(hermes_home: &Path) -> Result<()> {
    let pre_adopt_path = hermes_home.join(".pre-adopt-target");
    if !pre_adopt_path.exists() {
        bail!("no .pre-adopt-target found — nothing to undo");
    }

    let old_target = std::fs::read_to_string(&pre_adopt_path)?;
    let old_target = old_target.trim();

    let link_dir = find_command_link_dir()?;
    let symlink_path = link_dir.join("hermes");

    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&symlink_path);
        std::os::unix::fs::symlink(old_target, &symlink_path)?;
    }

    let _ = std::fs::remove_file(&pre_adopt_path);

    println!("✓ Adoption undone");
    println!("  Symlink: {} → {}", symlink_path.display(), old_target);

    Ok(())
}

/// Read the git SHA of a checkout.
fn read_checkout_sha(checkout: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(checkout)
        .output()
        .context("failed to run git rev-parse")?;

    if !output.status.success() {
        bail!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Detect the current platform string (e.g., "linux-x64").
fn detect_platform() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let plat_os = match os {
        "linux" => "linux",
        "macos" => "darwin",
        "windows" => "win",
        _ => bail!("unsupported OS: {}", os),
    };

    let plat_arch = match arch {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        _ => bail!("unsupported arch: {}", arch),
    };

    Ok(format!("{}-{}", plat_os, plat_arch))
}

/// Find the command link directory (~/.local/bin, /usr/local/bin, etc.)
fn find_command_link_dir() -> Result<PathBuf> {
    // Check common locations
    let home = dirs::home_dir().context("cannot find home directory")?;

    // Try ~/.local/bin first
    let local_bin = home.join(".local").join("bin");
    if local_bin.exists() {
        return Ok(local_bin);
    }

    // Try /usr/local/bin
    let usr_local = PathBuf::from("/usr/local/bin");
    if usr_local.exists() && usr_local.is_dir() {
        return Ok(usr_local);
    }

    // Fallback: create ~/.local/bin
    std::fs::create_dir_all(&local_bin)?;
    Ok(local_bin)
}

/// Unpack a .tar.zst bundle into a directory.
fn unpack_bundle(archive: &Path, dest: &Path) -> Result<()> {
    let output = std::process::Command::new("tar")
        .arg("--zstd")
        .arg("-xf")
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .output()
        .context("failed to run tar")?;

    if !output.status.success() {
        bail!(
            "tar extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_platform() {
        let platform = detect_platform().unwrap();
        assert!(
            platform.contains("linux") || platform.contains("darwin") || platform.contains("win")
        );
    }

    #[test]
    fn test_read_checkout_sha_invalid_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = read_checkout_sha(tmp.path());
        // Not a git repo — should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_find_command_link_dir() {
        let dir = find_command_link_dir().unwrap();
        assert!(dir.is_dir() || dir.parent().is_some());
    }

    #[test]
    fn test_adopt_undo_fails_without_pre_adopt() {
        let tmp = tempfile::tempdir().unwrap();
        let result = adopt_undo(tmp.path());
        assert!(result.is_err());
    }
}
