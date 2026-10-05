//! Filesystem paths + logging setup.
//!
//! Mirrors `hermes_constants.get_hermes_home()` from the Python CLI:
//!   Windows: %LOCALAPPDATA%\hermes
//!   macOS:   ~/.hermes
//!   Linux:   ~/.hermes  (override via $HERMES_HOME)
//!
//! NOTE (macOS): Python's get_hermes_home(), scripts/install.sh, and the
//! Electron desktop's resolveHermesHome() ALL use ~/.hermes on macOS — there
//! is no ~/Library/Application Support branch anywhere else. An earlier
//! version of this file used Application Support, which drifted from every
//! other component: the installer wrote the install to one dir and the
//! desktop looked for it in another, so first launch never found the backend.
//!
//! IMPORTANT: this must match exactly. Drift here means install.ps1
//! writes to one place and the installer reads from another, breaking
//! the bootstrap-complete check.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::process::Command;
use tracing_appender::non_blocking::WorkerGuard;

/// Returns the canonical Hermes home directory, respecting $HERMES_HOME if set.
pub fn hermes_home() -> PathBuf {
    if let Ok(override_path) = std::env::var("HERMES_HOME") {
        if !override_path.trim().is_empty() {
            return PathBuf::from(override_path);
        }
    }

    #[cfg(target_os = "windows")]
    {
        // %LOCALAPPDATA%\hermes — matches scripts/install.ps1's $HermesHome.
        if let Some(local_app_data) = dirs::data_local_dir() {
            return local_app_data.join("hermes");
        }
    }

    // macOS + Linux + fallback: ~/.hermes (matches Python get_hermes_home(),
    // install.sh, and the Electron desktop's resolveHermesHome()).
    if let Some(home) = dirs::home_dir() {
        return home.join(".hermes");
    }

    // Last resort — current dir, almost certainly wrong but at least
    // doesn't panic.
    PathBuf::from(".hermes")
}

pub fn log_dir() -> PathBuf {
    hermes_home().join("logs")
}

pub fn log_path() -> PathBuf {
    log_dir().join("bootstrap-installer.log")
}

pub fn bootstrap_cache_dir() -> PathBuf {
    hermes_home().join("bootstrap-cache")
}

/// Stable location the installer copies itself to after a successful install.
/// The desktop app re-invokes this with `--update`, and the start-menu /
/// desktop shortcuts can point users back to it. Lives directly under
/// HERMES_HOME so it survives repo checkout deletion (unlike anything under
/// hermes-agent/).
///
/// On Windows this is `%LOCALAPPDATA%\hermes\hermes-setup.exe`; on other
/// platforms the extension differs but the directory is the same.
pub fn installer_dest() -> PathBuf {
    let name = if cfg!(target_os = "windows") {
        "hermes-setup.exe"
    } else {
        "hermes-setup"
    };
    hermes_home().join(name)
}

/// Marker the updater writes for the duration of an in-app update and removes
/// when it finishes (see update.rs `UpdateMarkerGuard`). A freshly-launched
/// desktop checks this before spawning its own local backend: spawning one
/// mid-update re-locks the venv shim and triggers `force_kill_other_hermes`,
/// which then kills that legitimate backend in a respawn loop (#50238).
///
/// Lives directly under HERMES_HOME (same rationale as `installer_dest`) so the
/// Electron desktop — which resolves HERMES_HOME identically and pins it into
/// the updater's env — agrees on the exact path.
pub fn update_in_progress_marker() -> PathBuf {
    hermes_home().join(".hermes-update-in-progress")
}

/// Copy the currently-running installer binary to `installer_dest()` so it's
/// available for future `--update` runs and shortcut launches.
///
/// No-ops (returns Ok) when the running exe is ALREADY the destination — which
/// is exactly the case during an `--update` run (the desktop launched us FROM
/// that path), where copying onto ourselves would be a Windows sharing
/// violation. Best-effort: a failure here must not fail the install, so the
/// caller logs and continues.
///
/// NOTE: because of that no-op, a user's staged installer is only ever written
/// by a full install/repair. Every later `--update` runs the ORIGINAL binary,
/// so an installer-protocol change can strand the whole installed base on a
/// binary that predates it (see `restage_from_checkout`, which repairs this
/// from the freshly-updated checkout).
pub fn copy_self_to_hermes_home() -> std::io::Result<()> {
    let src = std::env::current_exe()?;
    let dest = installer_dest();

    // Skip if we're already running from the destination (update re-invocation
    // or a prior copy). canonicalize both so symlinks / 8.3 short paths / case
    // differences don't trick us into a self-copy.
    let same = match (src.canonicalize(), dest.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => src == dest,
    };
    if same {
        tracing::info!(?dest, "installer already at destination; skipping self-copy");
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&src, &dest)?;
    repair_macos_installer_helper(&dest);
    tracing::info!(?src, ?dest, "copied installer to HERMES_HOME");
    Ok(())
}

#[cfg(target_os = "macos")]
fn repair_macos_installer_helper(path: &Path) {
    // The staged helper may inherit quarantine from the downloaded installer.
    // Desktop later launches this exact file for in-app updates, so make it
    // executable before the update handoff reaches LaunchServices/Gatekeeper.
    let _ = Command::new("/usr/bin/xattr")
        .args(["-cr"])
        .arg(path)
        .status();

    let verify = Command::new("/usr/bin/codesign")
        .arg("--verify")
        .arg(path)
        .status();

    if !matches!(verify, Ok(status) if status.success()) {
        let _ = Command::new("/usr/bin/codesign")
            .args(["--force", "--sign", "-"])
            .arg(path)
            .status();
    }
}

#[cfg(not(target_os = "macos"))]
fn repair_macos_installer_helper(_path: &Path) {}

/// Where the bootstrap-complete marker lives (existence-only for the Rust
/// installer fast path; JSON schema-checked by the Electron app). Per main.ts:
///   const BOOTSTRAP_COMPLETE_MARKER = path.join(ACTIVE_HERMES_ROOT, '.hermes-bootstrap-complete')
/// We don't always know ACTIVE_HERMES_ROOT until install.ps1 reports it, so
/// this is a probe helper, not a definitive path.
pub fn likely_bootstrap_marker(install_root: &Path) -> PathBuf {
    install_root.join(".hermes-bootstrap-complete")
}

/// Initializes tracing to bootstrap-installer.log under HERMES_HOME/logs/.
/// Returns a guard that flushes the appender on drop — keep it alive for
/// the lifetime of the process.
pub fn init_logging() -> Option<WorkerGuard> {
    let dir = log_dir();
    if let Err(err) = std::fs::create_dir_all(&dir) {
        // No log dir → log to stderr only. Don't panic; the installer
        // should still be usable on an exotic filesystem.
        eprintln!("[hermes-setup] could not create log dir {dir:?}: {err}");
        return None;
    }

    let file_appender = tracing_appender::rolling::never(&dir, "bootstrap-installer.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::try_from_env("HERMES_BOOTSTRAP_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(true)
        .init();

    Some(guard)
}

// ---------------------------------------------------------------------------
// Path validation
// ---------------------------------------------------------------------------

/// Reject path-traversal payloads in a user/env-controlled path string.
///
/// Checked twice: once as the native OS parses it, once with backslashes
/// normalized to `/` so a Windows-style `..\` payload cannot slip through a
/// Unix (or otherwise non-NTFS) parser. `.` components are harmless (they
/// normalize away); any `..` component is a traversal attempt. Windows NTFS
/// alternate data streams (`name:ads`) and NUL bytes are rejected outright —
/// neither can be a legitimate Hermes home.
pub(crate) fn contains_traversal(path: &str) -> bool {
    contains_traversal_parsed(path) || contains_traversal_parsed(&path.replace('\\', "/"))
}

fn contains_traversal_parsed(path: &str) -> bool {
    // A Windows drive prefix ("C:") is only a prefix when it is a single
    // ASCII letter followed by ':' — anything longer ("foo:stream") is an
    // NTFS ADS name and stays inside the component scan below.
    let path = if path.len() >= 2
        && path.as_bytes()[0].is_ascii_alphabetic()
        && path.as_bytes()[1] == b':'
    {
        &path[2..]
    } else {
        path
    };

    for raw in Path::new(path).components() {
        match raw {
            // "." only survives parsing when it is the whole input.
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => return true,
            std::path::Component::Normal(part) => {
                // Reject Windows ADS separators anywhere in a normal
                // component ("foo:stream"), plus NUL (truncation attacks).
                let part = part.to_string_lossy();
                if part.contains(':') || part.contains('\0') {
                    return true;
                }
            }
            // Root prefix (e.g. `/` on unix, `C:\`-prefix or UNC server on
            // windows) is fine in itself; the `..`/ADS checks above cover the
            // dangerous parts.
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {}
        }
    }
    false
}

/// Resolve a path to its canonical absolute form, creating the directory if
/// needed, and reject symlink escapes.
///
/// Symlinks are resolved by `canonicalize()`, so a link pointing outside
/// `base` is caught by the prefix check below (a lexical join alone would
/// miss it).
fn canonicalize_within(dir: &Path, base: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("could not create directory {}", dir.display()))?;
    let canonical_dir = dir
        .canonicalize()
        .with_context(|| format!("could not canonicalize {}", dir.display()))?;
    let canonical_base = base
        .canonicalize()
        .with_context(|| format!("could not canonicalize {}", base.display()))?;
    if !canonical_dir.starts_with(&canonical_base) {
        anyhow::bail!(
            "path {} resolves outside the allowed base directory {}",
            canonical_dir.display(),
            canonical_base.display()
        );
    }
    Ok(canonical_dir)
}

/// Validate a user/env-supplied Hermes home directory and return the
/// canonical `install_root` (`<hermes_home>/hermes-agent`) for the bootstrap.
///
/// Defense in depth against path traversal:
///   1. Lexical check: reject `..` components, NTFS ADS separators, NULs.
///   2. Canonical check: `create_dir_all` + `canonicalize` both the install
///      root and the base, then require the root to stay within the base.
///      This defeats symlink redirection that the lexical check cannot see.
///
/// The base is `crate::paths::hermes_home()` when no explicit override is
/// supplied (the env-var default), or the override string itself — an
/// operator may legitimately install to a custom root, so the requirement is
/// only that the final path stays *within* the requested base after
/// canonicalization, never that it equals the OS default.
pub(crate) fn validate_install_root(hermes_home_override: &str) -> Result<PathBuf> {
    let trimmed = hermes_home_override.trim();
    if trimmed.is_empty() {
        anyhow::bail!("hermes home override is empty");
    }
    if contains_traversal(trimmed) {
        anyhow::bail!("hermes home override contains a path traversal sequence: {trimmed:?}");
    }

    let base = PathBuf::from(trimmed);
    let install_root = base.join("hermes-agent");
    canonicalize_within(&install_root, &base)
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn get_log_path() -> String {
    log_path().to_string_lossy().into_owned()
}

#[tauri::command]
pub fn get_hermes_home() -> String {
    hermes_home().to_string_lossy().into_owned()
}

#[tauri::command]
pub fn open_log_dir(app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let path = log_dir();
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Tests ()
// ---------------------------------------------------------------------------

#[cfg(test)]
mod path_validation_tests {
    use super::*;

    #[test]
    fn contains_traversal_rejects_parent_components() {
        assert!(contains_traversal(".."));
        assert!(contains_traversal("../etc/passwd"));
        assert!(contains_traversal("a/../b"));
        assert!(contains_traversal("..\\windows\\system32"));
        assert!(contains_traversal("C:\\..\\..\\Windows"));
        assert!(contains_traversal("foo:stream"));
        assert!(contains_traversal("C:\\foo:ads"));
        assert!(contains_traversal("foo\0bar"));
    }

    #[test]
    fn contains_traversal_allows_ordinary_paths() {
        assert!(!contains_traversal("/home/user/.hermes"));
        assert!(!contains_traversal(".hermes"));
        assert!(!contains_traversal("./hermes"));
        assert!(!contains_traversal("C:\\Users\\dev\\.hermes"));
        assert!(!contains_traversal("~/.hermes"));
    }

    #[test]
    fn validate_install_root_accepts_normal_override() {
        let dir =
            std::env::temp_dir().join(format!("hermes-validate-ok-test-{}", std::process::id()));
        let result = validate_install_root(dir.to_string_lossy().as_ref())
            .expect("plain override must validate");
        let canonical_dir = dir.canonicalize().unwrap();
        assert_eq!(result, canonical_dir.join("hermes-agent"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_install_root_accepts_dot_components() {
        // `.` components normalize away and must not be rejected (the OS
        // default path form itself can contain none, but a hand-typed
        // override might).
        let dir =
            std::env::temp_dir().join(format!("hermes-validate-dot-test-{}", std::process::id()));
        let with_dots = format!("{}/./sub", dir.to_string_lossy());
        let result = validate_install_root(&with_dots).expect("dot components must validate");
        let expected = dir.join("sub").canonicalize().unwrap();
        assert_eq!(result, expected.join("hermes-agent"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_install_root_rejects_traversal() {
        // Lexical rejection: must fire before any directory is created.
        for evil in [
            "/tmp/hermes-evil-base/../../etc",
            "..",
            "C:\\..\\..\\Windows",
            "foo:stream",
        ] {
            let err = validate_install_root(evil)
                .expect_err(&format!("traversal payload {evil:?} must be rejected"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains("path traversal sequence"),
                "unexpected error for {evil:?}: {msg}"
            );
        }
        // Nothing outside tmp may have been created by the rejected inputs.
    }

    #[test]
    fn validate_install_root_rejects_symlink_escape() {
        // Canonical containment: a symlinked base whose target lives outside
        // must still pass only if the install root stays within the resolved
        // base; here the *install root* escapes via symlink.
        let base = std::env::temp_dir().join(format!("hermes-validate-sym-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let link = base.join("hermes-agent");
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        #[cfg(not(unix))]
        {
            let _ = &link;
            std::fs::remove_dir_all(&base).unwrap();
            return;
        }

        let err = validate_install_root(base.to_string_lossy().as_ref())
            .expect_err("symlink escaping the base must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("outside the allowed base directory"),
            "unexpected error: {msg}"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn validate_install_root_rejects_empty_override() {
        let err = validate_install_root("   ").expect_err("empty override must be rejected");
        assert!(format!("{err:#}").contains("empty"));
    }
}
