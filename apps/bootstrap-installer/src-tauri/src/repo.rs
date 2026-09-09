//! Locate the North Forge checkout this installer should bootstrap, and
//! derive the sibling `-venv` / `-data` paths the bootstrap will create.
//!
//! North Forge's model (unlike upstream Hermes, which git-clones the repo
//! into a fixed `%LOCALAPPDATA%` path): the checkout is ALREADY on a drive.
//! We only have to find it. Resolution order:
//!   1. `$NORTH_FORGE_REPO_ROOT` / `$HERMES_SETUP_DEV_REPO_ROOT` env override.
//!   2. Walk up from the installer .exe's own directory.
//!   3. Scan each drive root for exactly one child folder that is a checkout.
//!
//! A directory is a North Forge checkout when it contains BOTH
//! `scripts/bootstrap-north-forge.ps1` and `pyproject.toml`.

use std::path::{Path, PathBuf};

use serde::Serialize;

pub const BOOTSTRAP_REL: &str = "scripts/bootstrap-north-forge.ps1";
pub const LAUNCHER_REL: &str = "north-forge.cmd";

/// Everything the frontend needs to show the welcome / location screen and
/// to kick off (or launch) a bootstrap.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoInfo {
    /// The resolved checkout, if one was found (or accepted via `set_repo_root`).
    pub repo_root: Option<String>,
    /// `<parent>` of the checkout.
    pub parent: Option<String>,
    /// The checkout's leaf folder name.
    pub leaf: Option<String>,
    /// `<parent>\<leaf>-venv` — what the bootstrap will create.
    pub venv_dir: Option<String>,
    /// `<parent>\<leaf>-data` — the HERMES_HOME the launcher will use.
    pub data_dir: Option<String>,
    /// `<repo>\scripts\bootstrap-north-forge.ps1`, only when it exists.
    pub bootstrap_script: Option<String>,
    /// `<repo>\north-forge.cmd`, only when it exists.
    pub launcher_cmd: Option<String>,
    /// `<venv>\.nf-bootstrapped` is already present.
    pub bootstrapped: bool,
    /// The checkout sits on the OS system drive. The drive picker refuses to
    /// TARGET the system drive (North Forge is meant to travel on an external
    /// drive), but an already-present checkout there is surfaced with a
    /// warning rather than hard-blocked.
    pub on_system_drive: bool,
    /// How the checkout was found: "env", "exe", "scan", "picked", or "none".
    pub source: String,
    /// Every drive root we can see, for the location picker.
    pub drives: Vec<DriveInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DriveInfo {
    /// e.g. `"D:"`.
    pub letter: String,
    /// e.g. `"D:\\"`.
    pub path: String,
    pub is_system: bool,
    /// A North Forge checkout was found at the drive root or one level under it.
    pub has_checkout: bool,
    pub checkout_path: Option<String>,
}

/// True when `dir` looks like a North Forge checkout.
pub fn is_checkout(dir: &Path) -> bool {
    dir.join(BOOTSTRAP_REL).is_file() && dir.join("pyproject.toml").is_file()
}

fn system_drive_prefix() -> String {
    std::env::var("SystemDrive")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "C:".to_string())
        .trim_end_matches('\\')
        .to_ascii_uppercase()
}

/// Case-insensitive "is `path` the drive named by `sys_prefix` (e.g. `C:`) or
/// under it". Pure — split out so the policy is testable without touching the
/// real `SystemDrive` env var.
fn is_under_drive_prefix(path: &Path, sys_prefix: &str) -> bool {
    let up = path.to_string_lossy().to_ascii_uppercase();
    let sp = sys_prefix.trim_end_matches('\\').to_ascii_uppercase();
    up == sp || up.starts_with(&format!("{sp}\\"))
}

fn on_system_drive(path: &Path) -> bool {
    is_under_drive_prefix(path, &system_drive_prefix())
}

/// Pure pre-spawn policy, given the facts already gathered. Kept separate from
/// [`validate_target`] so the "real checkout, not the system drive" rules can
/// be unit-tested without a real filesystem or a real system drive.
fn target_policy(root: &Path, is_checkout: bool, on_system_drive: bool) -> Result<(), String> {
    if !is_checkout {
        return Err(format!(
            "{} is missing scripts\\bootstrap-north-forge.ps1 or pyproject.toml — \
             the checkout looks incomplete.",
            root.display()
        ));
    }
    if on_system_drive {
        return Err(format!(
            "{} is on the system drive. North Forge is meant to run from a separate \
             drive that can travel between machines — move the checkout to another \
             drive and pick it there.",
            root.display()
        ));
    }
    Ok(())
}

/// Enumerate `A:\` .. `Z:\` that exist. No winapi — a bare `is_dir()` on the
/// root is enough to tell a mounted volume from an empty slot, and drive-type
/// classification isn't needed for the lightweight tier.
pub fn list_drives() -> Vec<DriveInfo> {
    let sys = system_drive_prefix();
    let mut out = Vec::new();
    for c in b'A'..=b'Z' {
        let letter = format!("{}:", c as char);
        let root = format!("{letter}\\");
        if !Path::new(&root).is_dir() {
            continue;
        }
        let is_system = letter.eq_ignore_ascii_case(&sys);
        let checkout = find_checkout_on_drive(Path::new(&root));
        out.push(DriveInfo {
            letter: letter.clone(),
            path: root,
            is_system,
            has_checkout: checkout.is_some(),
            checkout_path: checkout.map(|p| p.to_string_lossy().into_owned()),
        });
    }
    out
}

/// Every checkout at `root` itself or one level under it. `root` first: if the
/// picked path *is* a checkout that's the answer regardless of its children.
fn checkouts_under(root: &Path) -> Vec<PathBuf> {
    if is_checkout(root) {
        return vec![root.to_path_buf()];
    }
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && is_checkout(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// The drive root itself, or exactly one immediate child, that is a checkout.
/// `None` for zero *or* more than one (ambiguous — the caller must make the
/// user choose).
fn find_checkout_on_drive(root: &Path) -> Option<PathBuf> {
    let mut found = checkouts_under(root);
    if found.len() == 1 {
        found.pop()
    } else {
        None
    }
}

/// Walk up from `start` (inclusive) up to `max_up` levels looking for a checkout.
fn walk_up_for_checkout(start: &Path, max_up: usize) -> Option<PathBuf> {
    let mut cur = Some(start);
    let mut steps = 0;
    while let Some(dir) = cur {
        if is_checkout(dir) {
            return Some(dir.to_path_buf());
        }
        if steps >= max_up {
            break;
        }
        steps += 1;
        cur = dir.parent();
    }
    None
}

/// Resolve the checkout by env override → exe-relative walk-up → drive scan.
/// Returns `(repo_root, source_tag)`.
///
/// The drive scan auto-picks **only when exactly one checkout exists across
/// every visible drive**. Zero → `None` ("none"); two or more → `None`
/// ("ambiguous") so the UI forces an explicit choice instead of silently
/// grabbing the first drive letter (PC-2026-09-08-003). The env override and
/// the exe-relative walk-up are explicit, unambiguous signals and still win
/// outright.
fn resolve_checkout() -> Option<(PathBuf, String)> {
    for var in ["NORTH_FORGE_REPO_ROOT", "HERMES_SETUP_DEV_REPO_ROOT"] {
        if let Ok(v) = std::env::var(var) {
            let p = PathBuf::from(v.trim());
            if !v.trim().is_empty() && is_checkout(&p) {
                return Some((p, "env".to_string()));
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(root) = walk_up_for_checkout(dir, 6) {
                return Some((root, "exe".to_string()));
            }
        }
    }

    let mut candidates: Vec<PathBuf> = list_drives()
        .into_iter()
        .filter_map(|d| d.checkout_path.map(PathBuf::from))
        .collect();
    candidates.sort();
    candidates.dedup();
    match candidates.len() {
        1 => Some((candidates.pop().unwrap(), "scan".to_string())),
        _ => None,
    }
}

/// Shared pre-spawn gate. A target path — auto-detected or handed in by the
/// frontend — must be canonicalized, must resolve to a real North Forge
/// checkout (`scripts/bootstrap-north-forge.ps1` **and** `pyproject.toml`),
/// and must not sit on the OS system drive. Both `set_repo_root` (the picker)
/// and `run_bootstrap` (immediately before spawning PowerShell) call this, so
/// a value coming from the webview can never skip the checks
/// (PC-2026-09-08-002).
pub fn validate_target(picked: &Path) -> Result<PathBuf, String> {
    if !picked.is_dir() {
        return Err(format!("{} is not a folder.", picked.display()));
    }

    let found = checkouts_under(picked);
    let root = match found.len() {
        1 => found.into_iter().next().unwrap(),
        0 => {
            return Err(format!(
                "No North Forge checkout in {}. Pick the folder that contains \
                 scripts\\bootstrap-north-forge.ps1 and pyproject.toml (or its parent).",
                picked.display()
            ));
        }
        n => {
            return Err(format!(
                "{n} North Forge checkouts under {} — pick the exact folder, not its parent.",
                picked.display()
            ));
        }
    };

    let root = std::fs::canonicalize(&root)
        .map(strip_unc_prefix)
        .unwrap_or(root);

    target_policy(&root, is_checkout(&root), on_system_drive(&root))?;
    Ok(root)
}

/// Build a full [`RepoInfo`] for a known (or unknown) checkout root.
pub fn describe(repo_root: Option<PathBuf>, source: &str) -> RepoInfo {
    let drives = list_drives();

    let Some(root) = repo_root else {
        return RepoInfo {
            repo_root: None,
            parent: None,
            leaf: None,
            venv_dir: None,
            data_dir: None,
            bootstrap_script: None,
            launcher_cmd: None,
            bootstrapped: false,
            on_system_drive: false,
            source: source.to_string(),
            drives,
        };
    };

    let root = std::fs::canonicalize(&root)
        .map(strip_unc_prefix)
        .unwrap_or(root);
    let parent = root.parent().map(|p| p.to_path_buf());
    let leaf = root
        .file_name()
        .map(|s| s.to_string_lossy().into_owned());

    let (venv_dir, data_dir) = match (&parent, &leaf) {
        (Some(p), Some(l)) => (
            Some(p.join(format!("{l}-venv"))),
            Some(p.join(format!("{l}-data"))),
        ),
        _ => (None, None),
    };

    let bootstrap_script = {
        let s = root.join(BOOTSTRAP_REL);
        s.is_file().then(|| s)
    };
    let launcher_cmd = {
        let s = root.join(LAUNCHER_REL);
        s.is_file().then(|| s)
    };
    let bootstrapped = venv_dir
        .as_ref()
        .map(|v| v.join(".nf-bootstrapped").is_file())
        .unwrap_or(false);

    RepoInfo {
        repo_root: Some(root.to_string_lossy().into_owned()),
        parent: parent.map(|p| p.to_string_lossy().into_owned()),
        leaf,
        venv_dir: venv_dir.map(|p| p.to_string_lossy().into_owned()),
        data_dir: data_dir.map(|p| p.to_string_lossy().into_owned()),
        bootstrap_script: bootstrap_script.map(|p| p.to_string_lossy().into_owned()),
        launcher_cmd: launcher_cmd.map(|p| p.to_string_lossy().into_owned()),
        bootstrapped,
        on_system_drive: on_system_drive(&root),
        source: source.to_string(),
        drives,
    }
}

/// `\\?\D:\foo` → `D:\foo`. `canonicalize` returns a verbatim path on Windows
/// which PowerShell `-File` and display both dislike.
fn strip_unc_prefix(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }
    p
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

/// Auto-detect the checkout (env → exe → drive scan) and describe it.
///
/// When nothing is auto-resolved, the `source` tag tells the UI *why*:
/// `"ambiguous"` (more than one checkout across the drives — the user must
/// choose) vs `"none"` (nothing found).
#[tauri::command]
pub fn detect_repo() -> RepoInfo {
    match resolve_checkout() {
        Some((root, source)) => describe(Some(root), &source),
        None => {
            let total: usize = list_drives().iter().filter(|d| d.has_checkout).count();
            describe(None, if total > 1 { "ambiguous" } else { "none" })
        }
    }
}

/// Accept a user-picked folder (from the Browse dialog or a drive quick-pick).
/// The path may be the checkout itself or a drive/parent containing exactly
/// one. Runs the shared [`validate_target`] gate — real checkout, not the
/// system drive — so the picker and the bootstrap enforce identical rules.
#[tauri::command]
pub fn set_repo_root(path: String) -> Result<RepoInfo, String> {
    let root = validate_target(&PathBuf::from(path.trim()))?;
    Ok(describe(Some(root), "picked"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Throwaway dir under the OS temp root; removed on drop.
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            let base = std::env::temp_dir().join(format!(
                "nf-setup-test-{tag}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&base).unwrap();
            TmpDir(base)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Make `dir` look like a real North Forge checkout.
    fn seed_checkout(dir: &Path) {
        std::fs::create_dir_all(dir.join("scripts")).unwrap();
        std::fs::write(dir.join(BOOTSTRAP_REL), b"# stub").unwrap();
        std::fs::write(dir.join("pyproject.toml"), b"[project]\n").unwrap();
    }

    #[test]
    fn is_checkout_needs_both_marker_files() {
        let tmp = TmpDir::new("ischeck");
        let d = tmp.path();
        assert!(!is_checkout(d));
        std::fs::create_dir_all(d.join("scripts")).unwrap();
        std::fs::write(d.join(BOOTSTRAP_REL), b"x").unwrap();
        assert!(!is_checkout(d), "bootstrap script alone is not enough");
        std::fs::write(d.join("pyproject.toml"), b"x").unwrap();
        assert!(is_checkout(d));
    }

    #[test]
    fn is_under_drive_prefix_is_case_insensitive_and_boundary_safe() {
        assert!(is_under_drive_prefix(Path::new("c:\\Users\\x"), "C:"));
        assert!(is_under_drive_prefix(Path::new("C:"), "c:"));
        assert!(is_under_drive_prefix(Path::new("C:\\"), "C:"));
        assert!(!is_under_drive_prefix(Path::new("D:\\north-forge-agent"), "C:"));
        // "C:" must not match a "CX:" drive by prefix.
        assert!(!is_under_drive_prefix(Path::new("CX:\\thing"), "C:"));
    }

    #[test]
    fn target_policy_rejects_non_checkout() {
        let err = target_policy(Path::new("D:\\somewhere"), false, false).unwrap_err();
        assert!(err.contains("incomplete"), "{err}");
    }

    #[test]
    fn target_policy_rejects_system_drive_checkout() {
        let err = target_policy(Path::new("C:\\north-forge-agent"), true, true).unwrap_err();
        assert!(err.contains("system drive"), "{err}");
    }

    #[test]
    fn target_policy_accepts_real_offsystem_checkout() {
        assert!(target_policy(Path::new("E:\\north-forge-agent"), true, false).is_ok());
    }

    #[test]
    fn checkouts_under_counts_zero_one_many() {
        let tmp = TmpDir::new("count");
        let root = tmp.path();
        assert_eq!(checkouts_under(root).len(), 0);
        assert!(find_checkout_on_drive(root).is_none());

        seed_checkout(&root.join("alpha"));
        assert_eq!(checkouts_under(root).len(), 1);
        assert_eq!(
            find_checkout_on_drive(root).unwrap().file_name().unwrap(),
            "alpha"
        );

        seed_checkout(&root.join("beta"));
        assert_eq!(checkouts_under(root).len(), 2);
        assert!(
            find_checkout_on_drive(root).is_none(),
            "two checkouts under one root must be ambiguous, not first-wins"
        );
    }

    #[test]
    fn validate_target_errors_are_total_over_bad_input() {
        // not a dir
        assert!(validate_target(Path::new("D:\\nf-setup-test-does-not-exist-xyz")).is_err());

        // a dir, but no checkout in it
        let tmp = TmpDir::new("nocheck");
        let e = validate_target(tmp.path()).unwrap_err();
        assert!(e.contains("No North Forge checkout"), "{e}");

        // a parent holding two checkouts → explicit-choice error, never a silent pick
        let tmp2 = TmpDir::new("two");
        seed_checkout(&tmp2.path().join("one"));
        seed_checkout(&tmp2.path().join("two"));
        let e2 = validate_target(tmp2.path()).unwrap_err();
        assert!(e2.contains("pick the exact folder"), "{e2}");
    }

    #[test]
    fn validate_target_accepts_a_real_checkout_when_not_on_system_drive() {
        let tmp = TmpDir::new("accept");
        let root = tmp.path().join("north-forge-agent");
        seed_checkout(&root);
        // Skip on machines whose temp dir is the system drive — the policy
        // (correctly) refuses that; `target_policy_*` tests cover the rule.
        if on_system_drive(&root) {
            return;
        }
        let ok = validate_target(&root).expect("real off-system checkout should validate");
        assert!(is_checkout(&ok));
    }
}
