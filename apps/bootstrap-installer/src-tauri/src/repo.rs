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

fn on_system_drive(path: &Path) -> bool {
    let sys = system_drive_prefix();
    path.to_string_lossy()
        .to_ascii_uppercase()
        .starts_with(&format!("{sys}\\"))
        || path.to_string_lossy().to_ascii_uppercase() == sys
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

/// The drive root itself, or exactly one immediate child, that is a checkout.
fn find_checkout_on_drive(root: &Path) -> Option<PathBuf> {
    if is_checkout(root) {
        return Some(root.to_path_buf());
    }
    let mut found: Option<PathBuf> = None;
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        if is_checkout(&p) {
            if found.is_some() {
                // More than one — ambiguous; let the user pick explicitly.
                return None;
            }
            found = Some(p);
        }
    }
    found
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

    for d in list_drives() {
        if let Some(path) = d.checkout_path {
            return Some((PathBuf::from(path), "scan".to_string()));
        }
    }

    None
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
#[tauri::command]
pub fn detect_repo() -> RepoInfo {
    match resolve_checkout() {
        Some((root, source)) => describe(Some(root), &source),
        None => describe(None, "none"),
    }
}

/// Accept a user-picked folder (from the Browse dialog or a drive quick-pick).
/// The path may be the checkout itself or a drive/parent containing exactly
/// one. Rejects the system drive as an explicit target.
#[tauri::command]
pub fn set_repo_root(path: String) -> Result<RepoInfo, String> {
    let picked = PathBuf::from(path.trim());
    if !picked.is_dir() {
        return Err(format!("{} is not a folder.", picked.display()));
    }

    let root = if is_checkout(&picked) {
        picked.clone()
    } else if let Some(found) = find_checkout_on_drive(&picked) {
        found
    } else {
        return Err(format!(
            "No North Forge checkout found in {}. Pick the folder that contains \
             scripts\\bootstrap-north-forge.ps1 (or its parent).",
            picked.display()
        ));
    };

    if on_system_drive(&root) {
        return Err(format!(
            "{} is on the system drive. North Forge is meant to run from a \
             separate drive that can travel between machines — move the checkout \
             to another drive and pick it there.",
            root.display()
        ));
    }

    Ok(describe(Some(root), "picked"))
}
