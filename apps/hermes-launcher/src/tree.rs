//! Tree-root resolution + environment setup.
//!
//! The launcher self-locates its tree by walking up from the binary's real
//! path (symlinks resolved) to find one of:
//!   - `current.txt` → managed root (resolve the active version, recurse
//!     into `versions/<v>/`)
//!   - `manifest.json` → slot (bundle layout)
//!   - `pyproject.toml` + `.git` → checkout (source tree)
//!
//! See docs/updater-world.md §2.5.1.

use std::path::{Path, PathBuf};

/// What kind of tree the launcher is running from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeKind {
    /// A managed slot: `<bundle_root>/` with `manifest.json`.
    /// Code in `runtime/venv/site-packages`, assets in `app/`, `ui/`, etc.
    Slot,

    /// A source checkout: has `pyproject.toml` + `.git`.
    /// Code in the tree itself (editable install), assets alongside.
    Checkout,
}

/// The resolved tree root + its kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTree {
    pub root: PathBuf,
    pub kind: TreeKind,
}

/// Walk up from `exe_path` to find the tree root.
///
/// `exe_path` should be the result of `std::env::current_exe()` (symlinks
/// resolved) or a mock for testing.
///
/// Returns `Err` if no tree root is found (neither manifest.json nor
/// pyproject.toml + .git in any ancestor).
pub fn resolve_tree_root(exe_path: &Path) -> anyhow::Result<ResolvedTree> {
    let start = exe_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("cannot get parent of exe path"))?;

    for dir in start.ancestors() {
        // Slot: manifest.json present
        if dir.join("manifest.json").is_file() {
            return Ok(ResolvedTree {
                root: dir.to_path_buf(),
                kind: TreeKind::Slot,
            });
        }

        // Checkout: pyproject.toml + .git (dir or file for worktrees)
        if dir.join("pyproject.toml").is_file() && has_git(dir) {
            return Ok(ResolvedTree {
                root: dir.to_path_buf(),
                kind: TreeKind::Checkout,
            });
        }
    }

    Err(anyhow::anyhow!(
        "no hermes tree root found (no manifest.json or pyproject.toml+.git in ancestors of {})",
        exe_path.display()
    ))
}

/// Check if a directory has a `.git` (either a directory for regular
/// clones, or a FILE for worktrees — `gitdir: /path/to/main/.git/worktrees/name`).
fn has_git(dir: &Path) -> bool {
    let git = dir.join(".git");
    git.is_dir() || git.is_file()
}

/// Build the environment for a child process (the venv python).
///
/// Contract from §2.5.1:
/// - PATH: prepend `<tree>/runtime/tools` + `<tree>/runtime/node/bin` +
///   `<tree>/runtime/python/bin` (slot) or `$HERMES_HOME/node/bin` +
///   `$HERMES_HOME/bin` (checkout)
/// - VIRTUAL_ENV: `<tree>/runtime/venv` (slot) or `<tree>/.venv` (checkout)
/// - UV_PYTHON: same as VIRTUAL_ENV's python
/// - UV_NO_CONFIG: 1
/// - Remove PYTHONPATH, PYTHONHOME
pub fn build_child_env(tree: &ResolvedTree) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = Vec::new();

    // Start with the current environment, filtering out vars we'll set/replace.
    let skip_keys = |k: &str| {
        k == "PYTHONPATH"
            || k == "PYTHONHOME"
            || k == "PATH"
            || k == "VIRTUAL_ENV"
            || k == "UV_PYTHON"
            || k == "UV_NO_CONFIG"
    };
    for (k, v) in std::env::vars() {
        if !skip_keys(&k) {
            env.push((k, v));
        }
    }

    match tree.kind {
        TreeKind::Slot => {
            let tools = tree.root.join("runtime").join("tools");
            let node_bin = tree.root.join("runtime").join("node").join("bin");
            let python_bin = tree.root.join("runtime").join("python").join("bin");
            let venv = tree.root.join("runtime").join("venv");

            // Prepend to PATH
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!(
                "{}:{}:{}:{}",
                tools.display(),
                node_bin.display(),
                python_bin.display(),
                current_path
            );
            env.push(("PATH".to_string(), new_path));
            env.push(("VIRTUAL_ENV".to_string(), venv.to_string_lossy().into()));
            env.push(("UV_PYTHON".to_string(), venv.to_string_lossy().into()));
        }
        TreeKind::Checkout => {
            let hermes_home = std::env::var("HERMES_HOME").unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap()
                    .join(".hermes")
                    .to_string_lossy()
                    .into()
            });
            let node_bin = format!("{}/node/bin", hermes_home);
            let bin_dir = format!("{}/bin", hermes_home);
            let venv = tree.root.join(".venv");

            // Prepend to PATH
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{}:{}:{}", node_bin, bin_dir, current_path);
            env.push(("PATH".to_string(), new_path));
            env.push(("VIRTUAL_ENV".to_string(), venv.to_string_lossy().into()));
            env.push(("UV_PYTHON".to_string(), venv.to_string_lossy().into()));
        }
    }

    env.push(("UV_NO_CONFIG".to_string(), "1".to_string()));

    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    /// Create a slot layout under `root`.
    /// The exe path is at `<root>/bin/hermes`.
    fn make_slot_layout(root: &Path) -> PathBuf {
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let exe = bin.join("hermes");
        fs::write(&exe, "# stub").unwrap();
        fs::write(root.join("manifest.json"), "{}").unwrap();
        // Create runtime dirs so env building doesn't panic
        for d in [
            "runtime/tools",
            "runtime/node/bin",
            "runtime/python/bin",
            "runtime/venv",
        ] {
            fs::create_dir_all(root.join(d)).unwrap();
        }
        exe
    }

    /// Create a checkout layout under `root`.
    /// The exe path is at `<root>/bin/hermes`.
    fn make_checkout_layout(root: &Path) -> PathBuf {
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let exe = bin.join("hermes");
        fs::write(&exe, "# stub").unwrap();
        fs::write(
            root.join("pyproject.toml"),
            "[project]\nname = \"hermes-agent\"",
        )
        .unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        fs::create_dir_all(root.join(".venv")).unwrap();
        exe
    }

    /// Create a worktree layout (`.git` is a FILE, not a dir).
    fn make_worktree_layout(root: &Path) -> PathBuf {
        let exe = make_checkout_layout(root);
        // Replace .git dir with a .git file
        fs::remove_dir_all(root.join(".git")).unwrap();
        let mut f = fs::File::create(root.join(".git")).unwrap();
        writeln!(f, "gitdir: /some/main/repo/.git/worktrees/foo").unwrap();
        exe
    }

    #[test]
    fn test_slot_layout_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = make_slot_layout(tmp.path());
        let tree = resolve_tree_root(&exe).unwrap();
        assert_eq!(tree.kind, TreeKind::Slot);
        assert_eq!(tree.root, tmp.path());
    }

    #[test]
    fn test_checkout_layout_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = make_checkout_layout(tmp.path());
        let tree = resolve_tree_root(&exe).unwrap();
        assert_eq!(tree.kind, TreeKind::Checkout);
        assert_eq!(tree.root, tmp.path());
    }

    #[test]
    fn test_worktree_layout_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = make_worktree_layout(tmp.path());
        let tree = resolve_tree_root(&exe).unwrap();
        assert_eq!(tree.kind, TreeKind::Checkout);
        assert_eq!(tree.root, tmp.path());
    }

    #[test]
    fn test_no_tree_root_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("hermes");
        fs::write(&exe, "# stub").unwrap();
        let result = resolve_tree_root(&exe);
        assert!(result.is_err());
    }

    #[test]
    fn test_slot_env_has_correct_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = make_slot_layout(tmp.path());
        let tree = resolve_tree_root(&exe).unwrap();
        let env = build_child_env(&tree);

        let path: String = env
            .iter()
            .find(|(k, _)| k == "PATH")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert!(path.contains("runtime/tools"));
        assert!(path.contains("runtime/node/bin"));
        assert!(path.contains("runtime/python/bin"));

        let venv: String = env
            .iter()
            .find(|(k, _)| k == "VIRTUAL_ENV")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert!(venv.ends_with("runtime/venv"));

        assert!(env.iter().any(|(k, _)| k == "UV_NO_CONFIG"));
    }

    #[test]
    fn test_env_removes_pythonpath_pythonhome() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = make_checkout_layout(tmp.path());
        let tree = resolve_tree_root(&exe).unwrap();

        // Set the env vars before building
        std::env::set_var("PYTHONPATH", "/should/be/removed");
        std::env::set_var("PYTHONHOME", "/should/be/removed");

        let env = build_child_env(&tree);
        assert!(!env.iter().any(|(k, _)| k == "PYTHONPATH"));
        assert!(!env.iter().any(|(k, _)| k == "PYTHONHOME"));

        // Clean up
        std::env::remove_var("PYTHONPATH");
        std::env::remove_var("PYTHONHOME");
    }
}
