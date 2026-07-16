//! Launch verb — resolve tree, build env, exec venv python.
//!
//! See docs/updater-world.md §2.5.1.

use crate::tree::{build_child_env, resolve_tree_root, TreeKind};
use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::Command;

/// Launch the Hermes agent: resolve tree → build env → exec venv python.
///
/// On POSIX, uses `execvp` (replaces the process). On Windows, spawns +
/// waits + mirrors the exit code.
pub fn launch(args: Vec<String>) -> Result<()> {
    let exe = std::env::current_exe().context("cannot get current exe")?;
    let tree = resolve_tree_root(&exe)
        .context("cannot determine hermes tree root — are you running from a slot or checkout?")?;

    // Find the venv python
    let venv = match tree.kind {
        TreeKind::Slot => tree.root.join("runtime").join("venv"),
        TreeKind::Checkout => tree.root.join(".venv"),
    };
    let python = venv.join("bin").join("python");
    // Fallback for Windows
    let python = if python.exists() {
        python
    } else {
        let win_path = venv.join("Scripts").join("python.exe");
        if win_path.exists() {
            win_path
        } else {
            python
        }
    };

    // Self-check: verify venv python exists
    if !python.exists() {
        eprintln!("hermes: this tree's virtualenv is missing or broken.");
        eprintln!("  tree: {}", tree.root.display());
        match tree.kind {
            TreeKind::Checkout => {
                eprintln!("  fix:  hermes dev sync        (source checkout)");
            }
            TreeKind::Slot => {
                eprintln!("  fix:  hermes-updater apply   (managed install)");
            }
        }
        std::process::exit(3);
    }

    // Self-check: verify core imports work (cached via .launcher-ok stamp)
    let stamp_ok = check_launcher_stamp(&tree, &python);
    if !stamp_ok {
        let result = Command::new(&python)
            .arg("-c")
            .arg("import hermes_cli")
            .output();
        match result {
            Ok(output) if output.status.success() => {
                write_launcher_stamp(&tree, &python);
            }
            Ok(_) | Err(_) => {
                eprintln!("hermes: this tree's virtualenv is missing or broken.");
                eprintln!("  tree: {}", tree.root.display());
                match tree.kind {
                    TreeKind::Checkout => {
                        eprintln!("  fix:  hermes dev sync        (source checkout)");
                    }
                    TreeKind::Slot => {
                        eprintln!("  fix:  hermes-updater apply   (managed install)");
                    }
                }
                std::process::exit(3);
            }
        }
    }

    // Build the environment
    let env = build_child_env(&tree);

    // Execute: python -m hermes_cli.main <args...>
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let mut cmd = Command::new(&python);
        cmd.arg("-m").arg("hermes_cli.main");
        cmd.args(&args);
        for (k, v) in &env {
            cmd.env(k, v);
        }
        let err = cmd.exec();
        bail!("failed to exec {}: {}", python.display(), err);
    }

    #[cfg(not(unix))]
    {
        let mut cmd = Command::new(&python);
        cmd.arg("-m").arg("hermes_cli.main");
        cmd.args(&args);
        for (k, v) in &env {
            cmd.env(k, v);
        }
        let status = cmd.status().context("failed to spawn python")?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

/// Compute the stamp key: sha256 of (pyvenv.cfg + uv.lock + interpreter path).
fn stamp_key(tree: &crate::tree::ResolvedTree, python: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();

    // pyvenv.cfg
    let pyvenv_cfg = match tree.kind {
        TreeKind::Slot => tree.root.join("runtime").join("venv").join("pyvenv.cfg"),
        TreeKind::Checkout => tree.root.join(".venv").join("pyvenv.cfg"),
    };
    if let Ok(content) = std::fs::read(&pyvenv_cfg) {
        hasher.update(&content);
    }

    // uv.lock
    let uv_lock = tree.root.join("uv.lock");
    if let Ok(content) = std::fs::read(&uv_lock) {
        hasher.update(&content);
    }

    // interpreter path
    hasher.update(python.to_string_lossy().as_bytes());

    format!("{:x}", hasher.finalize())
}

/// Check if the .launcher-ok stamp is current.
fn check_launcher_stamp(tree: &crate::tree::ResolvedTree, python: &std::path::Path) -> bool {
    let stamp_path = stamp_path(tree);
    let key = stamp_key(tree, python);
    std::fs::read_to_string(&stamp_path)
        .map(|content| content.trim() == key)
        .unwrap_or(false)
}

/// Write the .launcher-ok stamp.
fn write_launcher_stamp(tree: &crate::tree::ResolvedTree, python: &std::path::Path) {
    let stamp_path = stamp_path(tree);
    let key = stamp_key(tree, python);
    let _ = std::fs::write(&stamp_path, key);
}

/// Path to the .launcher-ok stamp file.
fn stamp_path(tree: &crate::tree::ResolvedTree) -> PathBuf {
    match tree.kind {
        TreeKind::Slot => tree.root.join("runtime").join("venv").join(".launcher-ok"),
        TreeKind::Checkout => tree.root.join(".venv").join(".launcher-ok"),
    }
}
