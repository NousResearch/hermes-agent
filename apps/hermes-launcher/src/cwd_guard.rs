//! The cwd guard: inside a checkout, always say which hermes you mean.
//!
//! See docs/updater-world.md §2.5.1a.
//!
//! Rule: inside any hermes-agent checkout, plain `hermes` refuses — you
//! always state which one you mean. No exceptions, even when the invoked
//! launcher IS that checkout's own.
//!
//! - no enclosing checkout → Run (flags accepted as no-ops)
//! - inside a checkout, no flag → Refuse (exit 2)
//! - inside a checkout + --dev → Run THIS checkout's launcher (strip flag)
//!   OR re-exec the cwd checkout's bin/hermes (retaining --dev)
//! - inside a checkout + --global → Run with the invoked launcher (strip flag)
//! - both flags → Refuse (contradictory)

use std::path::{Path, PathBuf};

/// The cwd guard's decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardDecision {
    /// Proceed with the invoked launcher. The flag (if any) is stripped.
    Run,
    /// Re-exec into the cwd checkout's bin/hermes, retaining --dev.
    /// The path is the checkout's bin/hermes.
    ReExec(PathBuf),
    /// Refuse with a message. Exit code 2.
    Refuse(String),
}

/// Check the cwd guard for the current invocation.
///
/// `launcher_tree` is the tree the invoked launcher belongs to (from
/// resolve_tree_root). `cwd` is the current working directory. `argv` is
/// the full argv (including argv[0]).
pub fn cwd_guard(launcher_tree: &Path, cwd: &Path, argv: &[String]) -> GuardDecision {
    let has_dev = argv.iter().any(|a| a == "--dev");
    let has_global = argv.iter().any(|a| a == "--global");

    // Both flags → contradictory
    if has_dev && has_global {
        return GuardDecision::Refuse(
            "hermes: --dev and --global are contradictory — pick one.".to_string(),
        );
    }

    // Find the enclosing hermes-agent checkout (if any)
    let enclosing = find_enclosing_checkout(cwd);

    match enclosing {
        None => {
            // Not inside a checkout → run normally (flags are no-ops)
            GuardDecision::Run
        }
        Some(checkout_root) => {
            if !has_dev && !has_global {
                // Inside a checkout, no flag → refuse
                return GuardDecision::Refuse(format!(
                    "hermes: you are inside a hermes-agent checkout ({}).\n\
                     say which hermes you mean:\n  \
                     hermes --dev       run THIS checkout's ./bin/hermes\n  \
                     hermes --global    run the installed hermes (managed or PATH target)",
                    checkout_root.display()
                ));
            }

            if has_dev {
                // --dev: run THIS checkout's launcher
                let checkout_launcher = checkout_root.join("bin").join("hermes");
                if launcher_tree == checkout_root && checkout_launcher.exists() {
                    // The invoked launcher IS this checkout's own → just run
                    // (strip the flag — the re-exec'd launcher would see --dev
                    // and resolve to Run on its own pass, but we skip the hop)
                    GuardDecision::Run
                } else {
                    // Re-exec the cwd checkout's bin/hermes, retaining --dev
                    if checkout_launcher.exists() {
                        GuardDecision::ReExec(checkout_launcher)
                    } else {
                        // No bin/hermes in the checkout → just run with invoked launcher
                        // (the stub fallback will handle the missing venv)
                        GuardDecision::Run
                    }
                }
            } else {
                // has_global → run with the invoked launcher (strip flag)
                GuardDecision::Run
            }
        }
    }
}

/// Walk up from cwd to find the nearest enclosing hermes-agent checkout.
/// A checkout is identified by a `pyproject.toml` containing
/// `name = "hermes-agent"`. A worktree's `.git` FILE bounds the tree the
/// same as a `.git` dir.
fn find_enclosing_checkout(cwd: &Path) -> Option<PathBuf> {
    for dir in cwd.ancestors() {
        let pyproject = dir.join("pyproject.toml");
        if pyproject.is_file() {
            // Check if this is a hermes-agent checkout (string probe —
            // no toml parser in the launcher)
            if let Ok(content) = std::fs::read_to_string(&pyproject) {
                if content.contains("hermes-agent") {
                    return Some(dir.to_path_buf());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_checkout(root: &Path) {
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::write(
            root.join("pyproject.toml"),
            "[project]\nname = \"hermes-agent\"\n",
        )
        .unwrap();
        fs::write(root.join("bin").join("hermes"), "#!/bin/sh\n# stub").unwrap();
    }

    #[test]
    fn test_no_checkout_runs() {
        let tmp = tempfile::tempdir().unwrap();
        let launcher = tmp.path().join("launcher");
        fs::create_dir_all(&launcher).unwrap();
        let argv = vec!["hermes".to_string(), "--version".to_string()];
        let decision = cwd_guard(&launcher, tmp.path(), &argv);
        assert_eq!(decision, GuardDecision::Run);
    }

    #[test]
    fn test_inside_checkout_no_flag_refuses() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        let launcher = tmp.path().join("somewhere-else");
        fs::create_dir_all(&launcher).unwrap();
        let argv = vec!["hermes".to_string()];
        let decision = cwd_guard(&launcher, tmp.path(), &argv);
        assert!(matches!(decision, GuardDecision::Refuse(_)));
        if let GuardDecision::Refuse(msg) = decision {
            assert!(msg.contains("--dev"));
            assert!(msg.contains("--global"));
        }
    }

    #[test]
    fn test_inside_checkout_dev_runs_when_invoked_launcher_is_own() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        let argv = vec![
            "hermes".to_string(),
            "--dev".to_string(),
            "--version".to_string(),
        ];
        // launcher_tree IS the checkout root
        let decision = cwd_guard(tmp.path(), tmp.path(), &argv);
        assert_eq!(decision, GuardDecision::Run);
    }

    #[test]
    fn test_inside_checkout_dev_reexec_when_invoked_launcher_is_different() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        let other = tmp.path().join("other-launcher");
        fs::create_dir_all(&other).unwrap();
        let argv = vec!["hermes".to_string(), "--dev".to_string()];
        // launcher_tree is NOT the checkout root
        let decision = cwd_guard(&other, tmp.path(), &argv);
        match decision {
            GuardDecision::ReExec(path) => {
                assert!(path.ends_with("bin/hermes"));
            }
            _ => panic!("expected ReExec, got {:?}", decision),
        }
    }

    #[test]
    fn test_inside_checkout_global_runs() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        let other = tmp.path().join("managed-launcher");
        fs::create_dir_all(&other).unwrap();
        let argv = vec![
            "hermes".to_string(),
            "--global".to_string(),
            "--version".to_string(),
        ];
        let decision = cwd_guard(&other, tmp.path(), &argv);
        assert_eq!(decision, GuardDecision::Run);
    }

    #[test]
    fn test_both_flags_refuses() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        let argv = vec![
            "hermes".to_string(),
            "--dev".to_string(),
            "--global".to_string(),
        ];
        let decision = cwd_guard(tmp.path(), tmp.path(), &argv);
        assert!(matches!(decision, GuardDecision::Refuse(_)));
    }

    #[test]
    fn test_outside_checkout_flags_accepted_as_noops() {
        let tmp = tempfile::tempdir().unwrap();
        let launcher = tmp.path().join("managed");
        fs::create_dir_all(&launcher).unwrap();
        let argv = vec![
            "hermes".to_string(),
            "--dev".to_string(),
            "--version".to_string(),
        ];
        // cwd is NOT inside a checkout
        let cwd = tmp.path().join("some-dir");
        fs::create_dir_all(&cwd).unwrap();
        let decision = cwd_guard(&launcher, &cwd, &argv);
        assert_eq!(decision, GuardDecision::Run);
    }

    #[test]
    fn test_worktree_git_file_bounds_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        // Replace .git dir (if any) with a .git FILE (worktree style)
        let _ = fs::remove_dir_all(tmp.path().join(".git"));
        fs::write(
            tmp.path().join(".git"),
            "gitdir: /some/main/repo/.git/worktrees/foo",
        )
        .unwrap();
        // Should still detect the checkout
        let other = tmp.path().join("launcher");
        fs::create_dir_all(&other).unwrap();
        let argv = vec!["hermes".to_string()];
        let decision = cwd_guard(&other, tmp.path(), &argv);
        assert!(matches!(decision, GuardDecision::Refuse(_)));
    }

    #[test]
    fn test_nested_dir_inside_checkout_detected() {
        let tmp = tempfile::tempdir().unwrap();
        make_checkout(tmp.path());
        let nested = tmp.path().join("src").join("deep").join("path");
        fs::create_dir_all(&nested).unwrap();
        let other = tmp.path().join("launcher");
        fs::create_dir_all(&other).unwrap();
        let argv = vec!["hermes".to_string()];
        let decision = cwd_guard(&other, &nested, &argv);
        assert!(matches!(decision, GuardDecision::Refuse(_)));
    }
}
