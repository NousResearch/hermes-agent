use std::process::Command;

fn main() {
    // -----------------------------------------------------------------
    // Bake the install.ps1 pin into the binary at compile time.
    //
    // BUILD_PIN_COMMIT and BUILD_PIN_BRANCH are read by bootstrap.rs's
    // `option_env!()` macro to default the install-script reference.
    // Precedence (matches install.ps1's own arg precedence): commit > branch.
    //
    // Resolution order:
    //   1. Env var override at build time (HERMES_BUILD_PIN_COMMIT, etc.).
    //      Useful for CI builds that want to pin to a tagged release SHA
    //      rather than whatever the checkout's HEAD happens to be.
    //   2. `git rev-parse HEAD` + `git rev-parse --abbrev-ref HEAD` against
    //      the repo this build.rs lives in. Default for `cargo tauri build`
    //      from a dev machine — pins the produced .exe to your current
    //      checkout state.
    //   3. Last-resort fallback: hardcoded `main` branch, no commit. The
    //      installer will fetch HEAD-of-main at runtime. Used when the
    //      build is happening outside a git checkout (e.g. cargo install
    //      from a packaged crate, unlikely for this binary but defensive).
    //
    // Build script reruns on git HEAD change so a new commit triggers
    // a rebuild without `cargo clean`.
    // -----------------------------------------------------------------

    let commit = resolve_commit_pin();
    let branch = resolve_branch_pin();

    if let Some(c) = &commit {
        println!("cargo:rustc-env=BUILD_PIN_COMMIT={c}");
        println!("cargo:warning=hermes-bootstrap: pinning to commit {}", short(c));
    }
    if let Some(b) = &branch {
        println!("cargo:rustc-env=BUILD_PIN_BRANCH={b}");
        println!("cargo:warning=hermes-bootstrap: pinning to branch {b}");
    }
    if commit.is_none() && branch.is_none() {
        // Fail loudly rather than silently produce a binary that errors
        // at runtime with "no install-script pin supplied". A build that
        // can't resolve a pin almost certainly indicates a misconfigured
        // build environment.
        println!(
            "cargo:warning=hermes-bootstrap: no pin resolved at build time; binary will fail at runtime without HERMES_SETUP_DEV_REPO_ROOT or runtime args"
        );
    }

    // Rerun build.rs when HEAD moves so successive builds pick up new
    // commits without needing `cargo clean`. .git/HEAD changes on every
    // commit / branch switch / rebase.
    let git_dir = locate_git_dir();
    if let Some(gd) = &git_dir {
        println!("cargo:rerun-if-changed={}/HEAD", gd.display());
        // .git/HEAD often points at a ref (e.g. `ref: refs/heads/bb/gui`);
        // also watch the ref itself so a new commit on the same branch
        // re-triggers.
        if let Ok(head) = std::fs::read_to_string(gd.join("HEAD")) {
            if let Some(rest) = head.trim().strip_prefix("ref: ") {
                println!("cargo:rerun-if-changed={}/{}", gd.display(), rest);
            }
        }
    }
    println!("cargo:rerun-if-env-changed=HERMES_BUILD_PIN_COMMIT");
    println!("cargo:rerun-if-env-changed=HERMES_BUILD_PIN_BRANCH");

    // -----------------------------------------------------------------
    // Tauri windows manifest. See hermes-setup.manifest for rationale —
    // declares level="asInvoker" so Windows's installer-detection
    // heuristic doesn't refuse to launch us without UAC elevation.
    // -----------------------------------------------------------------
    #[cfg(target_os = "windows")]
    let attrs = {
        let manifest = include_str!("hermes-setup.manifest");
        let win = tauri_build::WindowsAttributes::new().app_manifest(manifest);
        tauri_build::Attributes::new().windows_attributes(win)
    };

    #[cfg(not(target_os = "windows"))]
    let attrs = tauri_build::Attributes::new();

    tauri_build::try_build(attrs).expect("failed to run tauri-build");
}

fn resolve_commit_pin() -> Option<String> {
    if let Ok(v) = std::env::var("HERMES_BUILD_PIN_COMMIT") {
        if !v.trim().is_empty() {
            return Some(v.trim().to_string());
        }
    }
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn resolve_branch_pin() -> Option<String> {
    if let Ok(v) = std::env::var("HERMES_BUILD_PIN_BRANCH") {
        if !v.trim().is_empty() {
            return Some(v.trim().to_string());
        }
    }
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    // "HEAD" is what you get on a detached checkout — no meaningful branch
    // to pin to. The commit pin still applies; just don't emit a branch.
    if s.is_empty() || s == "HEAD" {
        None
    } else {
        Some(s)
    }
}

fn locate_git_dir() -> Option<std::path::PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(s))
}

fn short(commit: &str) -> &str {
    if commit.len() >= 12 {
        &commit[..12]
    } else {
        commit
    }
}
