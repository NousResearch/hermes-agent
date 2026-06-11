<!--
    Roadmap for reducing Hermes Agent installation dependencies through incremental Rust components.
-->

# Rust Migration Roadmap

## Goal

Reduce install-time dependency complexity and make install, repair, update, and uninstall faster and more reliable,
while preserving every existing user-facing feature and keeping a safe fallback during each transition.

## Non-Negotiable Constraints

- Do not remove existing Python, Electron, shell, or installer functionality until the Rust replacement has parity.
- Do not require Rust for normal end users.
- Release packages may require Rust during CI/release build, but the built artifacts must include the needed binaries.
- Every phase must be independently shippable and reversible.
- User data under Hermes home must be preserved unless the user explicitly chooses full uninstall.
- Windows path, lock, symlink, junction, and current-directory deletion hazards must be tested before expanding scope.

## Phase 0: Safety and Inventory

**Purpose:** Know exactly what can be moved before moving it.

**Work:**
- Inventory install-time dependencies from `scripts/install.ps1`, `scripts/install.sh`, desktop bootstrap, and
  bootstrap installer.
- Classify dependencies into:
  - runtime-required features,
  - install-only tooling,
  - release-package build tooling,
  - optional platform integrations.
- Record which features require Python packages, Node packages, uv, git, PowerShell, bash, or platform tools.

**Exit Criteria:**
- A dependency matrix exists.
- Each dependency has an owner path and a planned Rust replacement or explicit reason to keep it.

## Phase 1: Rust Install Manager Foundation

**Status:** Implemented on this branch.

**Purpose:** Add a small, testable Rust binary that safely owns Hermes-managed runtime paths.

**Work:**
- Add `apps/hermes-manager`.
- Implement path resolution, bundled manifest validation, installed-files manifest, lite uninstall, repair cleanup,
  and safety checks.
- Package manager into desktop resources when present.
- Let desktop lite uninstall use manager without reducing existing Python cleanup behavior.
- Add CI for Rust manager and desktop platform tests.

**Exit Criteria:**
- `cargo test --manifest-path apps/hermes-manager/Cargo.toml` passes.
- `npm --workspace apps/desktop run test:desktop:platforms` passes.
- Desktop lite uninstall preserves Python parity and can still clean when the venv is missing.

## Phase 2: Release Build Closure

**Purpose:** Ensure published desktop packages always include the Rust manager.

**Work:**
- Add a desktop release-only script to compile `apps/hermes-manager` with Cargo.
- Wire `pack`, `dist`, `dist:win`, `dist:mac`, and `dist:linux` to build the manager before staging.
- Keep normal `npm run build` fallback-friendly so local desktop development does not require Rust.
- Fail release packaging clearly if a target manager binary cannot be produced.

**Exit Criteria:**
- Release build scripts produce and stage `hermes-manager(.exe)` deterministically.
- CI checks manager build logic.
- Staging never ships stale manager binaries.

## Phase 3: Install Metadata Integration

**Status:** Implemented on this branch.

**Purpose:** Make the Rust manager aware of what the installer actually created.

**Work:**
- Call `hermes-manager install-metadata` from successful desktop bootstrap and bootstrap installer paths.
- Extend metadata beyond the default `hermes-agent` runtime directory only when ownership is clear.
- Keep shell/Python install scripts as the source of truth until Rust metadata has enough coverage.
- Add repair behavior for missing or old metadata.

**Exit Criteria:**
- Fresh desktop installs write `$HERMES_HOME/manager/installed-files.json`.
- Lite uninstall works with and without metadata.
- Metadata never includes user config, sessions, `.env`, logs, or other user data.

## Phase 4: Rust Bootstrap Orchestrator

**Status:** In progress on this branch.

**Purpose:** Replace shell-driven orchestration with a self-contained Rust bootstrapper while still delegating complex
language-specific setup where needed.

**Work:**
- Add Rust commands for:
  - download with checksum,
  - archive extraction,
  - atomic file replacement,
  - PATH and environment probing,
  - install-state reporting.
- Make desktop/bootstrap installer call Rust orchestration first.
- Keep `install.ps1` and `install.sh` as fallback for stages not yet ported.

**Implemented so far:**
- Bootstrap installer emits a Rust-side install state and stage plan report before running script-backed stages.
- Downloaded installer artifacts use a Rust helper for HTTP download, optional SHA-256 verification, and atomic cache
  writes.
- Rust ZIP extraction exists with path traversal protection for future repository/archive fallback replacement.
- Rust repository archive fallback primitives can build GitHub ZIP URLs, strip the archive's single top-level directory,
  and refuse to overwrite an existing install root.
- Bootstrap installer can use the Rust repository archive path for Windows fresh installs, while existing install roots
  still fall back to the script-backed Git update path.
- Windows fresh installs now defer the script-backed `git` stage while the Rust repository archive path is available.
  If the archive path fails before creating the install root, the bootstrapper installs Git through the existing script
  stage and then falls back to the script-backed repository stage.
- Archive-created checkouts write `.hermes-source.json` with the GitHub archive owner, repo, ref, branch/commit,
  cached archive path, and best-effort Git initialization status, giving the update path an explicit source marker for
  future no-Git refresh support.
- The Tauri updater now detects archive-created checkouts that are missing `.git` and logs the need for Git checkout
  preparation before handing off to the existing `hermes update` flow.
- `bootstrap-marker` now runs as a native Rust stage in the Tauri bootstrapper.
- `config-templates` now runs as a native Rust stage while preserving Python `tools/skills_sync.py` when available and
  retaining the existing bundled-skill copy fallback.
- CI runs bootstrap-installer Rust unit tests in addition to the manager and desktop platform tests.

**Still script-backed:**
- Language/runtime setup: uv, Python, venv, Python dependencies, Node, npm dependencies, desktop build, and platform SDK
  verification.
- Repository clone/update stage execution until the Git/ZIP fallback matrix has a parity suite and native stage wiring.
- PATH mutation and shell/profile integration, which belongs in Phase 5 platform integration.

**Exit Criteria:**
- First-launch desktop bootstrap can complete the platform/file-management stages without shell scripts.
- Existing install scripts still work independently.
- Failures report actionable errors and leave repairable state.

## Phase 5: Platform Integration in Rust

**Status:** Foundation in progress on this branch.

**Purpose:** Move fragile platform-specific install/uninstall operations out of ad hoc scripts.

**Work:**
- Windows:
  - PATH mutation,
  - Start Menu/Desktop shortcuts,
  - portable Git ownership checks,
  - process-lock-aware cleanup.
- macOS/Linux:
  - symlink creation/removal,
  - shell profile PATH hints,
  - app bundle/AppImage cleanup.
- Add dry-run and machine-readable JSON output for desktop UI.

**Implemented so far:**
- `hermes-manager uninstall-lite` and `repair-clean` support dry-run planning without deleting files.
- Cleanup commands can emit machine-readable JSON via `--json`, while keeping the existing text output for desktop
  cleanup script compatibility.
- `hermes-manager plan-path` computes side-effect-free PATH updates and Unix shell profile hints for future desktop UI
  and OS-specific apply commands.
- `hermes-manager write-profile-hint` can idempotently write a managed Hermes PATH block to an explicitly provided
  shell profile file, with dry-run and JSON output support.
- `hermes-manager write-user-path` can dry-run or write the current user's Windows `Path` registry value, using the
  registry as the default source of truth and broadcasting an environment-change notification after apply.
- Bootstrap installer now runs the Windows `path` stage natively, preserving user `Path` and `HERMES_HOME` setup while
  leaving Unix symlink/profile behavior script-backed until full parity exists.
- `hermes-manager plan-shortcuts` reports Start Menu and Desktop `.lnk` targets for the packaged Windows desktop app,
  including working directory and icon location, without mutating user state.
- `hermes-manager write-shortcuts` can dry-run or create those `.lnk` files through the built-in Windows shortcut COM
  API, keeping shortcut setup in the Rust-managed command surface.
- Tauri bootstrap now passes an internal `-SkipDesktopShortcuts` flag for Windows desktop stages and creates Start
  Menu/Desktop shortcuts through the Rust manager after the desktop build succeeds, while direct `install.ps1`
  invocations keep the legacy PowerShell fallback.
- Windows desktop lite uninstall now asks `hermes-manager uninstall-lite --shortcuts` to remove Rust-managed Start
  Menu/Desktop shortcuts. The manager only removes planned `.lnk` files whose shortcut target still points at the
  packaged Hermes desktop executable.

**Exit Criteria:**
- Rust manager can perform platform cleanup with parity to Python/shell uninstall.
- Existing Python/shell uninstall remains fallback for one release cycle.

## Phase 6: Dependency Bundle Strategy

**Status:** Foundation in progress on this branch.

**Purpose:** Reduce user install-time downloads and dependency setup.

**Work:**
- Decide which artifacts belong in release packages:
  - manager binary,
  - pinned install scripts,
  - checksummed runtime manifests,
  - optional prebuilt helper tools.
- Avoid bundling large or frequently patched dependencies unless there is a clear user-install win.
- For Python/uv dependencies, prefer reproducible cache or wheelhouse strategy before attempting a Rust rewrite of
  Python functionality.

**Exit Criteria:**
- Installer downloads fewer moving pieces.
- Release manifest documents every bundled binary and checksum.
- Security update path remains clear.

**Implemented so far:**
- Desktop release staging writes `build/hermes-manager/bundled-manifest.json` beside the packaged Rust manager.
- The manifest records schema version, Hermes desktop version, source commit, manager resource path, and SHA-256.
- `hermes-manager doctor --manifest <path>` validates the generated manifest successfully in a local staging smoke.
- Commit-pinned bootstrap installers now compile `scripts/install.ps1` and `scripts/install.sh` into the Rust binary,
  so the first run does not need to download the orchestration script from GitHub.
- Branch-following bootstrap builds still resolve install scripts from GitHub raw, preserving HEAD-tracking behavior
  for development and non-immutable builds.
- Bootstrap logs now include an embedded install-script resource summary with size and SHA-256 prefix for diagnostics
  and future release manifest integration.
- Desktop release staging writes install-script metadata into `embedded_resources`, so release manifests document
  embedded script names, sizes, and SHA-256 values without treating them as standalone files.

## Phase 7: Larger Runtime Rust Candidates

**Purpose:** Only after install is stable, consider deeper Rust replacements.

**Candidate Areas:**
- File/archive/download helpers currently duplicated across shell, Python, and Electron.
- Local process supervision and health checks.
- Gateway-adjacent low-level utilities if they are dependency-heavy and have stable APIs.

**Do Not Move Yet:**
- Model/provider orchestration.
- Plugin/skill runtime.
- Web UI/Electron renderer.
- Fast-changing Python features where Rust would slow product iteration.

**Exit Criteria:**
- Each candidate has a parity test suite and measurable dependency or reliability benefit.

## Execution Order

1. Finish Phase 2 release build closure.
2. Add Phase 3 metadata integration for desktop bootstrap.
3. Expand manager metadata and repair commands only after real installs write the manifest.
4. Start Phase 4 with download/checksum/extract helpers, not with Python feature rewrites.
5. Revisit Phase 6 bundle strategy after one release using the manager.

## Review Gates

Every phase needs:

- A focused implementation plan.
- Unit tests for safety boundaries.
- At least one end-to-end install/update/uninstall smoke path.
- A fallback path for one release.
- A short release note explaining changed install behavior.

