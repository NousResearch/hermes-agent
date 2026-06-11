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

## Highest-Level Rust Plan

The target is not "rewrite Hermes in Rust." The target is a smaller and more predictable release package:
users should download one installer, the installer should bring the platform-specific tools it owns, and shell/Python
scripts should become recovery paths instead of the normal first-run path.

**Target package shape:**
- Rust bootstrap installer owns install orchestration, repository archive refresh, PATH/profile changes, shortcuts,
  install metadata, repair cleanup, and uninstall planning.
- Release package bundles `hermes-manager`, pinned install scripts, a checksummed bootstrap resource manifest, and
  platform-specific tool archives that create real install-time savings.
- Bundled tool archives start with Node.js and `uv`, then expand only when the size/security trade-off is clear.
- Python application features stay Python. Rust should install, update, verify, and clean them rather than rewriting
  fast-moving agent logic.
- Electron renderer features stay TypeScript. Rust should only handle the native install/update shell around them.
- Git should not be required for normal fresh installs or archive-created updates. It remains available as fallback for
  source checkouts and advanced contributor workflows.
- Shell scripts remain supported for direct CLI installs and one-release fallback, but desktop bootstrap should prefer
  native Rust stages wherever parity exists.

**Dependency reduction order:**
1. Remove Git from fresh packaged installs and archive-based updates.
2. Bundle and install Node.js and `uv` from Rust before any shell script can download them.
3. Create Python 3.11, the virtual environment, and locked Python dependencies through Rust-invoked `uv`.
4. Install npm, Playwright, TUI, and desktop dependencies through Rust first, while preserving platform recovery
   fallbacks for Linux system libraries and Electron mirrors.
5. Move PATH/profile, shortcuts, install stamps, metadata, repair, and uninstall into `hermes-manager`.
6. Add release manifests and smoke tests so every bundled binary has an owner, checksum, and update path.
7. Only after installer parity, evaluate larger Rust runtime candidates with measurable dependency or reliability wins.

**Never-rust-first areas for this roadmap:**
- Model/provider orchestration, conversation state, plugin/skill execution, and fast-changing agent features.
- Messaging gateway platform behavior unless a low-level helper has a stable API and a clear dependency payoff.
- Web UI or Electron renderer code.

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

**Status:** Implemented on this branch.

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

**Implemented:**
- `build:release`, `pack`, and every platform `dist:*` target build `apps/hermes-manager` before packaging.
- Release builds set `HERMES_DESKTOP_REQUIRE_MANAGER=1`, so staging fails clearly if the target manager binary is
  missing instead of silently shipping the Python-only fallback.
- Plain `npm run build` remains fallback-friendly for desktop development when Rust is not installed.

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
- When an archive-created checkout without `.git` is updated, the Tauri updater can run the existing installer Git
  stage on demand, initialize the checkout as a Git repository, fetch the recorded archive ref, and reset to it before
  handing off to `hermes update`.
- For Windows archive-created checkouts, the Tauri updater now skips Git checkout preparation, refreshes the repository
  from a GitHub ZIP archive natively in Rust, and then calls `hermes update --finalize-only` so Python only refreshes
  dependencies, caches, and generated assets.
- Archive-created checkout updates now use the same native ZIP refresh path on Windows, Linux, and macOS, so those
  installs do not require Git just to refresh source files before dependency finalization.
- Fresh bootstrap repository stages now try the native Rust GitHub archive path before falling back to script-backed
  clone behavior on every platform. On Unix bootstrap runs, `install.sh` receives an internal archive signal so the
  prerequisites stage does not install or require Git when Rust will fetch the source archive.
- Fresh bootstrap runs now defer the separate Git stage on Windows, Linux, and macOS while the Rust repository archive
  path is available, so new archive-based installs do not install Git unless archive fallback or later update recovery
  actually needs it.
- Rust ZIP extraction now rejects symlink entries as well as path traversal entries before materializing repository
  archive contents.
- Rust stage planning now reports native-first, probe-only, and script-only coverage counts so later bootstrap work can
  target the remaining script-owned dependency stages explicitly.
- Bootstrap stage manifests are now generated in Rust for both Windows and Unix scripts, so setup no longer starts
  PowerShell or bash just to discover the stage list.
- Non-interactive post-install stages that require user input are now skipped in Rust with the same successful skipped
  result shape, so GUI bootstrap no longer starts shell processes for those no-op stages.
- Windows bootstrap tool stages for `uv`, `git`, `node`, and `system-packages` now use Rust preflight checks to skip
  the script process when the required tools are already available, while preserving script fallback when anything is
  missing or the detected Node.js version is too old for the desktop build.
- Windows `uv` now has a Rust native-first GitHub release ZIP path for x64, ARM64, and x86, installing `uv.exe` into
  `$HERMES_HOME/bin` and preserving the PowerShell astral installer as fallback for download, extraction, or version
  check failures.
- Windows `git` now has a Rust native-first path for the same pinned Git for Windows release used by `install.ps1`,
  downloading PortableGit or 32-bit MinGit into `$HERMES_HOME/git`, updating current/User PATH entries, and persisting
  `HERMES_GIT_BASH_PATH` when Bash is available; the PowerShell stage remains fallback for download/extraction/PATH
  failures.
- Windows `node` now has a Rust native-first portable ZIP path for Node.js v22, including official index resolution,
  ZIP download/extraction into `$HERMES_HOME/node`, current-process PATH update, and User PATH persistence; the
  PowerShell stage remains fallback for download, extraction, PATH, or version-verification failures.
- Unix `prerequisites` now runs a Rust native-first Node.js v22 tarball preflight before handing off to `install.sh`.
  Successful preflight installs `$HERMES_HOME/node`, updates the bootstrap process PATH, and creates Node/npm/npx
  symlinks, so the script can skip its own Node curl/tar path while still owning uv, Python, Git, system package, and
  network fallback behavior.
- Unix Node preflight now prefers matching bundled Node tarballs from `bootstrap-tools/` before downloading from
  nodejs.org, matching the Windows bundled archive behavior for packaged installers.
- Unix bootstrap manifests now expose Node.js as a separate native-first stage after `uv`, and Rust passes an internal
  skip signal so the prerequisites shell fallback does not repeat the managed Node.js check/install path.
- Unix `uv` now has a Rust native-first GitHub release tarball path for Linux and macOS x64/arm64, installing `uv` and
  `uvx` into `$HERMES_HOME/bin` while preserving `install.sh` fallback for unsupported platforms, Termux, download,
  extraction, or version-check failures.
- Unix bootstrap manifests now expose `uv` as a separate native-first stage before `prerequisites`, and Rust passes an
  internal skip signal so the prerequisites shell fallback does not repeat the managed `uv` install.
- Windows `python` now uses a Rust `uv python find 3.11` preflight to skip the PowerShell stage when the required
  runtime is already available, while preserving script fallback so missing Python can still be installed by uv.
- Unix bootstrap manifests now expose Python 3.11 as a separate native-first stage after Node.js, and Rust passes an
  internal skip signal so the prerequisites shell fallback does not repeat the Python check/install path.
- Unix bootstrap manifests now expose `system-packages` as a separate probe-then-script stage after Python, so Rust can
  skip the shell process when `rg` and `ffmpeg` are already available and preserve shell package-manager fallback.
- Normal Unix bootstrap manifests no longer include the legacy aggregate `prerequisites` stage after its tool and
  system-package responsibilities were split into explicit stages; `install.sh --stage prerequisites` remains
  available for compatibility.
- `python` now also runs native-first installation through Rust by invoking `uv python install 3.11`, with script
  fallback preserved if uv fails to install or locate the runtime.
- `venv` now runs native-first through Rust by invoking `uv venv venv --python 3.11` in the checkout, with script
  fallback preserved if native venv creation fails.
- Python dependency installation now has a Rust native-first lockfile path using `uv sync --extra all --locked` with
  `UV_PROJECT_ENVIRONMENT` pinned to `venv`, while the script keeps all PyPI fallback tiers.
- `node-deps` now uses a Rust no-op skip when npm is unavailable on every platform, matching the existing script
  behavior without starting PowerShell or bash for a stage that can only skip.
- Windows `node-deps` now has a Rust native-first path for root npm dependencies, Playwright Chromium, and TUI npm
  dependencies, while preserving the PowerShell stage as fallback for missing `npx` or failed npm/Playwright commands.
- macOS `node-deps` now uses the same Rust native-first npm/Playwright/TUI dependency path as Windows, while Linux
  now uses the same native-first path; Linux keeps script fallback so distribution-specific Playwright system-library
  recovery remains intact when the native npm/Playwright path fails.
- `desktop` now uses a Rust no-op skip when `apps/desktop/package.json` is absent, matching the existing script
  behavior without starting PowerShell or bash for a stage that can only skip.
- Windows `desktop` now has a Rust native-first build path for workspace npm install and `npm run pack`, verifies the
  produced `Hermes.exe`, and still creates shortcuts through the Rust manager; the PowerShell stage remains fallback
  for dependency/build failures so its cache purge and Electron mirror recovery are preserved.
- macOS `desktop` now uses the same Rust native-first workspace npm install and `npm run pack` path as Windows, verifies
  the produced `Hermes.app`, and preserves shell fallback for dependency or build recovery.
- Linux `desktop` now uses the same Rust native-first workspace npm install and `npm run pack` path, verifies the
  produced unpacked app, and configures Electron's `chrome-sandbox` helper while preserving script fallback for build
  recovery or privileged sandbox setup failures.
- Windows `platform-sdks` now skips natively when `.env` has no configured messaging platform tokens, and runs
  native-first SDK import checks plus targeted `pip install` recovery when tokens are present, while preserving script
  fallback if the native recovery path fails.
- Unix bootstrap manifests now expose the same `platform-sdks` stage after config preparation, so Linux and macOS GUI
  installs also skip SDK work when no messaging platform tokens are configured and run native-first targeted SDK
  recovery when tokens are present.
- `bootstrap-marker` now runs as a native Rust stage in the Tauri bootstrapper.
- `config-templates` and the Unix `config` stage now run as native Rust stages while preserving Python
  `tools/skills_sync.py` when available and retaining the existing bundled-skill copy fallback.
- CI runs bootstrap-installer Rust unit tests in addition to the manager and desktop platform tests.

**Still script-backed:**
- Language/runtime setup: Python dependency fallback tiers when `uv.lock` sync is unavailable, script fallback for
  Windows/macOS/Linux npm recovery, Windows uv, Windows Git, Windows Node, Windows/macOS/Linux desktop recovery, and
  platform SDK recovery.
- Repository clone/update stage execution until the Git/ZIP fallback matrix has a parity suite and native stage wiring.
- Remaining platform shell/profile edge cases that are not covered by the current Rust path-stage helpers.

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
- Bootstrap installer now runs the Windows `path` stage natively, preserving user `Path` and `HERMES_HOME` setup.
- Bootstrap installer now runs the Unix `path` stage natively for shell profile PATH setup, writing an idempotent
  Hermes-managed profile block through the Rust manager and refreshing the bootstrap process PATH. System-level
  symlink behavior remains script-backed until full parity exists.
- Bootstrap installer now runs the Unix `complete` stage natively for the install-method stamp, preserving the existing
  `git` value so status, dashboard, and update recommendations remain compatible with archive-created checkouts.
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
- Tauri bootstrap installer bundles a `bootstrap-tools/` resource directory and the Windows native Node, uv, and Git
  runtime stages prefer matching bundled archives before falling back to the download cache.
- Windows installer release workflow prepares x64 Node v22, uv, and pinned Git for Windows archives before Tauri
  packaging, then writes `bootstrap-tools-manifest.json` with archive URL, size, and SHA-256 metadata for review.
- The same release preparation helper now supports `--platform linux|macos` for x64/arm64 Node and uv tarballs, matching
  the Unix Rust Node/uv installer asset matrix when future macOS/Linux installer packaging wires in bundled tools.
- A manual Unix installer workflow now builds Linux and macOS Tauri setup artifacts with matching bundled Node/`uv`
  archives and uploads the generated `bootstrap-tools-manifest.json` alongside the installer artifacts.

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

1. Finish native-first runtime setup parity: make every platform's `uv`, Node, Python, venv, Python deps, npm deps, and
   desktop stages either native-first or explicitly script-only with a recorded reason.
2. Close the release bundle loop for Linux and macOS installers by staging Node/`uv` archives into `bootstrap-tools/`
   and validating them through `bootstrap-tools-manifest.json`.
3. Expand `hermes-manager` ownership of repair and uninstall metadata after real bootstrap paths write accurate
   installed-file manifests.
4. Add end-to-end smoke commands for fresh install, archive update, repair cleanup, and lite uninstall on Windows,
   Linux, and macOS.
5. Reduce shell usage in desktop bootstrap until scripts are only fallback or direct-install entry points.
6. After one release with native bootstrap enabled, evaluate larger Rust runtime candidates from Phase 7 using measured
   install-time dependency reduction, not rewrite preference.

## Review Gates

Every phase needs:

- A focused implementation plan.
- Unit tests for safety boundaries.
- At least one end-to-end install/update/uninstall smoke path.
- A fallback path for one release.
- A short release note explaining changed install behavior.

