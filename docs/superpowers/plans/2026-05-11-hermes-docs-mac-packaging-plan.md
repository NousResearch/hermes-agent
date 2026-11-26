# Hermes Docs — Mac Packaging Plan: Local Web App to DMG

Date: 2026-05-11
Status: PLAN — not yet implemented
Author: Claude Sonnet (kanban task t_2577aaf0)

---

## Purpose

This document defines the concrete installation and packaging plan for
Hermes Docs on macOS. It describes what the installer must do, where files
live, which runtime checks are mandatory, and how failures and rollbacks
are handled. It is intended as the actionable input for a future
implementation ticket.

No code is changed by this document. No DMG is built. No dependencies are
added.

---

## Two Distribution Modes

Hermes Docs ships in two distinct modes that must be kept conceptually
separate throughout implementation.

### Mode A — Local Web App (current)

The app runs as a local web service started by the Hermes Dashboard or via
a CLI command. The user navigates to it in any browser. There is no
dedicated macOS app bundle.

```
Start path:
  hermes dashboard   (or: hermes docs start)
  → Python backend starts on localhost:<port>
  → User opens http://localhost:<port>/docs in browser
  → Dashboard plugin shows status and Open button
```

This mode is appropriate for developers and users who already run the
Hermes Dashboard. It does not require any installer.

Limitations:
- No Dock icon, no Launchpad icon.
- Requires the user to know a URL or use the Dashboard.
- The app process lives and dies with the Dashboard process.
- First-run onboarding is triggered in-browser; it cannot prompt for
  system-level permissions automatically.

### Mode B — Native DMG Installer (target)

A macOS .dmg distributes an app bundle (`Hermes Docs.app`) built with a
lightweight native shell (Electron or a custom Swift/WKWebView wrapper)
that hosts the same local web backend and frontend. The DMG installs the
bundle to /Applications.

```
Installed layout:
  /Applications/Hermes Docs.app/
    Contents/
      MacOS/
        hermes-docs          (launcher binary)
      Resources/
        app/                 (bundled frontend assets)
        backend/             (Python backend, venv, kordoc binary)
        icons/
      Info.plist
      Entitlements.plist
```

The native wrapper:
- Exposes a Dock icon and Launchpad entry.
- Owns the process lifecycle (start/stop backend on app open/close).
- Displays a loading screen while the local backend starts.
- Forwards all document file permissions through the macOS Security
  Scoped Bookmark API so the browser sandbox gets trusted read/write
  access to registered workspace folders.
- Runs onboarding on first launch as a native sheet, not a separate URL.

The frontend and backend code are identical between Mode A and Mode B.
The difference is the host shell and the installer.

---

## Installation Phases

The installer (DMG background script or a first-launch onboarding flow)
executes the following phases in order. Each phase either passes or blocks
progress with a specific failure mode.

### Phase 0 — Pre-flight

Goal: confirm the host machine meets minimum requirements before writing
any files.

Checks:
1. macOS version >= 13.0 (Ventura). Older versions lack required
   WebKit API surface for Security Scoped Bookmarks.
2. At least 500 MB free on the target volume.
3. No conflicting Hermes Docs installation already running (check for
   lock file at ~/.hermes/docs.pid).

Failure modes:
- macOS too old: show "Hermes Docs requires macOS 13 or later" and abort.
- Disk full: show "Not enough space" and abort.
- Existing process running: offer "Quit existing Hermes Docs and
  continue" or "Cancel".

Files written: none.

### Phase 1 — Core App Install

Goal: place the application bundle and Dashboard plugin assets on disk.

Steps:
1. Copy `Hermes Docs.app` to /Applications (DMG mode) or skip (web-app
   mode, assets already present in the Hermes plugin directory).
2. Copy Dashboard plugin assets to:
   `~/.hermes/plugins/hermes-docs/`
     plugin.yaml
     frontend/   (compiled JS/CSS for the Dashboard tab)
3. Register the plugin with Hermes by appending to
   `~/.hermes/config.yaml` under `plugins.enabled` if not already
   present.

File locations written:
```
/Applications/Hermes Docs.app/          (DMG mode only)
~/.hermes/plugins/hermes-docs/
~/.hermes/config.yaml                   (plugins.enabled entry added)
```

Failure modes:
- /Applications not writable: prompt for admin password (macOS
  installer auth) or fall back to ~/Applications.
- Plugin directory already exists with a newer version: abort and
  inform the user.

### Phase 2 — Hermes Verification

Goal: confirm a working Hermes installation exists before bootstrapping
any persona or credential.

Checks:
1. `hermes --version` exits 0.
2. `~/.hermes/config.yaml` is present and parseable.
3. `~/.hermes/.env` is present (may be empty; signals the user has run
   Hermes setup).

Failure modes:
- Hermes not found in PATH: show "Hermes is required. Install it first
  at https://hermes.nousresearch.com" and abort. Do not install.
- config.yaml missing or unparseable: show "Hermes config is missing or
  corrupt. Run `hermes setup` first" and abort.

Files written: none.

### Phase 3 — Docs Persona Bootstrap

Goal: create the `docs` agent profile if it does not exist.

Steps:
1. Check for `~/.hermes/profiles/docs/config.yaml`.
2. If absent, create the profile directory and write the default
   persona config:
   ```
   ~/.hermes/profiles/docs/
     config.yaml         (model, system_prompt, enabled_toolsets)
     PERSONA.md          (persona instruction file)
   ```
3. If already present, validate that required toolsets (file, web,
   kordoc-mcp) are listed in config.yaml. Append missing ones.
4. Write a sentinel file `~/.hermes/profiles/docs/.hermes-docs-managed`
   so future installer runs know the profile was installed by Hermes
   Docs and can update it safely without overwriting user edits to
   optional fields.

Required config.yaml keys for the docs persona:
```yaml
profile: docs
model: ""                    # inherit from main config
enabled_toolsets:
  - file
  - web
  - kordoc-mcp
  - terminal                 # for document actions
system_prompt: |
  You are a document workspace assistant specialised in Markdown editing,
  review, brainstorming, Kordoc conversions, and careful source-file
  edits with preview-before-apply.
```

Failure modes:
- Profile directory not writable: show error and abort.
- Existing profile is missing required toolsets AND the sentinel is
  absent (user-managed profile): warn the user and ask permission to
  append. If denied, continue with a warning that side-chat may not
  function correctly.

### Phase 4 — Kordoc Verification

Goal: confirm the kordoc MCP server is available.

Checks:
1. Locate the kordoc binary: first in PATH (`which kordoc-mcp`), then
   in bundled backend resources (DMG mode only).
2. Run a smoke test: `kordoc-mcp --version` exits 0.
3. Confirm the kordoc MCP server is registered in
   `~/.hermes/config.yaml` under `mcp_servers`. If missing, add the
   entry pointing to the located binary.

MCP server entry to add if missing:
```yaml
mcp_servers:
  kordoc:
    command: kordoc-mcp
    args: []
    transport: stdio
```

Failure modes:
- kordoc not found and not bundled: show "Kordoc MCP server not found.
  Install it with: pip install kordoc-mcp" and continue with a warning.
  Do not abort; HWP/HWPX/PDF conversion will be unavailable but the
  rest of the app functions.
- kordoc --version fails: show version error and continue with warning.

### Phase 5 — Codex OAuth / Local Broker Readiness

Goal: ensure the local auth broker is ready for Codex-powered side-chat
and Doc Agent actions.

Steps:
1. Check whether Hermes exposes an OpenAI Codex credential or auth broker
   command in the installed version.
2. If a credential exists and is not expired, skip to step 5.
3. If no credential exists and the installed Hermes build has a Codex
   auth helper, start the device-code OAuth flow through that helper.
   The exact command is intentionally left as an installer integration
   requirement because the current Hermes CLI exposes `auth`/`login`
   commands, not a stable `hermes oauth ...` command.
4. On success, credentials are stored in the Hermes credential pool or
   provider-specific auth store selected by that helper. Browser code
   never receives or persists the raw token.
5. Verify the local broker endpoint is reachable:
   `GET http://localhost:<port>/api/broker/health`
   (port is determined by the backend config; default 8788).

Fallback provider path (if Codex OAuth is declined or unavailable):
- Installer writes `docs_provider_fallback: hermes` to the docs plugin
  config so the side-chat routes through the main configured Hermes
  provider.
- User sees a banner in the app: "Codex not connected. Using Hermes
  provider."

Failure modes:
- Device-code flow timed out: continue without Codex credential and
  activate fallback. User can retry from Settings.
- Network unavailable: skip OAuth. Activate fallback.
- Local broker not reachable after backend start: log error to
  `~/.hermes/logs/docs-install.log` and show "Local broker failed to
  start" with a retry button.

### Phase 6 — App Launcher Creation

Goal: make Hermes Docs easy to launch from the standard macOS locations.

DMG mode:
- The .app bundle in /Applications is the primary launcher. Dock and
  Launchpad discovery is automatic once the bundle is copied.
- The installer offers to add the bundle to the Dock by running:
  `defaults write com.apple.dock persistent-apps -array-add ...`
  followed by `killall Dock`. This step is optional and user-confirmed.

Web-app mode:
- Create a minimal macOS .app shell that opens the browser to the
  Hermes Docs URL when double-clicked. Place it in ~/Applications.
  The shell is a simple AppleScript app:
  ```applescript
  tell application "System Events"
    open location "http://localhost:8787/docs"
  end tell
  ```
  Compiled with: `osacompile -o "~/Applications/Hermes Docs.app" launcher.applescript`
- Optionally create a shell alias in ~/.zprofile:
  `alias hermes-docs="hermes dashboard --open-url /docs"`

File locations written:
```
~/Applications/Hermes Docs.app          (web-app mode launcher)
~/.hermes/plugins/hermes-docs/.app-launcher-installed   (sentinel)
```

Failure modes:
- Dock add command fails (non-fatal): log and skip. The .app is still
  available in Finder and Spotlight.
- osacompile not available (non-standard macOS): skip the web-app
  launcher and tell the user to bookmark the URL instead.

### Phase 7 — User Data Preservation

Goal: ensure existing workspace data and user settings survive
installation and upgrades.

Rules:
1. The installer NEVER deletes or overwrites:
   - `~/.hermes/docs-workspaces/` (workspace registry and session data)
   - `~/.hermes/profiles/docs/` unless the sentinel file is present
     AND the user explicitly consents to a reset.
   - Any file in the registered source workspace folders.
2. Before replacing the plugin assets, the installer moves the old
   assets to a timestamped backup:
   `~/.hermes/plugins/hermes-docs.backup.<timestamp>/`
3. The workspace registry format is versioned. The installer runs a
   migration if the version field is older than the installed version:
   `~/.hermes/docs-workspaces/registry.json` → field: `"schema_version"`

Backup file locations:
```
~/.hermes/plugins/hermes-docs.backup.<timestamp>/
~/.hermes/logs/docs-install.log                         (install log)
```

---

## Rollback and Uninstall

### Rollback (Failed Install)

If any phase from Phase 1 onward fails and the installer exits
non-zero, the installer runs cleanup in reverse order:

1. Remove `~/.hermes/plugins/hermes-docs/` if it was written in this run.
2. Restore the plugin backup if one was made.
3. Remove the `plugins.enabled: hermes-docs` entry from config.yaml.
4. Remove `~/.hermes/profiles/docs/.hermes-docs-managed` sentinel if
   written in this run. Do NOT remove the profile itself (user may
   have had one before install).
5. Remove `/Applications/Hermes Docs.app` if it was copied in this run.
6. Remove `~/Applications/Hermes Docs.app` if it was created.
7. Write a rollback summary to `~/.hermes/logs/docs-install.log`.

User data (`~/.hermes/docs-workspaces/`) is never touched during rollback.

### Uninstall

The uninstaller is a separate script: `~/.hermes/plugins/hermes-docs/uninstall.sh`.

Uninstall steps:
1. Confirm with the user (interactive prompt or --yes flag).
2. Stop the Hermes Docs backend if running (kill PID from docs.pid).
3. Remove `/Applications/Hermes Docs.app` (DMG mode).
4. Remove `~/Applications/Hermes Docs.app` (web-app launcher).
5. Remove `~/.hermes/plugins/hermes-docs/`.
6. Remove `plugins.enabled: hermes-docs` from config.yaml.
7. Optionally remove the `docs` profile (ask separately; default: keep).
8. Optionally remove `~/.hermes/docs-workspaces/` (ask separately;
   default: keep — this is the user's session history and annotations).
9. Optionally remove the OpenAI Codex credential from the Hermes
   credential/auth store if the installer created it (ask separately;
   default: keep — Codex credential may be shared with other Hermes tools).

All optional removals default to "keep" to prevent data loss on
accidental uninstall.

---

## Summary: File Locations

| Path | Owner | Keep on uninstall |
|---|---|---|
| `/Applications/Hermes Docs.app` | Installer (DMG) | No |
| `~/Applications/Hermes Docs.app` | Installer (web-app) | No |
| `~/.hermes/plugins/hermes-docs/` | Installer | No |
| `~/.hermes/plugins/hermes-docs.backup.<ts>/` | Installer | Optional |
| `~/.hermes/config.yaml` (plugin entry) | Installer | Remove entry |
| `~/.hermes/profiles/docs/` | Installer / User | Optional (ask) |
| `~/.hermes/profiles/docs/.hermes-docs-managed` | Installer | Removed with profile |
| Hermes credential/auth store for OpenAI Codex | OAuth flow | Optional (ask) |
| `~/.hermes/docs-workspaces/` | User data | Yes (default) |
| `~/.hermes/logs/docs-install.log` | Installer | Yes |
| `~/.hermes/docs.pid` | Runtime | Removed on stop |

---

## Summary: Mandatory Pre-launch Checks

These checks run every time the Hermes Docs app starts (not just at
install time). They are fast and non-destructive.

1. Hermes version >= minimum required (read from plugin.yaml: `requires_hermes`).
2. `docs` profile exists and has required toolsets.
3. Kordoc binary accessible (warn if not found; do not block start).
4. Workspace registry parseable (if corrupt, show recovery UI).
5. PID file not stale (if stale, remove and continue).

---

## Implementation Notes for the Follow-up Ticket

- The installer script should be written in bash with no dependencies
  beyond what macOS ships: bash, python3 (system), osascript, defaults,
  plutil.
- The DMG creation step (not covered here) will need create-dmg or
  hdiutil and a signing identity. This is explicitly deferred.
- The web-app launcher (.app via osacompile) is a stopgap. The target
  UX is the DMG mode .app bundle.
- Electron is the lowest-risk native shell choice given the web frontend;
  Tauri is a viable alternative with a smaller binary size but requires
  Rust toolchain in CI.
- Security Scoped Bookmarks for workspace folder access require the app
  to have the `com.apple.security.files.user-selected.read-write`
  entitlement and App Sandbox enabled (mandatory for Mac App Store;
  strongly recommended for direct distribution).
- The OpenAI Codex OAuth/device-code broker command must be verified in
  the target Hermes release before DMG implementation. On this machine,
  `hermes oauth ...` is not a valid command; the installer should discover
  and call the stable `auth`/credential broker API that exists at that time.
