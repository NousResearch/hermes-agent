# Hermes Desktop

Native Electron shell for Hermes. It packages the desktop renderer, a bundled Hermes source payload, and installer targets for macOS and Windows. Note: this doc needs updating.

## Setup

Install workspace dependencies from the repo root so `apps/desktop`, `web`, and `apps/shared` stay linked:

```bash
npm install
```

For Python, you have two options:

**Option A — let the desktop provision it for you (recommended for first-time setup):** just run `npm run dev`. On first launch the desktop creates a venv at `HERMES_HOME/hermes-agent/venv` and runs `pip install -e .` against the resolved Hermes source automatically. Requires Python 3.11+ on `PATH`.

**Option B — share an existing CLI install:** if you already ran `scripts/install.ps1` / `scripts/install.sh`, that's the same layout the desktop uses. The desktop reuses your existing venv and editable install — no extra steps. See [Runtime Bootstrap](#runtime-bootstrap) below for details.

If you're hacking on Hermes from a clone outside `HERMES_HOME/hermes-agent`, point the desktop at it explicitly:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/your/clone npm run dev
```

### Runtime prerequisites

Hermes Desktop needs:

- **Python 3.11+** — for the agent runtime, dashboard backend, and tool execution. (required)
- **Git for Windows** (Windows only) — provides Git Bash, which Hermes' terminal tool calls directly. Linux and macOS already ship a system bash. (required)
- **ripgrep** — used by Hermes' `search_files` tool for fast `.gitignore`-aware file/content search. Recommended on all platforms; Hermes falls back to `grep`/`find` if missing (works but slower and noisier).

The packaged Windows installer (`Hermes-*.exe`) detects all three at install time. Required items missing are auto-installed via `winget install -e --id Python.Python.3.11 --scope user` and `winget install -e --id Git.Git`. The recommended ripgrep is offered as `winget install -e --id BurntSushi.ripgrep.MSVC --scope user`. If `winget` isn't available the installer shows manual download URLs and lets you continue. The MSI installer (`Hermes-*.msi`) doesn't run the prereq page — enterprise deploys are expected to handle prereqs out-of-band.

For dev (`npm run dev`) the Python and Git Bash checks happen at first launch via the Electron bootstrapper, which throws a clear error if either prereq is missing. Manual install commands you can run yourself:

```powershell
winget install -e --id Python.Python.3.11 --scope user
winget install -e --id Git.Git
winget install -e --id BurntSushi.ripgrep.MSVC --scope user
```

## Development

```bash
cd apps/desktop
npm run dev
```

`npm run dev` starts Vite on `127.0.0.1:5174`, launches Electron, and lets Electron boot the Hermes backend (`hermes dashboard --no-open --tui`) on an open port in `9120-9199`. This path is for UI iteration and may still show Electron/dev identities in OS prompts.

Useful overrides:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/hermes-agent npm run dev
HERMES_DESKTOP_PYTHON=/path/to/python npm run dev
HERMES_DESKTOP_CWD=/path/to/project npm run dev
HERMES_DESKTOP_IGNORE_EXISTING=1 npm run dev
HERMES_HOME=/tmp/throwaway-hermes-home npm run dev
HERMES_DESKTOP_BOOT_FAKE=1 npm run dev
HERMES_DESKTOP_BOOT_FAKE=1 HERMES_DESKTOP_BOOT_FAKE_STEP_MS=900 npm run dev
```

`HERMES_DESKTOP_IGNORE_EXISTING=1` skips any `hermes` CLI already on `PATH`, which is useful when testing the factory-image bootstrap path.

`HERMES_HOME` overrides the install root (default: `%LOCALAPPDATA%\hermes` on Windows, `~/.hermes` elsewhere) — handy for sandboxed dev runs that shouldn't touch your real config.

`HERMES_DESKTOP_BOOT_FAKE=1` adds deterministic per-phase delays to desktop startup so you can validate the startup overlay and progress bar. For convenience, `npm run dev:fake-boot` enables fake mode with defaults.

On a fresh Hermes profile, Desktop shows a first-run setup overlay after boot. The overlay saves the minimum required provider credential (for example `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`) to the active Hermes `.env`, reloads the backend env, and then lets the user continue without opening Settings manually.

## Dashboard Dev

Run the Python dashboard backend with embedded chat enabled:

```bash
hermes dashboard --tui --no-open
```

For dashboard HMR, start Vite in another terminal:

```bash
cd web
npm run dev
```

Open the Vite URL. The dev server proxies `/api`, `/api/pty`, and plugin assets to `http://127.0.0.1:9119` and fetches the live dashboard HTML so the ephemeral session token matches the running backend.

## Build

```bash
npm run build
npm run pack          # unpacked app at release/mac-<arch>/Hermes.app
npm run dist:mac      # macOS DMG + zip
npm run dist:mac:dmg  # DMG only
npm run dist:mac:zip  # zip only
npm run dist:win      # NSIS + MSI
```

Before packaging, the desktop app no longer bundles a copy of the Hermes Agent Python source. Instead, the packaged Electron app will fetch and install Hermes Agent at first launch via `scripts/install.ps1`'s stage protocol (Windows) — see the bootstrap flow documented in `electron/main.cjs`. macOS and Linux packaged builds are temporarily non-functional until `install.sh` gains the same stage protocol; dev workflows on all three platforms continue to work since they resolve a sibling source checkout.

## Releases

Installers are built and uploaded manually (no CI auto-build). Build per platform
with the `dist:*` scripts above, then attach the artifacts to a GitHub Release and
point the website's download links at them via env (`NEXT_PUBLIC_HERMES_DL_*`).

Signing credentials are read from the environment at build time when present:

- macOS signing + notarization: `CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_API_KEY`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER`
- Windows signing: `WIN_CSC_LINK`, `WIN_CSC_KEY_PASSWORD`

## Icons

Desktop icons live in `assets/`:

- `assets/icon.icns`
- `assets/icon.ico`
- `assets/icon.png`

The builder config points at `assets/icon`. Replace these files directly if the app icon changes.

## Testing Install Paths

Use the package-local test scripts from this directory:

```bash
npm run test:desktop:all
npm run test:desktop:existing
npm run test:desktop:fresh
npm run test:desktop:dmg
npm run test:desktop:platforms
```

`test:desktop:existing` builds the packaged app and opens it normally. It should use an existing `hermes` CLI if one is on `PATH`, preserving the user’s real `~/.hermes` config.

`test:desktop:fresh` builds the packaged app and launches it in a throwaway fresh-install sandbox. It sets `HERMES_DESKTOP_IGNORE_EXISTING=1`, points Electron `userData` at a temp dir, points `HERMES_HOME` at a temp dir, and launches through the factory-image bootstrap path without touching your real desktop runtime or `~/.hermes`.

`test:desktop:dmg` builds and opens the DMG.

`test:desktop:platforms` runs platform bootstrap-path assertions, including:
- existing-CLI vs factory-image runtime path selection semantics
- WSL2 protection against Windows `.exe/.cmd/.bat/.ps1` overrides
- platform-specific runtime import checks (`winpty` vs `ptyprocess`)

For fast reruns without rebuilding:

```bash
HERMES_DESKTOP_SKIP_BUILD=1 npm run test:desktop:fresh
HERMES_DESKTOP_SKIP_BUILD=1 npm run test:desktop:existing
HERMES_DESKTOP_SKIP_BUILD=1 npm run test:desktop:dmg
```

## Installing Locally

```bash
npm run dist:mac:dmg
open release/Hermes-0.0.0-arm64.dmg
```

Drag `Hermes` to Applications. If testing repeated installs, replace the existing app.

## Runtime Bootstrap

Hermes Desktop shares its install layout with the CLI installers (`scripts/install.ps1`, `scripts/install.sh`) so a desktop-only user and a CLI-only user end up with the same files in the same places.

### Where things live

```text
HERMES_HOME/                       # %LOCALAPPDATA%\hermes (Windows)
                                   # ~/.hermes (macOS / Linux)
├── hermes-agent/                  # ACTIVE_HERMES_ROOT — git checkout
│   ├── .git/                      # canonical install is always a git checkout
│   ├── hermes_cli/, agent/, ...   # Python source
│   ├── pyproject.toml             # source of truth for deps
│   ├── venv/                      # virtualenv (Scripts\python.exe on Windows,
│   │                              #             bin/python elsewhere)
│   └── .hermes-bootstrap-complete # marker: first-launch install.ps1 succeeded
├── git/                           # PortableGit (Windows; installed by install.ps1)
├── config.yaml                    # user config
├── .env                           # API keys
└── logs/
    ├── desktop.log                # Electron-side boot log
    ├── agent.log
    ├── errors.log
    └── gateway.log
```

The packaged installer ships only the Electron app — Hermes Agent itself is fetched and installed at first launch by running `scripts/install.ps1` (Windows) against the git ref baked into the .exe at build time (see `apps/desktop/scripts/write-build-stamp.cjs`).

### Resolution order

The desktop resolves a Hermes backend in this order:

1. `HERMES_DESKTOP_HERMES_ROOT` — explicit dev override.
2. Repo source root — only when running `npm run dev` from a checkout. Takes precedence over `HERMES_HOME/hermes-agent` so devs always run their local edits.
3. `HERMES_HOME/hermes-agent` if the `.hermes-bootstrap-complete` marker is present. The marker attests that install.ps1 succeeded and the user finished initial configuration; we trust the install and skip the bootstrap flow on every launch after the first.
4. Existing `hermes` CLI on PATH (skipped when `HERMES_DESKTOP_IGNORE_EXISTING=1`).
5. Pip-installed `hermes_cli` module via system Python.
6. None of the above → bootstrap-needed sentinel. The desktop's first-launch wizard runs `scripts/install.ps1` stages, then writes the marker on success.

### First-launch flow on a packaged install

1. `resolveHermesBackend()` returns `kind: 'bootstrap-needed'`.
2. The renderer shows the install overlay; main fetches `scripts/install.ps1` from GitHub at the pinned commit (from `install-stamp.json`).
3. Main drives `install.ps1 -Manifest` to get the stage list, then iterates `install.ps1 -Stage <name> -NonInteractive -Json` with live progress events to the renderer.
4. On all stages succeeding, main writes `.hermes-bootstrap-complete` with `{ schemaVersion, pinnedCommit, pinnedBranch, completedAt, desktopVersion }`.
5. Renderer hands off to the existing onboarding overlay (API key / model / persona).
6. Subsequent launches see the marker and skip everything in steps 1-5.

### Updates

Once bootstrapped, the install is a real git checkout. Updates flow through the in-app update path (`applyUpdates()` → `git fetch && git pull --ff-only` against the configured branch) or `hermes update` from the CLI. Both check `pyproject.toml` drift and re-run `pip install -e .` only when needed.

A user who installed via `scripts/install.ps1` directly (so `HERMES_HOME/hermes-agent/.git` exists but no `.hermes-bootstrap-complete` marker) is detected via resolver step 4 (their `hermes` CLI on PATH) and the desktop reuses their install without re-running the bootstrap.

## Debugging

Desktop boot logs are written to:

```text
HERMES_HOME/logs/desktop.log     # %LOCALAPPDATA%\hermes\logs\desktop.log on Windows
                                  # ~/.hermes/logs/desktop.log on macOS / Linux
```

If the UI reports `Desktop boot failed`, check that log first. It includes the backend command output and recent Python traceback context.

To force a fresh first-launch bootstrap (rare — useful for development / dogfooding the install flow):

```bash
# macOS / Linux
rm "$HOME/.hermes/hermes-agent/.hermes-bootstrap-complete"

# Windows (PowerShell)
Remove-Item "$env:LOCALAPPDATA\hermes\hermes-agent\.hermes-bootstrap-complete"
```

For a full reset of just the Python venv (rare — usually only needed if the venv is broken):

```bash
# macOS / Linux
rm -rf "$HOME/.hermes/hermes-agent/venv"

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes\hermes-agent\venv"
```

To reset stale macOS microphone permission prompts:

```bash
tccutil reset Microphone com.github.Electron
tccutil reset Microphone com.nousresearch.hermes
```

## Verification

Run before handing off installer changes:

```bash
npm run fix
npm run type-check
npm run lint
npm run test:desktop:all
```

Current lint may report existing warnings, but it should exit with no errors.
