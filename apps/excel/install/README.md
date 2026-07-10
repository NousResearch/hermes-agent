# Hermes for Excel — install / packaging

Per-box, per-user installer for the Hermes for Excel add-in (Office.js task pane
+ local Node bridge). **Windows-first** (the fleet is Windows + WSL). Nothing
here requires elevation — everything lives under `%LOCALAPPDATA%\hermes\excel-addin`
and `HKCU`.

## Prerequisites

- **Windows 10/11**, Excel for Windows (desktop) with add-in sideloading allowed.
- **Node.js LTS** — the installer auto-installs it via `winget install OpenJS.NodeJS.LTS`
  if `node` is not on PATH. If `winget` is unavailable, install Node.js LTS
  manually from <https://nodejs.org> and re-run.
- A running **Hermes gateway** (`api_server` on `http://127.0.0.1:8642/v1`) for
  the add-in to be useful. The installer's Hermes check is non-fatal — install
  still succeeds if the gateway is down; you just won't get model replies until
  it's up.
- Optional: **Docling** for PDF/Office parsing. Default mode is `wsl` (distro
  `Ubuntu-24.04`); see env vars below.

## One-line install

```powershell
powershell -ExecutionPolicy Bypass -File install\apply.ps1
```

By default the installer selects the first free port starting at `8787` and
rewrites the installed manifest to match. Add `-Port 8790` to require a
specific port (the install fails if it is occupied), or
`-SkipSideload` to skip the Excel developer-catalog registration, or
`-SkipExcelAddin` to opt a box out entirely.

After install, **restart Excel** and open the pane from the **Hermes** ribbon
group.

## What each script does

| Script | Purpose |
| --- | --- |
| `apply.ps1` | Fleet entry point. Calls `addin-install.ps1`; `-SkipExcelAddin` opts a box out; passes `-Port` / `-SkipSideload` through. |
| `addin-install.ps1` | The installer. Ensures Node, copies the payload to `%LOCALAPPDATA%\hermes\excel-addin`, creates the data dir, unblocks mark-of-the-web, generates the bridge token, emits the launchers, registers autostart + sideload, then health-checks the bridge (fails loudly if it doesn't come up). |
| `run-bridge.cmd.template` | Template for the bridge launcher. The installer substitutes `__INSTALL_DIR__/__PORT__/__DATA_DIR__/__BRIDGE_TOKEN__/__DOCLING_MODE__/__WSL_DISTRO__`. Sets env, **kills any stale bridge from this install dir**, then `node <abs path>\broker\server.mjs`. |
| `run-bridge.vbs.template` | Hidden launcher for `run-bridge.cmd` (no console window, no elevation). |
| `register-task.ps1` | Registers the logon-triggered Scheduled Task `Hermes_Excel_Bridge` (per-user, hidden, no elevation) running the supervisor. `-Unregister` removes it. |
| `register-sideload.ps1` | Writes the manifest to the per-user catalog and sets `HKCU\...\WEF\Developer` so Excel sideloads it. Port-substitutes the manifest when `-Port` ≠ 8787. `-Unregister` reverses it. |
| `service\bridge-service.cmd` | Restart-loop supervisor: relaunches the bridge on exit, backs off (stands down) when it crashes instantly, logs restart events with timestamps to `bridge-restarts.log` under the **data dir**. |
| `rollback.ps1` | Reverses everything: stops/unregisters the task, removes any legacy Startup shortcut, removes the WEF value + catalog manifest, deletes the install dir (prompts unless `-Force`). |

## Environment variables (set by the launchers, read by the bridge)

| Var | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8787` | Bridge port. |
| `HERMES_EXCEL_DATA_DIR` | `%LOCALAPPDATA%\hermes\excel-addin\data` | Writable per-user data root for uploads/exports/logs, **outside** the served web root. |
| `HERMES_EXCEL_BRIDGE_TOKEN` | generated per install | Random session token the bridge requires on `/api/*` and the pane sends. Persisted to `data\.bridge-token` (user-only ACL) and the `User` env scope. |
| `HERMES_EXCEL_DOCLING_MODE` | `wsl` | `wsl` \| `native` \| `docker`. |
| `HERMES_EXCEL_WSL_DISTRO` | `Ubuntu-24.04` | WSL distro used when docling mode is `wsl`. |

## Manual sideload fallback

If the registry catalog isn't picked up (policy-locked box, or a different Excel
build), load the manifest by hand:

> **Excel → Home → Add-ins → More Add-ins → My Add-ins → Upload My Add-in →**
> browse to `%LOCALAPPDATA%\hermes\excel-addin\OfficeAddinManifests\hermes-excel-addin.xml`

The Upload-My-Add-in path works even when the Developer catalog does not.

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File install\rollback.ps1          # prompts before deleting data
powershell -ExecutionPolicy Bypass -File install\rollback.ps1 -Force   # no prompt
```

## Deployer assumptions to verify

- **Office registry version is `16.0`** (Office 2016/2019/2021/365). Older Excel
  uses a different `WEF\Developer` hive — pass `-OfficeVersion` to
  `register-sideload.ps1` / `rollback.ps1` if needed.
- **`winget` availability** — required only if Node isn't already installed.
- **WSL distro name** (`Ubuntu-24.04`) and **docling mode** match the box; retune
  the config block at the top of `addin-install.ps1`.
- The hardened `broker\server.mjs` must honor `HERMES_EXCEL_DATA_DIR` /
  `HERMES_EXCEL_BRIDGE_TOKEN` (coordinated with the main session). These scripts
  set them; the bridge consumes them.
