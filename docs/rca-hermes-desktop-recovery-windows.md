# Hermes Desktop Fails to Start / "Session Recovery Failed" on Windows — Root Cause & Fix

> **Environment**: Windows 10 (10.0.19045), Hermes Agent v0.20.0 (2026.8.3), install at `F:\hermes\hermes-agent`
> **Reported by**: @fangyuchuang

---

## 1. Summary

Hermes Desktop on Windows repeatedly failed to launch:

1. First symptom: the app stalled on the **"Set up Hermes Desktop"** first-run screen.
2. After rebuilding the virtualenv, the backend started but the UI reported:
   `恢复失败 (Recovery failed) — request timed out after 30s: session.resume`.
3. The Electron main process could also crash with:
   `SyntaxError: The requested module 'electron' does not provide an export named 'BrowserWindow'`.

All three symptoms had different underlying causes (see §4). After fixing each of them in turn,
Hermes Desktop starts cleanly: backend announces `HERMES_BACKEND_READY`, the UI loads normally,
and **no more `session.resume` timeouts occur**.

---

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows 10 Pro 10.0.19045.6466 |
| Hermes | Agent v0.20.0 (2026.8.3) |
| Install dir | `F:\hermes\hermes-agent` |
| Hermes Home | `F:\hermes` |
| Required Python | **3.11** (`.python-version` file) |
| Electron | 40.10.2 (packaged in `apps/desktop`) |
| Path shim | `F:\hermes\bin\hermes.BAT` → `venv\Scripts\hermes.exe` |

---

## 3. Observed Symptoms

### 3.1 Stage A — Stuck on "Set up Hermes Desktop"

The desktop app only offered two choices ("Connect to existing Hermes" / "Install Hermes locally")
and never detected the existing installation at `F:\hermes\hermes-agent`.

Key log lines (`F:\hermes\logs\desktop.log`):

```
[hermes] Ignoring existing Hermes CLI at F:\hermes\bin\hermes.BAT: --version probe failed; falling through to bootstrap.
[hermes] Ignoring system Python C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe: hermes_cli is not importable; falling through to bootstrap.
[hermes] [boot] Waiting for first-run setup choice
```

### 3.2 Stage B — Backend up, but "Recovery failed (session.resume timeout)"

After the virtualenv was rebuilt, the backend itself started fine:

```
HERMES_BACKEND_READY port=6xxxx
[hermes] [boot] Hermes backend is ready. Finalizing desktop startup
```

…but the renderer showed **"恢复失败" (Recovery failed)**:
`request timed out after 30s: session.resume`.

### 3.3 Stage C — Electron crashes with `BrowserWindow` import error

When launching Electron directly from a shell, the main process died immediately:

```
file:///F:/hermes/hermes-agent/apps/desktop/dist/electron-main.mjs:13106
import { BrowserWindow, screen } from "electron";
SyntaxError: The requested module 'electron' does not provide an export named 'BrowserWindow'
```

---

## 4. Root Causes (chained, each masked the next)

### 4.1 Python version mismatch — **the primary root cause**

- `F:\hermes\hermes-agent\.python-version` requires **Python 3.11**.
- The original `venv` had been created with **Python 3.13.14**.
- On 3.13 the dependency resolution still *worked*, but the environment was fragile:
  `import _ssl` eventually failed with
  `DLL load failed while importing _ssl: 另一个程序正在使用此文件，进程无法访问。` (os error 32),
  which killed the backend and made Desktop fall through to the first-run screen.

### 4.2 Lazy-installed deps block startup → `session.resume` timeout

Hermes lazy-installs heavy optional dependencies at **first startup** (see `agent.log`):

```
INFO tools.lazy_deps: Lazy-installing faster-whisper==1.2.1 sounddevice==0.5.5 numpy==2.4.3 for feature 'stt.faster_whisper'
INFO tools.lazy_deps: Lazy-installing anthropic==0.87.0 for feature 'provider.anthropic'
```

These installs (especially `faster-whisper`/`ctranslate2`) take several minutes and **block the
backend event loop**. The renderer's `session.resume` call times out after 30 s → "恢复失败".

### 4.3 `execFileSync` cannot probe a `.BAT` shim

`electron-main.mjs` probes a candidate CLI with:

```js
execFileSync(hermesCommand, ["--version"], { shell: shellForProbe, timeout: 15000, ... })
```

`findOnPath("hermes")` resolves to `F:\hermes\bin\hermes.BAT` (a `.bat` shim). The probe
fails, so Desktop logs `--version probe failed` and falls into bootstrap, even though the
underlying `venv\Scripts\hermes.exe` is perfectly fine.

### 4.4 `ELECTRON_RUN_AS_NODE=1` inherited from the shell

Launching Electron from a Git Bash session that had `ELECTRON_RUN_AS_NODE=1` in the
environment made `electron.exe` run in **pure Node.js mode**. The `electron` npm package
does not export `BrowserWindow`, so the ESM import at
`dist/electron-main.mjs:13106` threw `SyntaxError`.

---

## 5. Fixes Applied

### 5.1 Rebuild the venv with the required Python (3.11)

```bash
cd F:\hermes\hermes-agent
rm -rf venv
F:\hermes\bin\uv.exe venv --python 3.11 venv     # CPython 3.11.15
```

### 5.2 Install dependencies

- `pyproject.toml` sets `exclude-newer = "14 days"`; Chinese mirror indexes (Aliyun/Tsinghua)
  omit `upload-time` metadata, so uv mis-filters every package. Use the official PyPI index
  with a far-future cutoff:

```bash
UV_EXCLUDE_NEWER=2099-01-01 UV_LINK_MODE=copy \
  F:\hermes\bin\uv.exe pip install -e ".[dev]" --no-build-isolation --python venv
```

- `--no-build-isolation` avoids the uv build-sandbox `os error 32` (file locked by the
  sandbox Python) seen on this machine.
- Install any stragglers individually if a batch stalls:

```bash
F:\hermes\bin\uv.exe pip install uvicorn pywin32 pywinpty --python venv
```

### 5.3 Pre-install the lazy dependencies so startup never blocks

```bash
F:\hermes\bin\uv.exe pip install \
  faster-whisper==1.2.1 sounddevice==0.5.5 "numpy==2.4.3" \
  google-auth==2.55.1 pyasn1==0.6.4 anthropic==0.87.0 --python venv
```

### 5.4 Point Desktop at the real executable (avoid the `.bat` probe)

Persist a user-level environment variable:

```bat
setx HERMES_DESKTOP_HERMES "F:\hermes\hermes-agent\venv\Scripts\hermes.exe"
```

`resolveHermesBackend()` checks `process.env.HERMES_DESKTOP_HERMES` first and, because the
path is an absolute `.exe`, `unwrapWindowsVenvHermesCommand()` correctly maps it to the venv
Python, bypassing the `.bat` probe failure entirely.

### 5.5 Write the bootstrap-complete marker

The active runtime is usable, but the marker was missing, so Desktop entered the repair
flow. Create `F:\hermes\hermes-agent\.hermes-bootstrap-complete`:

```json
{
  "schemaVersion": 1,
  "pinnedCommit": null,
  "pinnedBranch": null,
  "completedAt": "2026-08-13T14:00:00.000Z",
  "desktopVersion": "0.20.0"
}
```

### 5.6 Launch Electron without `ELECTRON_RUN_AS_NODE`

```bash
cd F:\hermes\hermes-agent\apps\desktop
unset ELECTRON_RUN_AS_NODE
HERMES_DESKTOP_HERMES="F:\hermes\hermes-agent\venv\Scripts\hermes.exe" \
  node_modules\electron\dist\electron.exe .
```

> Note: launching Desktop from the Start Menu / desktop shortcut does **not** have this
> problem — the variable is only present in some shells (e.g. the WorkBuddy agent sandbox).

---

## 6. Verification

```text
F:\hermes\hermes-agent\venv\Scripts\hermes.exe --version
  Hermes Agent v0.20.0 (2026.8.3) — Python: 3.11.15, OpenAI SDK: 2.24.0

hermes serve
  HERMES_BACKEND_READY port=9119          (no warnings)

Desktop log (clean start):
  [hermes] [boot] Hermes runtime is ready
  [hermes] [boot] Starting Hermes backend via Hermes at F:\hermes\hermes-agent (venv: F:\hermes\hermes-agent\venv)
  [hermes] HERMES_BACKEND_READY port=61563
  [hermes] [boot] Hermes backend is ready. Finalizing desktop startup
  (no probe failures, no first-run gate, no session.resume timeout)
```

Dependency sanity check (all importable):

```
ssl fastapi uvicorn openai cryptography pydantic          ✓ core
faster_whisper sounddevice numpy google.auth anthropic    ✓ lazy
winpty win32api charset_normalizer                        ✓ windows/runtime
```

---

## 7. Suggestions for Maintainers

1. **Enforce the Python version at venv creation time.** The desktop bootstrap should refuse
   (or migrate) a venv built with a Python major/minor that differs from `.python-version`;
   a silent 3.13-vs-3.11 mismatch caused all downstream instability here.
2. **Pre-warm lazy deps during install** (or install them non-blocking / in a thread) so that
   first startup does not stall `session.resume` for minutes. Consider raising the
   `session.resume` client timeout or making resume independent of lazy-deps.
3. **`execFileSync` probing of `.bat` shims**: prefer resolving `venv\Scripts\hermes.exe`
   before probing the shim, or add the venv `Scripts` dir to `PATH` ahead of the shim dir.
4. **Clearer Electron bootstrap errors** would help: `BrowserWindow` export errors from
   `ELECTRON_RUN_AS_NODE=1` are cryptic.
5. Consider documenting `HERMES_DESKTOP_HERMES` and `HERMES_DESKTOP_HERMES_ROOT` as supported
   escape hatches (they are extremely useful for troubleshooting).

---

*Report generated after on-site diagnosis and verification on 2026-08-13.*
