# Windows Support — Status and Contribution Guide

> This document is a technical map of the current state of Windows compatibility in Hermes and a guide for contributors who want to help close the gaps. It was written to accompany [#2512](https://github.com/NousResearch/hermes-agent/issues/2512).

---

## Current State

Hermes was developed primarily on Linux/macOS, but a significant amount of Windows groundwork already exists in the codebase. This is not a greenfield effort.

### What already works

| Area | Status | Details |
|------|--------|---------|
| PTY / process spawning | Handled | `pywinpty` is a declared optional dep (`pip install hermes-agent[pty]`). `tools/process_registry.py` imports `winpty.PtyProcess` on Windows instead of `ptyprocess`. |
| Process group kill | Handled | `_kill_process_group()` in `tools/code_execution_tool.py` and `tools/environments/local.py` both check `_IS_WINDOWS` and use `proc.terminate()` / `proc.kill()` instead of `os.killpg`. |
| `preexec_fn` | Handled | All `subprocess.Popen` calls use `preexec_fn=None if _IS_WINDOWS else os.setsid`. |
| File locking | Handled | `hermes_cli/auth.py` and `cron/scheduler.py` both have `try: import fcntl / except: import msvcrt` with Windows fallback paths. |
| bash detection | Handled | `tools/environments/local.py` probes Git for Windows paths (`C:\Program Files\Git\bin\bash.exe`, etc.) and the `HERMES_GIT_BASH_PATH` env var. |
| venv path | Handled | `hermes_cli/main.py` uses `Scripts` instead of `bin` on `win32`. |
| Config/data paths | Handled | `hermes_cli/config.py` uses `platformdirs` (`_IS_WINDOWS`) for correct `%APPDATA%` placement. |

### What is broken or untested

#### 1. `tools/memory_tool.py` — hard `import fcntl` (no fallback)

```python
# Line 26 — bare import, no try/except
import fcntl
```

This will raise `ModuleNotFoundError` on Windows the moment any tool tries to use memory. **This is the most critical blocker.**

**Fix:** Mirror the pattern from `hermes_cli/auth.py`:

```python
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False
```

Then replace `fcntl.flock(fd, fcntl.LOCK_EX)` with a cross-platform helper:

```python
import msvcrt, os

def _lock(fd):
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_EX)
    else:
        msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)

def _unlock(fd):
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_UN)
    else:
        msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
```

#### 2. `curses` — no Windows fallback

`hermes_cli/curses_ui.py`, `hermes_cli/checklist.py`, `hermes_cli/setup.py`, `hermes_cli/tools_config.py`, `hermes_cli/main.py`, and `hermes_cli/mcp_config.py` all import `curses`. The standard `curses` module does not ship with CPython on Windows.

The imports are all wrapped in `try/except`, but the fallback paths typically just print a plain-text list with no interactive selection. This degrades the setup experience but won't crash.

**Options:**
- Accept the degraded UX (already implemented via `except` fallbacks).
- Add `windows-curses` as an optional dependency (`pip install hermes-agent[windows]`) and document it.
- Replace `curses` with a cross-platform TUI library (e.g. `prompt_toolkit`, which `rich` already uses indirectly via `textual`) — larger refactor, not required for initial support.

#### 3. Shell requirement — Git Bash / WSL

Hermes uses bash syntax in its command fencing (`$?`, semicolons, `printf`). On Windows this requires:
- **Git for Windows** (provides `bash.exe`) — already probed in `_find_bash()`
- **WSL** — not yet detected or documented

If neither is available, `hermes terminal` will fail with a clear error from `_find_bash()`. The error message already suggests `HERMES_GIT_BASH_PATH`.

**Recommendation:** Document these requirements prominently in the Windows install guide (see below). PowerShell support would require rewriting the command fencing — out of scope for initial support.

#### 4. `os.killpg` in `tools/environments/local.py` (lines 450)

Most kill calls are already guarded, but line 450 has:

```python
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
```

without a `_IS_WINDOWS` guard. Needs the same `proc.terminate()` fallback.

#### 5. `SIGALRM` in tests

`tests/conftest.py` already skips `SIGALRM` on Windows (line 111). No action needed, but worth knowing if you're running the test suite.

#### 6. Gateway platforms that use `os.setsid` / `os.killpg`

`gateway/platforms/whatsapp.py` has proper `_IS_WINDOWS` guards. Other gateway platforms should be audited when they're tested on Windows.

---

## Recommended Contribution Scope

For an initial "Windows works" PR, the minimum viable fix set is:

1. **Fix `tools/memory_tool.py`** — the hard `import fcntl` blocker (15 lines).
2. **Document Git for Windows / WSL as a prerequisite** — in `README.md` and `docs/install.md` (if it exists).
3. **Fix the unguarded `os.killpg` in `tools/environments/local.py` line 450**.
4. **Add a CI job** — even a basic smoke test (`hermes --help`) on `windows-latest` in GitHub Actions catches regressions.

Everything else (curses UX, PowerShell support, WSL detection) can be follow-up PRs.

---

## Setting Up a Dev Environment on Windows

### Prerequisites

- Python 3.10+ from [python.org](https://www.python.org/downloads/windows/) (not the Microsoft Store version — it lacks some stdlib headers)
- [Git for Windows](https://git-scm.com/download/win) — required for bash
- Optional: [Windows Terminal](https://aka.ms/terminal) for a better experience

### Install

```powershell
git clone https://github.com/NousResearch/hermes-agent
cd hermes-agent
python -m venv venv
.\venv\Scripts\activate
pip install -e ".[pty]"
```

If you want interactive setup UIs (curses):
```powershell
pip install windows-curses
```

### Required environment variable (if bash isn't on PATH)

```powershell
$env:HERMES_GIT_BASH_PATH = "C:\Program Files\Git\bin\bash.exe"
```

### WSL alternative

If you're on Windows 10/11 with WSL2, running Hermes inside WSL is the path of least resistance — full Linux compatibility, no bash shim needed.

```bash
# Inside WSL
pip install hermes-agent
hermes setup
```

---

## Testing Matrix Targets

When submitting a Windows PR, ideally test against:

| Scenario | Minimum | Ideal |
|----------|---------|-------|
| `hermes --help` | x | x |
| `hermes setup` (provider config) | x | x |
| `hermes` (interactive chat, no tools) | x | x |
| `terminal` tool (bash via Git for Windows) | x | x |
| `memory` tool (read/write) | x | x |
| PTY mode (`hermes --pty`) | | x |
| Gateway mode | | x |

---

## Related Issues and PRs

- [#2512](https://github.com/NousResearch/hermes-agent/issues/2512) — tracking issue for native Windows support
- `pyproject.toml` — `[project.optional-dependencies]` already defines `pty = ["ptyprocess>=0.7.0; sys_platform != 'win32'", "pywinpty>=2.0.0; sys_platform == 'win32'"]`
