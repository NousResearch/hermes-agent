# Windows QoL & Security Plan for Hermes
## Generated: 2026-03-22

Based on a full codebase audit (compatibility, security, existing Windows work).

---

## Phase 1: Stop the Crashes (Critical Fixes)
*Estimated effort: 1-2 sessions*

### 1.1 memory_tool.py — bare `import fcntl`
- **File:** `tools/memory_tool.py:26`
- **Problem:** Module-level `import fcntl` with no guard. Entire memory tool
  fails to import on Windows.
- **Fix:** Add try/except with msvcrt fallback (same pattern as auth.py,
  scheduler.py). Update `_file_lock()` to use msvcrt.locking().

### 1.2 gateway.py — signal.SIGKILL without platform guard
- **File:** `hermes_cli/gateway.py:883`
- **Problem:** `os.kill(pid, signal.SIGKILL)` in `_wait_for_gateway_exit()`.
  SIGKILL doesn't exist on Windows. AttributeError on any gateway shutdown.
- **Fix:** Guard with `signal.SIGTERM` on win32.

### 1.3 Hardcoded /tmp paths
- **Files:** `tools/environments/local.py:321`, `persistent_shell.py:47`
- **Problem:** `/tmp/hermes-local-*` and `/tmp/hermes-persistent-*` don't
  resolve on Windows.
- **Fix:** Replace with `tempfile.gettempdir()` + prefix.

### 1.4 agent/display.py — /dev/tty
- **Problem:** `os.open("/dev/tty", os.O_WRONLY)` crashes on Windows.
- **Fix:** Use `"CON"` on Windows, fallback to sys.stderr.

---

## Phase 2: Security Hardening (High Priority)
*Estimated effort: 2-3 sessions*

### 2.1 Credential storage — use Windows Credential Manager
- **Problem:** auth.json and .env store API keys, OAuth tokens, SUDO_PASSWORD
  in plaintext. Unix chmod(0o600) is a no-op on Windows — any local user can
  read them.
- **Fix:** Integrate `keyring` library (uses Windows Credential Manager/DPAPI
  automatically). Store secrets in WCM, keep only non-sensitive config in files.
- **Scope:** hermes_cli/auth.py, hermes_cli/setup.py, .env handling

### 2.2 MCP config secrets
- **Problem:** MCP server configs embed API keys directly in config.yaml.
- **Fix:** Support `$ENV{VAR_NAME}` references in MCP config, resolve at
  runtime. Keep secrets in env vars or WCM, not in the config file.

### 2.3 shell=True with user-configurable commands
- **File:** `cli.py:3694` (quick commands)
- **Problem:** User-configurable commands run with `shell=True`. On Windows,
  this invokes cmd.exe with unreliable escaping.
- **Fix:** Use `subprocess.run()` with list args where possible. For commands
  that need shell features, validate/sanitize input.

### 2.4 shlex.quote() on Windows
- **File:** `tools/transcription_tools.py`
- **Problem:** `shlex.quote()` produces Unix-style quoting that doesn't
  protect against cmd.exe injection.
- **Fix:** Add a `_win_quote()` helper that uses `^` escaping for cmd.exe
  special chars, or avoid shell=True entirely.

### 2.5 SMS webhook binds to 0.0.0.0
- **File:** `gateway/platforms/sms.py`
- **Problem:** Webhook server binds all interfaces, exposing it to the network.
- **Fix:** Default to 127.0.0.1, require explicit config to bind externally.

---

## Phase 3: Feature Parity (Medium Priority)
*Estimated effort: 3-5 sessions*

### 3.1 Gateway service management for Windows
- **Problem:** Gateway install/start/stop is entirely systemd-based (~700 lines).
  On Windows: "not supported".
- **Fix options (pick one):**
  a) **Task Scheduler** — `schtasks` for auto-start, simplest
  b) **Windows Service** via NSSM — most robust, like systemd
  c) **Startup folder shortcut** — lowest friction for non-technical users
- **Recommendation:** Start with Task Scheduler (a), add NSSM option later.

### 3.2 Gateway status on Windows
- **File:** `hermes_cli/status.py`
- **Problem:** Uses /proc/{pid}/stat (Linux procfs). Shows "N/A" on Windows.
- **Fix:** Use `tasklist` or `wmic` to find gateway process by name/PID.
  Or use `psutil` if it's already a dependency.

### 3.3 Code execution sandbox on Windows
- **Problem:** Sandbox uses Unix Domain Sockets, disabled entirely on Windows
  (`SANDBOX_AVAILABLE = sys.platform != "win32"`).
- **Fix:** Implement Named Pipes IPC for Windows. Or use TCP localhost as a
  simpler alternative that works cross-platform.

### 3.4 Process killing on Windows
- **File:** `tools/environments/local.py:350-353`
- **Problem:** `pkill -P` for killing shell children doesn't exist on Windows.
- **Fix:** Use `taskkill /F /T /PID` on Windows (kill process tree).

---

## Phase 4: Polish & Documentation
*Estimated effort: 1-2 sessions*

### 4.1 Create WINDOWS.md
- **Problem:** README says "Native Windows is not supported" but there's a
  full PowerShell installer and significant Windows code. Confusing.
- **Fix:** Write WINDOWS.md covering: installation, known limitations,
  workarounds, how to report Windows-specific issues.

### 4.2 Windows CI
- **Problem:** No Windows CI pipeline. Regressions go unnoticed.
- **Fix:** Add GitHub Actions `windows-latest` job running the test suite.
  Start with just `test_windows_compat.py` and clipboard tests.

### 4.3 Expand test_windows_compat.py
- **Problem:** Only tests 4 files for POSIX guards. Many gaps.
- **Fix:** Add tests for:
  - memory_tool fcntl fallback
  - clipboard Windows path
  - _find_bash() Windows discovery
  - gateway status on Windows
  - temp path resolution

### 4.4 Fix startup hint text
- Show Windows-specific keyboard shortcuts
- Show Windows-specific paths (LOCALAPPDATA vs ~/.hermes)
- Detect and warn about missing Git Bash

### 4.5 Voice temp file cleanup
- **Problem:** STT temp files persist in %TEMP% without cleanup.
- **Fix:** Use `tempfile.NamedTemporaryFile(delete=True)` or explicit
  cleanup in a finally block.

---

## Execution Order (Recommended)

```
Week 1:  Phase 1 (crashes)     — unblocks daily use
         Phase 2.1 (keyring)   — biggest security win
Week 2:  Phase 2.2-2.5         — remaining security
         Phase 3.4 (pkill)     — process management
Week 3:  Phase 3.1-3.2         — gateway on Windows
         Phase 4.1 (docs)      — WINDOWS.md
Week 4:  Phase 3.3 (sandbox)   — code execution
         Phase 4.2-4.5         — CI, tests, polish
```

---

## What's Already Working Well

These are done and don't need changes:

- ✔ Clipboard image/text paste (macOS, Windows, WSL, Linux)
- ✔ PowerShell path resolution with caching
- ✔ Ctrl+V image paste on Windows Terminal
- ✔ fcntl→msvcrt fallback in auth.py, scheduler.py
- ✔ _IS_WINDOWS guards in process_registry, local env, code_execution
- ✔ Git Bash discovery for shell tool
- ✔ pywinpty conditional dependency
- ✔ Platform-aware skill filtering
- ✔ PowerShell installer (scripts/install.ps1)
- ✔ WhatsApp platform Windows process management
- ✔ UTF-8 encoding on file I/O
