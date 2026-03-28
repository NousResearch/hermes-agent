# Windows QoL Branch Handoff

## Current State

**Branch:** `windows-qol-v2` (1 commit ahead of `origin/main` @ `02fb7c4a`)
**Commit:** `20294a32` — "feat: Windows QoL — full platform support rebased on latest upstream"
**Repo:** `/mnt/c/Users/Carlos/hermes-agent`
**Tests:** 94/94 pass (clipboard + compat)

### What's Done (committed on windows-qol-v2)

10 files with surgical Windows patches on top of latest upstream:

| File | Change |
|------|--------|
| `hermes_cli/clipboard.py` | Full Windows clipboard: ctypes win32 API (64-bit safe), text functions, PowerShell hardening, pwsh.exe support, .NET disposal, base64 line-break handling, -ExecutionPolicy Bypass |
| `agent/display.py` | `write_tty()` uses `open("CON", "w")` on Windows instead of `/dev/tty` |
| `tools/memory_tool.py` | `fcntl` wrapped in try/except, `msvcrt.locking` fallback for Windows file locking |
| `tools/voice_mode.py` | `PermissionError` handler in `cleanup_temp_recordings()` (Windows can't unlink open files) |
| `hermes_cli/gateway.py` | Windows Task Scheduler (`schtasks`): install/uninstall/start/stop, SIGTERM fallback (no SIGKILL on Windows) |
| `gateway/status.py` | `wmic process` queries for `_get_process_start_time()` and `_read_process_cmdline()`, `/proc/status` guard |
| `hermes_cli/config.py` | `icacls` ACL enforcement in `_secure_dir()` / `_secure_file()` (replaces no-op chmod on Windows) |
| `tools/environments/local.py` | `taskkill /F /T /PID` for child kill, `tempfile.gettempdir()`, Git Bash finder with Windows paths |
| `tools/environments/persistent_shell.py` | `tempfile.gettempdir()` instead of hardcoded `/tmp` |
| `tools/code_execution_tool.py` | TCP localhost sandbox fallback (`_detect_transport()`), `SYSTEMROOT`/`SYSTEMDRIVE`/`COMSPEC`/`WINDIR` in safe env prefixes |

### What Still Needs Doing

#### 1. cli.py Windows Patches (~30 lines, manual work)

The upstream cli.py changed heavily (imports, paste collapsing, new features). The Windows-specific additions from the old branch need to be surgically re-applied to the **new upstream cli.py**:

**a) Ctrl+V keybinding — text fallback** (around line ~6411 equivalent in new file)
The old branch's `c-v` handler falls back to `get_clipboard_text()` when no image found:
```python
@kb.add('c-v')
def handle_ctrl_v(event):
    if _attach_and_notify(event):
        return
    # No image — try text clipboard (critical for terminals that don't
    # send BracketedPaste, e.g., Windows Terminal with image-only clipboard)
    from hermes_cli.clipboard import get_clipboard_text
    txt = get_clipboard_text()
    if txt:
        event.current_buffer.insert_text(txt)
```

**b) Shift+Insert binding** (after the Alt+V handler)
```python
@kb.add('s-insert')
def handle_shift_insert(event):
    _attach_and_notify(event)
```

**c) `/paste` command — text fallback** (in `_handle_paste_command`)
After the image check fails, try `get_clipboard_text()`:
```python
from hermes_cli.clipboard import has_clipboard_image, get_clipboard_text
# ... existing image check ...
# If no image, try text
txt = get_clipboard_text()
if txt:
    # insert into buffer or show preview
```

**d) Platform-aware startup hint** (in the welcome banner area)
```python
if sys.platform == "win32":
    _cprint(f"  {_DIM}Paste image: Ctrl+V (or /paste){_RST}\n")
else:
    _cprint(f"  {_DIM}Paste image: Alt+V, Ctrl+V (or /paste){_RST}\n")
```

**How to find the right locations in the new cli.py:**
```bash
grep -n "c-v\|BracketedPaste\|escape.*v\|_handle_paste\|Paste image" cli.py
```

#### 2. Additive Files to Copy from Old Branch

These files exist on `windows-qol-local` and need to be brought over:

```bash
# Shell quoting utility (cross-platform shlex replacement)
git show windows-qol-local:tools/shell_quote.py > tools/shell_quote.py

# Windows documentation
git show windows-qol-local:WINDOWS.md > WINDOWS.md

# Windows compat test suite (37 tests)
git show windows-qol-local:tests/tools/test_windows_compat.py > tests/tools/test_windows_compat.py

# Windows CI workflow
git show windows-qol-local:.github/workflows/windows-tests.yml > .github/workflows/windows-tests.yml

# Windows PowerShell installer
git show windows-qol-local:scripts/install-windows.ps1 > scripts/install-windows.ps1
```

**Note:** `test_windows_compat.py` may need import adjustments if upstream moved functions. Compile-check after copying.

#### 3. _find_bash WSL Filtering (minor improvement)

The old branch had more careful filtering of `System32\bash.exe` and `WindowsApps\bash.exe` (WSL launcher) in `_find_bash()`. The current v2 branch's version uses `shutil.which("bash")` on Windows without filtering. The old version checked:
```python
found_lower = found.lower()
if "system32" not in found_lower and "windowsapps" not in found_lower:
    return found
```
This prevents accidentally launching WSL bash when Git Bash is intended. Consider adding this filter back.

#### 4. README.md Update

The old branch changed the README from "Native Windows is not supported" to "Native Windows 10/11 is supported" with install instructions and WINDOWS.md link.

### Reference: Old Branch

**Branch:** `windows-qol-local` (11 commits, merge base 193 commits behind main)
**Stash:** `upstream-drift-working-tree-20260328` (the 240-file upstream drift)

To compare what the old branch had vs what's on v2:
```bash
# See all files the old branch touched
git diff --name-only windows-qol-local~10..windows-qol-local

# See specific old code for reference
git show windows-qol-local:<filepath>
```

### Verification Commands

```bash
# Compile all Windows-modified files
for f in hermes_cli/clipboard.py cli.py agent/display.py tools/memory_tool.py \
         hermes_cli/gateway.py tools/environments/local.py \
         tools/environments/persistent_shell.py tools/code_execution_tool.py \
         hermes_cli/config.py gateway/status.py tools/voice_mode.py; do
    python3 -m py_compile "$f" && echo "OK: $f" || echo "FAIL: $f"
done

# Run test suites
python3 -m pytest tests/tools/test_clipboard.py -o "addopts=" -q --tb=short
python3 -m pytest tests/tools/test_windows_compat.py -o "addopts=" -q --tb=short
```

### Critical Context

- **logout corruption:** WSL terminal sessions inject `logout` into files. Check for it after every edit session. The skill `hermes-windows-development` has full details.
- **Patch tool indentation:** `mcp_patch` fuzzy matching can introduce 3-space indentation (should be 4). Always compile-check after patching.
- **cli.py is 7500+ lines:** Don't use full-file patches. Use targeted `mcp_patch` replace mode with enough surrounding context for uniqueness.
- **Never push without Carlos saying "push" three times.**
