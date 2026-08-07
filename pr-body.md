## What does this PR do?

When Hermes is started from PowerShell or cmd.exe on Windows, `os.getcwd()` returns a Windows native path (`C:\Users\...`) which gets stored as the session working directory. Both `init_session()` and `_wrap_command()` embed this path directly into bash scripts via `cd`, but Git Bash cannot parse Windows drive-letter paths.

**Impact:** Every terminal tool call fails with exit code 126 the first time any command runs. The agent cannot write files, search, or execute shell commands. It wastes tokens diagnosing the failure, may produce incorrect results, and can fail entirely on tasks requiring shell execution. This affects ALL Windows users launching Hermes from non-Msys terminals (PowerShell, cmd.exe).

The existing `_msys_to_windows_path()` in `local.py` handles reverse conversion (bash output -> Python `Popen` cwd). No forward helper existed.

This PR adds `_windows_to_msys_path()` to `BaseEnvironment` and applies it in both `init_session()` and `_wrap_command()`.

## Related Issue

Fixes #50594

## Type of Change

- [x] Bug fix (non-breaking change that fixes an issue)

## Changes Made

- `tools/environments/base.py` — Added `_windows_to_msys_path()` static method: `C:\Users\x` -> `/c/Users/x`. No-op on non-Windows or already-Msys paths
- `tools/environments/base.py` — `init_session()`: convert `self.cwd` via `_windows_to_msys_path()` before `shlex.quote()`
- `tools/environments/base.py` — `_wrap_command()`: convert `cwd` via `_windows_to_msys_path()` before `_quote_cwd_for_cd()`
- Added `import re` and `import sys` to module imports

## How to Test

1. Open PowerShell (not Git Bash)
2. `cd C:\Users\YourName`
3. `hermes`
4. Ask agent: list files in current directory
5. Command should succeed instead of failing with exit code 126

## Checklist

### Code

- [x] I have read the Contributing Guide
- [x] My commit messages follow Conventional Commits
- [x] I searched for existing PRs to make sure this is not a duplicate
- [x] My PR contains only changes related to this fix/feature
- [x] pytest 55 passed (1 pre-existing failure on Windows — POSIX root test)
- [ ] I have added tests for my changes
- [x] I have tested on my platform: Windows 10, PowerShell

### Documentation & Housekeeping

- [x] N/A — no docs or config keys affected
- [x] Cross-platform: `_windows_to_msys_path()` is a no-op on non-Windows
