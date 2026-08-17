# Windows-Specific Quirks

Hermes runs natively on Windows (PowerShell, cmd, Windows Terminal, git-bash
mintty, VS Code integrated terminal). Most of it just works, but a handful
of differences between Win32 and POSIX have bitten us — document new ones
here as you hit them so the next person (or the next session) doesn't
rediscover them from scratch.

### Input / Keybindings

**Alt+Enter doesn't insert a newline** — Windows Terminal (and mintty) grab it
for fullscreen before prompt_toolkit sees it. Use **Ctrl+Enter** instead (the
CLI binds it to newline on Windows; raw Ctrl+J does the same, harmlessly).
To inspect how your terminal reports a keystroke, run
`python scripts/keystroke_diagnostic.py` from the repo root.

### Config / Files

**HTTP 400 "No models provided" on first run** — `config.yaml` was saved with
a UTF-8 BOM (Notepad does this). Re-save as UTF-8 without BOM;
`hermes config edit` writes correctly.

### `execute_code` / Sandbox

**WinError 10106** from the sandbox child process — it can't create an
`AF_INET` socket. Root cause is usually Hermes's env scrubber dropping
`SYSTEMROOT`/`WINDIR`/`COMSPEC` (Python's `socket` needs `SYSTEMROOT` to find
`mswsock.dll`), not a broken Winsock LSP. The `_WINDOWS_ESSENTIAL_ENV_VARS`
allowlist in `tools/code_execution_tool.py` covers it; if you still hit it,
echo `os.environ` inside an `execute_code` block to confirm `SYSTEMROOT` is set.

### Testing on Windows

`scripts/run_tests.sh` is POSIX-only (expects `.venv/bin/activate`); the
Hermes-installed `venv/Scripts/` has no pip/pytest (stripped for size).
Install pytest into a system Python and run directly (the repo no longer
uses pytest-xdist; the canonical runner does per-file subprocess isolation,
which the POSIX-only wrapper handles):

```bash
"/c/Program Files/Python311/python" -m pip install --user pytest pyyaml
export PYTHONPATH="$(pwd)"
"/c/Program Files/Python311/python" -m pytest tests/foo/test_bar.py -v --tb=short
```

(POSIX-only tests need skip guards — see the cross-platform guard list in
`references/contributor-guide.md`.)

### Path / Filesystem

**Line endings.** Git may warn `LF will be replaced by CRLF`. Cosmetic — the
repo's `.gitattributes` normalizes. Don't let editors auto-convert committed
POSIX-newline files to CRLF.

**Forward slashes work almost everywhere.** `C:/Users/...` is accepted by
every Hermes tool and most Windows APIs. Prefer forward slashes in code
and logs — avoids shell-escaping backslashes in bash.

### Shell redirection & Windows reserved names

Hermes runs shell commands through the bundled **git-bash / MSYS** (POSIX)
shell on native Windows — NOT cmd.exe. Null-device redirection is
**not portable between shells**:

| Shell | Correct null redirection |
|-------|--------------------------|
| bash / git-bash / MSYS / POSIX | `>/dev/null`, `2>/dev/null` |
| cmd.exe (only inside `.bat`/cmd) | `>nul`, `2>nul` |
| PowerShell | `>$null`, `2>$null` |

**Never emit `2>nul`, `>nul`, or `1>nul` from an agent when running under
bash/git-bash/MSYS.** Bash parses `nul` as an ordinary relative filename and
writes a real file literally named `nul` — a Windows reserved device name.
That debris has broken unattended backups (CopyFileEx →
`ERROR_INVALID_PARAMETER` → modal hang → blocked system suspend). Use
`2>/dev/null` instead. cmd.exe syntax is valid *only* when you are certain
the command executes under cmd.exe; never copy it blindly between shells.

Never intentionally create files or directories named Windows reserved
device names: **NUL, CON, PRN, AUX, COM1–COM9, LPT1–LPT9** — they are
uncreatable/undeletable through normal filesystem APIs and break tools.

