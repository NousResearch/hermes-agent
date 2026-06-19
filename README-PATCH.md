# Hermes Agent — Native PowerShell Support for Windows

A patch that adds full native PowerShell 7+ support to [Hermes Agent](https://github.com/NousResearch/hermes-agent) on Windows.

## What This Fixes

Without this patch, Hermes Agent on Windows requires Git Bash (MSYS) as the terminal backend. All file operations, background process spawning, and shell quoting use Bash-specific commands that fail on PowerShell:

- `write_file` — uses `cat`, `mkdir -p`, `wc -c`, `mktemp`, `trap`, `mv`
- `read_file` — uses `cat`, `head -c`, `sed -n`, `wc -l`
- `patch_replace` — uses `cat` for pre-read and post-write verification
- `search` — uses `find`, `rg | head`, `set -o pipefail`, `ls -1 | head`
- Background processes — `spawn_local` uses `bash -lic "set +m; ..."`, `spawn_via_env` uses `nohup bash -lc`
- `shell_quote()` produces bash quoting on Windows
- System prompt hardcodes bash as the shell

## What This Patch Does

Detects the configured shell type (`terminal.shell: powershell` in `config.yaml`) and routes all shell-specific operations through PowerShell-compatible alternatives:

- **File operations** — Uses `python -c` snippets (stdlib only: `os`, `re`, `fnmatch`, `tempfile`, `shutil`, `pathlib`) instead of bash commands
- **Process spawning** — Uses `pwsh -NoProfile -Command` instead of `bash -lic`
- **Background tasks** — Uses `Start-Process` + `Set-Content` instead of `nohup bash -lc`
- **Shell quoting** — PowerShell single-quote escaping (`''` doubling) instead of `shlex.quote`
- **System prompts** — Dynamic shell hints that reflect the actual configured shell
- **Temp directories** — Windows paths (`C:\...`) instead of POSIX-only `/tmp`

## Installation

### Option A: Apply as Git Patch

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
git am 0001-feat-windows-native-PowerShell-support-for-file-ops-.patch
```

### Option B: Clone This Repo

```bash
git clone https://github.com/kosecom/hermes-agent-native-shell.git
cd hermes-agent-native-shell
git checkout feat/windows-powershell-support
```

## Configuration

Add to `config.yaml`:

```yaml
terminal:
  shell: powershell
```

Without this setting (or with `shell: bash`), all existing behavior is unchanged — **fully backward compatible**.

## Files Changed

```
 9 files changed, 990 insertions(+), 78 deletions(-)
 create mode 100644 tools/environments/powershell.py
```

| File | Changes |
|------|---------|
| `tools/environments/powershell.py` | **New** — `PowerShellEnvironment` class with session snapshot, command wrapping, pwsh detection |
| `tools/environments/base.py` | `shell_type = "bash"` default in `BaseEnvironment` |
| `tools/environments/local.py` | `_find_shell()` returns pwsh when PowerShell configured |
| `tools/file_operations.py` | All file ops shell-aware: Python snippets for PowerShell, bash preserved |
| `tools/process_registry.py` | Shell-aware `spawn_local`, `spawn_via_env`, `_env_temp_dir` |
| `tools/terminal_tool.py` | Dynamic tool descriptions, shell config detection, NameError fix |
| `agent/prompt_builder.py` | Dynamic shell hints in system prompt |
| `tools/transcription_tools.py` | Windows-compatible quoting for STT commands |
| `tools/code_execution_tool.py` | `shell_quote()` uses PowerShell escaping on Windows |

## Requirements

- PowerShell 7+ (`pwsh`) recommended, or Windows PowerShell 5.1 (`powershell.exe`) as fallback
- Python 3.8+ (for `python -c` snippets — already required by Hermes)

## Backward Compatibility

- **Fully backward compatible.** No changes to behavior when `terminal.shell` is `bash` or unset.
- All new code paths are guarded by shell-type detection.
- `shell_type` defaults to `"bash"` in `BaseEnvironment`, so Docker/SSH/Modal/Daytona/Singularity backends are unaffected.
- No new dependencies — Python snippets use stdlib only.

## Pull Request

PR opened against upstream: [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent/pull/new/kosecom:feat/windows-powershell-support)

---

Built by a Windows user, for Windows users.
