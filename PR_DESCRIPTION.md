# feat(windows): native PowerShell support for file ops, process registry, and shell detection

## Summary

This patch adds full native PowerShell support to Hermes Agent on Windows. Previously, all file operations, background process spawning, and shell quoting used Bash-specific commands (`cat`, `sed`, `wc`, `head`, `mkdir -p`, `nohup bash -lc`, `shlex.quote`, `set -o pipefail`, etc.) that fail on PowerShell. This patch detects the configured shell type (`terminal.shell: powershell` in `config.yaml`) and routes all shell-specific operations through PowerShell-compatible alternatives.

## Motivation

Hermes Agent on Windows required Git Bash (MSYS) as the terminal backend. While this works, many Windows users prefer PowerShell 7+ as their primary shell. Without this patch:

- `write_file` fails — uses `cat`, `mkdir -p`, `wc -c`, `mktemp`, `trap`, `mv`
- `read_file` fails — uses `cat`, `head -c`, `sed -n`, `wc -l`
- `patch_replace` fails — uses `cat` for pre-read and post-write verification
- `search` fails — uses `find`, `rg | head`, `set -o pipefail`, `ls -1 | head`
- Background processes fail — `spawn_local` uses `bash -lic "set +m; ..."`, `spawn_via_env` uses `nohup bash -lc`
- `shell_quote()` in the agent's code execution tool produces bash quoting on Windows
- System prompt hardcodes "Shell: on this Windows host your terminal tool runs commands through bash"

## Changes

### New file: `tools/environments/powershell.py`
- `PowerShellEnvironment` class with `shell_type = "powershell"`
- `init_session()` — captures env vars via `Get-ChildItem Env:`, CWD via `Get-Location`
- `_wrap_command()` — PowerShell-native command wrapping (`Set-Location`, `& { ... }`, `$LASTEXITCODE`)
- `_find_powershell()` — locates pwsh 7+ or falls back to Windows PowerShell 5.1
- `_kill_process()` — `proc.terminate()` on Windows (no process groups)

### `tools/environments/base.py`
- Added `shell_type: str = "bash"` class attribute to `BaseEnvironment` as default

### `tools/environments/local.py`
- `_find_shell()` changed from a static alias (`_find_shell = _find_bash`) to a function that returns `pwsh` when `terminal.shell: powershell` is configured, otherwise `bash`
- Added `shell_type = "bash"` to `LocalEnvironment`

### `tools/file_operations.py` (+547/-78 lines)
- `ShellFileOperations.__init__` detects `shell_type` from terminal env, sets `self._is_powershell`
- `_escape_shell_arg` — PowerShell single-quote escaping (`''` doubling) vs bash escaping
- `_has_command` — `Get-Command` on PowerShell, `command -v` on bash
- `_expand_path` — `$env:USERPROFILE` on PowerShell, `$HOME` on bash
- New Python-snippet methods for PowerShell:
  - `_atomic_write_python` — `tempfile.mkstemp` + `os.replace`
  - `_read_file_python` — paginated read with `open()` + `readlines()`
  - `_read_file_raw_python` — full file read
  - `_search_files_python` — `os.walk` + `fnmatch`
  - `_search_content_python` — `re.compile` + file iteration (replaces `rg`/`grep`)
- All existing methods (`write_file`, `read_file`, `read_file_raw`, `patch_replace`, `_check_lint`, `search`, `_suggest_similar_files`, `move_file`, `_python_delete`, `_detect_file_line_ending`, `_file_has_bom`) now branch on `self._is_powershell` and use Python snippets instead of bash commands
- Original bash behavior preserved for all non-PowerShell backends

### `tools/process_registry.py` (+109 lines)
- New `_is_powershell_shell()` — detects PowerShell config
- New `_find_configured_shell()` — returns pwsh or bash path
- `_env_temp_dir()` — accepts Windows paths (`C:\...`), falls back to `tempfile.gettempdir()` on Windows instead of `/tmp`
- `spawn_local()` — uses `pwsh -NoProfile -NoLogo -Command` instead of `bash -lic "set +m; ..."` when PowerShell is configured (both PTY and Popen paths)
- `spawn_via_env()` — uses `Start-Process` + `Set-Content` instead of `nohup bash -lc` for PowerShell backends

### `tools/terminal_tool.py` (+72 lines)
- `_get_terminal_shell_config()` — reads `terminal.shell` from `config.yaml`, returns `"powershell"` or `"bash"`
- `_get_terminal_tool_description()` — returns PowerShell or Bash tool description dynamically
- Fixed `NameError` caused by function definition ordering

### `agent/prompt_builder.py` (+28 lines)
- `_WINDOWS_POWERSHELL_SHELL_HINT` — system prompt hint for PowerShell
- `build_environment_hints()` — dynamically selects between Bash and PowerShell hints based on config

### `tools/transcription_tools.py` (+19 lines)
- `_get_local_command_template()` — uses `subprocess.list2cmdline()` on Windows instead of `shlex.quote`
- `_transcribe_local_command()` — uses `_quote_command_stt_placeholder()` for Windows-compatible quoting

### `tools/code_execution_tool.py` (+5 lines)
- `shell_quote()` — uses PowerShell single-quote escaping (`''` doubling) on Windows instead of `shlex.quote`

## Configuration

Add to `config.yaml`:
```yaml
terminal:
  shell: powershell
```

This activates the PowerShell backend. Without this setting (or with `shell: bash`), all existing behavior is unchanged.

## Testing

- All files compile cleanly (`py_compile` verified)
- Original bash behavior is preserved for all non-PowerShell backends (every change is guarded by `self._is_powershell` or `_is_powershell_shell()`)
- No new dependencies required — Python snippets use stdlib only (`os`, `re`, `fnmatch`, `tempfile`, `shutil`, `pathlib`)

## Backward Compatibility

- **Fully backward compatible.** No changes to behavior when `terminal.shell` is `bash` or unset.
- All new code paths are guarded by shell-type detection.
- `shell_type` defaults to `"bash"` in `BaseEnvironment`, so Docker/SSH/Modal/Daytona/Singularity backends are unaffected.
