## 2024-05-24 - Security Enhancement: YAML Parsing & Subprocess Execution
**Vulnerability:** Use of `subprocess.run(shell=True)` in local STT command execution.
**Learning:** `shell=True` can enable shell injection and should be replaced with argv invocation and `shell=False`. The YAML path already used `CSafeLoader`/`SafeLoader`; updates there are about preserving safe semantics and performance clarity, not fixing unsafe deserialization.
**Prevention:** Avoid `shell=True` in `subprocess` unless absolutely necessary, and prefer passing commands as argument lists. Keep YAML parsing on safe loaders, preferring `CSafeLoader` when available.

## 2024-05-24 - [Sanitize Subprocess Environments for `quick_commands` and `shell.exec`]
**Vulnerability:** The CLI and TUI Gateway executed user-defined `quick_commands` and arbitrary shell commands (`shell.exec`) using `subprocess.run(..., shell=True)` without sanitizing the environment variables passed to the child process.
**Learning:** This exposed sensitive API keys and credentials contained in the main Hermes process environment to these child processes, allowing for easy credential exfiltration by a malicious config or user interaction.
**Prevention:** Always use `tools.environments.local._sanitize_subprocess_env` to filter the environment before passing it to `subprocess` execution mechanisms when executing untrusted or user-supplied shell commands.
