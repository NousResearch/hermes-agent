## 2024-05-24 - [Sanitize Subprocess Environments for `quick_commands` and `shell.exec`]
**Vulnerability:** The CLI and TUI Gateway executed user-defined `quick_commands` and arbitrary shell commands (`shell.exec`) using `subprocess.run(..., shell=True)` without sanitizing the environment variables passed to the child process.
**Learning:** This exposed sensitive API keys and credentials contained in the main Hermes process environment to these child processes, allowing for easy credential exfiltration by a malicious config or user interaction.
**Prevention:** Always use `tools.environments.local._sanitize_subprocess_env` to filter the environment before passing it to `subprocess` execution mechanisms when executing untrusted or user-supplied shell commands.

## 2025-05-14 - [Sanitize Subprocess Environments in `hermes_cli/tools_config.py`]
**Vulnerability:** `subprocess.run` calls in `hermes_cli/tools_config.py` (specifically in `_pip_install` and `_run_post_setup` hooks) were executing without environment sanitization.
**Learning:** This could leak sensitive Hermes-managed API keys and secrets to external package managers (like `npm` or `pip`) or installation scripts (like the `cua-driver` curl-to-bash script).
**Prevention:** Always apply `_sanitize_subprocess_env` from `tools.environments.local` to the environment dictionary before passing it to `subprocess.run` or `subprocess.Popen`.
