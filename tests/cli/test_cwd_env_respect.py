"""Tests that load_cli_config() CWD resolution works correctly.

The rule:
- CLI/TUI on local backend: ALWAYS use os.getcwd() (config ignored).
- Gateway (TERMINAL_CWD pre-set to absolute path): respect it.
- Non-local backends with placeholder: pop cwd for backend default.
- Non-local backends with explicit path: keep it.

See issues #19214, #4672, #10225, #10817.
"""

import os
import pytest


# The sentinel values that mean "resolve at runtime"
_CWD_PLACEHOLDERS = (".", "auto", "cwd")


def _resolve_terminal_cwd(terminal_config: dict, defaults: dict, env: dict):
    """Simulate the CWD resolution logic from load_cli_config().

    This mirrors the code in cli.py that handles the CWD resolution
    based on mode (CLI vs gateway) and backend type.
    """
    _existing_cwd = env.get("TERMINAL_CWD", "")
    _is_gateway_import = (
        _existing_cwd
        and _existing_cwd not in _CWD_PLACEHOLDERS
        and os.path.isabs(_existing_cwd)
    )
    effective_backend = terminal_config.get("env_type", "local")

    if _is_gateway_import:
        # Gateway already resolved a real path — keep it.
        terminal_config["cwd"] = _existing_cwd
        defaults["terminal"]["cwd"] = _existing_cwd
    elif effective_backend == "local":
        # CLI/TUI on local backend: always use launch directory.
        terminal_config["cwd"] = "/fake/getcwd"  # stand-in for os.getcwd()
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        # Non-local backend with placeholder — pop for backend default.
        terminal_config.pop("cwd", None)
    # else: non-local backend with explicit path — keep as-is

    # Simulate the bridging loop: write terminal_config["cwd"] to env
    _file_has_terminal = defaults.get("_file_has_terminal", False)
    if "cwd" in terminal_config:
        if _file_has_terminal or "TERMINAL_CWD" not in env:
            env["TERMINAL_CWD"] = str(terminal_config["cwd"])

    return env.get("TERMINAL_CWD", "")


class TestLazyImportGuard:
    """TERMINAL_CWD resolved by gateway must survive a lazy cli.py import."""

    def test_gateway_resolved_cwd_survives(self):
        """Gateway set TERMINAL_CWD → lazy cli import must not clobber."""
        env = {"TERMINAL_CWD": "/home/user/workspace"}
        terminal_config = {"cwd": ".", "env_type": "local"}
        defaults = {"terminal": {"cwd": "."}, "_file_has_terminal": False}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/home/user/workspace"

    def test_gateway_resolved_cwd_survives_with_file_terminal(self):
        """Even when config.yaml has a terminal: section, resolved CWD survives."""
        env = {"TERMINAL_CWD": "/home/user/workspace"}
        terminal_config = {"cwd": ".", "env_type": "local"}
        defaults = {"terminal": {"cwd": "."}, "_file_has_terminal": True}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/home/user/workspace"

    def test_gateway_resolved_cwd_survives_even_with_explicit_config(self):
        """Gateway pre-set TERMINAL_CWD wins even when config has explicit path.

        This is the key scenario: config.yaml has terminal.cwd: /home/user
        (from hermes setup), but the gateway already resolved TERMINAL_CWD.
        The gateway's value must win.
        """
        env = {"TERMINAL_CWD": "/home/user/workspace"}
        terminal_config = {"cwd": "/home/user", "env_type": "local"}
        defaults = {"terminal": {"cwd": "/home/user"}, "_file_has_terminal": True}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/home/user/workspace"


class TestCliAlwaysUsesGetcwd:
    """CLI/TUI on local backend always uses os.getcwd(), ignoring config."""

    def test_explicit_config_cwd_ignored_on_local_cli(self):
        """terminal.cwd: /explicit/path is IGNORED for CLI on local backend.

        This is the #19214 fix — 'hermes setup' may have written an absolute
        path, but CLI always uses os.getcwd() (the user's launch directory).
        """
        env = {}  # No pre-set TERMINAL_CWD = CLI mode
        terminal_config = {"cwd": "/explicit/path", "env_type": "local"}
        defaults = {"terminal": {"cwd": "/explicit/path"}, "_file_has_terminal": True}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/fake/getcwd"  # os.getcwd(), NOT /explicit/path

    def test_dot_cwd_resolves_to_getcwd_when_no_prior(self):
        """With no pre-set TERMINAL_CWD, "." resolves to os.getcwd()."""
        env = {}
        terminal_config = {"cwd": "."}
        defaults = {"terminal": {"cwd": "."}, "_file_has_terminal": False}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/fake/getcwd"

    def test_home_dir_config_ignored_on_local_cli(self):
        """terminal.cwd: ~ (home dir from setup) is ignored for CLI."""
        env = {}
        terminal_config = {"cwd": "/home/daimon", "env_type": "local"}
        defaults = {"terminal": {"cwd": "/home/daimon"}, "_file_has_terminal": True}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/fake/getcwd"


class TestNonLocalBackends:
    """Non-local backends use config or per-backend defaults."""

    def test_remote_backend_pops_placeholder_cwd(self):
        """Remote backend + placeholder cwd → popped for backend default."""
        env = {}
        terminal_config = {"cwd": ".", "env_type": "docker"}
        defaults = {"terminal": {"cwd": "."}, "_file_has_terminal": False}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == ""  # cwd popped, no env var set

    def test_remote_backend_keeps_explicit_path(self):
        """Remote backend + explicit path → kept (e.g. SSH cwd: /srv/app)."""
        env = {}
        terminal_config = {"cwd": "/srv/myproject", "env_type": "ssh"}
        defaults = {"terminal": {"cwd": "/srv/myproject"}, "_file_has_terminal": True}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/srv/myproject"

    def test_remote_backend_with_prior_cwd_preserves(self):
        """Remote backend + pre-resolved TERMINAL_CWD → adopted."""
        env = {"TERMINAL_CWD": "/project"}
        terminal_config = {"cwd": ".", "env_type": "docker"}
        defaults = {"terminal": {"cwd": "."}, "_file_has_terminal": False}

        result = _resolve_terminal_cwd(terminal_config, defaults, env)
        assert result == "/project"
