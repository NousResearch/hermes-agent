"""Tests for CLI/TUI CWD resolution in load_cli_config().

Rules:
- Local backend CLI/TUI: always os.getcwd(), ignoring config and inherited env.
- Non-local with placeholder: pop cwd for backend default.
- Non-local with explicit path: keep as-is.
"""

import os


_CWD_PLACEHOLDERS = (".", "auto", "cwd")


def _resolve_cwd(terminal_config: dict, defaults: dict, env: dict):
    """Mirror the CWD resolution logic from cli.py load_cli_config()."""
    effective_backend = terminal_config.get("env_type", "local")

    if effective_backend == "local":
        terminal_config["cwd"] = "/fake/getcwd"
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        terminal_config.pop("cwd", None)

    # Bridge: TERMINAL_CWD always exported in CLI, skipped in gateway
    _is_gateway = env.get("_HERMES_GATEWAY") == "1"
    if "cwd" in terminal_config:
        if _is_gateway:
            pass  # don't touch env
        else:
            env["TERMINAL_CWD"] = str(terminal_config["cwd"])

    return env.get("TERMINAL_CWD", "")


class TestLocalBackendCli:
    """Local backend always uses os.getcwd()."""

    def test_explicit_config_ignored(self):
        env = {}
        tc = {"cwd": "/explicit/path", "env_type": "local"}
        d = {"terminal": {"cwd": "/explicit/path"}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"

    def test_inherited_env_overwritten(self):
        env = {"TERMINAL_CWD": "/parent/hermes"}
        tc = {"cwd": "/home/user", "env_type": "local"}
        d = {"terminal": {"cwd": "/home/user"}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"

    def test_placeholder_resolved(self):
        env = {}
        tc = {"cwd": "."}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"

    def test_env_and_no_config_file(self):
        env = {"TERMINAL_CWD": "/stale/value"}
        tc = {"cwd": ".", "env_type": "local"}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"


class TestNonLocalBackends:
    """Non-local backends use config or per-backend defaults."""

    def test_placeholder_popped(self):
        env = {}
        tc = {"cwd": ".", "env_type": "docker"}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == ""

    def test_explicit_path_kept(self):
        env = {}
        tc = {"cwd": "/srv/app", "env_type": "ssh"}
        d = {"terminal": {"cwd": "/srv/app"}}
        assert _resolve_cwd(tc, d, env) == "/srv/app"

    def test_auto_placeholder_popped(self):
        env = {}
        tc = {"cwd": "auto", "env_type": "modal"}
        d = {"terminal": {"cwd": "auto"}}
        assert _resolve_cwd(tc, d, env) == ""


class TestGatewayLazyImport:
    """Gateway lazy import of cli.py must not clobber TERMINAL_CWD."""

    def test_gateway_cwd_preserved(self):
        env = {"_HERMES_GATEWAY": "1", "TERMINAL_CWD": "/home/user/project"}
        tc = {"cwd": "/home/user", "env_type": "local"}
        d = {"terminal": {"cwd": "/home/user"}}
        result = _resolve_cwd(tc, d, env)
        assert result == "/home/user/project"

    def test_cli_overwrites_stale_env(self):
        env = {"TERMINAL_CWD": "/stale/from/dotenv"}
        tc = {"cwd": "/home/user", "env_type": "local"}
        d = {"terminal": {"cwd": "/home/user"}}
        result = _resolve_cwd(tc, d, env)
        assert result == "/fake/getcwd"


class TestDeletedCwd:
    """A deleted launch directory must not crash load_cli_config().

    os.getcwd() raises FileNotFoundError when the process's working directory
    is removed out from under it (e.g. a scratch workspace cleaned up
    mid-session). The local-backend cwd bridge must fall back safely instead
    of propagating the crash through CLI/TUI (and gateway lazy-import) startup.
    """

    def test_load_cli_config_falls_back_to_terminal_cwd(self, tmp_path, monkeypatch):
        import cli

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)

        def _boom():
            raise FileNotFoundError("[Errno 2] No such file or directory")

        monkeypatch.setattr(os, "getcwd", _boom)
        monkeypatch.setenv("TERMINAL_CWD", "/srv/fallback")

        cfg = cli.load_cli_config()

        assert cfg["terminal"]["cwd"] == "/srv/fallback"
        assert os.environ["TERMINAL_CWD"] == "/srv/fallback"

    def test_load_cli_config_falls_back_to_home(self, tmp_path, monkeypatch):
        import cli

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)

        def _boom():
            raise FileNotFoundError()

        monkeypatch.setattr(os, "getcwd", _boom)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda p: "/home/me")

        cfg = cli.load_cli_config()

        assert cfg["terminal"]["cwd"] == "/home/me"
