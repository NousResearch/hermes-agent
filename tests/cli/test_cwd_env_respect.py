"""Tests for CLI/TUI CWD resolution in load_cli_config().

Rules:
- Local backend CLI/TUI: always os.getcwd(), ignoring config and inherited env.
- Non-local with placeholder: pop cwd for backend default.
- Non-local with explicit path: keep as-is.
"""


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


# ---------------------------------------------------------------------------
# Kanban worker workspace pin (real load_cli_config, not the mirror above).
#
# The dispatcher pins TERMINAL_CWD to the task workspace in
# hermes_cli/kanban_db.py::_default_spawn (#41312 / #34619). The child CLI's
# config bridge then force-exports the assignee profile's terminal.cwd over
# it, which silently replaces the task-scoped boundary — and with
# docker_mount_cwd_to_workspace that becomes a broad host mount (#73556).
# ---------------------------------------------------------------------------


def _write_terminal_config(hermes_home, cwd, backend="docker"):
    import yaml

    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": backend, "cwd": cwd}}),
        encoding="utf-8",
    )


class TestKanbanWorkerWorkspacePin:
    def test_profile_cwd_does_not_override_task_workspace(self, tmp_path, monkeypatch):
        """A dispatcher-owned worker keeps its task workspace as TERMINAL_CWD."""
        import os

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        workspace = tmp_path / "task-repair"
        workspace.mkdir()
        profile_cwd = tmp_path / "home-example"
        profile_cwd.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_terminal_config(hermes_home, str(profile_cwd))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
        monkeypatch.setenv("TERMINAL_CWD", str(workspace))

        import cli

        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cli.load_cli_config()

        assert os.environ["TERMINAL_CWD"] == str(workspace)

    def test_plain_cli_still_exports_profile_cwd(self, tmp_path, monkeypatch):
        """Without a dispatcher task the profile cwd keeps winning."""
        import os

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_cwd = tmp_path / "home-example"
        profile_cwd.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_terminal_config(hermes_home, str(profile_cwd))
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", "/stale/from/dotenv")

        import cli

        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cli.load_cli_config()

        assert os.environ["TERMINAL_CWD"] == str(profile_cwd)

    def test_unusable_workspace_falls_back_to_profile_cwd(self, tmp_path, monkeypatch):
        """Only a real absolute directory pins — mirrors _default_spawn's own rule."""
        import os

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_cwd = tmp_path / "home-example"
        profile_cwd.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_terminal_config(hermes_home, str(profile_cwd))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "does-not-exist"))
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        import cli

        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cli.load_cli_config()

        assert os.environ["TERMINAL_CWD"] == str(profile_cwd)

    def test_docker_mount_source_is_the_task_workspace(self, tmp_path, monkeypatch):
        """The bind-mount source follows TERMINAL_CWD, so pin it end-to-end.

        ``docker_mount_cwd_to_workspace`` mounts ``os.getenv("TERMINAL_CWD")``
        into the container at /workspace (rw). If the profile cwd wins, the
        worker gets the whole profile directory instead of its task worktree.
        """
        import os

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        workspace = tmp_path / "task-repair"
        workspace.mkdir()
        profile_cwd = tmp_path / "home-example"
        profile_cwd.mkdir()

        import yaml

        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({
                "terminal": {
                    "backend": "docker",
                    "cwd": str(profile_cwd),
                    "docker_mount_cwd_to_workspace": True,
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
        monkeypatch.setenv("TERMINAL_CWD", str(workspace))

        import cli

        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cli.load_cli_config()

        from tools.terminal_tool import _get_env_config

        cfg = _get_env_config()
        assert cfg["host_cwd"] == str(workspace)
        assert cfg["docker_mount_cwd_to_workspace"] is True
