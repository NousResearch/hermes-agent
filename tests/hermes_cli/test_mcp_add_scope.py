"""RED tests for #123621: ``hermes mcp add`` profile-scope disclosure + ``--all-profiles``.

On unfixed main:
- ``cmd_mcp_add`` prints only the home path, never naming the profile scope.
- ``cmd_mcp_add`` ignores ``all_profiles`` (no such parser flag, no fan-out write).
- Bearer tokens stay in the active profile's ``.env`` only.
"""

import argparse
from pathlib import Path

import pytest


@pytest.fixture()
def _isolate_home(tmp_path, monkeypatch):
    """Redirect home I/O to a temp HERMES_HOME with a live sibling profile.

    Pure ``HERMES_HOME``-env isolation (no ``hermes_cli.config`` attribute patches) so the
    per-profile ``HERMES_HOME``-override writes in ``--all-profiles`` hit the real code path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    coder = tmp_path / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "profile.yaml").write_text("display_name: Coder\n", encoding="utf-8")
    return tmp_path


def _make_args(**kwargs):
    defaults = {
        "name": "ink",
        "url": "https://mcp.ml.ink/mcp",
        "mcp_command": None,
        "args": None,
        "auth": None,
        "preset": None,
        "env": None,
        "connect_timeout": None,
        "all_profiles": False,
        "mcp_action": "add",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _mock_probe_ok(monkeypatch):
    def mock_probe(name, config, **kw):
        return [("create_service", "Deploy from repo"), ("list_services", "List all")]

    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", mock_probe)


def _read_servers(home: Path) -> dict:
    import hermes_yaml as yaml

    path = home / "config.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    servers = data.get("mcp_servers")
    return servers if isinstance(servers, dict) else {}


class TestMcpAddScopeDisclosure:
    def test_success_names_active_profile_scope(self, tmp_path, capsys, monkeypatch, _isolate_home):
        """With >1 live profile, success must state the scope it wrote to."""
        _mock_probe_ok(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _: next(_inputs))
        _inputs = iter(["n", ""])

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args())
        out = capsys.readouterr().out
        assert "Saved 'ink'" in out
        assert "Scoped to profile 'default'" in out
        # Default stays single-profile: the sibling must not silently gain the server.
        assert "ink" not in _read_servers(tmp_path / "profiles" / "coder")

    def test_disabled_save_path_names_scope(self, tmp_path, capsys, monkeypatch, _isolate_home):
        """The probe-failure early return must disclose scope too (half-configured risk)."""
        def failing_probe(name, config, **kw):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", failing_probe)
        _inputs = iter(["n", "y"])  # no auth; then save anyway (disabled)
        monkeypatch.setattr("builtins.input", lambda _: next(_inputs))

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args())
        out = capsys.readouterr().out
        assert "disabled" in out
        assert "Scoped to profile 'default'" in out

    def test_single_profile_host_stays_quiet(self, tmp_path, capsys, monkeypatch, _isolate_home):
        """One live profile: no scope noise (nothing to contrast against)."""
        import shutil

        shutil.rmtree(tmp_path / "profiles")
        _mock_probe_ok(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _: next(_inputs))
        _inputs = iter(["n", ""])

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args())
        out = capsys.readouterr().out
        assert "Saved 'ink'" in out
        assert "Scoped to profile" not in out


class TestMcpAddAllProfiles:
    def test_parser_exposes_all_profiles_flag(self):
        """`hermes mcp add --all-profiles` must parse."""
        import argparse as _argparse

        from hermes_cli.subcommands.mcp import build_mcp_parser

        parser = _argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_mcp_parser(subparsers, cmd_mcp=lambda args: args)
        ns = parser.parse_args(["mcp", "add", "ink", "--url", "https://mcp.ml.ink/mcp", "--all-profiles"])
        assert ns.name == "ink"
        assert ns.all_profiles is True

    def test_all_profiles_writes_every_served_profile(
        self, tmp_path, capsys, monkeypatch, _isolate_home
    ):
        """--all-profiles fans the same server entry out to every served profile."""
        _mock_probe_ok(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _: next(_inputs))
        _inputs = iter(["n", ""])

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(all_profiles=True))
        out = capsys.readouterr().out
        assert "ink" in _read_servers(tmp_path)
        assert "ink" in _read_servers(tmp_path / "profiles" / "coder")
        assert "2 profiles" in out
        assert "default" in out and "coder" in out
        # Finding 3: the fan-out names its one-command undo.
        assert "hermes mcp remove ink --all-profiles" in out

    def test_all_profiles_propagates_bearer_token(
        self, tmp_path, capsys, monkeypatch, _isolate_home
    ):
        """A bearer token captured during add must land in every served profile's .env."""
        from hermes_cli.cli_output import prompt as _unused  # noqa: F401 — ensures module path exists

        _mock_probe_ok(monkeypatch)
        monkeypatch.delenv("MCP_INK_API_KEY", raising=False)
        monkeypatch.setattr("hermes_cli.cli_output.prompt", lambda *a, **k: "sekret-token")
        monkeypatch.setattr("builtins.input", lambda _: next(_inputs))
        _inputs = iter(["y", ""])

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(all_profiles=True))
        capsys.readouterr()
        coder_env = tmp_path / "profiles" / "coder" / ".env"
        assert coder_env.exists()
        assert "MCP_INK_API_KEY" in coder_env.read_text(encoding="utf-8")


def _seed_server(home: Path, name: str) -> None:
    """Write a ``config.yaml`` carrying *name* under ``mcp_servers``."""
    import hermes_yaml as yaml

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": {name: {"url": "https://mcp.ml.ink/mcp"}}}),
        encoding="utf-8")


def _provision_home() -> None:
    """Materialize HERMES_HOME's subdirs while the install is still unmanaged.

    A managed install assumes a pre-provisioned home — ``initialize_home`` passes
    ``create=not managed`` — so a fresh tmp home raises ``HomeInitializationError`` on the
    first ``load_config()``. Initialize it unmanaged first, then flip ``HERMES_MANAGED``."""
    from hermes_cli.config import ensure_hermes_home

    ensure_hermes_home()


class TestMcpAllProfilesManaged:
    """A managed install (Nix / package manager) must write NO profile — active or sibling.

    Base behaviour: ``save_config`` refuses the active profile and no sibling was ever touched
    (the fan-out did not exist). The sibling ``atomic_config_write`` path must not resurrect the
    write for the profiles the operator was refused in the active one."""

    def test_add_all_profiles_writes_no_sibling_and_reports_nothing(
        self, tmp_path, capsys, monkeypatch, _isolate_home
    ):
        _provision_home()
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        _mock_probe_ok(monkeypatch)
        _inputs = iter(["n", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(_inputs))

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(all_profiles=True))
        out, err = capsys.readouterr()
        assert "ink" not in _read_servers(tmp_path)
        assert "ink" not in _read_servers(tmp_path / "profiles" / "coder")
        # Finding 2: the reported set must be the written set — nothing saved, nothing claimed.
        assert "Saved 'ink'" not in out
        assert "in 2 profiles" not in out
        assert "managed" in err.lower()

    def test_remove_all_profiles_refuses_and_writes_nothing(
        self, tmp_path, capsys, monkeypatch, _isolate_home
    ):
        _provision_home()
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        _seed_server(tmp_path, "ink")
        _seed_server(tmp_path / "profiles" / "coder", "ink")
        monkeypatch.setattr("builtins.input", lambda _: "y")

        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="ink", all_profiles=True))
        out, err = capsys.readouterr()
        assert "ink" in _read_servers(tmp_path)
        assert "ink" in _read_servers(tmp_path / "profiles" / "coder")
        assert "Removed" not in out
        assert "managed" in err.lower()


class TestMcpRemoveAllProfiles:
    def test_parser_exposes_all_profiles_flag(self):
        """`hermes mcp remove --all-profiles` must parse."""
        import argparse as _argparse

        from hermes_cli.subcommands.mcp import build_mcp_parser

        parser = _argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_mcp_parser(subparsers, cmd_mcp=lambda args: args)
        ns = parser.parse_args(["mcp", "remove", "ink", "--all-profiles"])
        assert ns.name == "ink"
        assert ns.all_profiles is True

    def test_remove_all_profiles_undoes_the_fan_out(
        self, tmp_path, capsys, monkeypatch, _isolate_home
    ):
        """`add --all-profiles` must be undoable in one command."""
        _seed_server(tmp_path, "ink")
        _seed_server(tmp_path / "profiles" / "coder", "ink")
        monkeypatch.setattr("builtins.input", lambda _: "y")

        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="ink", all_profiles=True))
        out = capsys.readouterr().out
        assert "ink" not in _read_servers(tmp_path)
        assert "ink" not in _read_servers(tmp_path / "profiles" / "coder")
        assert "2 profiles" in out
        assert "default" in out and "coder" in out
