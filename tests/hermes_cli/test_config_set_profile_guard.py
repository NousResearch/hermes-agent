"""``hermes config set`` / ``unset`` profile-safety guards.

- ``HERMES_PROFILE=<other>`` with no ``-p`` must NOT silently write the resolved profile: refused with
  a hint to use ``hermes -p <name> config set``. A matching or absent value is fine.
- ``mcp_servers.<new>.tools.*`` on a server that has no ``url``/``command`` is refused (it used to
  create an invalid partial server block); ``--force`` still allows it, and defining the route first
  makes the follow-up write succeed.
"""

import pytest
import yaml

from hermes_cli import config as cfg


def _read(tmp_path):
    return yaml.safe_load((tmp_path / "config.yaml").read_text()) or {}


class TestStrayProfileEnv:
    def test_env_naming_other_profile_is_refused(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "work")
        monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
        with pytest.raises(SystemExit):
            cfg.set_config_value("agent.max_turns", "5")
        err = capsys.readouterr().err
        assert "HERMES_PROFILE" in err and "hermes -p work config set" in err
        assert not (tmp_path / "config.yaml").exists() or "agent" not in _read(tmp_path)

    def test_env_matching_resolved_profile_writes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "default")
        monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
        cfg.set_config_value("agent.max_turns", "5")
        assert _read(tmp_path)["agent"]["max_turns"] == 5

    def test_env_unset_writes_as_before(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        cfg.set_config_value("agent.max_turns", "5")
        assert _read(tmp_path)["agent"]["max_turns"] == 5

    def test_unset_is_guarded_too(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        cfg.set_config_value("agent.max_turns", "5")
        monkeypatch.setenv("HERMES_PROFILE", "work")
        monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
        with pytest.raises(SystemExit):
            cfg.unset_config_value("agent.max_turns")
        assert _read(tmp_path)["agent"]["max_turns"] == 5


class TestMcpLeafGuard:
    def test_tools_include_on_unknown_server_is_refused(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        with pytest.raises(SystemExit):
            cfg.set_config_value("mcp_servers.ghost.tools.include", '["FOO"]')
        err = capsys.readouterr().err
        assert "mcp_servers.ghost is not defined" in err
        assert not (tmp_path / "config.yaml").exists() or "ghost" not in (_read(tmp_path).get("mcp_servers") or {})

    def test_defining_url_first_then_tools_write_works(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        cfg.set_config_value("mcp_servers.real.url", "https://mcp.example.test/mcp")
        cfg.set_config_value("mcp_servers.real.tools.include", '["FOO", "BAR"]')
        data = _read(tmp_path)["mcp_servers"]["real"]
        assert data["url"] == "https://mcp.example.test/mcp" and data["tools"]["include"] == ["FOO", "BAR"]

    def test_other_nested_keys_on_unknown_server_only_warn(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        cfg.set_config_value("mcp_servers.demo.env", '{"terminal": "off"}')
        assert _read(tmp_path)["mcp_servers"]["demo"]["env"] == {"terminal": "off"}
        assert "has no url/command yet" in capsys.readouterr().out

    def test_force_writes_partial_block_with_warning(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        cfg.set_config_value("mcp_servers.ghost.tools.include", '["FOO"]', True)
        assert _read(tmp_path)["mcp_servers"]["ghost"]["tools"]["include"] == ["FOO"]
        assert "--force" in capsys.readouterr().out
