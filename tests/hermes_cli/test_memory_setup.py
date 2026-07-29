from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_cli.memory_setup as memory_setup
from hermes_cli.memory_setup import _CANCELLED, _curses_select


def test_curses_select_cancel_defaults_to_selected(monkeypatch):
    captured = {}

    def fake_radiolist(title, items, selected=0, *, cancel_returns=None):
        captured.update({
            "title": title,
            "items": items,
            "selected": selected,
            "cancel_returns": cancel_returns,
        })
        return cancel_returns

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radiolist)

    result = _curses_select("Pick one", [("first", "desc"), ("second", "")], default=1)

    assert result == 1
    assert captured == {
        "title": "Pick one",
        "items": ["first - desc", "second"],
        "selected": 1,
        "cancel_returns": 1,
    }


def test_cmd_setup_top_level_cancel_writes_nothing(monkeypatch):
    save_config = MagicMock()
    load_config = MagicMock(side_effect=AssertionError("cancel should not load config"))

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("fake", "local", object())])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: kwargs["cancel_returns"])
    monkeypatch.setattr("hermes_cli.config.load_config", load_config)
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    load_config.assert_not_called()
    save_config.assert_not_called()


def test_cmd_status_prefers_provider_status_config(monkeypatch, capsys):
    class StatusProvider:
        def get_status_config(self, provider_config):
            assert provider_config["endpoint"] == "http://stale.local"
            return {
                "use_ovcli_config": True,
                "ovcli_config_path": "/tmp/ovcli.conf.VPS_ROOT",
                "endpoint": "https://vps.example",
                "account": "acct",
                "user": "alice",
                "agent": "hermes",
            }

        def is_available(self):
            return True

    config = {
        "memory": {
            "provider": "openviking",
            "openviking": {
                "use_ovcli_config": True,
                "ovcli_config_path": "/tmp/ovcli.conf.VPS_ROOT",
                "endpoint": "http://stale.local",
            },
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("openviking", "API key / local", StatusProvider())])

    memory_setup.cmd_status(SimpleNamespace())

    output = capsys.readouterr().out
    assert "endpoint: https://vps.example" in output
    assert "http://stale.local" not in output


def test_cmd_setup_generic_choice_cancel_writes_nothing(tmp_path, monkeypatch):
    class ChoiceProvider:
        def __init__(self):
            self.save_config = MagicMock()

        def get_config_schema(self):
            return [{
                "key": "mode",
                "description": "Mode",
                "default": "one",
                "choices": ["one", "two"],
            }]

    provider = ChoiceProvider()
    selections = iter([0, _CANCELLED])
    save_config = MagicMock()
    install_dependencies = MagicMock()

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("fake", "local", provider)])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr(memory_setup, "_install_dependencies", install_dependencies)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    install_dependencies.assert_called_once_with("fake")
    save_config.assert_not_called()
    provider.save_config.assert_not_called()
    assert not (tmp_path / ".env").exists()


def test_write_env_vars_strips_line_separators_and_nul(tmp_path):
    """A pasted secret with embedded CR/LF/NUL must not inject an extra
    KEY=VALUE line into .env (mirrors the openviking plugin's writer)."""
    env_path = tmp_path / ".env"

    memory_setup._write_env_vars(
        env_path,
        {"PROVIDER_API_KEY": "good\nINJECTED_KEY=attacker\r\u2028\x00tail"},
    )

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["PROVIDER_API_KEY=goodINJECTED_KEY=attackertail"]
    parsed = dict(line.split("=", 1) for line in lines if "=" in line)
    assert set(parsed) == {"PROVIDER_API_KEY"}


def test_write_env_vars_plain_value_roundtrips(tmp_path):
    env_path = tmp_path / ".env"
    memory_setup._write_env_vars(env_path, {"PROVIDER_API_KEY": "sk-plain-1234"})
    assert env_path.read_text(encoding="utf-8") == "PROVIDER_API_KEY=sk-plain-1234\n"


# ---------------------------------------------------------------------------
# _provider_pip_dependencies — mode-aware dep expansion (#70636)
# ---------------------------------------------------------------------------

def test_provider_pip_dependencies_passthrough_for_non_hindsight():
    deps = memory_setup._provider_pip_dependencies("mem0", ["mem0ai>=2.0.10,<3"])
    assert deps == ["mem0ai>=2.0.10,<3"]


def test_provider_pip_dependencies_legacy_local_alias(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "hindsight").mkdir()
    (tmp_path / "hindsight" / "config.json").write_text(
        json.dumps({"mode": "local"}), encoding="utf-8"
    )
    deps = memory_setup._provider_pip_dependencies("hindsight", ["hindsight-client>=0.6.1"])
    assert "hindsight-all" in deps


def test_install_dependencies_force_reinstalls_versioned_specs(tmp_path, monkeypatch):
    """force=True hands every declared spec (version ranges intact) to pip,
    so a downgraded/stripped bridge package is restored on hermes update."""
    import yaml as _yaml

    plugin_dir = tmp_path / "mem0"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        _yaml.safe_dump({"pip_dependencies": ["mem0ai>=2.0.10,<3"]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )

    installed = []

    def fake_install_specs(specs, timeout=120):
        installed.append(list(specs))
        return SimpleNamespace(ok=True, blocked=False, reason="", stderr="")

    monkeypatch.setattr("tools.lazy_deps.install_specs", fake_install_specs)

    memory_setup._install_dependencies("mem0", force=True)

    assert installed, "force=True must reach the install step"
    assert any("mem0ai>=2.0.10,<3" in specs for specs in installed)
