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


def test_curses_select_accepts_explicit_cancel_value(monkeypatch):
    captured = {}

    def fake_radiolist(title, items, selected=0, *, cancel_returns=None):
        captured["cancel_returns"] = cancel_returns
        return cancel_returns

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radiolist)

    result = _curses_select("Pick one", [("first", "")], default=0, cancel_returns=_CANCELLED)

    assert result == _CANCELLED
    assert captured["cancel_returns"] == _CANCELLED


def test_curses_select_clears_after_picker_returns(monkeypatch):
    events = []

    def fake_radiolist(title, items, selected=0, *, cancel_returns=None):
        events.append("picker")
        return selected

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radiolist)
    monkeypatch.setattr(memory_setup, "_clear_interactive_transition", lambda: events.append("clear"))

    result = _curses_select("Pick one", [("first", "")], default=0)

    assert result == 0
    assert events == ["picker", "clear"]


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


def test_cmd_setup_builtin_selection_still_saves_builtin(monkeypatch):
    save_config = MagicMock()
    config = {"memory": {"provider": "openviking"}}
    providers = [("fake", "local", object())]

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: providers)
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: len(providers))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    assert config["memory"]["provider"] == ""
    save_config.assert_called_once_with(config)


def test_cmd_setup_clears_interactive_picker_before_provider_post_setup(monkeypatch):
    events = []

    class PostSetupProvider:
        def post_setup(self, hermes_home, config):
            events.append("post_setup")

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("openviking", "local", PostSetupProvider())])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: events.append("select") or 0)
    monkeypatch.setattr(memory_setup, "_clear_interactive_transition", lambda: events.append("clear"), raising=False)
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: events.append("install"))
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: "/tmp/hermes-test")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})

    memory_setup.cmd_setup(SimpleNamespace())

    assert events == ["select", "clear", "install", "post_setup"]


def test_cmd_setup_provider_clears_before_provider_post_setup(monkeypatch):
    events = []

    class PostSetupProvider:
        def post_setup(self, hermes_home, config):
            events.append("post_setup")

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("openviking", "local", PostSetupProvider())])
    monkeypatch.setattr(memory_setup, "_clear_interactive_transition", lambda: events.append("clear"), raising=False)
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: events.append("install"))
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: "/tmp/hermes-test")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})

    memory_setup.cmd_setup_provider("openviking")

    assert events == ["clear", "install", "post_setup"]


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


def test_cmd_status_warns_when_inactive_provider_is_configured(monkeypatch, capsys):
    """A provider can have its own config file fully populated (e.g. honcho.json
    with enabled=true) without ever being activated via memory.provider in
    config.yaml. is_available() only checks the provider's own config, so this
    mismatch previously surfaced nowhere — cmd_status should flag it."""

    class ConfiguredButInactiveProvider:
        def is_available(self):
            return True

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [
            ("honcho", "Memory with social cognition", ConfiguredButInactiveProvider())
        ],
    )

    memory_setup.cmd_status(SimpleNamespace())

    output = capsys.readouterr().out
    assert "configured but not active" in output
    assert "memory.provider: honcho" in output


def test_cmd_status_silent_for_inactive_unconfigured_provider(monkeypatch, capsys):
    """The common case — an installed plugin nobody has configured yet —
    should not print the warning; only genuinely-configured-but-inactive
    providers should."""

    class UnconfiguredProvider:
        def is_available(self):
            return False

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("mem0", "Vector memory", UnconfiguredProvider())],
    )

    memory_setup.cmd_status(SimpleNamespace())

    output = capsys.readouterr().out
    assert "configured but not active" not in output


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
