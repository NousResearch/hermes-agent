import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
    monkeypatch.setattr(memory_setup, "_get_installable_providers", lambda providers: [])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: len(providers))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    assert config["memory"]["provider"] == ""
    save_config.assert_called_once_with(config)


def test_get_installable_providers_filters_catalog_entries_already_installed():
    installed = [("openbrain", "API key / local", object())]

    assert memory_setup._get_installable_providers(installed) == []


def test_cmd_setup_lists_catalogued_provider_when_not_installed(monkeypatch):
    captured = {}
    save_config = MagicMock()

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    def select_cancel(title, items, **kwargs):
        captured["items"] = items
        return kwargs["cancel_returns"]

    monkeypatch.setattr(memory_setup, "_curses_select", select_cancel)
    monkeypatch.setattr("hermes_cli.config.load_config", MagicMock())
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    assert captured["items"][0][0] == "openbrain"
    assert "install standalone plugin" in captured["items"][0][1]
    save_config.assert_not_called()


def test_cmd_setup_installs_catalogued_provider_then_runs_post_setup(monkeypatch):
    events = []

    class PostSetupProvider:
        def post_setup(self, hermes_home, config):
            events.append(("post_setup", hermes_home, config))

    provider = PostSetupProvider()
    calls = iter([[], [("openbrain", "API key / local", provider)]])

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: next(calls))
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: 0)
    monkeypatch.setattr(memory_setup, "_clear_interactive_transition", lambda: events.append("clear"))
    monkeypatch.setattr(
        memory_setup,
        "_install_standalone_provider",
        lambda entry: events.append(("install", entry["name"])) or "openbrain",
    )
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: events.append(("deps", name)))
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: "/tmp/hermes-test")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})

    memory_setup.cmd_setup(SimpleNamespace())

    assert events == [
        "clear",
        ("install", "openbrain"),
        "clear",
        ("deps", "openbrain"),
        ("post_setup", "/tmp/hermes-test", {"memory": {}}),
    ]


def test_catalog_install_refreshes_real_discovery_then_runs_post_setup(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")

    hermes_home = tmp_path / "hermes-home"
    repo = tmp_path / "fixture-memory-provider"
    repo.mkdir()
    (repo / "plugin.yaml").write_text(
        "name: fixture_memory\n"
        "manifest_version: 1\n"
        "description: Filesystem-backed memory fixture\n",
        encoding="utf-8",
    )
    (repo / "__init__.py").write_text(
        "from pathlib import Path\n\n"
        "class FixtureMemoryProvider:\n"
        "    def is_available(self):\n"
        "        return True\n\n"
        "    def get_config_schema(self):\n"
        "        return []\n\n"
        "    def post_setup(self, hermes_home, config):\n"
        "        Path(hermes_home, 'post-setup-ran').write_text('ok')\n\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(FixtureMemoryProvider())\n",
        encoding="utf-8",
    )

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hermes test",
        "GIT_AUTHOR_EMAIL": "hermes-test@example.invalid",
        "GIT_COMMITTER_NAME": "Hermes test",
        "GIT_COMMITTER_EMAIL": "hermes-test@example.invalid",
    }
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=git_env)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=git_env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repo,
        check=True,
        env=git_env,
    )

    from hermes_cli import memory_provider_catalog

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        memory_provider_catalog,
        "INSTALLABLE_MEMORY_PROVIDERS",
        [{
            "name": "fixture_memory",
            "label": "fixture_memory",
            "setup_hint": "install test plugin",
            "identifier": repo.as_uri(),
            "description": "Filesystem-backed memory fixture",
        }],
    )

    module_prefix = "_hermes_user_memory.fixture_memory"
    try:
        memory_setup.cmd_setup_provider("fixture_memory")

        assert (hermes_home / "plugins" / "fixture_memory" / "__init__.py").is_file()
        assert (hermes_home / "post-setup-ran").read_text() == "ok"
    finally:
        for module_name in list(sys.modules):
            if module_name == module_prefix or module_name.startswith(f"{module_prefix}."):
                sys.modules.pop(module_name, None)


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
