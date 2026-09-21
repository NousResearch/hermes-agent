import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_cli.memory_setup as memory_setup
from hermes_cli.memory_setup import _CANCELLED








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


# _write_env_vars's CR/LF-stripping, denylist, and plain-value-roundtrip
# behavior is covered by tests/hermes_cli/test_memory_setup_env_denylist.py,
# which exercises the current save_env_value-routed signature
# (env_writes, hermes_home=None) \u2014 these three tests pinned the prior direct
# Path.write_text(env_path, env_writes) signature/implementation and were
# removed along with it (#60587).


# ---------------------------------------------------------------------------
# provider extras — mode-aware expansion (#70636)
# ---------------------------------------------------------------------------





def test_install_dependencies_prepares_declared_extra_even_if_importable(tmp_path, monkeypatch):
    """PM, not ambient importability, decides whether constraints are current."""
    import hermes_yaml as _yaml

    plugin_dir = tmp_path / "mem0"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        _yaml.safe_dump({"extra": "mem0"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )

    synced = []

    import pm

    monkeypatch.setattr(pm, "available", lambda extra: True)
    monkeypatch.setattr(
        pm, "sync_venv",
        lambda extras=None, explicit=False: synced.append((list(extras or []), explicit)),
    )

    memory_setup._install_dependencies("mem0")

    assert synced == [(["mem0"], True)]


def test_cmd_status_memory_tool_gate_disabled(capsys, monkeypatch):
    """When both memory stores are disabled, Memory status reports memory tool as disabled."""
    _cfg = {"memory": {"memory_enabled": False, "user_profile_enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg)
    # check_memory_requirements() reads the readonly loader, not load_config.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _cfg, raising=False
    )
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert re.search(r"Memory tool:\s+disabled", captured)
    assert re.search(r"Memory injection:\s+disabled", captured)
    assert re.search(r"User profile:\s+disabled", captured)


def test_cmd_status_memory_tool_gate_enabled(capsys, monkeypatch):
    """When at least one memory store is enabled, Memory status reports memory tool as enabled."""
    _cfg = {"memory": {"memory_enabled": True, "user_profile_enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _cfg, raising=False
    )
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert re.search(r"Memory tool:\s+enabled", captured)
    assert re.search(r"Memory injection:\s+enabled", captured)
    assert re.search(r"User profile:\s+disabled", captured)


# ---------------------------------------------------------------------------
# provider answers reach a writer — memory.<name> when the plugin overrides nothing
# ---------------------------------------------------------------------------


def _real_provider(schema, *, own_writer):
    """A provider built on the real ABC, because that is what every bundled plugin is:
    MemoryProvider.save_config exists with an EMPTY body, so a plain hasattr() check cannot
    tell a plugin that persists from one that silently drops."""
    from agent.memory_provider import MemoryProvider

    class _Provider(MemoryProvider):
        name = "test-provider"

        def initialize(self, session_id, **kwargs):
            pass

        def is_available(self):
            return True

        def get_tool_schemas(self):
            return []

        def get_config_schema(self):
            return schema

    if own_writer:
        _Provider.save_config = MagicMock()
    return _Provider()


def _setup_harness(monkeypatch, tmp_path, provider, name, selections, answer, saved):
    """Drive cmd_setup with a scripted picker/prompt, deep-copying what save_config receives so a
    mutation made after the save cannot masquerade as a persisted value."""
    from copy import deepcopy

    picks = iter(selections)
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [(name, "local", provider)])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *a, **k: next(picks))
    monkeypatch.setattr(memory_setup, "_install_dependencies", MagicMock())
    monkeypatch.setattr(memory_setup, "_prompt", lambda *a, **k: answer)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg, **kw: saved.update(deepcopy(cfg)))


def test_setup_persists_answers_when_the_plugin_does_not_override_save_config(tmp_path, monkeypatch):
    """The retaindb/byterover shape: subclasses MemoryProvider but overrides no writer, so the
    inherited no-op would swallow the values. They belong in memory.<name>, which is where those
    providers read them (plugins/memory/retaindb:41, plugins/memory/byterover:41)."""
    provider = _real_provider(
        [
            {"key": "project", "description": "Project identifier", "default": ""},
            {"key": "mode", "description": "Mode", "default": "one", "choices": ["one", "two"]},
        ],
        own_writer=False,
    )
    saved = {}
    _setup_harness(monkeypatch, tmp_path, provider, "brv", [0, 1], "team-alpha", saved)

    memory_setup.cmd_setup(SimpleNamespace())

    assert saved["memory"]["provider"] == "brv"
    assert saved["memory"].get("brv") == {"project": "team-alpha", "mode": "two"}


def test_setup_does_not_give_a_self_writing_provider_a_second_config_home(tmp_path, monkeypatch):
    """A plugin that really overrides save_config (mem0.json, honcho.json, ...) keeps exactly one
    source of truth: its own writer is used and memory.<name> is left alone."""
    provider = _real_provider([{"key": "host", "description": "Host", "default": ""}], own_writer=True)
    saved = {}
    _setup_harness(monkeypatch, tmp_path, provider, "mem", [0], "https://example.invalid", saved)

    memory_setup.cmd_setup(SimpleNamespace())

    type(provider).save_config.assert_called_once()
    assert type(provider).save_config.call_args[0][0] == {"host": "https://example.invalid"}
    assert "mem" not in saved["memory"]
