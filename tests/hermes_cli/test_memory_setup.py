import re
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_cli import memory_setup
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
    monkeypatch.setattr(memory_setup, "_get_available_providers", list)

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
    monkeypatch.setattr(memory_setup, "_get_available_providers", list)

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert re.search(r"Memory tool:\s+enabled", captured)
    assert re.search(r"Memory injection:\s+enabled", captured)
    assert re.search(r"User profile:\s+disabled", captured)


# ---------------------------------------------------------------------------
# re-run setup must offer the provider's SAVED config (#123571)
# ---------------------------------------------------------------------------


def test_saved_provider_config_falls_back_to_memory_block_when_reader_raises():
    class Broken:
        def load_saved_config(self):
            raise RuntimeError("boom")

    config = {"memory": {"x": {"a": "1"}}}
    assert memory_setup._saved_provider_config(Broken(), "x", config) == {"a": "1"}


def test_saved_provider_config_reader_is_authoritative():
    """#123571: a provider that overrides ``save_config`` keeps its values outside
    ``memory.<name>``, so its ``load_saved_config`` read must win over the block."""
    class Reader:
        def load_saved_config(self):
            return {"db_path": "/data/facts.db"}

    config = {"memory": {"x": {"a": "1"}}}
    assert memory_setup._saved_provider_config(Reader(), "x", config) == {"db_path": "/data/facts.db"}


def test_saved_provider_config_ignores_non_dict_reader_result():
    class Weird:
        def load_saved_config(self):
            return "not-a-dict"

    config = {"memory": {"x": {"a": "1"}}}
    assert memory_setup._saved_provider_config(Weird(), "x", config) == {"a": "1"}


def test_saved_provider_config_falls_back_without_reader():
    class Plain:
        pass

    config = {"memory": {"x": {"a": "1"}}}
    assert memory_setup._saved_provider_config(Plain(), "x", config) == {"a": "1"}
