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


def test_schema_prompt_accepting_structured_defaults_keeps_their_type(monkeypatch):
    schema = [
        {"key": "sync_roles", "default": ["user"]},
        {"key": "reflect", "default": {"disabled_for_cron": True, "max_calls_per_session": 3}},
        {"key": "mode", "default": "auto"},
    ]
    monkeypatch.setattr(memory_setup, "_prompt", lambda label, default=None, secret=False: default or "")
    config = {}

    assert memory_setup._prompt_schema_fields("mnemosyne", schema, config, {})

    assert config == {field["key"]: field["default"] for field in schema}


def test_schema_prompt_edited_structured_fields_parse_or_are_rejected(monkeypatch):
    schema = [
        {"key": "sync_roles", "default": ["user", "assistant"]},
        {"key": "bad_roles", "default": ["user"]},
    ]
    answers = {"sync_roles": "[]", "bad_roles": "user,assistant"}
    monkeypatch.setattr(memory_setup, "_prompt", lambda label, default=None, secret=False: answers[label])
    config = {"bad_roles": ["assistant"]}

    assert memory_setup._prompt_schema_fields("mnemosyne", schema, config, {})

    assert config == {"sync_roles": [], "bad_roles": ["assistant"]}


def test_schema_prompt_edits_follow_structured_schema_default_over_saved_string(monkeypatch, capsys):
    schema = [
        {"key": "sync_roles", "default": ["user"]},
        {"key": "kept_roles", "default": ["user"]},
        {"key": "broken_roles", "default": ["user"]},
    ]
    answers = {"sync_roles": '["user"]', "kept_roles": "user,assistant", "broken_roles": "[broken"}
    monkeypatch.setattr(memory_setup, "_prompt", lambda label, default=None, secret=False: answers[label])
    config = {"sync_roles": "user,assistant", "kept_roles": "user,assistant", "broken_roles": "user,assistant"}

    assert memory_setup._prompt_schema_fields("mnemosyne", schema, config, {})

    # Edited JSON parses to the schema's list shape; accepting the saved CSV keeps it;
    # malformed input is rejected and the saved value survives.
    assert config == {"sync_roles": ["user"], "kept_roles": "user,assistant", "broken_roles": "user,assistant"}
    assert 'Ignoring broken_roles: expected a list such as ["user"]' in capsys.readouterr().out
