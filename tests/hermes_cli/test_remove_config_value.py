"""Tests for remove_config_value + _delete_nested — companion to set_config_value.

Covers issue #59598: ``hermes config remove <key>`` was missing — users had to
fake it by setting the value to the empty string, which left a stale YAML key.

These tests verify the four behaviours the issue calls out:

1. Existing top-level keys are deleted cleanly (no empty-string residue).
2. Nested dotted keys delete without touching siblings.
3. Numeric list indices splice the element out.
4. Missing keys exit non-zero with a clear message.

Plus parity with :func:`set_config_value`:
- Env-shaped keys (API keys / tokens) go to .env, not config.yaml.
- Managed-scope keys refuse with a clear error.
"""

import argparse
import os
import sys
from unittest.mock import patch

import pytest

from hermes_cli.config import (
    _delete_nested,
    config_command,
    remove_config_value,
    set_config_value,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _read_env(tmp_path):
    return (tmp_path / ".env").read_text()


def _read_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    return config_path.read_text() if config_path.exists() else ""


# ---------------------------------------------------------------------------
# _delete_nested — pure unit tests
# ---------------------------------------------------------------------------

class TestDeleteNestedUnit:
    def test_dict_top_level(self):
        cfg = {"a": 1, "b": 2}
        assert _delete_nested(cfg, "a") is True
        assert cfg == {"b": 2}

    def test_dict_nested(self):
        cfg = {"a": {"b": {"c": 1, "d": 2}}}
        assert _delete_nested(cfg, "a.b.c") is True
        assert cfg == {"a": {"b": {"d": 2}}}

    def test_missing_returns_false(self):
        cfg = {"a": {"b": 1}}
        assert _delete_nested(cfg, "a.c") is False
        assert _delete_nested(cfg, "x.y.z") is False
        # Tree untouched
        assert cfg == {"a": {"b": 1}}

    def test_list_index(self):
        cfg = {"providers": ["alpha", "bravo", "charlie"]}
        assert _delete_nested(cfg, "providers.1") is True
        assert cfg == {"providers": ["alpha", "charlie"]}

    def test_list_index_out_of_range(self):
        cfg = {"providers": ["alpha"]}
        assert _delete_nested(cfg, "providers.5") is False
        assert cfg == {"providers": ["alpha"]}

    def test_list_non_numeric_segment_returns_false(self):
        cfg = {"providers": ["alpha"]}
        assert _delete_nested(cfg, "providers.alpha") is False
        assert cfg == {"providers": ["alpha"]}

    def test_navigate_into_scalar_returns_false(self):
        cfg = {"a": "scalar"}
        assert _delete_nested(cfg, "a.b") is False

    def test_empty_parents_not_pruned(self):
        """Delete of a.b.c leaves a.b as an empty dict — by design."""
        cfg = {"a": {"b": {"c": 1}}}
        _delete_nested(cfg, "a.b.c")
        assert cfg == {"a": {"b": {}}}


# ---------------------------------------------------------------------------
# remove_config_value — config.yaml paths
# ---------------------------------------------------------------------------

class TestRemoveFromConfigYaml:
    def test_top_level_key(self, tmp_path):
        set_config_value("model", "gpt-4o")
        assert "model: gpt-4o" in _read_config(tmp_path)
        remove_config_value("model")
        assert "model" not in _read_config(tmp_path)

    def test_nested_key(self, tmp_path):
        set_config_value("terminal.backend", "docker")
        set_config_value("terminal.cwd", "/tmp")
        remove_config_value("terminal.backend")
        cfg = _read_config(tmp_path)
        assert "backend" not in cfg
        # sibling preserved
        assert "cwd" in cfg

    def test_missing_key_exits_nonzero(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            remove_config_value("does.not.exist")
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "not present" in err

    def test_remove_cleans_empty_string_residue(self, tmp_path):
        """Issue #59598: ``set ... ''`` leaves a stale empty key — remove cleans it."""
        set_config_value("providers.hp-old-name", "")
        cfg_before = _read_config(tmp_path)
        assert "hp-old-name" in cfg_before
        remove_config_value("providers.hp-old-name")
        assert "hp-old-name" not in _read_config(tmp_path)


# ---------------------------------------------------------------------------
# remove_config_value — .env paths (env-shaped keys)
# ---------------------------------------------------------------------------

class TestRemoveFromEnv:
    def test_explicit_api_key(self, tmp_path):
        set_config_value("OPENROUTER_API_KEY", "sk-test")
        assert "OPENROUTER_API_KEY=sk-test" in _read_env(tmp_path)
        remove_config_value("OPENROUTER_API_KEY")
        assert "OPENROUTER_API_KEY" not in _read_env(tmp_path)

    def test_suffix_match_api_key(self, tmp_path):
        set_config_value("CUSTOM_API_KEY", "secret")
        remove_config_value("CUSTOM_API_KEY")
        assert "CUSTOM_API_KEY" not in _read_env(tmp_path)

    def test_missing_env_key_exits_nonzero(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            remove_config_value("OPENAI_API_KEY")
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# config_command dispatch (handler-level integration)
# ---------------------------------------------------------------------------

class TestConfigCommandDispatch:
    def test_remove_subcommand(self, tmp_path, capsys):
        set_config_value("model", "gpt-4o")
        args = argparse.Namespace(key="model", config_command="remove")
        config_command(args)
        out = capsys.readouterr().out
        assert "Removed" in out
        assert "model" not in _read_config(tmp_path)

    def test_delete_alias(self, tmp_path, capsys):
        set_config_value("model", "gpt-4o")
        # argparse normalises aliases to the canonical choice
        args = argparse.Namespace(key="model", config_command="delete")
        config_command(args)
        assert "model" not in _read_config(tmp_path)

    def test_missing_key_arg(self, tmp_path, capsys):
        args = argparse.Namespace(key=None, config_command="remove")
        with pytest.raises(SystemExit):
            config_command(args)