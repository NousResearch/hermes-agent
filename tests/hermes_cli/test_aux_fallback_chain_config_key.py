"""`auxiliary.<task>.fallback_chain` must be a registered config key.

The per-task fallback chain is genuinely honoured by ``_try_configured_fallback_chain``
(agent/auxiliary_client.py), but it was absent from DEFAULT_CONFIG, so every write through
``hermes config set`` warned "not a recognized config key -- it was saved anyway, but Hermes
may not read it". The value WAS read; the schema just did not declare it, so operators had no
way to tell a working key from a typo without reading the source.

These tests pin the schema half: the key validates, it survives a round-trip as a real list of
dicts, and a genuine typo in the same position is still caught.
"""

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import _validate_config_key, get_config_value, set_config_value


@pytest.fixture
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so the test never touches real config."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _read_config(home):
    return (home / "config.yaml").read_text(encoding="utf-8")


class TestAuxFallbackChainIsRegistered:
    def test_key_validates_against_schema(self):
        for task in ("vision", "compression", "title_generation", "skills_hub", "mcp"):
            is_known, suggestion = _validate_config_key(f"auxiliary.{task}.fallback_chain")
            assert is_known, f"auxiliary.{task}.fallback_chain rejected (suggestion={suggestion!r})"

    def test_set_writes_without_warning(self, _isolated_hermes_home, capsys):
        set_config_value(
            "auxiliary.vision.fallback_chain",
            '[{"provider": "openrouter", "model": "qwen/qwen3.8-27b:free"}]',
        )

        captured = capsys.readouterr()
        assert "not a recognized config key" not in captured.out
        assert "not a recognized config key" not in captured.err

    def test_round_trips_as_real_list_of_dicts(self, _isolated_hermes_home):
        """A list must not be stored as a quoted string -- the reader does
        ``chain[i] if isinstance(chain, list) else None``, so a string silently disables
        every rung."""
        set_config_value(
            "auxiliary.vision.fallback_chain",
            '[{"provider": "openrouter", "model": "qwen/qwen3.8-27b:free"},'
            ' {"provider": "openrouter", "model": "inclusionai/ling-3.0-flash-vl"}]',
        )

        chain = yaml.safe_load(_read_config(_isolated_hermes_home))["auxiliary"]["vision"]["fallback_chain"]
        assert isinstance(chain, list)
        assert [e["model"] for e in chain] == [
            "qwen/qwen3.8-27b:free",
            "inclusionai/ling-3.0-flash-vl",
        ]
        assert all(e["provider"] == "openrouter" for e in chain)

    def test_get_does_not_flag_it_as_phantom(self, _isolated_hermes_home, capsys):
        """``hermes config get`` has its own phantom-key notice for a nested path under a known
        section that the schema does not define; the same key must be clean on the read path."""
        set_config_value(
            "auxiliary.vision.fallback_chain",
            '[{"provider": "openrouter", "model": "qwen/qwen3.8-27b:free"}]',
        )
        capsys.readouterr()

        from hermes_cli.config import get_config_value

        get_config_value("auxiliary.vision.fallback_chain")

        captured = capsys.readouterr()
        assert "not a recognized config key" not in captured.out
        assert "not a recognized config key" not in captured.err

    def test_real_typo_is_still_caught(self):
        """Registering the key must not open the whole auxiliary block to typos."""
        is_known, suggestion = _validate_config_key("auxiliary.vision.fallback_chan")
        assert not is_known
        assert suggestion and "fallback_chain" in suggestion
