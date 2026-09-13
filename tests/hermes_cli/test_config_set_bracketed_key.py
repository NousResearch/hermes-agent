"""Regression tests for #109611: `hermes config set` with bracket list-indices.

Two defects reported in one call:
  (a) ``fallback_providers[1].provider`` was written as a literal
      ``fallback_providers[1]`` top-level mapping — an address the runtime
      never reads — while the command printed ``✓ Set`` and exited 0.
  (b) The whole-file re-serialization (plain PyYAML dump) deleted trailing
      comment lines from config.yaml.

Covered here: bracket keys canonicalize to numeric segments `_set_nested`
navigates; malformed brackets are a hard error, never a literal write; the
write preserves comments; get/unset canonicalize the same way.
"""

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import (
    _canonicalize_list_index_key,
    get_config_value,
    set_config_value,
    unset_config_value,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _write_config(home, text: str) -> None:
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _read_config(home) -> str:
    path = home / "config.yaml"
    return path.read_text(encoding="utf-8") if path.exists() else ""


LIST_CONFIG = """\
model:
  default: x
fallback_providers:
  - model: a
    provider: deepseek
  - model: b
    provider: ollama-cloud

# \u2500\u2500 Fallback Model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Automatic provider failover when primary is unavailable.
"""


class TestCanonicalizeListIndexKey:
    def test_bracket_index_translated(self):
        assert _canonicalize_list_index_key("a[0].b") == "a.0.b"
        assert _canonicalize_list_index_key("list[12]") == "list.12"
        assert _canonicalize_list_index_key("a[1][2].c") == "a.1.2.c"

    def test_plain_keys_untouched(self):
        assert _canonicalize_list_index_key("model.default") == "model.default"
        assert _canonicalize_list_index_key("a.0.b") == "a.0.b"  # already numeric
        assert _canonicalize_list_index_key("plain") == "plain"

    def test_malformed_brackets_rejected(self, capsys):
        for bad in ("a[x].b", "a[].b", "a[-1].b", "a[0", "a0].b", "list[[0]]"):
            with pytest.raises(SystemExit) as exc:
                _canonicalize_list_index_key(bad)
            assert exc.value.code == 1
            assert "Invalid config key" in capsys.readouterr().err


class TestSetAddressesListElement:
    def test_bracketed_key_writes_the_list_element(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, LIST_CONFIG)
        set_config_value("fallback_providers[1].provider", "openrouter")
        data = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert data["fallback_providers"][1]["provider"] == "openrouter"
        # The dead literal key is NOT written.
        assert "fallback_providers[1]" not in data
        # Sibling element untouched.
        assert data["fallback_providers"][0]["provider"] == "deepseek"

    def test_trailing_comments_survive_the_write(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, LIST_CONFIG)
        set_config_value("fallback_providers[1].provider", "openrouter")
        text = _read_config(_isolated_hermes_home)
        assert "# Automatic provider failover when primary is unavailable." in text, (
            "trailing comment block was deleted by the write (#109611 defect b)")

    def test_numeric_segment_form_still_works(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, LIST_CONFIG)
        set_config_value("fallback_providers.1.model", "c")
        data = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert data["fallback_providers"][1]["model"] == "c"

    def test_out_of_range_index_fails_loudly(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, LIST_CONFIG)
        with pytest.raises(SystemExit):
            set_config_value("fallback_providers[5].provider", "x")


class TestGetUnsetCanonicalization:
    def test_get_reads_the_same_key_set_wrote(self, _isolated_hermes_home, capsys):
        _write_config(_isolated_hermes_home, LIST_CONFIG)
        set_config_value("fallback_providers[1].provider", "openrouter")
        get_config_value("fallback_providers[1].provider")
        assert "openrouter" in capsys.readouterr().out

    def test_unset_removes_the_addressed_leaf(self, _isolated_hermes_home, capsys):
        _write_config(_isolated_hermes_home, LIST_CONFIG)
        unset_config_value("fallback_providers[1].model")
        data = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert "model" not in data["fallback_providers"][1]
        assert data["fallback_providers"][1] == {"provider": "ollama-cloud"}
