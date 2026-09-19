"""Regression tests: ``hermes config set`` / ``config unset`` preserve comments and order.

Both commands route through ``hermes_cli/config.py::_write_user_config``, which used to emit
the user's raw config with ``atomic_yaml_write`` (a PyYAML dump). That destroyed every comment,
commented-out example block, key-ordering nuance and quote style in config.yaml on the next
set/unset. ``_write_user_config`` now uses the comment-preserving round-trip writer
(``utils.atomic_roundtrip_yaml_save``). These pin the two behaviours it must keep.
Regression for #63039; #50698 reports the same command.
Cases for the commented-out block, YAML 1.1 ambiguous scalars and file mode come from #111241.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import set_config_value, unset_config_value


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so the CLI never touches real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def commented_config_path(_isolated_hermes_home):
    """A config.yaml with a header comment, a commented-out example block, an inline comment,
    and top-level keys in a deliberate (non-alphabetical) author-intended order."""
    path = Path(_isolated_hermes_home) / "config.yaml"
    path.write_text(
        "# my hermes config -- do not clobber\n"
        "model:\n"
        "  # preferred model\n"
        "  default: claude-opus-4-7\n"
        "# provider tuning\n"
        "providers: {}\n"
        "agent:\n"
        "  max_turns: 90  # cap the loop\n"
        "display:\n"
        "  personality: noir\n"
        "  skin: default\n"
        "# A commented-out example block, deliberately left disabled.\n"
        "# fallback_model:\n"
        "#   provider: openrouter\n"
        "#   model: anthropic/claude-sonnet-4\n",
        encoding="utf-8",
    )
    return path


def _top_level_order(text: str):
    """Top-level mapping keys in file order (comments stripped)."""
    return [
        line.split(":", 1)[0].strip()
        for line in text.splitlines()
        if line and not line.startswith(" ") and not line.startswith("#")
    ]


def _comment_lines(text: str):
    """Comment lines in file order, so a dropped or reordered comment fails the test."""
    return [line for line in text.splitlines() if line.strip().startswith("#")]


def test_config_set_preserves_comments_and_order(_isolated_hermes_home, commented_config_path):
    config_path = Path(_isolated_hermes_home) / "config.yaml"
    before = config_path.read_text(encoding="utf-8")
    config_path.chmod(0o600)

    set_config_value("display.skin", "mono")
    set_config_value("approvals.mode", "off")

    text = config_path.read_text(encoding="utf-8")
    # Comment lines survive in file order; a dropped or reordered comment fails here.
    assert _comment_lines(text) == _comment_lines(before)
    # Inline comments on data lines are pinned separately; the full-line filter cannot see them.
    assert "max_turns: 90  # cap the loop" in text
    # Top-level order is the author's, not alphabetical (approvals is a new key, appended last).
    assert _top_level_order(text) == ["model", "providers", "agent", "display", "approvals"]
    # The new value is present and parseable in PyYAML.
    assert yaml.safe_load(text)["display"]["skin"] == "mono"
    # The writer must quote YAML 1.1 ambiguous words (reader is PyYAML/YAML 1.1, writer ruamel/YAML 1.2).
    value = yaml.safe_load(text)["approvals"]["mode"]
    assert value == "off"
    assert isinstance(value, str)
    # File mode survives the write.
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_config_unset_removes_key_and_preserves_comments(_isolated_hermes_home, commented_config_path):
    config_path = Path(_isolated_hermes_home) / "config.yaml"
    before = config_path.read_text(encoding="utf-8")

    unset_config_value("display.personality")

    text = config_path.read_text(encoding="utf-8")
    # Key removed.
    assert "personality" not in yaml.safe_load(text).get("display", {})
    assert "personality" not in text
    # Comment lines survive in file order.
    assert _comment_lines(text) == _comment_lines(before)
    # Inline comments on data lines are pinned separately; the full-line filter cannot see them.
    assert "max_turns: 90  # cap the loop" in text
    # Order preserved.
    assert _top_level_order(text) == ["model", "providers", "agent", "display"]
    # Remaining keys intact.
    assert yaml.safe_load(text)["model"]["default"] == "claude-opus-4-7"
