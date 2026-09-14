"""Regression tests: ``hermes config set`` / ``config unset`` preserve comments and order.

Both commands route through ``hermes_cli/config.py::_write_user_config``, which used to emit
the user's raw config with ``atomic_yaml_write`` (a PyYAML dump). That destroyed every comment,
commented-out example block, key-ordering nuance and quote style in config.yaml on the next
set/unset. ``_write_user_config`` now uses the comment-preserving round-trip writer
(``utils.atomic_roundtrip_yaml_save``). These pin the two behaviours it must keep.
"""

import os
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
        "  skin: default\n",
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


def test_config_set_preserves_comments_and_order(_isolated_hermes_home, commented_config_path):
    set_config_value("display.skin", "mono")

    text = Path(_isolated_hermes_home / "config.yaml").read_text(encoding="utf-8")
    # Comments survive.
    assert "# my hermes config -- do not clobber" in text
    assert "# preferred model" in text
    assert "# provider tuning" in text
    assert "# cap the loop" in text
    # Top-level order is the author's, not alphabetical.
    assert _top_level_order(text) == ["model", "providers", "agent", "display"]
    # The new value is present and parseable in PyYAML.
    assert yaml.safe_load(text)["display"]["skin"] == "mono"


def test_config_unset_removes_key_and_preserves_comments(_isolated_hermes_home, commented_config_path):
    unset_config_value("display.personality")

    text = Path(_isolated_hermes_home / "config.yaml").read_text(encoding="utf-8")
    # Key removed.
    assert "personality" not in yaml.safe_load(text).get("display", {})
    assert "personality" not in text
    # Comments survive.
    assert "# my hermes config -- do not clobber" in text
    assert "# preferred model" in text
    assert "# provider tuning" in text
    assert "# cap the loop" in text
    # Order preserved.
    assert _top_level_order(text) == ["model", "providers", "agent", "display"]
    # Remaining keys intact.
    assert yaml.safe_load(text)["model"]["default"] == "claude-opus-4-7"