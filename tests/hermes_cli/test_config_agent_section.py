"""A non-mapping ``agent:`` section must not crash config loading (#144067).

``agent: fast`` / ``true`` / ``90`` / ``[max_turns]`` all died inside
``dict(config.get("agent") or {})`` with a raw ValueError/TypeError, taking
down every config-loading command — including ``hermes config show`` and
``hermes status``, the two a user needs to fix it. The malformed section is
now ignored with a warning naming the key, matching how the other sections
(``display``, ``terminal``, ``memory``) tolerate bad spellings.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import hermes_cli.config as config_module
from hermes_cli.config import load_config


@pytest.fixture()
def isolated_home(monkeypatch):
    home = tempfile.mkdtemp()
    monkeypatch.setenv("HERMES_HOME", home)
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    yield home
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()


@pytest.mark.parametrize(
    "agent_spelling",
    ["fast", "true", "90", "[max_turns]"],
    ids=["string", "bool", "int", "list"],
)
def test_non_mapping_agent_section_is_ignored_with_warning(
    isolated_home, agent_spelling, caplog
):
    Path(os.path.join(isolated_home, "config.yaml")).write_text(
        f"agent: {agent_spelling}\n", encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        config = load_config()

    assert isinstance(config.get("agent"), dict) or "agent" not in config
    assert any("agent" in record.message for record in caplog.records)


def test_mapping_agent_section_still_loads(isolated_home):
    Path(os.path.join(isolated_home, "config.yaml")).write_text(
        "agent:\n  max_turns: 7\n", encoding="utf-8"
    )

    config = load_config()

    assert config["agent"]["max_turns"] == 7


def test_absent_agent_section_still_loads(isolated_home):
    Path(os.path.join(isolated_home, "config.yaml")).write_text(
        "display:\n  color: true\n", encoding="utf-8"
    )

    config = load_config()

    assert "max_turns" not in config  # unchanged default path


def test_legacy_root_max_turns_still_relocates(isolated_home):
    Path(os.path.join(isolated_home, "config.yaml")).write_text(
        "max_turns: 5\n", encoding="utf-8"
    )

    config = load_config()

    assert config["agent"]["max_turns"] == 5
