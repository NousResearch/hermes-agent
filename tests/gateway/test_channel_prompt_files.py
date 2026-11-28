"""File-backed ``channel_prompts`` entries, end to end through the real gateway config loader.

Regression for #48175: a channel prompt kept in its own markdown file (``{prompt?, file}``) must
reach ``resolve_channel_prompt`` as text, exactly like an inline string does.
"""

import json
import logging

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.config_loader import load_gateway_config_file
from gateway.platforms.base import resolve_channel_prompt


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "channel-prompts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_config(home, cfg):
    (home / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")


@pytest.mark.parametrize("platform", ["discord", "slack", "telegram"])
def test_file_entry_resolves_to_inline_prompt_then_file_text(home, platform):
    (home / "channel-prompts" / "qa.md").write_text("# QA persona\n\nWrite failing tests first.\n", encoding="utf-8")
    _write_config(home, {platform: {"channel_prompts": {
        "100": {"prompt": "You are a Senior QA Engineer.", "file": "channel-prompts/qa.md"},
        "200": {"file": str(home / "channel-prompts" / "qa.md")},
        "300": "Inline prompts are unchanged.",
    }}})

    extra = load_gateway_config().platforms[Platform(platform)].extra

    assert resolve_channel_prompt(extra, "100") == (
        "You are a Senior QA Engineer.\n\n# QA persona\n\nWrite failing tests first.")
    assert resolve_channel_prompt(extra, "200") == "# QA persona\n\nWrite failing tests first."
    assert resolve_channel_prompt(extra, "300") == "Inline prompts are unchanged."
    assert resolve_channel_prompt(extra, "thread-1", parent_id="100").startswith("You are a Senior QA Engineer.")


def test_unreadable_file_keeps_inline_part_and_blank_falls_back_to_parent(home, caplog):
    _write_config(home, {"discord": {"channel_prompts": {
        "parent": "Parent channel prompt.",
        "thread-missing": {"file": "channel-prompts/missing.md"},
        "inline-kept": {"prompt": "Inline part survives.", "file": "channel-prompts/missing.md"},
    }}})

    with caplog.at_level(logging.WARNING, logger="gateway.config"):
        extra = load_gateway_config().platforms[Platform.DISCORD].extra

    assert resolve_channel_prompt(extra, "thread-missing", parent_id="parent") == "Parent channel prompt."
    assert resolve_channel_prompt(extra, "inline-kept") == "Inline part survives."
    assert any("missing.md" in r.getMessage() for r in caplog.records)


def test_gateway_config_file_expands_block_level_entries(home, tmp_path):
    (home / "channel-prompts" / "dev.md").write_text("Ship small, reviewed changes.", encoding="utf-8")
    config_file = tmp_path / "gateway.yaml"
    config_file.write_text(json.dumps({"platforms": {"discord": {"channel_prompts": {
        "100": {"prompt": "You are a Senior Full Stack Engineer.", "file": "channel-prompts/dev.md"},
    }}}}), encoding="utf-8")

    extra = load_gateway_config_file(str(config_file)).platforms[Platform.DISCORD].extra

    assert resolve_channel_prompt(extra, "100") == (
        "You are a Senior Full Stack Engineer.\n\nShip small, reviewed changes.")
