"""A bare `home_channel` chat id is honoured, in both config spellings (issue #33141).

`hermes config set platforms.<name>.home_channel <chat_id>` writes a plain scalar, and the root
`<platform>:` settings block takes one too. The loader only ever read the mapping form, so a bare
value resolved to `None` with no diagnostic: `deliver="<platform>"`, cron `--deliver <platform>`
and the host-wide notices silently kept going to the old destination (or nowhere).

Contracts asserted here (not snapshots of the current code):
- a mapping-shaped home still wins over a bare one wherever they meet;
- a bare id in `platforms.<name>` and in the root `<name>:` block both resolve;
- a bare value never *creates* a home for a platform with no block, and an empty/blank/bool
  value is not a chat id;
- a richer nested mapping is never clobbered by a root-level bare value.

The scalar cases run through the real loader against a throwaway ``HERMES_HOME`` (config
propagation is exactly what unit-testing `from_dict` alone cannot prove).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config

CHAT_ID = "1532136816336044092"  # 19 digits: must survive as a string end to end


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME; returns a writer for its config.yaml."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    def write(config: dict) -> Path:
        path = root / "config.yaml"
        # JSON is valid YAML — keeps the test free of a YAML dependency the repo does not ship.
        path.write_text(json.dumps(config, indent=2))
        return path

    return write


def _home_of(platform: Platform):
    return load_gateway_config().get_home_channel(platform)


# ── unit: PlatformConfig.from_dict ────────────────────────────────────────────


def test_bare_scalar_needs_the_platform_that_only_the_caller_knows():
    """Without a platform there is nothing to build a HomeChannel from — keep returning None."""
    cfg = PlatformConfig.from_dict({"home_channel": CHAT_ID})
    assert cfg.home_channel is None


def test_bare_scalar_is_read_with_the_platform():
    cfg = PlatformConfig.from_dict({"home_channel": CHAT_ID}, platform=Platform.DISCORD)
    assert cfg.home_channel is not None
    assert cfg.home_channel.chat_id == CHAT_ID
    assert cfg.home_channel.platform == Platform.DISCORD


def test_mapping_form_is_unchanged():
    cfg = PlatformConfig.from_dict(
        {"home_channel": {"platform": "discord", "chat_id": CHAT_ID, "name": "ai-agents"}},
        platform=Platform.DISCORD,
    )
    assert (cfg.home_channel.chat_id, cfg.home_channel.name) == (CHAT_ID, "ai-agents")


@pytest.mark.parametrize("value", ["", "   ", None, True, False, [], 0])
def test_non_chat_id_values_are_not_homes(value):
    assert PlatformConfig.from_dict({"home_channel": value}, platform=Platform.DISCORD).home_channel is None


@pytest.mark.parametrize("value", [{}, {"chat_id": "123"}, {"platform": "discord"}])
def test_malformed_mapping_is_ignored_not_fatal(value):
    """A typo in the mapping must not take the whole gateway config down (issue #33141's crash)."""
    assert PlatformConfig.from_dict({"home_channel": value}, platform=Platform.DISCORD).home_channel is None


def test_mapping_wins_over_a_bare_scalar_in_the_same_block():
    cfg = PlatformConfig.from_dict(
        {"home_channel": {"platform": "discord", "chat_id": "111", "name": "named"}},
        platform=Platform.DISCORD,
    )
    assert cfg.home_channel.chat_id == "111"


# ── end to end: the two config spellings the CLI and hand-edits produce ───────


def test_bare_id_under_platforms_resolves(home):
    home({"platforms": {"discord": {"home_channel": CHAT_ID}}})
    resolved = _home_of(Platform.DISCORD)
    assert resolved is not None and resolved.chat_id == CHAT_ID


def test_bare_id_in_the_root_section_resolves(home):
    """`hermes config set discord.home_channel <id>` writes this shape."""
    home({"discord": {"require_mention": True, "home_channel": CHAT_ID}})
    resolved = _home_of(Platform.DISCORD)
    assert resolved is not None and resolved.chat_id == CHAT_ID


def test_root_section_mapping_resolves_too(home):
    home({"discord": {"home_channel": {"platform": "discord", "chat_id": CHAT_ID, "name": "ai-agents"}}})
    resolved = _home_of(Platform.DISCORD)
    assert resolved is not None and resolved.chat_id == CHAT_ID


def test_nested_mapping_is_not_clobbered_by_a_root_scalar(home):
    home(
        {
            "platforms": {"discord": {"home_channel": {"platform": "discord", "chat_id": "999", "name": "nested"}}},
            "discord": {"home_channel": "111"},
        }
    )
    resolved = _home_of(Platform.DISCORD)
    assert resolved is not None and (resolved.chat_id, resolved.name) == ("999", "nested")


def test_a_bare_home_does_not_enable_a_platform(home):
    """A home channel is a destination, never an enablement signal."""
    home({"discord": {"home_channel": CHAT_ID}})
    cfg = load_gateway_config()
    platform = cfg.platforms.get(Platform.DISCORD)
    assert platform is not None and platform.enabled is False


def test_blank_bare_home_is_not_a_home(home):
    home({"platforms": {"telegram": {"home_channel": "   "}}})
    assert _home_of(Platform.TELEGRAM) is None