"""Tests for the nested TopicBackfillConfig and its load_gateway_config bridge.

The v1 design assumed ``GatewayConfig.from_dict`` would auto-resolve a nested
``gateway.topic_backfill`` block; it does NOT (from_dict reads the TOP-LEVEL
key). So beyond the dataclass parse/roundtrip checks, the critical test here is
that ``load_gateway_config`` actually bridges a config.yaml
``gateway.topic_backfill`` value onto the loaded GatewayConfig.
"""

from pathlib import Path

import pytest

from gateway.config import GatewayConfig, TopicBackfillConfig, load_gateway_config


def test_defaults():
    cfg = TopicBackfillConfig()
    assert cfg.enabled is True
    assert cfg.max_messages == 15
    assert cfg.max_age_hours == 24


def test_from_dict_empty_returns_defaults():
    cfg = TopicBackfillConfig.from_dict({})
    assert cfg.enabled is True
    assert cfg.max_messages == 15
    assert cfg.max_age_hours == 24


def test_from_dict_parses_nested_values():
    cfg = TopicBackfillConfig.from_dict(
        {"enabled": False, "max_messages": 5, "max_age_hours": 2}
    )
    assert cfg.enabled is False
    assert cfg.max_messages == 5
    assert cfg.max_age_hours == 2


def test_from_dict_coerces_bad_types():
    cfg = TopicBackfillConfig.from_dict(
        {"enabled": "false", "max_messages": "not-a-number", "max_age_hours": None}
    )
    assert cfg.enabled is False
    assert cfg.max_messages == 15  # falls back to default on bad value
    assert cfg.max_age_hours == 24


def test_roundtrip_to_from_dict():
    original = TopicBackfillConfig(enabled=False, max_messages=7, max_age_hours=12)
    restored = TopicBackfillConfig.from_dict(original.to_dict())
    assert restored == original


def test_gateway_config_has_topic_backfill_field_by_default():
    gw = GatewayConfig()
    assert isinstance(gw.topic_backfill, TopicBackfillConfig)
    assert gw.topic_backfill.enabled is True


def test_gateway_config_from_dict_reads_top_level_topic_backfill():
    gw = GatewayConfig.from_dict(
        {"topic_backfill": {"enabled": False, "max_messages": 3}}
    )
    assert gw.topic_backfill.enabled is False
    assert gw.topic_backfill.max_messages == 3


def test_gateway_config_to_dict_includes_topic_backfill():
    gw = GatewayConfig(
        topic_backfill=TopicBackfillConfig(enabled=False, max_messages=9)
    )
    d = gw.to_dict()
    assert d["topic_backfill"]["enabled"] is False
    assert d["topic_backfill"]["max_messages"] == 9


# ---------------------------------------------------------------------------
# The bug v1 missed: the explicit load_gateway_config bridge for the nested
# gateway.topic_backfill block.
# ---------------------------------------------------------------------------


def _write_config_yaml(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_load_gateway_config_bridges_nested_gateway_topic_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``gateway.topic_backfill.*`` in config.yaml must land on the config.

    This is exactly the path ``hermes config set gateway.topic_backfill.*``
    writes, and the v1 bug was that from_dict never read it. The explicit
    bridge in load_gateway_config (mirroring gateway.streaming) fixes it.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config_yaml(
        home,
        "gateway:\n"
        "  topic_backfill:\n"
        "    enabled: false\n"
        "    max_messages: 4\n"
        "    max_age_hours: 6\n",
    )

    gw = load_gateway_config()

    assert gw.topic_backfill.enabled is False
    assert gw.topic_backfill.max_messages == 4
    assert gw.topic_backfill.max_age_hours == 6


def test_load_gateway_config_bridges_top_level_topic_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A top-level ``topic_backfill`` block is also honored."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config_yaml(
        home,
        "topic_backfill:\n"
        "  enabled: true\n"
        "  max_messages: 8\n",
    )

    gw = load_gateway_config()

    assert gw.topic_backfill.enabled is True
    assert gw.topic_backfill.max_messages == 8


def test_load_gateway_config_defaults_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No topic_backfill block -> defaults (enabled, 15, 24)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config_yaml(home, "gateway:\n  strict: false\n")

    gw = load_gateway_config()

    assert gw.topic_backfill.enabled is True
    assert gw.topic_backfill.max_messages == 15
    assert gw.topic_backfill.max_age_hours == 24
