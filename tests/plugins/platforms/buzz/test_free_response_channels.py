"""Buzz: ``free_response_channels`` exempts listed channels from ``require_mention``.

Parity with Discord's ``DISCORD_FREE_RESPONSE_CHANNELS``: a community keeps the
mention requirement everywhere except a dedicated bot-help channel. The exemption
is evaluated in the adapter gate, before dispatch, so an unaddressed message in a
mention-required channel neither reaches the agent nor earns the "seen" reaction —
the gateway-hook workaround could suppress the reply but not the reaction.
"""
from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.buzz.adapter import _YAML_BRIDGE, BuzzAdapter

FREE = "d2b59f59-de60-40a4-bece-78bdc93972da"
OTHER = "5cc7032e-951f-529b-a028-5c438ad86d2e"


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch, extra: dict | None = None, env: dict | None = None
) -> BuzzAdapter:
    for var in ("BUZZ_REQUIRE_MENTION", "BUZZ_FREE_RESPONSE_CHANNELS", "BUZZ_CHANNELS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BUZZ_RELAY_URL", "https://relay.example.test")
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    cfg = PlatformConfig(enabled=True, token="", extra=extra or {})
    return BuzzAdapter(cfg)


def test_default_requires_mention_in_every_channel(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    assert adapter.require_mention is True
    assert adapter._free_response_channels == set()
    assert adapter._mention_required(FREE) is True
    assert adapter._mention_required(OTHER) is True


def test_extra_list_exempts_only_listed_channels(monkeypatch):
    adapter = _make_adapter(monkeypatch, {"free_response_channels": [FREE]})
    assert adapter._mention_required(FREE) is False
    assert adapter._mention_required(OTHER) is True


def test_env_csv_is_split_and_trimmed(monkeypatch):
    adapter = _make_adapter(monkeypatch, env={"BUZZ_FREE_RESPONSE_CHANNELS": f" {FREE} , {OTHER},"})
    assert adapter._free_response_channels == {FREE, OTHER}
    assert adapter._mention_required(OTHER) is False


def test_env_wins_over_extra_like_channels(monkeypatch):
    adapter = _make_adapter(
        monkeypatch, {"free_response_channels": [OTHER]}, env={"BUZZ_FREE_RESPONSE_CHANNELS": FREE}
    )
    assert adapter._free_response_channels == {FREE}


def test_require_mention_off_makes_every_channel_free(monkeypatch):
    adapter = _make_adapter(monkeypatch, {"require_mention": False, "free_response_channels": [FREE]})
    assert adapter._mention_required(FREE) is False
    assert adapter._mention_required(OTHER) is False


def test_yaml_bridge_maps_the_new_key():
    assert ("free_response_channels", "BUZZ_FREE_RESPONSE_CHANNELS", "csv") in _YAML_BRIDGE
