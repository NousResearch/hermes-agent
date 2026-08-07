"""Per-channel mention-gating config tests for BuzzAdapter.

``require_mention`` is a single global switch per agent. ``mention_channels``
refines it: channel UUIDs listed there require a mention even when
``require_mention`` is False. This enables the "offices + meeting room"
topology — an agent that answers everything in its own channel but only
speaks when addressed in a shared channel.

These tests cover config parsing (extra + env override) and the gating
predicate, without any relay connection.
"""
from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.buzz.adapter import BuzzAdapter

_OFFICE = "9b8e7e29-a385-4180-9acf-14bf7df05fcf"
_SHARED = "f28ceae7-c1c9-4190-b36e-4cd7d5461e55"


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch, extra: dict | None = None
) -> BuzzAdapter:
    monkeypatch.setenv("BUZZ_RELAY_URL", "https://example.communities.buzz.xyz")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "00" * 32)
    monkeypatch.delenv("BUZZ_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("BUZZ_MENTION_CHANNELS", raising=False)
    monkeypatch.delenv("BUZZ_CHANNELS", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra=extra or {})
    return BuzzAdapter(cfg)


def _needs_mention(adapter: BuzzAdapter, channel_id: str) -> bool:
    """Mirror of the inbound gate: global flag OR per-channel override."""
    return adapter.require_mention or (
        channel_id.strip().lower() in adapter.mention_channels
    )


def test_default_no_mention_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, {"require_mention": False})
    assert adapter.mention_channels == set()
    assert not _needs_mention(adapter, _OFFICE)


def test_extra_list_requires_mention_only_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(
        monkeypatch,
        {"require_mention": False, "mention_channels": [_SHARED]},
    )
    assert not _needs_mention(adapter, _OFFICE)
    assert _needs_mention(adapter, _SHARED)


def test_global_flag_still_wins_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(
        monkeypatch,
        {"require_mention": True, "mention_channels": [_SHARED]},
    )
    assert _needs_mention(adapter, _OFFICE)
    assert _needs_mention(adapter, _SHARED)


def test_env_override_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUZZ_MENTION_CHANNELS", f"{_SHARED}, {_OFFICE.upper()}")
    adapter = _make_adapter(monkeypatch, {"require_mention": False})
    # Values are trimmed and lower-cased; empty entries dropped.
    assert adapter.mention_channels == {_SHARED, _OFFICE}


def test_case_and_whitespace_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(
        monkeypatch,
        {"require_mention": False, "mention_channels": [f"  {_SHARED.upper()}  ", ""]},
    )
    assert adapter.mention_channels == {_SHARED}
    assert _needs_mention(adapter, _SHARED.upper())
