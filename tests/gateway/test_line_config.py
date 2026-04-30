"""Tests for LineAdapter env-var configuration loading + global integration checks."""
import pytest

from gateway.platforms.line import LineAdapter
from tests.gateway.conftest import make_line_platform_config


def test_init_parses_csv_allowlists(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_ALLOWED_USERS", "U1,U2")
    monkeypatch.setenv("LINE_ALLOWED_GROUPS", "C1")
    monkeypatch.setenv("LINE_ALLOWED_ROOMS", "")
    adapter = LineAdapter(make_line_platform_config(token="t"))
    assert adapter._allowed_users == ["U1", "U2"]
    assert adapter._allowed_groups == ["C1"]
    assert adapter._allowed_rooms == []


def test_init_strips_whitespace(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_ALLOWED_USERS", " U1 , U2 , ")
    adapter = LineAdapter(make_line_platform_config(token="t"))
    assert adapter._allowed_users == ["U1", "U2"]


def test_init_no_secret_yields_outbound_only(monkeypatch):
    """Outbound-only mode: LINE_CHANNEL_ACCESS_TOKEN alone is enough for
    Push API + cron deliveries; webhook receiver returns 401 on every
    inbound (verify_signature rejects when secret is empty)."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
    adapter = LineAdapter(make_line_platform_config(token="t"))
    assert adapter._channel_access_token == "t"
    assert adapter._channel_secret == ""


def test_init_reads_optional_overrides(monkeypatch):
    """All optional env overrides should round-trip through __init__."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_SLOW_RESPONSE_THRESHOLD", "30")
    monkeypatch.setenv("LINE_CACHE_TTL", "7200")
    monkeypatch.setenv("LINE_REQUIRE_MENTION", "true")
    monkeypatch.setenv("LINE_BOT_DISPLAY_NAME", "Samantha")
    monkeypatch.setenv("LINE_FREE_RESPONSE_GROUPS", "Caaa,Cbbb")
    monkeypatch.setenv("LINE_FREE_RESPONSE_ROOMS", "R111")

    adapter = LineAdapter(make_line_platform_config(token="t"))

    assert adapter._slow_response_threshold == 30.0
    assert adapter._cache_ttl == 7200
    assert adapter._require_mention is True
    assert adapter._bot_display_name == "Samantha"
    assert adapter._free_response_groups == ["Caaa", "Cbbb"]
    assert adapter._free_response_rooms == ["R111"]


def test_init_defaults_when_overrides_missing(monkeypatch):
    """__init__ applies sensible defaults when optional overrides are unset."""
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    for key in ("LINE_SLOW_RESPONSE_THRESHOLD", "LINE_CACHE_TTL",
                "LINE_REQUIRE_MENTION", "LINE_BOT_DISPLAY_NAME",
                "LINE_FREE_RESPONSE_GROUPS", "LINE_FREE_RESPONSE_ROOMS"):
        monkeypatch.delenv(key, raising=False)

    adapter = LineAdapter(make_line_platform_config(token="t"))

    assert adapter._slow_response_threshold == 45.0  # documented LINE token margin
    assert adapter._cache_ttl == 3600
    assert adapter._require_mention is False
    assert adapter._bot_display_name == ""
    assert adapter._free_response_groups == []
    assert adapter._free_response_rooms == []
