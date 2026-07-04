"""Tests for AIRIES subscription and web fetch."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def aries_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_subscription_usage_limits(aries_env):
    from hermes_cli.airies_subscription import AriesSubscriptionManager

    mgr = AriesSubscriptionManager(aries_env / "sub.db")
    for _ in range(50):
        mgr.record_event("turn")
    ok, msg = mgr.check_turn_allowed()
    assert not ok
    assert "limit" in msg.lower()


def test_subscription_snapshot(aries_env):
    from hermes_cli.airies_subscription import AriesSubscriptionManager

    mgr = AriesSubscriptionManager(aries_env / "sub.db")
    snap = mgr.get_snapshot()
    assert snap.tier == "free"
    assert snap.turns_limit == 50


def test_airies_fetch_text_extractor():
    from plugins.web.airies_fetch.provider import _TextExtractor

    html = "<html><body><script>ignore()</script><p>Hello AIRIES</p></body></html>"
    parser = _TextExtractor()
    parser.feed(html)
    assert "Hello AIRIES" in parser.text()
    assert "ignore" not in parser.text()


def test_airies_defaults():
    from hermes_cli.config import DEFAULT_CONFIG, cfg_get

    assert cfg_get(DEFAULT_CONFIG, "agent", "product") == "airies-agent"
    assert cfg_get(DEFAULT_CONFIG, "display", "skin") == "airies"
    assert cfg_get(DEFAULT_CONFIG, "web", "backend") == "airies_fetch"
    assert cfg_get(DEFAULT_CONFIG, "subscription", "enabled") is True


def test_airies_skin_branding():
    from hermes_cli.skin_engine import load_skin

    skin = load_skin("airies")
    assert skin.get_branding("agent_name") == "AIRIES Agent"
