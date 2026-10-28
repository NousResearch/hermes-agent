"""Provider-scoped usage-limit holds (agent/usage_hold.py)."""

import json
import time

from agent import usage_hold


def test_unknown_or_past_reset_never_creates_a_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert usage_hold.record_hold("openai-codex", until=None) is None
    assert usage_hold.record_hold("openai-codex", until=time.time() - 5) is None
    usage_hold.record_from_plan({"eligible": False, "reason": "rate_limit"}, provider="openai-codex")
    usage_hold.record_from_plan({"resume_at": time.time() + 600, "reason": "billing"}, provider="x")
    assert usage_hold.active_hold("openai-codex") is None
    assert usage_hold.active_hold("x") is None


def test_hold_is_bounded_persisted_scoped_and_released(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    usage_hold.record_hold("custom", base_url="http://127.0.0.1:9/v1/", until=now + 10 * 365 * 86400)
    # Survives a restart: a fresh read of the profile's file sees it, clamped to the bound.
    stored = json.loads((tmp_path / "usage_holds.json").read_text())
    hold = stored["custom:http://127.0.0.1:9/v1"]
    assert hold["until"] <= now + usage_hold.MAX_HOLD_SECONDS + 1
    assert usage_hold.active_hold("custom", "http://127.0.0.1:9/v1") is not None
    # Another endpoint / provider is not held.
    assert usage_hold.active_hold("custom", "http://127.0.0.1:10/v1") is None
    assert usage_hold.active_hold("anthropic") is None
    # Exactly one notice per hold.
    assert usage_hold.claim_notice("custom", "http://127.0.0.1:9/v1") is True
    assert usage_hold.claim_notice("custom", "http://127.0.0.1:9/v1") is False
    usage_hold.release_hold("custom", "http://127.0.0.1:9/v1")
    assert usage_hold.active_hold("custom", "http://127.0.0.1:9/v1") is None
