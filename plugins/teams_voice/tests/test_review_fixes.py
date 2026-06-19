"""Tests for the code-review fixes: allowlist, pending TTL, SSRF, group gate."""

from __future__ import annotations

import asyncio
import time

import pytest

from plugins.teams_voice import handlers
from plugins.teams_voice.config import caller_allowed, resolve_config
from plugins.teams_voice.outbound import OutboundError, place_call
from plugins.teams_voice.realtime.openai_client import RealtimeConfig, RealtimeSession


# ── allowlist (AAD-only by default) ──────────────────────────────────────────


def test_allowlist_empty_allows_all():
    cfg = resolve_config(extra={"shared_secret": "s"})
    assert caller_allowed(cfg, "anyone", "Anyone")


def test_allowlist_matches_aad_not_name_by_default():
    cfg = resolve_config(extra={"shared_secret": "s", "allowlist": ["aad-1", "Bob"]})
    assert caller_allowed(cfg, "AAD-1", "Whoever")  # aad match (case-insensitive)
    assert not caller_allowed(cfg, "other", "Bob")  # name match OFF by default


def test_allowlist_name_match_opt_in():
    cfg = resolve_config(extra={"shared_secret": "s", "allowlist": ["bob"], "allowlist_allow_names": True})
    assert caller_allowed(cfg, "other", "Bob")


# ── pending-outbound TTL ─────────────────────────────────────────────────────


def test_pending_set_pop_roundtrip():
    handlers._PENDING_OUTBOUND.clear()
    handlers._pending_set("call-1", "the result")
    assert handlers._pending_pop("call-1") == "the result"
    assert handlers._pending_pop("call-1") is None  # single-use


def test_pending_entry_expires():
    handlers._PENDING_OUTBOUND.clear()
    handlers._PENDING_OUTBOUND["old"] = ("stale", time.monotonic() - 1.0)  # already expired
    assert handlers._pending_pop("old") is None
    assert "old" not in handlers._PENDING_OUTBOUND  # pruned


# ── outbound SSRF guard ──────────────────────────────────────────────────────


def test_place_call_refuses_non_loopback_worker():
    with pytest.raises(OutboundError, match="non-loopback"):
        asyncio.run(
            place_call(
                user_object_id="u", tenant_id="t", shared_secret="s",
                worker_base_url="http://evil.example.com:9440",
            )
        )


def test_place_call_allows_loopback():
    # Loopback passes the guard (then fails on connection, not on the guard).
    with pytest.raises(OutboundError) as ei:
        asyncio.run(
            place_call(
                user_object_id="u", tenant_id="t", shared_secret="s",
                worker_base_url="http://127.0.0.1:9", timeout_s=0.2,
            )
        )
    assert "non-loopback" not in str(ei.value)  # got past the SSRF guard


# ── realtime group-gate auto-response control ────────────────────────────────


def _session():
    sent: list[dict] = []
    s = RealtimeSession(RealtimeConfig(api_key="x"))
    s._closed = False

    async def fake_send(obj):
        sent.append(obj)

    s._send = fake_send  # type: ignore[assignment]
    return s, sent


def test_set_auto_response_toggles_and_is_idempotent():
    s, sent = _session()
    s._auto_response = True
    asyncio.run(s.set_auto_response(False))
    assert sent[-1]["session"]["turn_detection"]["create_response"] is False
    sent.clear()
    asyncio.run(s.set_auto_response(False))  # no-op when unchanged
    assert sent == []


def test_create_response_guarded_on_active():
    s, sent = _session()
    s._response_active = True
    asyncio.run(s.create_response())
    assert sent == []  # nothing created while a response is active
    s._response_active = False
    asyncio.run(s.create_response())
    assert sent[-1]["type"] == "response.create"
