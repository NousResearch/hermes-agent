"""Tests for the dialogue-depth pieces: echo guard, vision store, outbound."""

from __future__ import annotations

import hashlib
import hmac

from plugins.teams_voice import hmac_auth, realtime_tools
from plugins.teams_voice.echo_guard import EchoGuard
from plugins.teams_voice.vision_store import StoredFrame, VisionStore


# ── echo guard ───────────────────────────────────────────────────────────────


def test_echo_guard_passes_when_silent():
    g = EchoGuard()
    # nothing played out yet → not speaking → always allow
    assert g.allow_input(0.001, now=1000.0) is True


def test_echo_guard_suppresses_own_playback_before_first_turn():
    g = EchoGuard(tail_window_ms=600, barge_in_rms=0.04)
    g.note_output(1000.0, now=0.0)  # queued 1s of audio at t=0 → playout_end=1000
    # at t=500 we're "speaking"; before the first caller turn, even loud is dropped
    assert g.allow_input(0.9, now=500.0) is False


def test_echo_guard_allows_loud_barge_in_after_first_turn():
    g = EchoGuard(tail_window_ms=600, barge_in_rms=0.04)
    g.mark_caller_turn()
    g.note_output(1000.0, now=0.0)
    assert g.allow_input(0.9, now=500.0) is True  # loud → barge-in
    assert g.allow_input(0.01, now=500.0) is False  # quiet echo → dropped


def test_echo_guard_collapse_stops_guarding():
    g = EchoGuard()
    g.mark_caller_turn()
    g.note_output(1000.0, now=0.0)
    g.collapse(now=500.0)  # barge-in accepted; playback cut
    assert g.allow_input(0.01, now=510.0) is True


def test_echo_guard_disabled_passes_everything():
    g = EchoGuard(enabled=False)
    g.note_output(1000.0, now=0.0)
    assert g.allow_input(0.0, now=100.0) is True


# ── vision store ─────────────────────────────────────────────────────────────


def _frame(source, ts):
    return StoredFrame(source=source, data_base64="AAA", mime="image/jpeg", ts=ts)


def test_vision_store_latest_prefers_screenshare():
    s = VisionStore()
    s.store(_frame("camera", 1))
    s.store(_frame("screenshare", 2))
    assert s.latest().source == "screenshare"
    assert s.latest("camera").source == "camera"


def test_vision_store_history_ring_bounded():
    s = VisionStore(history=3)
    for i in range(5):
        s.store(_frame("screenshare", i))
    hist = s.history(limit=10)
    assert len(hist) == 3  # ring capped
    assert [f.ts for f in hist] == [2, 3, 4]  # oldest-first, newest retained


def test_vision_store_data_url():
    f = _frame("camera", 1)
    assert f.data_url() == "data:image/jpeg;base64,AAA"


# ── outbound signing ─────────────────────────────────────────────────────────


def test_outbound_signature_matches_userobjectid_payload():
    # The outbound endpoint signs "{ts}.{userObjectId}" (not callId) — same helper.
    secret, uid, ts = "s3cret", "user-123", 1700000000000
    sig = hmac_auth.sign(secret, ts, uid)
    expected = hmac.new(
        secret.encode(), f"{ts}.{uid}".encode(), hashlib.sha256
    ).hexdigest()
    assert sig == expected == sig.lower()


# ── tool surface ─────────────────────────────────────────────────────────────


def test_default_tools_present_and_well_formed():
    names = {t["name"] for t in realtime_tools.default_tools()}
    assert names == {
        "hermes_agent_consult",
        "hermes_agent_task",
        "look_at_screen",
        "show_to_caller",
        "call_me_back",
        "post_meeting_minutes",
    }
    for tool in realtime_tools.default_tools():
        assert tool["type"] == "function"
        assert "parameters" in tool and tool["parameters"]["type"] == "object"
