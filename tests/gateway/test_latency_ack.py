"""Tests for complexity-based latency hint helpers."""

import json
import time

from gateway import run


def test_router_payload_maps_reasoning_and_tier_to_latency_classes():
    assert run._latency_class_from_router_payload({"decision": "trivial", "reasoning_effort": "low"}) == "quick"
    assert run._latency_class_from_router_payload({"decision": "simple", "reasoning_effort": "low"}) == "quick"
    assert run._latency_class_from_router_payload({"decision": "normal", "reasoning_effort": "medium"}) == "medium"
    assert run._latency_class_from_router_payload({"decision": "complex", "reasoning_effort": "high"}) == "long"
    assert run._latency_class_from_router_payload({"decision": "flagship", "reasoning_effort": "xhigh"}) == "long"


def test_router_payload_reasoning_can_escalate_unknown_tier():
    assert run._latency_class_from_router_payload({"decision": "custom", "reasoning_config": {"effort": "high"}}) == "long"
    assert run._latency_class_from_router_payload({"decision": "custom", "reasoning_config": {"effort": "medium"}}) == "medium"
    assert run._latency_class_from_router_payload({"decision": "custom", "reasoning_config": {"effort": "low"}}) == "quick"


def test_latency_ack_messages_follow_first_version_rule():
    assert run._latency_ack_message("quick") is None
    assert run._latency_ack_message("unknown") is None
    assert run._latency_ack_message("medium") == "收到，我確認一下，請稍等…"
    assert run._latency_ack_message("long") == "收到，這需要一點時間，我處理完再回覆您…"


def test_read_router_payload_prefers_fresh_per_session_route(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(run, "_hermes_home", tmp_path / "home")
    session_id = "sess-123"
    route_file = tmp_path / f"hermes_route_{session_id}"
    route_file.write_text(json.dumps({"decision": "complex", "reasoning_effort": "high"}))

    payload = run._read_model_router_payload(session_id)

    assert payload["decision"] == "complex"
    assert payload["reasoning_effort"] == "high"


def test_read_router_payload_ignores_cross_session_last_route(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    home = tmp_path / "home"
    state_dir = home / "state.d"
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(run, "_hermes_home", home)
    (state_dir / "last_route.json").write_text(json.dumps({
        "session_id": "other-session",
        "decision": "flagship",
        "reasoning_effort": "xhigh",
    }))

    assert run._read_model_router_payload("sess-123") is None


def test_read_router_payload_ignores_stale_route(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(run, "_hermes_home", tmp_path / "home")
    session_id = "sess-123"
    route_file = tmp_path / f"hermes_route_{session_id}"
    route_file.write_text(json.dumps({"decision": "complex", "reasoning_effort": "high"}))

    old = time.time() - run._ROUTER_ROUTE_MAX_AGE_SECS - 10
    import os
    os.utime(route_file, (old, old))

    assert run._read_model_router_payload(session_id) is None


def test_read_router_payload_requires_current_turn_mtime_when_provided(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(run, "_hermes_home", tmp_path / "home")
    session_id = "sess-123"
    route_file = tmp_path / f"hermes_route_{session_id}"
    route_file.write_text(json.dumps({"decision": "complex", "reasoning_effort": "high"}))

    import os
    previous_turn = time.time() - 5
    os.utime(route_file, (previous_turn, previous_turn))

    assert run._read_model_router_payload(session_id, min_mtime=time.time()) is None
