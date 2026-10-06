"""Regression tests for gateway.ready capability negotiation (#130702).

A mobile client could not tell "no recovery guarantee" from "guaranteed
recovery" because the greeting carried no capability advertisement. The
greeting now carries ``capabilities`` + ``protocol``, and ``prompt.submit``
accepts ``client_turn_id`` for idempotency/turn identity.

Covers (as behaviour contracts, not snapshots):
- old greeting payloads still validate (backwards compatibility)
- new greeting payloads validate, on both transports' shapes
- the advertised capabilities distinguish "no recovery" (explicit False)
  from "guaranteed recovery", and the live WS greeting carries them
- prompt.submit accepts client_turn_id on the wire (no 4000), coerces it,
  retains it on the inflight turn / queue envelope, echoes it back, and
  treats a live retry of the same id as the same turn instead of queueing
  a duplicate.
"""

import asyncio
import json
import threading
import types

import pytest
from pydantic import ValidationError

from tui_gateway import server
from tui_gateway import ws as ws_mod
from tui_gateway.contracts import registry as contracts
from tui_gateway.contracts.events import GatewayReadyPayload
from tui_gateway.contracts.prompt_voice import PromptSubmitParams, PromptSubmitResult
from tui_gateway.contracts.sessions import InflightTurn
from tui_gateway.gateway_capabilities import (
    GATEWAY_PROTOCOL_VERSION,
    gateway_capabilities,
    gateway_protocol,
    gateway_ready_payload,
)


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "key-130702",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


# ── contract: backwards compatibility ────────────────────────────────────────


def test_ready_payload_without_new_fields_still_validates():
    """Pre-capability greetings validate with the documented defaults."""
    payload = GatewayReadyPayload.model_validate(
        {"skin": {}, "change_events": True, "replay_epoch": "epoch-1"}
    )
    assert payload.capabilities == {}
    assert payload.protocol is None
    contracts.check_payload(
        "gateway.ready",
        {"skin": {}, "change_events": True, "replay_epoch": "epoch-1"},
    )


def test_ready_payload_with_capabilities_validates():
    payload = GatewayReadyPayload.model_validate(
        {
            "skin": {},
            "change_events": True,
            "replay_epoch": "epoch-1",
            "capabilities": gateway_capabilities(),
            "protocol": gateway_protocol(),
        }
    )
    assert payload.capabilities["turn_recovery"] is False
    contracts.check_payload(
        "gateway.ready",
        {
            "skin": {},
            "change_events": True,
            "replay_epoch": "epoch-1",
            "capabilities": gateway_capabilities(),
            "protocol": gateway_protocol(),
        },
    )


def test_ready_payload_still_requires_its_identity_fields():
    """The new optionals did not loosen the fields clients branch on."""
    with pytest.raises(ValidationError):
        GatewayReadyPayload.model_validate(
            {"skin": {}, "change_events": True, "capabilities": {}}
        )


# ── capabilities: the no-recovery distinction ────────────────────────────────


def test_capabilities_mark_turn_recovery_as_explicitly_unsupported():
    """The whole point of #130702: absent vs False must not be ambiguous."""
    caps = gateway_capabilities()
    assert "turn_recovery" in caps
    assert caps["turn_recovery"] is False
    for name in ("event_replay", "change_events", "turn_lease", "client_turn_id"):
        assert caps[name] is True


def test_protocol_carries_a_version_clients_can_gate_on():
    proto = gateway_protocol()
    assert proto["version"] == GATEWAY_PROTOCOL_VERSION
    assert isinstance(proto["version"], int) and proto["version"] >= 1


def test_ready_helper_stdio_shape_has_no_heartbeat():
    payload = gateway_ready_payload({"name": "default"}, "epoch-7")
    assert "heartbeat" not in payload
    assert payload["change_events"] is True
    assert payload["replay_epoch"] == "epoch-7"
    assert payload["capabilities"]["turn_recovery"] is False
    assert payload["protocol"]["version"] >= 1
    contracts.check_payload("gateway.ready", payload)


def test_ready_helper_ws_shape_carries_heartbeat():
    payload = gateway_ready_payload({"name": "default"}, "epoch-7", heartbeat=True)
    assert payload["heartbeat"] is True
    contracts.check_payload("gateway.ready", payload)


def test_ws_ready_frame_advertises_capabilities(monkeypatch):
    """End to end: the first WS frame carries the negotiation fields."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    sent = []
    inbound = iter(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "ping-1",
                    "method": "gateway.ping",
                    "params": {},
                }
            )
        ]
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            sent.append(json.loads(line))

        async def receive_text(self):
            try:
                return next(inbound)
            except StopIteration:
                raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))

    ready = sent[0]["params"]
    assert ready["type"] == "gateway.ready"
    payload = ready["payload"]
    assert payload["heartbeat"] is True
    assert payload["capabilities"]["turn_recovery"] is False
    assert payload["capabilities"]["client_turn_id"] is True
    assert payload["capabilities"]["event_replay"] is True
    assert payload["protocol"]["version"] >= 1
    GatewayReadyPayload.model_validate(payload)


# ── prompt.submit: client_turn_id on the wire ─────────────────────────────────


def test_prompt_submit_params_accept_client_turn_id():
    contract = contracts.METHODS["prompt.submit"]
    _, problem = contracts.validate_params(
        contract, {"session_id": "sid", "text": "hi", "client_turn_id": "turn-1"}
    )
    assert problem is None
    _, problem = contracts.validate_params(
        contract, {"session_id": "sid", "text": "hi"}
    )
    assert problem is None
    PromptSubmitParams.model_validate(
        {"session_id": "sid", "text": "hi", "client_turn_id": "turn-1"}
    )


def test_coerce_client_turn_id_accepts_and_rejects():
    ok, err = server._coerce_client_turn_id("r", {"session_id": "s"})
    assert (ok, err) == (None, None)
    ok, err = server._coerce_client_turn_id(
        "r", {"session_id": "s", "client_turn_id": "turn-1"}
    )
    assert ok == "turn-1" and err is None
    for bad in ("", "   ", 123, ["turn-1"], "x" * 257):
        _, err = server._coerce_client_turn_id(
            "r", {"session_id": "s", "client_turn_id": bad}
        )
        assert err is not None and err["error"]["code"] == 4004


def test_inflight_turn_carries_client_identity():
    session = _session()
    server._start_inflight_turn(session, "hello", client_turn_id="turn-9")
    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None and snapshot["client_turn_id"] == "turn-9"
    InflightTurn.model_validate(snapshot)


def test_inflight_turn_without_identity_omits_the_key():
    session = _session()
    server._start_inflight_turn(session, "hello")
    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None and "client_turn_id" not in snapshot
    InflightTurn.model_validate(snapshot)


def test_busy_queue_stores_and_echoes_client_turn_id(monkeypatch):
    """A mid-turn submit with an id queues once and the reply echoes the id."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(running=True)
    session["inflight_turn"] = {"user": "live turn", "assistant": "", "streaming": True}

    resp = server._handle_busy_submit(
        "r1", "sid", session, "follow-up", "ws-1", client_turn_id="turn-3"
    )

    assert resp["result"] == {"status": "queued", "client_turn_id": "turn-3"}
    assert session["queued_prompt"]["client_turn_id"] == "turn-3"
    PromptSubmitResult.model_validate(resp["result"])


def test_live_retry_of_same_turn_id_does_not_queue_a_duplicate(monkeypatch):
    """Idempotent retry: the same id while its turn is live answers the live
    status instead of queueing a second turn."""
    monkeypatch.setattr(
        server, "_ensure_active_session_slot", lambda sid, session: None
    )
    session = _session(running=True)
    session["inflight_turn"] = {
        "user": "hello",
        "assistant": "",
        "streaming": True,
        "client_turn_id": "turn-1",
    }
    server._sessions["sid-130702"] = session
    try:
        resp = server._methods["prompt.submit"](
            "r1",
            {"session_id": "sid-130702", "text": "hello", "client_turn_id": "turn-1"},
        )
    finally:
        server._sessions.pop("sid-130702", None)

    assert resp["result"] == {"status": "streaming", "client_turn_id": "turn-1"}
    assert session.get("queued_prompt") is None
    PromptSubmitResult.model_validate(resp["result"])


def test_drained_queued_turn_keeps_client_turn_id(monkeypatch):
    """The queued envelope's identity survives the drain into the next turn."""
    dispatched = []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **kwargs: dispatched.append(
            (rid, sid, text, kwargs)
        )
        or True,
    )
    session = _session()
    server._enqueue_prompt(session, "B", "ws-1", client_turn_id="turn-5")
    session["running"] = False
    assert server._drain_queued_prompt("drain-b", "sid", session) is True
    assert dispatched and dispatched[0][3].get("client_turn_id") == "turn-5"


def test_queued_retry_of_same_turn_id_does_not_merge_a_duplicate(monkeypatch):
    """Idempotent retry in the accept→start window: the same id while its turn
    is still queued answers the queued status instead of text-merging a second
    copy into the envelope (#130947)."""
    monkeypatch.setattr(
        server, "_ensure_active_session_slot", lambda sid, session: None
    )
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(running=True)
    session["inflight_turn"] = {
        "user": "live turn",
        "assistant": "",
        "streaming": True,
        "client_turn_id": "turn-live",
    }
    session["queued_prompt"] = {"text": "hello", "client_turn_id": "turn-1"}
    server._sessions["sid-130947"] = session
    try:
        resp = server._methods["prompt.submit"](
            "r1",
            {"session_id": "sid-130947", "text": "hello", "client_turn_id": "turn-1"},
        )
    finally:
        server._sessions.pop("sid-130947", None)

    assert resp["result"] == {"status": "queued", "client_turn_id": "turn-1"}
    assert session["queued_prompt"]["text"] == "hello", (
        "the retry must not text-merge a duplicate into the envelope")
    PromptSubmitResult.model_validate(resp["result"])
