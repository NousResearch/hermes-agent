"""Live-shape peer provider tests (CR-603 regression).

The real ``hermes_peer.tools.peer_read_inbox`` returns JSON
``{"messages": [...]}``. The provider must parse that shape — this test pins
the parse against the actual tool contract and exercises the full
provider → router → executor lifecycle with the real hermes_peer modules
patched at the module boundary (manager is process-global; only the inbox
tool and manager presence are faked).
"""
from __future__ import annotations

import json

import pytest

from control_room.actions import ControlRoomActionRouter
from control_room.contract import ActionTarget, ControlRoomAction
from control_room.executors import default_executors, peer_inbox_action_executor
from control_room.service import _peer_provider


@pytest.fixture(autouse=True)
def _patch_peer_modules(monkeypatch):
    """Patch the real hermes_peer modules so the provider's local imports
    resolve to controllable fakes. Kept faithful to the tool contract:
    peer_read_inbox returns JSON with a ``messages`` list of dicts."""
    import hermes_peer.plugin as hp
    import hermes_peer.tools as ht

    inbox_rows = [
        {
            "message_id": "m-abc123",
            "peer_id": "p-xyz",
            "content": "hello from remii",
            "state": "held",
            "from": "remii",
            "created_at": "2026-08-12T18:00:00Z",
        },
        {
            "message_id": "m-def456",
            "peer_id": "p-qqq",
            "content": "status request",
            "state": "queued",
            "from": "wesker",
            "created_at": "2026-08-12T18:01:00Z",
        },
    ]

    def fake_read_inbox(args):
        return json.dumps({"messages": inbox_rows})

    monkeypatch.setattr(ht, "peer_read_inbox", fake_read_inbox)
    monkeypatch.setattr(hp, "get_manager", lambda: object())
    yield inbox_rows


class TestPeerProviderJsonParse:
    def test_parses_real_json_message_shape(self):
        rows = _peer_provider({})
        assert len(rows) == 2
        first = rows[0]
        assert first["id"] == "m-abc123"
        assert first["title"] == "hello from remii"
        assert first["state"] == "held"
        assert first["sender"] == "remii"
        assert first["kind"] == "peer_message"

    def test_queued_state_preserved(self):
        rows = _peer_provider({})
        assert rows[1]["state"] == "queued"
        assert rows[1]["sender"] == "wesker"

    def test_malformed_tool_output_degrades_empty_not_crash(self, monkeypatch):
        import hermes_peer.tools as ht

        monkeypatch.setattr(ht, "peer_read_inbox", lambda args: "{not json")
        assert _peer_provider({}) == []


class TestPeerLifecycleThroughRouter:
    """Message receipt → inbox action → lifecycle, visible via Control Room
    primitives: the provider lists held messages; the router's peer_inbox
    executor acts on them through the same public tool surface."""

    def test_router_inbox_action_uses_public_tool(self):
        router = ControlRoomActionRouter(
            executors={"peer_inbox": peer_inbox_action_executor()},
            scope_profile="default",
        )
        # The executor calls hermes_peer.tools.peer_read_inbox with an action
        # param — patch it to record the call, proving the router path reaches
        # the public API.
        import hermes_peer.tools as ht

        called = {}

        def fake_inbox_action(args):
            called.update(args)
            return json.dumps({"released": True, "message_id": args.get("message_id")})

        ht.peer_read_inbox = fake_inbox_action

        result = router.dispatch(
            ControlRoomAction(
                id="act-1",
                target=ActionTarget(kind="peer_inbox", id="m-abc123"),
                parameters={"action": "release"},
                confirmation="required",
            ),
            confirmed=True,
        )
        assert result.status == "completed"
        assert called.get("action") == "release"
        assert called.get("message_id") == "m-abc123"

    def test_default_executors_include_peer_inbox(self):
        executors = default_executors()
        assert "peer_inbox" in executors
