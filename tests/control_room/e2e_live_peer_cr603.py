"""CR-603 live peer lifecycle E2E (manager path).

Two real PeerSessionManager instances (the Hermes Peer plugin manager) over
real AF_UNIX sockets and the shared runtime root. alpha sends to bravo;
bravo's policy=hold persists the message to its real MessageStore. Control
Room then attaches to bravo's manager and reads the held message through the
REAL public tools (peer_read_inbox), then releases it through the router.

This proves: message receipt visible through Control Room, inbox action
through the router, and lifecycle (held -> released) through the real store.

Run: python tests/control_room/e2e_live_peer_cr603.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WALKIE = Path("/home/kensei/repos/hermes-walkie-talkie")
sys.path.insert(0, str(WALKIE))
sys.path.insert(0, str(REPO))

from agent_peer.identity import generate_instance_id, generate_peer_id  # noqa: E402
from agent_peer.models import PeerRecord  # noqa: E402
from agent_peer.policy import PolicyEngine  # noqa: E402
from agent_peer.runtime import PeerRuntimeManager  # noqa: E402
from hermes_peer.config import PeerConfig  # noqa: E402
from hermes_peer.sessions import PeerSessionManager  # noqa: E402


class FakeCtx:
    """Minimal public PluginContext stand-in (register_hook / inject_message)."""

    def __init__(self):
        self.hooks: dict[str, list] = {}
        self.injected: list[tuple] = []

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)

    def register_tool(self, *a, **kw):
        pass

    def register_command(self, *a, **kw):
        pass

    def inject_message(self, content, role="user", *, mode="queue", target_session=None):
        self.injected.append((content, role, mode, target_session))
        return True


def wait_for(predicate, timeout: float = 15.0, interval: float = 0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="cr603-")
    runtime = Path(tmp) / "runtime"
    state = Path(tmp) / "state"
    runtime.mkdir(mode=0o700)
    state.mkdir(mode=0o700)
    os.environ["XDG_RUNTIME_DIR"] = str(Path(tmp) / "xdg-runtime")
    os.environ["XDG_STATE_HOME"] = str(state)

    # alpha: accepts (forwards to harness). bravo: holds (persists in store).
    ctx_a, ctx_b = FakeCtx(), FakeCtx()
    mgr_a = PeerSessionManager(ctx_a, runtime_root=runtime, config=PeerConfig(inbound="accept"))
    mgr_b = PeerSessionManager(ctx_b, runtime_root=runtime, config=PeerConfig(inbound="hold"))

    try:
        # Register sessions exactly as the plugin does via the public
        # lifecycle hook: on_session_start registers a peer and wires the
        # policy-driven inbound pipeline.
        mgr_a.on_session_start("session-alpha", platform="cli", profile="cr603")
        mgr_b.on_session_start("session-bravo", platform="cli", profile="cr603")
        a_rec = mgr_a.resolve_peer(mgr_a.peer_id_for_session("session-alpha"))
        b_rec = mgr_b.resolve_peer(mgr_b.peer_id_for_session("session-bravo"))
        assert a_rec and b_rec, "session peers not registered"
        # Ensure both are discoverable before sending.
        assert wait_for(lambda: len(mgr_a.list_peers()) >= 2), "peers not discoverable"

        # alpha -> bravo over the real socket.
        env = mgr_a._make_envelope(recipient=b_rec.peer_id, content="hello from alpha (live CR-603)")
        receipt = mgr_a._runtime.send(env)
        print(f"[live] send receipt: {receipt.state.value} {receipt.message_id}")
        assert receipt.state.value == "held", receipt

        # The held message must persist in bravo's REAL store.
        assert wait_for(lambda: len(mgr_b.read_inbox()) >= 1), "held message not in bravo store"
        held = mgr_b.read_inbox()[0]
        print(f"[live] bravo store sees: {held.get('message_id')} state={held.get('state')} from={held.get('from') or held.get('sender_peer_id')}")

        # Control Room attaches to bravo's manager (the session owning the
        # peer), routes the real public tools to it, and reads the inbox.
        import hermes_peer.plugin as hp
        import hermes_peer.tools as peer_tools

        hp._manager = mgr_b

        raw = peer_tools.peer_read_inbox({"limit": 10})
        payload = json.loads(raw) if isinstance(raw, str) else raw
        assert payload.get("messages"), f"Control Room saw empty inbox: {raw}"
        print(f"[control-room] inbox visible: {payload['messages']}")

        # Control Room provider parses the real JSON shape.
        from control_room.service import _peer_provider

        rows = _peer_provider({})
        assert rows, "provider saw nothing"
        print(f"[control-room] provider rows: {rows}")
        assert rows[0]["state"] in ("held", "queued")
        assert rows[0]["id"] == held.get("message_id")[:24]

        # Router releases through the real tool.
        from control_room.actions import ControlRoomActionRouter
        from control_room.contract import ActionTarget, ControlRoomAction
        from control_room.executors import peer_inbox_action_executor

        router = ControlRoomActionRouter(
            executors={"peer_inbox": peer_inbox_action_executor()}, scope_profile="default",
        )
        result = router.dispatch(
            ControlRoomAction(
                id="cr603-act-1",
                target=ActionTarget(kind="peer_inbox", id=held["message_id"]),
                parameters={"action": "release"},
                confirmation="required",
            ),
            confirmed=True,
        )
        print(f"[control-room] release: {result.status} {result.message}")
        assert result.status == "completed", result

        # Lifecycle: release transitions held -> queued (delivered to host) and
        # the host receives the peer-wrapped content through the public seam.
        assert wait_for(lambda: any(m["state"] == "queued" for m in mgr_b.read_inbox())), "message not queued after release"
        raw2 = peer_tools.peer_read_inbox({"limit": 10})
        payload2 = json.loads(raw2) if isinstance(raw2, str) else raw2
        released = [m for m in payload2.get("messages") or [] if m.get("message_id") == held["message_id"]]
        assert released, "released message vanished"
        assert released[0]["state"] == "queued", released[0]
        assert any("peer_message" in str(i[0]) for i in ctx_b.injected), ctx_b.injected
        print(f"[control-room] post-release: state=queued, host injected {len(ctx_b.injected)} message(s) — lifecycle complete")

        print("\nCR603 LIVE PEER E2E (manager path): PASS")
        return 0
    finally:
        try:
            mgr_a.shutdown()
        except Exception:
            pass
        try:
            mgr_b.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
