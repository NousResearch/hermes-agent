from __future__ import annotations

import asyncio
import json
import threading

import pytest

from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.bridge import capture_to_payload
from tools.computer_use.desktop_bridge import (
    DesktopBridgeBroker,
    DesktopBridgeScope,
    DesktopComputerUseBridgeBackend,
    desktop_bridge_computer_use_status,
    reset_desktop_bridge_caller,
    set_desktop_bridge_caller,
)


class _FakeWs:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = asyncio.Queue()

    async def accept(self):
        return None

    async def send_text(self, text):
        await self.sent.put(text)

    async def receive_text(self):
        item = await self.incoming.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def reply(self, frame, result):
        await self.incoming.put(json.dumps({
            "id": frame["id"],
            "ok": True,
            "result": result,
        }))

    async def disconnect(self):
        await self.incoming.put(RuntimeError("closed"))


def test_desktop_bridge_broker_isolates_two_principals_and_rejects_unmatched_scope():
    async def scenario():
        broker = DesktopBridgeBroker()
        alice = DesktopBridgeScope("stub", "alice", "default")
        bob = DesktopBridgeScope("stub", "bob", "default")
        unknown = DesktopBridgeScope("stub", "mallory", "default")
        wrong_profile = DesktopBridgeScope("stub", "alice", "work")
        alice_ws = _FakeWs()
        bob_ws = _FakeWs()
        alice_handler = asyncio.create_task(broker.handle_ws(alice_ws, alice))
        bob_handler = asyncio.create_task(broker.handle_ws(bob_ws, bob))
        await asyncio.sleep(0)

        alice_request = asyncio.create_task(
            broker.request_async({"type": "status"}, timeout=1, scope=alice)
        )
        alice_frame = json.loads(await alice_ws.sent.get())
        assert alice_frame["type"] == "status"
        assert bob_ws.sent.empty(), "Alice's call must never reach Bob's Desktop"
        await alice_ws.reply(alice_frame, {"owner": "alice"})
        assert await alice_request == {"owner": "alice"}

        bob_request = asyncio.create_task(
            broker.request_async({"type": "status"}, timeout=1, scope=bob)
        )
        bob_frame = json.loads(await bob_ws.sent.get())
        await bob_ws.reply(bob_frame, {"owner": "bob"})
        assert await bob_request == {"owner": "bob"}

        with pytest.raises(RuntimeError, match="not connected for the authenticated"):
            await broker.request_async({"type": "status"}, timeout=1, scope=unknown)
        with pytest.raises(RuntimeError, match="not connected for the authenticated"):
            await broker.request_async(
                {"type": "status"}, timeout=1, scope=wrong_profile
            )

        await alice_ws.disconnect()
        await bob_ws.disconnect()
        await asyncio.gather(alice_handler, bob_handler)

    asyncio.run(scenario())


def test_desktop_bridge_broker_rejects_ambiguous_scope_then_cleans_up_disconnect():
    async def scenario():
        broker = DesktopBridgeBroker()
        scope = DesktopBridgeScope("stub", "alice", "work")
        old_ws = _FakeWs()
        new_ws = _FakeWs()
        old_handler = asyncio.create_task(broker.handle_ws(old_ws, scope))
        new_handler = asyncio.create_task(broker.handle_ws(new_ws, scope))
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="ambiguous"):
            await broker.request_async({"type": "status"}, timeout=1, scope=scope)

        await old_ws.disconnect()
        await old_handler
        request = asyncio.create_task(
            broker.request_async({"type": "status"}, timeout=1, scope=scope)
        )
        frame = json.loads(await new_ws.sent.get())
        await new_ws.reply(frame, {"ready": True})
        assert await request == {"ready": True}

        await new_ws.disconnect()
        await new_handler
        assert broker.is_connected(scope) is False

    asyncio.run(scenario())


def test_tool_backend_lookup_dispatches_only_to_current_principal_and_falls_back(
    monkeypatch,
):
    import tools.computer_use.desktop_bridge as desktop_bridge
    from tools.computer_use import tool as computer_use_tool

    class LocalFallback:
        def start(self):
            return None

        def stop(self):
            return None

        def is_available(self):
            return True

        def list_apps(self):
            return [{"name": "backend-local"}]

    async def scenario():
        broker = DesktopBridgeBroker()
        alice = DesktopBridgeScope("stub", "alice", "default")
        bob = DesktopBridgeScope("stub", "bob", "default")
        alice_ws = _FakeWs()
        bob_ws = _FakeWs()
        alice_handler = asyncio.create_task(broker.handle_ws(alice_ws, alice))
        bob_handler = asyncio.create_task(broker.handle_ws(bob_ws, bob))
        await asyncio.sleep(0)

        monkeypatch.setattr(desktop_bridge, "_BROKER", broker)
        monkeypatch.setattr(
            computer_use_tool, "configured_computer_use_backend", lambda: "cua"
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(
            "tools.computer_use.cua_backend.CuaDriverBackend", LocalFallback
        )

        async def dispatch_for(subject):
            caller_token = set_desktop_bridge_caller(("stub", subject))
            try:
                return await asyncio.to_thread(
                    computer_use_tool.handle_computer_use,
                    {"action": "list_apps"},
                )
            finally:
                reset_desktop_bridge_caller(caller_token)

        alice_call = asyncio.create_task(dispatch_for("alice"))
        alice_status = json.loads(await alice_ws.sent.get())
        assert bob_ws.sent.empty()
        await alice_ws.reply(alice_status, {"ready": True, "checks": []})
        alice_list = json.loads(await alice_ws.sent.get())
        await alice_ws.reply(alice_list, {"apps": [{"name": "alice-desktop"}]})
        assert json.loads(await alice_call)["apps"] == [{"name": "alice-desktop"}]

        bob_call = asyncio.create_task(dispatch_for("bob"))
        bob_status = json.loads(await bob_ws.sent.get())
        assert alice_ws.sent.empty()
        await bob_ws.reply(bob_status, {"ready": True, "checks": []})
        bob_list = json.loads(await bob_ws.sent.get())
        await bob_ws.reply(bob_list, {"apps": [{"name": "bob-desktop"}]})
        assert json.loads(await bob_call)["apps"] == [{"name": "bob-desktop"}]

        unmatched = json.loads(await dispatch_for("mallory"))
        assert unmatched["apps"] == [{"name": "backend-local"}]
        assert alice_ws.sent.empty()
        assert bob_ws.sent.empty()

        computer_use_tool.reset_backend_for_tests()
        monkeypatch.setattr(
            computer_use_tool,
            "configured_computer_use_backend",
            lambda: "desktop",
        )
        fail_closed = json.loads(await dispatch_for("mallory"))
        assert "not connected for the authenticated principal" in fail_closed["error"]

        await alice_ws.disconnect()
        await alice_handler
        assert broker.is_connected(bob) is True

        computer_use_tool.reset_backend_for_tests()
        bob_again = asyncio.create_task(dispatch_for("bob"))
        bob_status = json.loads(await bob_ws.sent.get())
        await bob_ws.reply(bob_status, {"ready": True, "checks": []})
        bob_list = json.loads(await bob_ws.sent.get())
        await bob_ws.reply(bob_list, {"apps": [{"name": "bob-still-live"}]})
        assert json.loads(await bob_again)["apps"] == [{"name": "bob-still-live"}]

        await bob_ws.disconnect()
        await bob_handler

    try:
        asyncio.run(scenario())
    finally:
        computer_use_tool.reset_backend_for_tests()


def test_desktop_bridge_backend_round_trips_capture(monkeypatch):
    import tools.computer_use.desktop_bridge as desktop_bridge

    class FakeBroker:
        def is_connected(self, scope=None):
            return True

        def connection_info(self, scope=None):
            return {"connected": True, "client_id": "desktop-test", "pending": 0}

        def request(self, payload, timeout=None, scope=None):
            if payload["type"] == "status":
                return {"ready": True, "checks": []}
            assert payload == {"type": "computer-use", "method": "capture", "args": {"mode": "ax", "app": "Finder"}}
            return capture_to_payload(
                CaptureResult(
                    mode="ax",
                    width=800,
                    height=600,
                    elements=[UIElement(index=1, role="AXButton", label="OK")],
                    app="Finder",
                )
            )

    monkeypatch.setattr(desktop_bridge, "_BROKER", FakeBroker())

    backend = DesktopComputerUseBridgeBackend()
    backend.start()
    capture = backend.capture(mode="ax", app="Finder")

    assert capture.app == "Finder"
    assert capture.elements[0].label == "OK"


def test_desktop_bridge_status_reports_offline_without_secret(monkeypatch):
    import tools.computer_use.desktop_bridge as desktop_bridge

    class OfflineBroker:
        def is_connected(self, scope=None):
            return False

        def connection_info(self, scope=None):
            return {
                "connected": False,
                "error": "Desktop Computer Use bridge is not connected",
            }

    monkeypatch.setattr(desktop_bridge, "_BROKER", OfflineBroker())

    status = desktop_bridge_computer_use_status()

    assert status["platform"] == "desktop-bridge"
    assert status["ready"] is False
    assert status["bridge"] == {"kind": "desktop", "connected": False}
    assert "token" not in json.dumps(status).lower()


def test_tui_branch_turn_keeps_parent_profile_bridge_scope(monkeypatch, tmp_path):
    """A real branch + prompt turn must use the parent's profile-scoped socket."""
    import hermes_cli.profiles as profiles
    import tools.computer_use.desktop_bridge as desktop_bridge
    from tui_gateway import server
    from tui_gateway.transport import bind_transport, reset_transport

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "work"
    profile_home.mkdir(parents=True)
    parent_key = "20260713_100000_parent"
    child_key = "20260713_100001_child"
    principal = ("ticket", "alice")

    class _Transport:
        authenticated_principal = principal
        desktop_bridge_profile = "work"
        allow_desktop_bridge_profile_override = False

        def write(self, _obj):
            return True

    class _Worker:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class _DB:
        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} 2"

        def create_session(self, *_args, **_kwargs):
            return None

        def append_message(self, **_kwargs):
            return None

        def set_session_title(self, *_args):
            return None

        def get_session(self, _key):
            return None

        def update_session_cwd(self, *_args):
            return None

    turn_done = threading.Event()
    observed = {}

    class _Agent:
        model = "test/model"

        def run_conversation(self, _message, conversation_history=None, **_kwargs):
            try:
                observed["status"] = desktop_bridge.desktop_bridge_computer_use_status(
                    timeout=2
                )
                return {
                    "final_response": "",
                    "messages": list(conversation_history or []),
                }
            finally:
                turn_done.set()

        def close(self):
            return None

    broker = DesktopBridgeBroker()
    work_scope = DesktopBridgeScope(*principal, "work")
    default_scope = DesktopBridgeScope(*principal, "default")
    work_ws = _FakeWs()
    default_ws = _FakeWs()
    transport = _Transport()

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(desktop_bridge, "_BROKER", broker)
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *_a, **_k: (None, None))
    monkeypatch.setattr(server, "_new_session_key", lambda: child_key)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda _sid, _key, **_kwargs: _Agent(),
    )
    monkeypatch.setattr(server, "_SlashWorker", _Worker)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, session=None: {"profile_home": (session or {}).get("profile_home")},
    )

    import tools.approval as approval

    monkeypatch.setattr(approval, "register_gateway_notify", lambda *_a, **_k: None)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)

    async def scenario():
        work_handler = asyncio.create_task(broker.handle_ws(work_ws, work_scope))
        default_handler = asyncio.create_task(
            broker.handle_ws(default_ws, default_scope)
        )
        await asyncio.sleep(0)

        parent_sid = "parent01"
        server._sessions[parent_sid] = {
            "agent": object(),
            "authenticated_principal": principal,
            "cols": 80,
            "cwd": str(tmp_path),
            "history": [{"role": "user", "content": "hello"}],
            "history_lock": threading.Lock(),
            "profile_home": str(profile_home),
            "profile": "work",
            "session_key": parent_key,
            "source": "desktop",
            "transport": transport,
        }

        token = bind_transport(transport)
        try:
            branched = server._methods["session.branch"](
                "branch", {"session_id": parent_sid}
            )
        finally:
            reset_transport(token)

        assert "error" not in branched, branched
        child_sid = branched["result"]["session_id"]
        child = server._sessions[child_sid]
        assert child["authenticated_principal"] == principal
        assert child["profile"] == "work"
        assert child["transport"] is transport

        token = bind_transport(transport)
        try:
            submitted = server._methods["prompt.submit"](
                "turn", {"session_id": child_sid, "text": "check desktop"}
            )
        finally:
            reset_transport(token)
        assert submitted["result"]["status"] == "streaming"

        frame = json.loads(await asyncio.wait_for(work_ws.sent.get(), timeout=2))
        assert frame["type"] == "status"
        assert default_ws.sent.empty(), "branch turn must not dispatch to the default profile"
        await work_ws.reply(frame, {"ready": True, "checks": []})
        assert await asyncio.to_thread(turn_done.wait, 2)
        assert observed["status"]["ready"] is True
        assert default_ws.sent.empty()

        await work_ws.disconnect()
        await default_ws.disconnect()
        await asyncio.gather(work_handler, default_handler)

    try:
        asyncio.run(scenario())
    finally:
        for session in list(server._sessions.values()):
            server._teardown_session(session)
        server._sessions.clear()
