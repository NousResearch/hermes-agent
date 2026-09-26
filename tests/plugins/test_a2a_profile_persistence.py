"""Multiplex A2A writes stay with the execution profile across HTTP thread hops."""
import asyncio
import json
import threading
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.tools import a2a_history


def test_multiplex_adapters_persist_under_own_homes(tmp_path, monkeypatch):
    launch, a, b = (tmp_path / name for name in ("launch", "a", "b"))
    for home in (launch, a, b):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.delenv("A2A_PORT", raising=False)
    adapters = []
    for home in (a, b, a):
        token = set_hermes_home_override(home)
        try:
            adapters.append(A2AAdapter(SimpleNamespace(extra={})))
        finally:
            reset_hermes_home_override(token)

    def incoming(adapter, context):
        params = {"message": protocol.text_message(protocol.ROLE_USER, context, context_id=context)}
        adapter._prepare_task(params, "authenticated-peer")
        adapter._record_outcome("completed-task", context, "authenticated-peer", protocol.STATE_COMPLETED, "ACK")

    # New OS threads have no caller ContextVar: they see the multiplex launch home.
    for adapter, context in zip(adapters, ("to-a", "to-b", "back-to-a")):
        thread = threading.Thread(target=incoming, args=(adapter, context))
        thread.start()
        thread.join()
    for home, own, foreign in ((a, ("to-a", "back-to-a"), "to-b"), (b, ("to-b",), "to-a")):
        for context in own:
            rows = [json.loads(line) for line in (home / "a2a_conversations" / f"{context}.jsonl").read_text().splitlines()]
            assert [row["role"] for row in rows] == ["user", "agent"]
            assert [row["text"] for row in rows] == [context, "ACK"]
        assert not (home / "a2a_conversations" / f"{foreign}.jsonl").exists()
        audit = [json.loads(line) for line in (home / "a2a_audit.jsonl").read_text().splitlines()]
        assert len(audit) == 2 * len(own)
    assert not (launch / "a2a_conversations").exists()
    assert not (launch / "a2a_audit.jsonl").exists()


def test_local_completion_and_forwarded_roundtrip_isolated_after_restart(tmp_path, monkeypatch):
    launch, a, b = (tmp_path / name for name in ("launch", "a", "b"))
    for home in (launch, a, b):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in ("a", "b"))
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: {"a": a, "b": b}[name])
    from plugins.platforms.a2a import adapter as adapter_module
    monkeypatch.setattr(adapter_module, "_active_profile_name", lambda: "a")
    token = set_hermes_home_override(a)
    try:
        listener_a = A2AAdapter(SimpleNamespace(extra={"agents": {"b": {"profile": "b", "local": True}}}))
    finally:
        reset_hermes_home_override(token)
    monkeypatch.setattr(adapter_module, "_active_profile_name", lambda: "b")
    token = set_hermes_home_override(b)
    try:
        listener_b = A2AAdapter(SimpleNamespace(extra={"agents": {"a": {"profile": "a"}}}))
    finally:
        reset_hermes_home_override(token)
    assert listener_a._agents["b"]["local"] is False  # misconfigured local cannot enter A's turn

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    listener_a._loop = loop
    listener_a._message_handler = object()  # type: ignore[assignment]

    async def local_handler(event):
        await listener_a.send(event.source.chat_id, "ACK local", metadata={"notify": True})

    listener_a.handle_message = local_handler
    monkeypatch.setattr(listener_a, "_forward_to_profile", lambda *_: ("ACK b", protocol.STATE_COMPLETED))
    monkeypatch.setattr(listener_b, "_forward_to_profile", lambda *_: ("ACK a", protocol.STATE_COMPLETED))
    failures = []

    def send(listener, context, routed=None):
        try:
            params = {"message": protocol.text_message(protocol.ROLE_USER, context, context_id=context)}
            response = listener._rpc_message_send(1, params, "trusted-peer", agent=routed)
            assert response["result"]["status"]["state"] == protocol.STATE_COMPLETED
        except BaseException as exc:
            failures.append(exc)

    try:
        for listener, context, routed in (
            (listener_a, "a-local", None),
            (listener_a, "a-to-b", listener_a._agents["b"]),
            (listener_b, "b-to-a", listener_b._agents["a"]),
        ):
            thread = threading.Thread(target=send, args=(listener, context, routed))
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert not failures, failures
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=10)
        loop.close()

    # Fresh scope in a fresh OS thread models a new profile session after restart.
    def readback(home, own, foreign):
        try:
            token = set_hermes_home_override(home)
            try:
                for context, reply in own:
                    history = a2a_history({"context_id": context})
                    assert f"[user] {context}" in history
                    assert f"[agent] {reply}" in history
                for context in foreign:
                    assert "No persisted conversation" in a2a_history({"context_id": context})
            finally:
                reset_hermes_home_override(token)
            audit = [json.loads(line) for line in (home / "a2a_audit.jsonl").read_text().splitlines()]
            assert len(audit) == 2 * len(own)
            assert {row["direction"] for row in audit} == {"inbound", "outbound"}
        except BaseException as exc:
            failures.append(exc)

    for home, own, foreign in (
        (a, (("a-local", "ACK local"), ("b-to-a", "ACK a")), ("a-to-b",)),
        (b, (("a-to-b", "ACK b"),), ("a-local", "b-to-a")),
        (launch, (), ("a-local", "a-to-b", "b-to-a")),
    ):
        if not own:
            assert not (home / "a2a_conversations").exists()
            assert not (home / "a2a_audit.jsonl").exists()
            continue
        thread = threading.Thread(target=readback, args=(home, own, foreign))
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not failures, failures


def test_forwarded_missing_profile_fails_without_creating_history(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: False)
    adapter = A2AAdapter(SimpleNamespace(extra={"agents": {"missing": {"profile": "missing"}}}))
    results = []

    def send():
        params = {"message": protocol.text_message(protocol.ROLE_USER, "private", context_id="missing-ctx")}
        results.append(adapter._rpc_message_send(1, params, "peer", agent=adapter._agents["missing"]))

    thread = threading.Thread(target=send)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert results[0]["result"]["status"]["state"] == protocol.STATE_FAILED
    assert not (launch / "a2a_conversations").exists()
    assert not (launch / "a2a_audit.jsonl").exists()
