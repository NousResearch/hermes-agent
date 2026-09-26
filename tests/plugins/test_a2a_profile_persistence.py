"""Multiplex A2A writes stay with the execution profile across HTTP thread hops."""
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import threading
from types import SimpleNamespace

from tests.e2e.core.security._helpers import hermetic_env, run_python, write_home
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
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


def test_local_completion_and_forwarded_roundtrip_isolated_in_threads(tmp_path, monkeypatch):
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

    # A new OS thread checks scope isolation; the real subprocess/readback test below
    # checks process boundaries and persisted sessions after process exit.
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


def test_real_forwarded_profiles_survive_fresh_process_readback(tmp_path, monkeypatch):
    """HTTP worker threads route real CLI turns; a new interpreter reads each home."""
    root = tmp_path / "operator"
    launch = root / ".hermes"
    a, b = (launch / "profiles" / name for name in ("a", "b"))
    root.mkdir()
    launcher = tmp_path / "bin"
    launcher.mkdir()
    hermes = launcher / "hermes"
    hermes.write_text(f"#! /bin/sh\nexec '{sys.executable}' -m hermes_cli.main \"$@\"\n")
    hermes.chmod(0o700)
    with FakeLLMServer([Text("ACK b"), Text("ACK a")], api_key="fixture-key") as model:
        for home in (a, b):
            write_home(home, model.base_url, api_key="fixture-key")
        # The launcher has no model config or A2A state. All inference must occur in a or b.
        env = hermetic_env(root, {"PATH": str(launcher) + os.pathsep + os.environ["PATH"]})
        monkeypatch.setattr(os, "environ", env)
        adapters = []
        for home, source, target in ((a, "a", "b"), (b, "b", "a")):
            monkeypatch.setenv("HERMES_PROFILE", source)
            token = set_hermes_home_override(home)
            try:
                adapters.append(A2AAdapter(SimpleNamespace(extra={"agents": {target: {"profile": target}}})))
            finally:
                reset_hermes_home_override(token)
        monkeypatch.delenv("HERMES_PROFILE")

        failures = []
        for adapter, target, context in zip(adapters, ("b", "a"), ("to-b", "to-a")):
            def send():
                try:
                    params = {"message": protocol.text_message(protocol.ROLE_USER, context, context_id=context)}
                    response = adapter._rpc_message_send(1, params, "trusted-peer", agent=adapter._agents[target])
                    assert response["result"]["status"]["state"] == protocol.STATE_COMPLETED, response
                except BaseException as exc:
                    failures.append(exc)
            thread = threading.Thread(target=send)
            thread.start()
            thread.join(timeout=60)
            assert not thread.is_alive()
        assert not failures, failures
        assert len(model.main_requests()) == 2  # both turns reached the real target CLI

        # Fresh interpreters after the worker subprocesses exit read persisted transcripts,
        # not adapter caches or a cloned ContextVar.
        for home, context, foreign, reply in ((a, "to-a", "to-b", "ACK a"),
                                               (b, "to-b", "to-a", "ACK b")):
            code = ("import json; from plugins.platforms.a2a.tools import a2a_history; "
                    f"print(json.dumps([a2a_history({{'context_id': {context!r}}}), "
                    f"a2a_history({{'context_id': {foreign!r}}})]))")
            readback = run_python(code, root, extra_env={"HERMES_HOME": str(home)})
            assert readback.returncode == 0, readback.stderr
            own, other = json.loads(readback.stdout.strip().splitlines()[-1])
            assert f"[user] {context}" in own and f"[agent] {reply}" in own
            assert "No persisted conversation" in other
            audit = [json.loads(line) for line in (home / "a2a_audit.jsonl").read_text().splitlines()]
            assert [row["direction"] for row in audit] == ["inbound", "outbound"]
            with sqlite3.connect(home / "state.db") as db:
                sessions = db.execute("SELECT id, source FROM sessions WHERE source = 'a2a'").fetchall()
            assert len(sessions) == 1 and sessions[0][0]
        assert not (launch / "a2a_conversations").exists()
        assert not (launch / "a2a_audit.jsonl").exists()
        assert not (launch / "state.db").exists()


def test_forwarded_session_identity_survives_restart_without_peer_or_context_alias(tmp_path, monkeypatch):
    """Real target CLI sessions cannot be resumed by another peer or colliding context."""
    root = tmp_path / "operator"
    launch = root / ".hermes"
    target = launch / "profiles" / "receiver"
    root.mkdir()
    launcher = tmp_path / "bin"
    launcher.mkdir()
    hermes = launcher / "hermes"
    hermes.write_text(f"#! /bin/sh\nexec '{sys.executable}' -m hermes_cli.main \"$@\"\n")
    hermes.chmod(0o700)
    # These exact IDs used to collapse to the same sanitized session title.
    contexts = ("shared/id", "shared id", "x" * 96 + "one", "x" * 96 + "two")
    turns = (("peer-one", contexts[0]), ("peer-two", contexts[0]),
             ("peer-one", contexts[1]), ("peer-one", contexts[2]),
             ("peer-one", contexts[3]), ("peer-one", contexts[0]))
    with FakeLLMServer([Text(f"ACK {i}") for i in range(len(turns))], api_key="fixture-key") as model:
        write_home(target, model.base_url, api_key="fixture-key")
        env = hermetic_env(root, {"PATH": str(launcher) + os.pathsep + os.environ["PATH"]})
        monkeypatch.setattr(os, "environ", env)
        def new_listener():
            monkeypatch.setenv("HERMES_PROFILE", "sender")
            token = set_hermes_home_override(launch)
            try:
                return A2AAdapter(SimpleNamespace(extra={"agents": {"receiver": {"profile": "receiver"}}}))
            finally:
                reset_hermes_home_override(token)
                monkeypatch.delenv("HERMES_PROFILE")

        listener = new_listener()
        for i, (peer, context) in enumerate(turns):
            if i == len(turns) - 1:
                listener = new_listener()  # empty cache; title must find only this peer/context
            errors = []
            def send():
                try:
                    params = {"message": protocol.text_message(protocol.ROLE_USER, f"turn {i}", context_id=context)}
                    response = listener._rpc_message_send(1, params, peer, agent=listener._agents["receiver"])
                    assert response["result"]["status"]["state"] == protocol.STATE_COMPLETED, response
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=send)
            thread.start()
            thread.join(timeout=60)
            assert not thread.is_alive()
            assert not errors, errors

        assert len(model.main_requests()) == len(turns)
        with sqlite3.connect(target / "state.db") as db:
            sessions = db.execute("SELECT id, title FROM sessions WHERE source = 'a2a'").fetchall()
            by_title = {title: session_id for session_id, title in sessions}
            for peer, context in turns:
                identity = ("receiver", "receiver", peer, context)
                title = "a2a-" + hashlib.sha256(
                    json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                assert title in by_title
                user_turns = [row[0] for row in db.execute(
                    "SELECT content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id",
                    (by_title[title],))]
                expected = [f"turn {j}" for j, pair in enumerate(turns) if pair == (peer, context)]
                assert len(user_turns) == len(expected)
                assert all(text in turn for text, turn in zip(expected, user_turns))
        assert len(sessions) == len(turns) - 1  # only peer-one's exact context continues
        assert len({title for _, title in sessions}) == len(sessions)
        assert all(title.startswith("a2a-") for _, title in sessions)
        # Fresh process reads the receiver's durable transcript; launch profile stays clean.
        for context in contexts:
            code = ("from plugins.platforms.a2a.tools import a2a_history; "
                    f"print(a2a_history({{'context_id': {context!r}}}))")
            readback = run_python(code, root, extra_env={"HERMES_HOME": str(target)})
            assert readback.returncode == 0, readback.stderr
            assert "[user]" in readback.stdout and "[agent]" in readback.stdout
        assert not (launch / "state.db").exists()
        assert not (launch / "a2a_conversations").exists()
