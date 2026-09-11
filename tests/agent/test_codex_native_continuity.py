"""Real SQLite + stdio child lifecycle tests; the child is NOT vendor acceptance."""

import json
import subprocess
import sys
import threading
import time

import pytest

from agent.codex_runtime import _ensure_codex_session
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture
def wire_runtime(tmp_path, monkeypatch):
    script = tmp_path / "wire.py"
    log = tmp_path / "wire.jsonl"
    script.write_text('''import json, sys, uuid
from pathlib import Path
for line in sys.stdin:
    req = json.loads(line)
    with open(sys.argv[1], "a") as f:
        f.write(json.dumps(req) + "\\n")
    method, params = req.get("method"), req.get("params", {})
    if "id" not in req:
        continue
    blocked = Path(__file__).with_name("block-method")
    if blocked.exists() and blocked.read_text() == method:
        print(json.dumps({"method": "test/blocked"}), flush=True)
        continue
    result = {}
    if method == "thread/start":
        result = {"thread": {"id": str(uuid.uuid4())}}
    if method == "thread/resume":
        result = {"thread": {"id": params["threadId"]}}
        flag = Path(__file__).with_name("resume-mode")
        if flag.exists():
            if flag.read_text() == "unsupported":
                print(json.dumps({"id": req["id"], "error": {"code": -32601,
                    "message": "thread/resume unsupported"}}), flush=True)
                continue
            result = {"thread": {"id": "unexpected-replacement"}}
    if method == "turn/start":
        result = {"turn": {"id": str(uuid.uuid4())}}
    print(json.dumps({"id": req["id"], "result": result}), flush=True)
    if method == "turn/start":
        print(json.dumps({"method": "turn/completed", "params": {
            "threadId": params["threadId"],
            "turn": {"id": result["turn"]["id"], "status": "completed"}}}), flush=True)
''', encoding="utf-8")
    original = subprocess.Popen
    children = []

    def spawn(cmd, **kwargs):
        child = original([sys.executable, str(script), str(log)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("agent.transports.codex_app_server.subprocess.Popen", spawn)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    yield children, lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    for child in children:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("method", ["initialize", "thread/start", "thread/resume", "turn/start", "thread/compact/start"])
def test_stop_retires_real_stdio_child_and_releases_pending_rpc(tmp_path, wire_runtime, method):
    from agent.transports.codex_app_server import CodexAppServerClient
    from agent.transports.codex_app_server_session import CodexAppServerSession

    children, requests = wire_runtime
    entered, rpc_returned = threading.Event(), threading.Event()
    (tmp_path / "block-method").write_text(method)

    class Client(CodexAppServerClient):
        def _dispatch(self, message):
            if message.get("method") == "test/blocked":
                entered.set()
            super()._dispatch(message)

        def request(self, name, *args, **kwargs):
            try:
                return super().request(name, *args, **kwargs)
            finally:
                if name == method:
                    rpc_returned.set()

    session = CodexAppServerSession(cwd=str(tmp_path), client_factory=Client,
                                   thread_id="saved-thread" if method == "thread/resume" else None)
    results = []
    target = session.compact_thread if method == "thread/compact/start" else lambda: session.run_turn("blocked")
    worker = threading.Thread(target=lambda: results.append(target()), daemon=True)
    worker.start()
    try:
        assert entered.wait(5), "fixture child did not receive RPC"
        started = time.monotonic()
        session.request_interrupt()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert time.monotonic() - started < 5
        assert results[0].interrupted and results[0].should_retire
        assert not results[0].terminal_acknowledged
        assert children[-1].poll() is not None
        assert rpc_returned.wait(2), "closing the child left its RPC worker blocked"
        recorded = requests()
        assert recorded[-1]["method"] == method
        assert session.run_turn("must not restart").should_retire
        assert requests() == recorded
    finally:
        session.close()
        worker.join(timeout=12)


def test_retirement_and_database_reopen_resume_exact_native_threads(tmp_path, wire_runtime):
    children, requests = wire_runtime
    db_path = tmp_path / "state.db"
    bindings = {}
    for iteration in range(2):
        with SessionDB(db_path) as db:
            for sid in ("member-thread-a", "member-thread-b"):
                if iteration == 0:
                    db.create_session(session_id=sid, source="gui")
                agent = AIAgent.__new__(AIAgent)
                agent.session_id, agent._session_db = sid, db
                agent.session_cwd = str(tmp_path)
                agent.client = None
                _ensure_codex_session(agent)
                session = agent._codex_session
                tid = session.ensure_started()
                # Binding must be durable BEFORE any native model/tool work is admitted.
                binding = db.get_session_model_config_value(sid, "codex_native_session")
                assert binding and binding["thread_id"] == tid
                if iteration:
                    assert tid == bindings[sid]
                bindings[sid] = tid
                result = session.run_turn("fixture")
                assert result.error is None
                agent.release_clients()
                assert children[-1].poll() is not None
                assert agent._codex_session is None
    assert len(set(bindings.values())) == 2
    methods = [r["method"] for r in requests()]
    assert methods.count("thread/start") == methods.count("thread/resume") == 2


@pytest.mark.parametrize("assistant_content", ["Previous native answer", ""])
def test_persisted_history_without_binding_never_admits_native_work(tmp_path, wire_runtime, assistant_content):
    _, requests = wire_runtime
    db_path = tmp_path / "state.db"
    with SessionDB(db_path) as db:
        db.create_session(session_id="legacy", source="cli", model_config={"api_mode": "codex_app_server"})
        db.append_message("legacy", "user", "Previous question")
        db.append_message("legacy", "assistant", assistant_content)
    with SessionDB(db_path) as db:
        agent = AIAgent.__new__(AIAgent)
        agent.session_id, agent._session_db = "legacy", db
        agent.session_cwd, agent.client = str(tmp_path), None
        try:
            with pytest.raises(RuntimeError, match="history.*binding"):
                _ensure_codex_session(agent)
                agent._codex_session.run_turn("must not execute")
        finally:
            agent.release_clients()
        assert not any(r["method"] in {"thread/start", "turn/start"} for r in requests())
        assert db.get_session_model_config_value("legacy", "codex_native_session") is None


@pytest.mark.parametrize("states", [(False, True), (True, False), (True, None)])
def test_reused_session_refreshes_both_approval_policies_before_work(tmp_path, wire_runtime, monkeypatch, states):
    from unittest.mock import Mock

    _, requests = wire_runtime
    lookup = Mock(side_effect=[RuntimeError("lookup failed") if state is None else state for state in states])
    monkeypatch.setattr("tools.approval.is_approval_bypass_active", lookup)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(session_id="policy", source="cli")
        agent = AIAgent.__new__(AIAgent)
        agent.session_id, agent._session_db = "policy", db
        agent.session_cwd, agent.client = str(tmp_path), None
        sessions, threads = [], []
        try:
            for state in states:
                _ensure_codex_session(agent)
                session = agent._codex_session
                sessions.append(session)
                result = session.run_turn("fixture")
                assert result.error is None
                threads.append(result.thread_id)
                assert session._decide_exec_approval({"command": "fixture"}) == ("accept" if state else "decline")
                assert session._decide_apply_patch_approval({}) == ("accept" if state else "decline")
            assert lookup.call_count == len(states)
            assert sessions[0] is sessions[1]
            assert threads[0] == threads[1]
            sent = [r["params"] for r in requests() if r["method"] == "turn/start"]
            assert [p.get("approvalPolicy") for p in sent] == ["never" if state else "on-request" for state in states]
            assert all(p["threadId"] == threads[0] for p in sent)
            assert sum(r["method"] == "thread/start" for r in requests()) == 1
        finally:
            agent.release_clients()


@pytest.mark.parametrize("transition", ["resume", "new", "branch"])
def test_warm_session_reset_detaches_native_source(tmp_path, wire_runtime, transition):
    from types import SimpleNamespace
    from hermes_cli.cli_commands_mixin import _sync_agent_to_session

    children, requests = wire_runtime
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(session_id="source", source="cli")
        db.create_session(session_id="destination", source="cli", model_config=(
            {"_branched_from": "source"} if transition == "branch" else {}))
        agent = AIAgent.__new__(AIAgent)
        agent.session_id, agent._session_db = "source", db
        agent.session_cwd, agent.client = str(tmp_path), None
        _ensure_codex_session(agent)
        source = agent._codex_session.ensure_started()
        source_child = children[-1]
        source_binding = db.get_session_model_config_value("source", "codex_native_session")
        saved_destination = {**source_binding, "session_id": "destination", "thread_id": "saved-destination-thread"}
        if transition == "resume":
            db.patch_session_model_config("destination", {"codex_native_session": saved_destination})
        agent._invalidate_system_prompt = lambda: None
        if transition == "new":
            # Same-model /new keeps this AIAgent and calls this exact boundary.
            agent.session_id = "destination"
            agent.reset_session_state()
        else:
            cli = SimpleNamespace(agent=agent, conversation_history=[])
            _sync_agent_to_session(cli, "destination", parent_session_id="source", reason=transition)
        assert source_child.poll() is not None, "warm reset retained the source native process"
        assert db.get_session_model_config_value("source", "codex_native_session") == source_binding
        if transition == "branch":
            with pytest.raises(RuntimeError, match="native.*branch.*unsupported"):
                _ensure_codex_session(agent)
            assert not any(r["method"] == "turn/start" for r in requests())
            return
        _ensure_codex_session(agent)
        try:
            result = agent._codex_session.run_turn("destination fixture")
            assert result.error is None
            destination = result.thread_id
            assert destination != source
            if transition == "resume":
                assert destination == saved_destination["thread_id"]
            assert db.get_session_model_config_value("destination", "codex_native_session")["thread_id"] == destination
            assert [r["params"]["threadId"] for r in requests() if r["method"] == "turn/start"] == [destination]
        finally:
            agent.release_clients()


def test_native_image_input_crosses_stdio_and_unsupported_media_is_explicit(tmp_path, wire_runtime):
    from agent.transports.codex_app_server_session import CodexAppServerSession

    _, requests = wire_runtime
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    session = CodexAppServerSession(cwd=str(tmp_path))
    try:
        result = session.run_turn([
            {"type": "text", "text": "Describe the fixture"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        assert result.error is None
        sent = next(r["params"]["input"] for r in requests() if r["method"] == "turn/start")
        assert sent == [{"type": "text", "text": "Describe the fixture"},
                        {"type": "image", "url": data_url}]
        result = session.run_turn([{"type": "input_audio", "data": "fixture"}])
        assert "unsupported" in result.error.lower()
        assert sum(r["method"] == "turn/start" for r in requests()) == 1
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["unsupported", "wrong-thread"])
def test_resume_failure_never_submits_or_starts_a_replacement(tmp_path, wire_runtime, mode):
    from agent.transports.codex_app_server_session import CodexAppServerSession

    children, requests = wire_runtime
    (tmp_path / "resume-mode").write_text(mode)
    session = CodexAppServerSession(cwd=str(tmp_path), thread_id="saved-thread")
    result = session.run_turn("must not execute")
    assert result.error and result.should_retire
    assert not any(r["method"] in {"thread/start", "turn/start"} for r in requests())
    assert children[-1].poll() is not None
