"""Live compression-to-terminal identity, including a reused shell environment.

Summary, model-server detection, and automatic title generation are substituted; compression,
SQLite lineage, tool dispatch, and foreground/PTY subprocesses use the real runtime.
Caller-context variants exercise core APIs, not CLI or gateway entrypoints.
"""

import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from threading import Barrier, get_ident
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor
from agent.delegation_context import delegated_child_context
from gateway.session_context import (
    _SESSION_ID,
    get_session_env,
    scoped_current_session_id,
    set_session_vars,
)
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.process_registry import process_registry
from tools.terminal_tool_lifecycle import cleanup_vm, get_active_env


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    connections = []

    def reject_network(sock, address):
        connections.append(address)
        raise AssertionError("Identity regression must not connect to a network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    # Local-server discovery is unrelated to identity and otherwise probes HTTP.
    monkeypatch.setattr("agent.model_metadata.detect_local_server_type", lambda *a, **kw: None)
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test/model\n  base_url: http://127.0.0.1:1/v1\n"
        "  context_length: 100000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "inherited-session")
    monkeypatch.setattr("gateway.session_context._session_context_engaged", False)
    # Title generation is unrelated background LLM work, not part of this probe.
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    def summary(compressor, *args, **kwargs):
        compressor._identity_test_summary_thread = get_ident()
        return "Earlier turns discussed a generic example. Continue the task."

    monkeypatch.setattr(ContextCompressor, "_generate_summary", summary)
    # Exercise worker-to-caller rebinding, never the timeout-disabled direct path.
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda: (30, 60),
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()
        assert not connections


def _agent(db, sid):
    agent = AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test/model",
        quiet_mode=True, session_db=db, session_id=sid, enabled_toolsets=["terminal"],
        skip_context_files=True, skip_memory=True,
    )
    agent.compression_in_place = False
    agent._compression_feasibility_checked = True
    agent.context_compressor = ContextCompressor(
        model="test/model", config_context_length=100_000,
        protect_first_n=1, protect_last_n=3, quiet_mode=True,
    )
    agent.context_compressor.tail_token_budget = 10
    return agent


def _terminal(agent, key, *, pty=False):
    args = {"command": 'printf "IDENTITY=%s\\n" "$HERMES_SESSION_ID"'}
    if pty:
        args.update(background=True, pty=True)
    call = SimpleNamespace(id="identity-probe", type="function", function=SimpleNamespace(
        name="terminal", arguments=json.dumps(args)))
    messages = []
    agent._execute_tool_calls(SimpleNamespace(tool_calls=[call]), messages, key)
    result = json.loads(messages[-1]["content"])
    assert not result.get("error"), result
    if pty:
        process = process_registry.get(result["session_id"])
        assert process is not None
        result = process_registry.wait(process.id, timeout=15)
        assert result["status"] == "exited", result
    assert result["exit_code"] == 0, result
    return result["output"].strip()


def _compress(agent, db):
    previous = agent.session_id
    filler = "generic context " * 200
    messages = [{"role": "system", "content": "generic system prompt"}]
    for i in range(12):
        messages.extend([
            {"role": "user", "content": f"Question {i} {filler}"},
            {"role": "assistant", "content": f"Answer {i} {filler}"},
        ])
    compressed, _ = agent._compress_context(
        messages, "generic system prompt", approx_tokens=80_000, force=True,
    )
    assert agent.context_compressor._identity_test_summary_thread != get_ident()
    assert len(compressed) < len(messages)
    assert agent.session_id != previous
    assert db.get_session(previous)["ended_at"] is not None
    assert db.get_session(agent.session_id)["parent_session_id"] == previous
    assert db.get_session(agent.session_id)["ended_at"] is None
    assert get_session_env("HERMES_SESSION_ID") == agent.session_id


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("pty", [False, True])
@pytest.mark.parametrize("surface", ["unbound", "gateway-bound"])
def test_compression_refreshes_cached_terminal_identity(runtime, pty, surface):
    # The unbound variant models the core single-query caller context; it does
    # not launch the CLI. Both variants run real compression and tool dispatch.
    def run():
        key = f"identity-{surface}-{pty}"
        if surface == "gateway-bound":
            set_session_vars(session_id="parent-session", session_key=key, platform="telegram")
        agent = _agent(runtime, "parent-session")
        try:
            assert _terminal(agent, key, pty=pty) == "IDENTITY=parent-session"
            cached = get_active_env(key)
            assert cached is not None
            _compress(agent, runtime)
            assert _terminal(agent, key, pty=pty) == f"IDENTITY={agent.session_id}"
            assert get_active_env(key) is cached
        finally:
            cleanup_vm(key, force_remove=True)
    Context().run(run)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("pty", [False, True])
def test_concurrent_children_compress_without_replacing_parent(runtime, pty):
    Context().run(_concurrent_children, runtime, pty)


def _concurrent_children(runtime, pty):
    parent = _agent(runtime, "parent-session")
    barrier = Barrier(2)

    def child(index):
        key = f"identity-child-{index}-{pty}"
        with scoped_current_session_id(parent.session_id):
            before = _SESSION_ID.get()
            with delegated_child_context():
                agent = _agent(runtime, f"child-session-{index}")
            assert _SESSION_ID.get() == before
            try:
                with delegated_child_context(agent.session_id):
                    assert _terminal(agent, key, pty=pty) == f"IDENTITY={agent.session_id}"
                    cached = get_active_env(key)
                    barrier.wait(timeout=20)
                    _compress(agent, runtime)
                    barrier.wait(timeout=20)
                    assert _terminal(agent, key, pty=pty) == f"IDENTITY={agent.session_id}"
                    assert get_active_env(key) is cached
                    assert os.environ["HERMES_SESSION_ID"] == parent.session_id
                assert get_session_env("HERMES_SESSION_ID") == parent.session_id
            finally:
                cleanup_vm(key, force_remove=True)

    try:
        assert _terminal(parent, "identity-parent", pty=pty) == "IDENTITY=parent-session"
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(child, range(2)))
        assert get_session_env("HERMES_SESSION_ID") == parent.session_id
        assert _terminal(parent, "identity-parent", pty=pty) == "IDENTITY=parent-session"
    finally:
        cleanup_vm("identity-parent", force_remove=True)
