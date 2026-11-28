"""Opt-in stateful histories must not use Portal's shared delegate-tree identity."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import hermes_yaml as yaml

from agent.chat_completion_helpers import build_api_kwargs
from agent.opencode_affinity import opencode_session_headers
from agent.portal_tags import (
    get_conversation_context, reset_affinity_scope, reset_conversation_context,
    set_affinity_scope, set_conversation_context,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from run_agent import AIAgent

BASE = "http://127.0.0.1:4000/v1"
HEADER = "X-History-Session"
MESSAGES = [{"role": "user", "content": "synthetic task"}]


@contextmanager
def ambient(root="parent", declared=None):
    conversation = set_conversation_context(root)
    affinity = set_affinity_scope(declared)
    try:
        yield
    finally:
        reset_affinity_scope(affinity)
        reset_conversation_context(conversation)


@pytest.fixture
def history_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    home = tmp_path / "profile-a"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"providers": {"history-proxy": {
        "api": BASE, "api_key": "synthetic-key", "discover_models": False,
        "session_affinity_header": HEADER, "context_length": 256000,
    }}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def history_db(history_home):
    db = SessionDB(db_path=history_home / "state.db")
    db.create_session("parent", source="cli")
    for sid, marker in (("child-a", "_delegate_from"), ("child-b", "_delegate_from"),
                        ("branch", "_branched_from")):
        db.create_session(sid, source="subagent", parent_session_id="parent",
                          model_config={marker: "parent"})
    try:
        yield db
    finally:
        db.close()


def agent_for(db, sid, mode="chat_completions", **kwargs):
    base = kwargs.pop("base_url", BASE)
    agent = AIAgent(api_key="synthetic-key", base_url=base, provider="history-proxy",
                    model="test-model", api_mode=mode, session_id=sid, session_db=db,
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    enabled_toolsets=[], **kwargs)
    agent._anthropic_base_url = base
    return agent


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses", "anthropic_messages"])
def test_independent_children_keep_history_keys_through_compression(history_db, mode):
    db = history_db
    with ambient():
        agents = [agent_for(db, sid, mode) for sid in ("parent", "child-a", "child-b", "branch")]
        before = [build_api_kwargs(agent, MESSAGES)["extra_headers"][HEADER] for agent in agents]
        assert len(set(before)) == 4, before
        # Portal and OpenCode continue sharing the parent for attribution/sticky routing.
        assert get_conversation_context() == "parent"
        assert opencode_session_headers("opencode-go", None, "child-a")["x-opencode-session"] == "parent"
        db.end_session("child-a", "compression")
        db.create_session("child-a-rotated", source="subagent", parent_session_id="child-a")
        agents[1].session_id = "child-a-rotated"
        assert build_api_kwargs(agents[1], MESSAGES)["extra_headers"][HEADER] == before[1]
        # A fresh process/agent resumes the same compression lineage.
        assert build_api_kwargs(agent_for(db, "child-a-rotated", mode), MESSAGES)["extra_headers"][HEADER] == before[1]


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses", "anthropic_messages"])
@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from"])
def test_declared_host_key_forks_keep_history_through_compression_resume(history_db, mode, marker):
    from agent.prompt_cache_scope import declared_conversation_scope

    db = history_db
    # Same source and host key: falling back to the declaration would collide
    # with the parent, not merely change the child's history to another key.
    for sid in ("declared-child", "declared-sibling"):
        db.create_session(sid, source="cli", parent_session_id="parent",
                          model_config={marker: "parent"})
    parent = agent_for(db, "parent", mode, gateway_session_key="synthetic-chat")
    child = agent_for(db, "declared-child", mode, gateway_session_key="synthetic-chat")
    sibling = agent_for(db, "declared-sibling", mode, gateway_session_key="synthetic-chat")
    declared = declared_conversation_scope(parent)
    assert declared and declared.startswith("gwk_")
    with ambient(declared=declared):
        def history_key(agent):
            return build_api_kwargs(agent, MESSAGES)["extra_headers"][HEADER]

        before = [history_key(agent) for agent in (parent, child, sibling)]
        assert before == [declared, "declared-child", "declared-sibling"]
        observed = []
        # Compression successors do not carry a fresh fork marker. Resolve both
        # the live rotated agent and a new agent on a newly opened real DB.
        for sid in ("declared-child-1", "declared-child-2"):
            db.end_session(child.session_id, "compression")
            db.create_session(sid, source="cli", parent_session_id=child.session_id)
            child.session_id = sid
            observed.append(history_key(child))
            resumed_db = SessionDB(db_path=db.db_path)
            try:
                resumed = agent_for(resumed_db, sid, mode, gateway_session_key="synthetic-chat")
                observed.append(history_key(resumed))
            finally:
                resumed_db.close()
        # Exercise parent compression too: the fix must retain declarations for
        # non-fork conversations, not replace all host keys with physical ids.
        db.end_session("parent", "compression")
        db.create_session("parent-rotated", source="cli", parent_session_id="parent")
        parent.session_id = "parent-rotated"
        assert history_key(parent) == declared
        assert history_key(sibling) == before[2]
        assert opencode_session_headers("opencode-go", None, child.session_id)["x-opencode-session"] == declared
        assert get_conversation_context() == "parent"
        assert observed == [before[1]] * 4, {"before": before, "after": observed}


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses", "anthropic_messages"])
def test_case_insensitive_caller_pin_wins(history_db, mode):
    agent = agent_for(history_db, "child-a", mode)
    agent.request_overrides = {"extra_headers": {HEADER.lower(): "caller-pinned"}}
    with ambient():
        headers = build_api_kwargs(agent, MESSAGES)["extra_headers"]
    assert {k: v for k, v in headers.items() if k.lower() == HEADER.lower()} == {HEADER.lower(): "caller-pinned"}


def test_auxiliary_histories_are_one_shot_not_main_or_sibling_histories(history_db):
    from agent import auxiliary_client as aux

    with ambient(), aux.scoped_runtime_main({
        "provider": "history-proxy", "model": "test-model", "base_url": BASE,
        "session_id": "child-a", "cache_scope": "child-a",
    }):
        main_key = build_api_kwargs(agent_for(history_db, "child-a"), MESSAGES)["extra_headers"][HEADER]
        keys = [aux._build_call_kwargs("history-proxy", "test-model", MESSAGES,
                base_url=BASE, task=task)["extra_headers"][HEADER]
                for task in ("compression", "title_generation", "compression")]
        assert main_key not in keys
        assert len(set(keys)) == 3
        # Sticky OpenCode routing still uses the shared parent.
        assert opencode_session_headers("opencode-go", None)["x-opencode-session"] == "parent"


@pytest.fixture
def history_endpoint(history_home):
    """Loopback-only synthetic protocol peer; no provider/model calls."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            response = {"id": "synthetic-response", "object": "chat.completion", "created": 1,
                        "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Synthetic answer."}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            if body.get("stream"):
                chunk = {"id": "synthetic-response", "object": "chat.completion.chunk", "created": 1,
                         "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                         "delta": {"role": "assistant", "content": "Synthetic answer."}}]}
                data = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                data = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    config = yaml.safe_load((history_home / "config.yaml").read_text())
    config["providers"]["history-proxy"]["api"] = base
    (history_home / "config.yaml").write_text(yaml.safe_dump(config))
    try:
        yield base, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_real_agent_turns_and_auxiliary_wire_do_not_rebase_other_histories(history_db, history_endpoint):
    from agent.auxiliary_client import call_llm, scoped_runtime_main

    base, requests = history_endpoint
    a = agent_for(history_db, "child-a", base_url=base, platform="subagent")
    b = agent_for(history_db, "child-b", base_url=base, platform="subagent")
    result_a = a.run_conversation("Synthetic task A")
    result_b = b.run_conversation("Synthetic task B")
    for result in (result_a, result_b):
        assert result["final_response"] == "Synthetic answer."
    import time

    with ambient(), scoped_runtime_main({"provider": "history-proxy", "model": "test-model",
        "base_url": base, "api_key": "synthetic-key", "session_id": "child-a", "cache_scope": "child-a"}):
        for task in ("compression", "title_generation", "compression"):
            if task == "compression":
                assert a.context_compressor._call_summary_llm(
                    "Synthetic independent summary request", time.monotonic()) == "Synthetic answer."
            else:
                response = call_llm(task=task, messages=MESSAGES, provider="history-proxy", model="test-model",
                                    base_url=base, api_key="synthetic-key", timeout=10)
                assert response.choices[0].message.content == "Synthetic answer."
    resumed = a.run_conversation("Synthetic continuation A", conversation_history=result_a["messages"])
    assert resumed["final_response"] == "Synthetic answer."
    headers = [next(value for key, value in header.items() if key.lower() == HEADER.lower())
               for _, header, _ in requests]
    assert headers[:2] == ["child-a", "child-b"]
    assert headers[-1] == "child-a"
    assert len(set(headers[2:-1])) == 3
    assert not set(headers[2:-1]) & {"child-a", "child-b", "parent"}
    assert all(path == "/v1/chat/completions" for path, _, _ in requests)
    first_history = requests[0][2]["messages"]
    assert requests[-1][2]["messages"][:len(first_history)] == first_history
    assert get_conversation_context() is None


def test_real_config_profile_a_b_a_does_not_leak_opt_in_headers(history_home, history_db, tmp_path):
    from agent import auxiliary_client as aux

    b_home = tmp_path / "profile-b"
    b_home.mkdir()
    (b_home / "config.yaml").write_text(yaml.safe_dump({"providers": {"history-proxy": {
        "api": BASE, "api_key": "synthetic-key", "discover_models": False,
        "context_length": 256000,
    }}}))
    with ambient():
        assert build_api_kwargs(agent_for(history_db, "child-a"), MESSAGES)["extra_headers"][HEADER] == "child-a"
        token = set_hermes_home_override(b_home)
        try:
            assert HEADER not in build_api_kwargs(agent_for(history_db, "child-a"), MESSAGES).get("extra_headers", {})
            assert HEADER not in aux._build_call_kwargs("history-proxy", "test-model", MESSAGES,
                                                       base_url=BASE).get("extra_headers", {})
        finally:
            reset_hermes_home_override(token)
        assert build_api_kwargs(agent_for(history_db, "child-a"), MESSAGES)["extra_headers"][HEADER] == "child-a"
        assert HEADER not in build_api_kwargs(agent_for(history_db, "child-a", base_url="https://unconfigured.invalid/v1"),
                                              MESSAGES).get("extra_headers", {})


def test_host_declared_conversation_scope_is_preserved_but_children_are_separate(history_db):
    from agent.prompt_cache_scope import declared_conversation_scope

    parent = agent_for(history_db, "parent", gateway_session_key="synthetic-chat")
    child = agent_for(history_db, "child-a", gateway_session_key="synthetic-chat")
    declared = declared_conversation_scope(parent)
    assert declared and declared.startswith("gwk_")
    with ambient(declared=declared):
        assert build_api_kwargs(parent, MESSAGES)["extra_headers"][HEADER] == declared
        assert build_api_kwargs(child, MESSAGES)["extra_headers"][HEADER] == "child-a"
        assert opencode_session_headers("opencode-go", None, "child-a")["x-opencode-session"] == declared
    assert get_conversation_context() is None


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses", "anthropic_messages"])
def test_cache_parity_forks_keep_cache_routing_but_not_stateful_histories(history_db, mode):
    from agent.background_review import build_cache_parity_fork
    from agent.prompt_cache_scope import resolve_prompt_cache_scope

    parent = agent_for(history_db, "child-a", mode)
    first, _, _ = build_cache_parity_fork(parent, {}, max_iterations=1)
    second, _, _ = build_cache_parity_fork(parent, {}, max_iterations=1)
    # Cache-parity forks intentionally reuse even the physical session ID.
    assert first.session_id == second.session_id == parent.session_id
    assert resolve_prompt_cache_scope(first) == resolve_prompt_cache_scope(parent)
    with ambient():
        keys = [build_api_kwargs(agent, MESSAGES)["extra_headers"][HEADER]
                for agent in (parent, first, second)]
        assert len(set(keys)) == 3
        assert build_api_kwargs(first, MESSAGES)["extra_headers"][HEADER] == keys[1]
        assert resolve_prompt_cache_scope(first) == resolve_prompt_cache_scope(parent)
        assert opencode_session_headers("opencode-go", None, first.session_id)["x-opencode-session"] == "parent"
