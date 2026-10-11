"""Opt-in CLI cache reuse preserves content addressing and transcript isolation (#136359)."""

from types import SimpleNamespace

import pytest

from agent.prompt_cache_scope import declared_conversation_scope, resolve_prompt_cache_scope
from agent.transports import get_transport
from hermes_cli.cli_cache_scope import bind_cli_cache_scope, configure_cli_cache_scope
from hermes_state import SessionDB


def _agent(sid, label):
    agent = SimpleNamespace(session_id=sid, _session_db=None, _gateway_session_key=None)
    cli = SimpleNamespace()
    configure_cli_cache_scope(cli, label)
    bind_cli_cache_scope(cli, agent)
    return agent


def _request(agent, *, instructions="Stable instructions", tools=None, query="first"):
    return get_transport("codex_responses").build_kwargs(
        model="gpt-5.4", provider="openai-codex", is_codex_backend=True,
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": query}],
        tools=tools, session_id=agent.session_id,
        cache_scope_id=resolve_prompt_cache_scope(agent),
        cache_affinity_id=getattr(agent, "_cli_prompt_cache_scope", None),
    )


def test_opt_in_reuses_key_and_codex_header_but_not_session_identity():
    first, second = _agent("physical-a", "drafts"), _agent("physical-b", "drafts")
    a, b = _request(first), _request(second, query="different query")
    assert a["prompt_cache_key"] == b["prompt_cache_key"]
    assert a["extra_headers"]["session_id"] == b["extra_headers"]["session_id"]
    assert a["extra_headers"]["x-client-request-id"] == a["prompt_cache_key"]
    assert first.session_id != second.session_id
    assert declared_conversation_scope(first) is None
    assert first._gateway_session_key is second._gateway_session_key is None


def test_absent_option_preserves_per_session_body_and_header():
    first, second = _agent("physical-a", None), _agent("physical-b", None)
    a, b = _request(first), _request(second)
    assert a["prompt_cache_key"] != b["prompt_cache_key"]
    assert a["extra_headers"]["session_id"] == "physical-a"
    assert b["extra_headers"]["session_id"] == "physical-b"


def test_scope_instructions_and_tools_still_split_keys():
    agent = _agent("physical-a", "drafts")
    baseline = _request(agent)
    other_scope = _request(_agent("physical-b", "another-label"))
    other_instructions = _request(agent, instructions="Other instructions")
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    other_tools = _request(agent, tools=[tool])
    assert len({request["prompt_cache_key"] for request in (baseline, other_scope, other_instructions, other_tools)}) == 4


def test_compression_rotation_stays_in_explicit_scope():
    agent = _agent("physical-a", "drafts")
    before = _request(agent)
    agent.session_id = "rotated-a"
    assert _request(agent)["prompt_cache_key"] == before["prompt_cache_key"]


def test_independent_children_do_not_inherit_opt_in():
    parent = _agent("parent", "drafts")
    child = SimpleNamespace(session_id="delegate-child", _session_db=None)
    assert resolve_prompt_cache_scope(child) != resolve_prompt_cache_scope(parent)
    assert _request(child)["extra_headers"]["session_id"] == "delegate-child"


def test_non_codex_route_does_not_receive_codex_affinity_header():
    agent = _agent("physical-a", "drafts")
    request = get_transport("codex_responses").build_kwargs(
        model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1",
        messages=[{"role": "user", "content": "Hi"}],
        session_id=agent.session_id, cache_scope_id=resolve_prompt_cache_scope(agent),
        cache_affinity_id=agent._cli_prompt_cache_scope,
    )
    assert "session_id" not in request.get("extra_headers", {})


def test_real_agents_build_shared_wire_requests_in_profile_a_b_a(tmp_path, monkeypatch):
    """Real constructor, request assembly and SQLite; no live provider credentials/API."""
    from run_agent import AIAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    requests = []
    for index, profile in enumerate(("a", "b", "a")):
        home = tmp_path / profile
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(db_path=home / "state.db")
        agent = AIAgent(
            model="gpt-5.4", provider="custom", api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex", api_key="synthetic-test-key",
            session_id=f"physical-{index}", session_db=db, quiet_mode=True,
            enabled_toolsets=[], disabled_toolsets=["all"], save_trajectories=False,
            skip_memory=True, skip_context_files=True, skip_background_review=True,
        )
        try:
            cli = SimpleNamespace()
            configure_cli_cache_scope(cli, "drafts")
            bind_cli_cache_scope(cli, agent)
            request = agent._build_api_kwargs([
                {"role": "system", "content": "Stable instructions"},
                {"role": "user", "content": f"query-{index}"},
            ], [])
            requests.append(request)
            db.create_session(agent.session_id, source="cli")
            db.append_message(agent.session_id, "user", content=f"query-{index}")
            assert agent._gateway_session_key is None
            assert db.get_session(agent.session_id)["session_key"] is None
            assert [m["content"] for m in db.get_messages(agent.session_id)] == [f"query-{index}"]
        finally:
            agent.close()
            db.close()
    assert requests[0]["prompt_cache_key"] == requests[2]["prompt_cache_key"] != requests[1]["prompt_cache_key"]
    assert requests[0]["extra_headers"]["session_id"] == requests[2]["extra_headers"]["session_id"]
