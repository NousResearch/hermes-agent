"""Final SDK verification and trusted rotation retain exact attachment authority."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from tests.agent.test_run_agent import agent as _agent_fixture

agent = _agent_fixture


def _prepare(agent, tmp_path):
    from agent.context_references import preprocess_context_references
    from agent.source_provenance import admit_agent_context_sources, provenance_kwargs_for_agent
    source = tmp_path / "source.py"
    source.write_text("my_value = other_value + 123\n")
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent.api_mode = "chat_completions"
    agent.session_id = "parent"
    agent._llm_egress_state_dir = tmp_path / "egress"
    expanded = preprocess_context_references(
        "Inspect @file:source.py:1-1", cwd=tmp_path, allowed_root=tmp_path,
        context_length=240_000, **provenance_kwargs_for_agent(agent, establish_turn=True),
    )
    turn = agent._source_provenance_pending_turn_id
    agent._current_turn_id = turn
    agent._relay_pending_turn_id = turn
    agent._source_provenance_pending_turn_id = None
    admit_agent_context_sources(agent, turn_id=turn, prepared_session_id="parent")
    return source, expanded.message, turn


def _sdk(agent, captured):
    def io(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "lifecycle-completion", "object": "chat.completion", "created": 0,
            "model": agent.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })
    return OpenAI(api_key="synthetic-key", base_url=agent.base_url,
                  http_client=httpx.Client(transport=httpx.MockTransport(io)))


def _dispatch(agent, sdk, message, turn, number):
    from agent.llm_egress_runtime import dispatch_provider_request
    agent._current_api_request_id = f"{turn}:api:{number}"
    kwargs = agent._build_api_kwargs([{"role": "user", "content": message}], tools_for_api=[])
    return dispatch_provider_request(agent, kwargs,
                                     lambda request: sdk.chat.completions.create(**request))


@pytest.mark.parametrize("after", ["admission", "api2"])
def test_existing_context_grant_reverifies_before_first_or_repeated_sdk_io(
    agent, tmp_path, after,
):
    from agent.source_provenance import SourceProvenanceError
    source, message, turn = _prepare(agent, tmp_path)
    captured = []
    with _sdk(agent, captured) as sdk:
        if after == "api2":
            _dispatch(agent, sdk, message, turn, 1)
            _dispatch(agent, sdk, message, turn, 2)
        source.write_text("my_value = changed_value + 456\n")
        before = len(captured)
        with pytest.raises(SourceProvenanceError, match="content_mismatch"):
            _dispatch(agent, sdk, message, turn, 1 if after == "admission" else 2)
        assert len(captured) == before


@pytest.mark.parametrize("transition", ["recovery", "publication"])
def test_real_compression_transition_transfers_only_current_raw_bound_context(
    agent, tmp_path, transition,
):
    from agent.conversation_compression import (
        _publish_rotated_compaction, recover_rotated_compression_session,
    )
    from hermes_state import SessionDB
    source, message, turn = _prepare(agent, tmp_path)
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("parent", source="tui")
        db.append_message("parent", "user", "before compression")
        agent._session_db = db
        if transition == "recovery":
            db.end_session("parent", "compression")
            db.create_session("child", source="tui", parent_session_id="parent")
            db.append_message("child", "user", "compressed history")
            assert recover_rotated_compression_session(agent)
            assert agent.session_id == "child"
        else:
            _publish_rotated_compaction(
                agent, [], [{"role": "user", "content": "compressed history"}],
                new_system_prompt="", old_session_id="parent",
                compressed_user_turn_outcome="already_present",
                lease=SimpleNamespace(holder=None, ttl=10, watermark=None),
            )
            assert agent.session_id != "parent"
        captured = []
        with _sdk(agent, captured) as sdk:
            _dispatch(agent, sdk, message, turn, 1)
        assert len(captured) == 1
        grants = agent._source_provenance_registry.grants_for_request(f"{turn}:api:1")
        assert len(grants) == 1 and grants[0].session_id == agent.session_id
        assert grants[0].canonical_path == source.resolve()
        receipts = [json.loads(line) for line in
                    (tmp_path / "egress/llm-egress-receipts.jsonl").read_text().splitlines()]
        assert receipts[-1]["source_grant_count"] == 1


def test_cli_worker_callback_failure_clears_pending_after_thread_handoff(
    tmp_path, monkeypatch,
):
    from cli import HermesCLI, _ChatTurn
    from agent.context_references import preprocess_context_references
    from agent.source_provenance import provenance_kwargs_for_agent
    consumer = SimpleNamespace(session_id="parent")
    source = tmp_path / "source.py"
    source.write_text("my_value = 123\n")
    preprocess_context_references(
        "Inspect @file:source.py:1-1", cwd=tmp_path, allowed_root=tmp_path,
        context_length=240_000, **provenance_kwargs_for_agent(consumer, establish_turn=True),
    )
    pending = consumer._source_provenance_pending_turn_id
    host = HermesCLI.__new__(HermesCLI)
    host.agent = consumer
    host._sudo_password_callback = lambda *_a: None
    def failed_callback(*_a):
        raise RuntimeError("worker callback failed")
    monkeypatch.setattr("cli.set_sudo_password_callback", failed_callback)
    with pytest.raises(RuntimeError, match="worker callback failed"):
        host._chat_run_agent(_ChatTurn(), "prepared input")
    assert consumer._source_provenance_pending_turn_id is None
    assert not consumer._source_provenance_registry.grants_for_request(f"{pending}:api:1")
