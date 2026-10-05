"""Codex false-stop regression after clarification (related to #69778).

Provider responses and the platform's clarification callback are scripted;
discovery, client bootstrap and unrelated UI hooks are isolated. The AIAgent
loop, Responses adapter, tool round, file read and SQLite persistence are real.
All files belong to pytest's temporary home; no implementation target runs.
A visible final_answer/stop must not mistake an accepted correction for
completion, nor recover through a user's decline or intentional wait.
"""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools.clarify_tool import CLARIFY_SCHEMA


USER = "Proceed with implementing the selection panel behavior in the repository."
CORRECTION = "Trace selection means route highlighting, not viewport tracking."
ACK = (
    "Understood—Trace selection controls route highlighting, not viewport tracking. "
    "I’ll show that highlight only while its panel is open, keep it subtle in normal "
    "lighting, and avoid bright links in dark mode. Inspect + Focus remains a "
    "separate camera feature."
)
GOVERNED_ACK = "Understood. I’ll inspect the repository and implement the panel behavior."
TAIL_ACK = "Understood. I’ll now inspect the repository."
QUESTION = "Does Trace selection control viewport tracking?"


def _response(text=None, *, name=None, arguments=None, call_id="clarification"):
    if name:
        item = SimpleNamespace(
            type="function_call", id=f"fc_{call_id}", call_id=call_id,
            name=name, arguments=json.dumps(arguments),
        )
    else:
        item = SimpleNamespace(
            type="message", role="assistant", phase="final_answer", status="completed",
            content=[SimpleNamespace(type="output_text", text=text)],
        )
    return SimpleNamespace(
        output=[item], status="completed", model="gpt-5-codex", usage=None,
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_kw: None)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("clarification-stop", source="cli")
    fixture = tmp_path / "panel-fixture.txt"
    fixture.write_text("selection-panel-fixture\n", encoding="utf-8")
    schemas = [
        {"type": "function", "function": CLARIFY_SCHEMA},
        {"type": "function", "function": {
            "name": "read_file", "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        }},
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=schemas),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            model="gpt-5-codex", provider="openai-compat", api_mode="codex_responses",
            base_url="https://example.invalid/v1", api_key="test-key",
            session_id="clarification-stop", session_db=db, max_iterations=10,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "Read only the supplied temporary fixture when work resumes."
    agent.compression_enabled = False
    agent.save_trajectories = False
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda *_a, **_kw: False)
    yield SimpleNamespace(agent=agent, db=db, fixture=fixture)
    db.close()


def _run(harness, texts, *, answer=CORRECTION, resume=False, interrupt=False,
         user=USER, history=None):
    agent = harness.agent
    questions = []

    def clarify(question, choices):
        questions.append((question, choices))
        return answer

    agent.clarify_callback = clarify
    responses = [_response(name="clarify", arguments={"questions": [{"question": QUESTION}]})]
    responses.extend(_response(text) for text in texts)
    if resume:
        responses.extend([
            _response(name="read_file", arguments={"path": str(harness.fixture)}, call_id="work"),
            _response("Read the supplied fixture; the next implementation step is identified."),
        ])
    # A forbidden extra call gets an ordinary answer, not an exhausted mock's
    # StopIteration/retry error. RED must identify the continuation decision.
    responses.append(_response("Unexpected automatic recovery reached the provider."))
    requests = []
    scripted = iter(responses)

    def provider(kwargs):
        requests.append(deepcopy(kwargs))
        response = next(scripted)
        if interrupt and len(requests) == 2:
            agent.interrupt(hard_cancel=True)
        return response

    agent._interruptible_api_call = provider
    result = agent.run_conversation(user, conversation_history=history)
    durable = harness.db.get_messages_as_conversation(agent.session_id)
    # Establish a genuine answered tool round before testing continuation policy.
    assert len(questions) == 1
    clarification = next(m for m in durable if m.get("tool_call_id") == "clarification")
    assert json.loads(clarification["content"])["responses"][0]["user_response"] == answer
    assert result["api_calls"] == len(requests)
    assert result.get("error") is None
    # Every request preserves the already-cached prefix and tool surface verbatim.
    for previous, request in zip(requests, requests[1:]):
        assert request["instructions"] == requests[0]["instructions"]
        assert request["tools"] == requests[0]["tools"]
        assert request["input"][:len(previous["input"])] == previous["input"]
    return result, durable, requests


def _assert_transcript_contract(result, durable):
    user_rows = {
        label: [m["content"] for m in rows if m["role"] == "user"]
        for label, rows in (("returned", result["messages"]), ("sqlite", durable))
    }
    assert user_rows == {"returned": [USER], "sqlite": [USER]}
    for rows in (result["messages"], durable):
        pending = set()
        previous = None
        for row in rows:
            role = row["role"]
            assert role != previous or role == "tool", "illegal adjacent roles"
            if role == "assistant":
                assert not pending, "tool calls must receive their results"
                pending.update(call["id"] for call in row.get("tool_calls", []))
            elif role == "tool":
                assert row["tool_call_id"] in pending
                pending.remove(row["tool_call_id"])
            previous = role
        assert not pending
    assert durable[-1]["content"] == result["final_response"]


@pytest.mark.parametrize("texts,resume", [
    pytest.param([], True, id="real-tool-control"),
    pytest.param([ACK], True, id="generic-correction"),
    pytest.param([GOVERNED_ACK], True, id="governed-action"),
    pytest.param([TAIL_ACK], True, id="existing-stall-path-hygiene"),
    pytest.param(["The scope is clear. Next, I inspect the repository."], True, id="existing-next-tail"),
    pytest.param([TAIL_ACK] * 3, False, id="bounded-two-continuations"),
])
def test_answered_clarification_resumes_work_with_bounded_clean_history(harness, texts, resume):
    result, durable, requests = _run(harness, texts, resume=resume)
    if resume:
        work = [m for m in durable if m.get("role") == "tool" and m.get("tool_call_id") == "work"]
        assert work, f"answered clarification ended as completed={result['completed']} after {len(requests)} calls"
        assert "selection-panel-fixture" in work[0]["content"]
        assert not texts or result["final_response"] != texts[-1]
    else:
        # One clarify call, the initial ack, then no more than two recoveries.
        assert len(requests) == 4
    _assert_transcript_contract(result, durable)


@pytest.mark.parametrize("answer,text,interrupt", [
    pytest.param(CORRECTION, "The requested task is complete. No changes remain.", False, id="complete"),
    pytest.param("Do not implement anything. Stop here.", "Understood. I’ll inspect the repository when you return.", False, id="declined"),
    pytest.param(CORRECTION, "I’ll inspect the repository after you approve the change.", False, id="approval-wait"),
    pytest.param(CORRECTION, "Which file should I inspect?", False, id="question"),
    pytest.param(CORRECTION, "The background task is still running. I’ll review the results when it finishes.", False, id="background-wait"),
    pytest.param(CORRECTION, "The panel behavior is documented. If you want, I’ll inspect the files later.", False, id="optional-offer"),
    pytest.param(CORRECTION, GOVERNED_ACK, True, id="cancel-race"),
    pytest.param("Do not implement anything. Stop here.", TAIL_ACK, False, id="tail-declined"),
    pytest.param("Wait for my decision.", TAIL_ACK, False, id="tail-wait"),
    pytest.param(CORRECTION, TAIL_ACK, True, id="tail-cancel-race"),
])
def test_clarification_terminal_boundaries_do_not_spend_recovery(harness, answer, text, interrupt):
    result, durable, requests = _run(harness, [text], answer=answer, interrupt=interrupt)
    assert len(requests) == 2, "a decline, wait, completion or cancellation must not be continued"
    assert not any(m.get("tool_call_id") == "work" for m in durable)
    if not interrupt:
        assert result["final_response"] == text
        _assert_transcript_contract(result, durable)


@pytest.mark.parametrize("live_work", [False, "process", "delegation"])
def test_conversational_resume_keeps_unstarted_task_context(harness, monkeypatch, live_work):
    history = [
        {"role": "user", "content": "Implement the selection panel in the repository."},
        {"role": "assistant", "content": "The panel can highlight the selected route without moving the viewport."},
    ]
    # Establish the discussion through the real loop/DB, not only an in-memory
    # history supplied to a fresh DB (which correctly persists only new rows).
    harness.agent._interruptible_api_call = lambda _kwargs: _response(history[1]["content"])
    history = harness.agent.run_conversation(history[0]["content"])["messages"]
    user = "That approach sounds reasonable; please proceed."
    if live_work == "process":
        from tools.process_registry import ProcessSession, process_registry
        harness.agent._process_owner_task_ids = {"earlier-task"}
        monkeypatch.setattr(process_registry, "_running", {
            "earlier-process": ProcessSession(
                id="earlier-process", command="test job", owner_task_id="earlier-task",
            ),
        })
    elif live_work == "delegation":
        from tools import async_delegation
        monkeypatch.setattr(async_delegation, "_records", {
            "earlier-child": {"parent_session_id": harness.agent.session_id, "status": "running"},
        })
    ack = TAIL_ACK if live_work else ACK
    result, durable, requests = _run(
        harness, [ack], user=user, history=history, resume=not live_work,
    )
    assert len(requests) == (2 if live_work else 4)
    for rows in (result["messages"], durable):
        assert [m["content"] for m in rows if m["role"] == "user"] == [history[0]["content"], user]
        assert sum(m.get("tool_call_id") == "work" for m in rows) == (0 if live_work else 1)
        assert all(a["role"] != b["role"] or a["role"] == "tool" for a, b in zip(rows, rows[1:]))
    if not live_work:
        assert requests[1] == requests[2]
    assert durable[-1]["content"] == result["final_response"]


def test_nonclarification_stall_still_resumes(harness):
    responses = iter([
        _response(TAIL_ACK),
        _response(name="read_file", arguments={"path": str(harness.fixture)}, call_id="work"),
        _response("Read the fixture."),
    ])
    harness.agent._interruptible_api_call = lambda _kwargs: next(responses)
    result = harness.agent.run_conversation(USER)
    assert result["api_calls"] == 3
    assert result["final_response"] == "Read the fixture."
    assert sum(m.get("tool_call_id") == "work" for m in result["messages"]) == 1
