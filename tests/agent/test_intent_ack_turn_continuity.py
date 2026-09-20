"""Regression for #69778: prior work and answered clarification are not execution.

The negative prose cases include ClaySecAI's governed-announcement examples
from #69779. Provider replies are scripted; the agent loop and tool dispatch are real.
"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


REQUEST = "Please implement the panel highlight in the project."
ACK = (
    "Understood—the selection controls the panel highlight, not the camera. "
    "I’ll show the highlight only while the panel is open and keep it restrained. "
    "Camera tracking remains separate."
)
ANSWER = {"responses": [{"question": "When should the highlight appear?",
                          "user_response": "Only while the panel is open."}]}


def _call(name, call_id="call_1", arguments=None):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {})}}


def _round(name, result, call_id="call_1"):
    return [
        {"role": "assistant", "content": "", "tool_calls": [_call(name, call_id)]},
        {"role": "tool", "tool_call_id": call_id, "name": name,
         "content": json.dumps(result)},
    ]


HISTORY = [
    {"role": "user", "content": "Read the project notes."},
    *_round("read_file", {"content": "Panel notes"}, "old_call"),
    {"role": "assistant", "content": "The notes describe a panel."},
]


@pytest.mark.parametrize("require_workspace", [True, False])
def test_only_unstarted_authorized_work_is_an_ack(require_workspace):
    agent = SimpleNamespace(_strip_think_blocks=lambda text: text)
    current = [{"role": "user", "content": REQUEST}]
    clarified = current + _round("clarify", ANSWER)
    cases = [
        (REQUEST, "Let me inspect the repository files first.", HISTORY + current, True),
        (REQUEST, ACK, HISTORY + clarified, True),
        (REQUEST, ACK, clarified, True),
        (REQUEST, ACK, current + _round("clarify", {**ANSWER, "timed_out": True}), False),
        (REQUEST, ACK, current + _round("clarify", {"responses": []}), False),
        (REQUEST, ACK, current + _round("clarify", {"error": "unavailable"}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": "Cancel the task."}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": "No, do not change it."}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": ""}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": "Wait for my approval."}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": "I need to supply credentials first."}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": "Declined."}), False),
        (REQUEST, ACK, current + _round("clarify", {"user_response": "The user did not provide a response within the time limit."}), False),
        ("Do not inspect the project yet.", "I'll inspect the files.", current, False),
        ("Explain the project options; do not edit anything.", ACK, clarified, False),
        (REQUEST, ACK, clarified + _round("write_file", {"success": True}, "work"), False),
        (REQUEST, ACK, clarified + _round("delegate_task", {"status": "running"}, "work"), False),
        (REQUEST, ACK, clarified + _round("terminal", {"session_id": "job"}, "work"), False),
        (REQUEST, "I'll inspect the files.", current + _round("read_file", {"content": "checked"}) + [
            {"role": "assistant", "content": "Checked the file."},
            {"role": "user", "content": "Verify the result.", "_verification_stop_synthetic": True},
        ], False),
        (REQUEST, ACK, current + _round("clarify", ANSWER)[:1], False),
    ]
    # Optional offers, refusals, quotations and completed answers are not retries.
    for text in (
        "Done. I'll inspect the remaining files another time.",
        "I'll inspect the files if you want.",
        "I'll inspect the files after your approval.",
        "I'll inspect the files once you provide credentials.",
        "I'll inspect the files, but first I need your password.",
        "I'll inspect the files after the background job finishes.",
        "I'll never run that command on prod.",
        "I'll admit, building rapport with a new team takes time.",
        "I'll check in with you next week.",
        "I'll run through my reasoning first.",
        "The example reply is: \"I'll inspect the files.\"",
        "> I'll inspect the files.",
        "`I'll inspect the files.`",
        "I'll remember that bread belongs in the pantry.",
        "Let me know if you'd like me to inspect the repository files.",
        "I'll inspect the files. The requested review is complete.",
        "I'll inspect the files. Should I proceed?",
        "I'll inspect the files once you give the go-ahead.",
        "I'll inspect the files when you send the details.",
        "Here is the example. 'I'll inspect the files.'",
        "Here is the example. ‘I'll inspect the files.’",
        "Here is the example: 'First step. I'll inspect the files.'",
        "Here is the example: ‘First step. I’ll inspect the files.’",
    ):
        cases.append((REQUEST, text, HISTORY + current, False))
    for request, text, messages, expected in cases:
        before = copy.deepcopy(messages)
        assert AIAgent._looks_like_codex_intermediate_ack(
            agent, request, text, messages, require_workspace=require_workspace
        ) is expected, (request, text, messages)
        assert messages == before


def _response(text="", tool_calls=None):
    output = [SimpleNamespace(
        type="function_call", id=call["id"], call_id=call["id"],
        name=call["function"]["name"], arguments=call["function"]["arguments"],
    ) for call in tool_calls or []]
    if text:
        output.append(SimpleNamespace(
            type="message", status="completed", phase="final_answer",
            content=[SimpleNamespace(type="output_text", text=text)],
        ))
    return SimpleNamespace(
        output=output, status="completed", model="test/model", usage=None,
    )


@pytest.mark.parametrize("scenario", [
    "history", "clarified", "declined", "timeout", "cap", "process", "delegation", "off", "executed",
])
def test_real_loop_retries_without_rewriting_history_or_repeating_work(tmp_path, monkeypatch, scenario):
    from tools.clarify_tool import CLARIFY_SCHEMA

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    # A harmless real file read proves the recovery reaches dispatch exactly once.
    target = tmp_path / "panel.txt"
    target.write_text("highlight only while open", encoding="utf-8")
    from tools.file_tools import READ_FILE_SCHEMA
    with (
        patch("model_tools.get_tool_definitions", return_value=[
            {"type": "function", "function": schema} for schema in [CLARIFY_SCHEMA, READ_FILE_SCHEMA]
        ]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://example.invalid/v1", provider="openai-compat",
            model="test/model", max_iterations=8, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    agent.api_mode = "codex_responses"
    agent._cached_system_prompt = "Stable test system prompt."
    agent._session_db = None
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent.tool_delay = 0
    if scenario == "off":
        agent._intent_ack_continuation = False
    if scenario == "process":
        from tools.process_registry import ProcessSession, process_registry
        agent._process_owner_task_ids = {"earlier-task"}
        monkeypatch.setattr(process_registry, "_running", {
            "earlier-process": ProcessSession(
                id="earlier-process", command="test job", owner_task_id="earlier-task",
            ),
        })
    if scenario == "delegation":
        from tools import async_delegation
        monkeypatch.setattr(async_delegation, "_records", {
            "earlier-child": {"parent_session_id": agent.session_id, "status": "running"},
        })
    answer = "Cancel the task." if scenario == "declined" else "Only while the panel is open."
    def clarify_callback(question, choices, **kwargs):
        return {"answers": {"q0": answer}, "timed_out": scenario == "timeout"}
    agent.clarify_callback = clarify_callback
    clarify_call = _call("clarify", arguments={"questions": [{"question": "When should the highlight appear?"}]})
    read_call = _call("read_file", "read_panel", {"path": str(target)})
    ack = "Let me inspect the repository files first." if scenario == "history" else ACK
    stages = [] if scenario == "history" else [_response(tool_calls=[clarify_call])]
    if scenario == "executed":
        stages += [_response(tool_calls=[read_call])]
    stages += [_response(ack)] * (3 if scenario == "cap" else 1)
    if scenario in {"history", "clarified"}:
        stages += [_response(tool_calls=[read_call]), _response("The panel setting has been checked.")]
    responses = iter(stages)
    sent = []
    def model_call(kwargs):
        sent.append(copy.deepcopy(kwargs))
        return next(responses)
    agent._interruptible_api_call = model_call
    history = copy.deepcopy(HISTORY)
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation(REQUEST, conversation_history=history)
    assert len(sent) == len(stages)
    assert history == HISTORY
    if scenario in {"history", "clarified"}:
        assert result["final_response"] == "The panel setting has been checked."
        reads = [m for m in result["messages"] if m.get("tool_call_id") == "read_panel"]
        assert len(reads) == 1
        assert "highlight only while open" in reads[0]["content"]
    else:
        assert result["final_response"] == ack
    assert sum(m.get("tool_call_id") == "read_panel" for m in result["messages"]) == (
        1 if scenario in {"history", "clarified", "executed"} else 0
    )
    # Recovery replays the existing request, never a fabricated user instruction.
    ack_index = 0 if scenario == "history" else 1
    if scenario in {"history", "clarified", "cap"}:
        assert sent[ack_index] == sent[ack_index + 1]
    rows = result["messages"]
    assert [m["content"] for m in rows if m["role"] == "user"] == [HISTORY[0]["content"], REQUEST]
    assert all(a["role"] != b["role"] or a["role"] == "tool" for a, b in zip(rows, rows[1:]))
