"""Tool calls written as text instead of structured ``tool_calls``.

Local llama.cpp / vLLM routes serving Nemotron sometimes end a reply with
``finish_reason="stop"``, no ``tool_calls`` and the intended call left as markup in
``reasoning``/``content``. Before the fix the reasoning was promoted to the final answer (or the
prose accepted) and the turn ended "complete" with the action never run. The detector fires only
on markup naming an offered tool; the final-response path re-prompts (bounded) and never executes
the recovered text.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from agent.leaked_tool_call import find_leaked_tool_call

TOOLS = {"search_files", "terminal", "read_file"}


# ── detector ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, name, fmt", [
    ('Searching.\n<tool_call>\n{"name": "search_files", "arguments": {"path": "/app", '
     '"pattern": "test_|test\\\\.py|verify|smoke"}}\n</tool_call>', "search_files", "hermes_json"),
    ('<tool_call>{"name": "terminal", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>',
     "terminal", "hermes_json"),
    ("<tool_call>\n<function=search_files>\n<parameter=path>\n/app\n</parameter>\n"
     "<parameter=pattern>\ntest_|verify\n</parameter>\n</function>\n</tool_call>", "search_files", "xml_function"),
    ("I should look. <function=read_file><parameter=path>/a.py</parameter></function>", "read_file", "xml_function"),
    ('<|tool_call>call:search_files{path:<|"|>/app<|"|>,pattern:<|"|>a{b<|"|>}<tool_call|>',
     "search_files", "gemma"),
    ('<|start|>assistant<|channel|>commentary to=functions.terminal <|constrain|>json'
     '<|message|>{"command": "ls"}<|call|>', "terminal", "harmony"),
    ('Next step:\n{"name": "read_file", "arguments": {"path": "/etc/hosts"}}', "read_file", "bare_json"),
    ('<tool_call>{"name": "functions.terminal", "arguments": {}}</tool_call>', "terminal", "hermes_json"),
])
def test_complete_calls_are_detected(text, name, fmt):
    leak = find_leaked_tool_call(text, TOOLS)
    assert leak is not None
    assert (leak.name, leak.fmt, leak.complete) == (name, fmt, True)
    assert leak.snippet


@pytest.mark.parametrize("text, name, fmt", [
    # The eval transcript shape: the run ended on a dangling </parameter>.
    ("<function=search_files>\n<parameter=path>\n/app\n</parameter>\n<parameter=pattern>\ntest_\n</parameter>",
     "search_files", "xml_function"),
    ("<function=search_files><parameter=path>/app", "search_files", "xml_function"),
    ('<tool_call>\n{"name": "search_files", "arguments": {"path": "/a', "search_files", "hermes_json"),
    ('<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}', "terminal", "hermes_json"),
    ('<|tool_call>call:terminal{command:<|"|>ls', "terminal", "gemma"),
    ("to=functions.terminal", "terminal", "harmony"),
    ('{"name": "read_file", "arguments": {"path": "/etc', "read_file", "bare_json"),
])
def test_truncated_calls_are_detected_as_incomplete(text, name, fmt):
    leak = find_leaked_tool_call(text, TOOLS)
    assert leak is not None
    assert (leak.name, leak.fmt, leak.complete) == (name, fmt, False)


@pytest.mark.parametrize("text", [
    '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>',
    "<function=launch_rockets><parameter=n>3</parameter></function>",
    "to=functions.unknown_tool {}",
    '{"name": "other", "arguments": {}}',
    # Prose that merely mentions a tool.
    "I will use search_files on /app with the pattern test_ to find the tests.",
    "The terminal tool returned exit code 0, so the build passed.",
    # Markdown code: an example, not a call.
    '```xml\n<function=search_files><parameter=path>/app</parameter></function>\n```',
    'Example:\n```json\n{"name": "terminal", "arguments": {"command": "ls"}}\n```\nDone.',
    "Wrap calls like `<tool_call>{\"name\": \"terminal\", \"arguments\": {}}</tool_call>` in tags.",
    # A JSON answer that names a tool but is not call-shaped.
    '{"tool": "terminal", "status": "ok"}',
    "",
])
def test_non_calls_are_ignored(text):
    assert find_leaked_tool_call(text, TOOLS) is None


def test_no_tools_offered_never_fires():
    assert find_leaked_tool_call('<tool_call>{"name": "terminal", "arguments": {}}</tool_call>', ()) is None
    assert find_leaked_tool_call(None, TOOLS) is None


# Hosted Nemotron 3.5 Bolt eval archive: the reasoning starts mid-call, so only the TAIL survives.
FRAGMENT_TAILS = [
    "The debug script ran but produced no output. Let me check what happened.\n</parameter>",
    "The lake build works, which means mathlib is available. Let me find where the mathlib source "
    "actually is.\n</parameter>\n</function>\n</tool_call>\n",
    "print(res)\ncat(\"out_q sum:\", sum(res$out_q), \"\\n\")\n'\n</parameter>\n</function>\n</tool_call>\n",
    "    def __repr__(self) -> str:\n        return f\"Graph(...)\"\n\n\nclass DiGraph:\n"
    "</parameter>\n</function>\n</tool_call>\n",
]


@pytest.mark.parametrize("text", FRAGMENT_TAILS)
def test_trailing_call_closers_without_an_opener_are_a_fragment(text):
    leak = find_leaked_tool_call(text, TOOLS)
    assert leak is not None
    assert (leak.name, leak.fmt, leak.complete) == ("", "xml_fragment", False)
    assert leak.snippet.endswith(text.rstrip()[-20:])
    assert "cut-off" in leak.label


@pytest.mark.parametrize("text", [
    # Closers mid-text, followed by prose.
    "Ran it.\n</parameter>\n</function>\nThe output shows three failing tests; fix them next.",
    "</tool_call> is the closing tag Hermes expects; anything after it is ignored.",
    # Closers inside code.
    "The template ends like this:\n```xml\n</parameter>\n</function>\n</tool_call>\n```",
    "Unclosed fence at the end:\n```\n</parameter>\n</function>",
    "Close the block with `</tool_call>`",
    # An opener is present (unknown tool): not a fragment, and not a named leak either.
    "<function=launch_rockets><parameter=n>3</parameter></function>",
])
def test_closers_that_are_not_a_trailing_fragment_are_ignored(text):
    assert find_leaked_tool_call(text, TOOLS) is None


@pytest.mark.parametrize("xml", [
    '<function name="describe">This is XML output, not a tool call.</function>',
    '<parameter name="format">xml</parameter>',
])
def test_complete_xml_elements_are_not_cutoff_tool_fragments(xml):
    # A legitimate final XML answer must not consume a tool-call retry.
    assert find_leaked_tool_call(xml, TOOLS) is None


def test_full_named_call_keeps_its_name_over_the_fragment_case():
    text = "<function=read_file>\n<parameter=path>\n/a.py\n</parameter>\n</function>\n</tool_call>\n"
    leak = find_leaked_tool_call(text, TOOLS)
    assert (leak.name, leak.fmt) == ("read_file", "xml_function")


def test_earliest_call_wins_across_formats():
    text = ('to=functions.read_file {"path": "/x"}\n'
            '<tool_call>{"name": "terminal", "arguments": {}}</tool_call>')
    assert find_leaked_tool_call(text, TOOLS).name == "read_file"


# ── final-response path ─────────────────────────────────────────────────────────────────────

LEAKED_REASONING = (
    "The tests probably live under /app. I will search for them.\n"
    '<tool_call>\n{"name": "search_files", "arguments": {"path": "/app", '
    '"pattern": "test_|test\\\\.py|verify|smoke"}}\n</tool_call>'
)


@pytest.fixture()
def loop_agent():
    """Promotion-trusted route: local vLLM serving Nemotron-3.5-Lightning (as
    test_reasoning_only_stop_persistence)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            provider="vllm",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.valid_tool_names = set(TOOLS)
        return agent


def _run(agent, responses, user_message="find the tests"):
    agent.client.chat.completions.create.side_effect = list(responses)
    statuses = []
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_emit_diagnostic_status", side_effect=statuses.append),
    ):
        result = agent.run_conversation(user_message)
    return result, statuses


def _sent(agent, call_index):
    return agent.client.chat.completions.create.call_args_list[call_index].kwargs["messages"]


def test_reasoning_only_leaked_call_is_not_promoted_and_gets_the_nudge(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    result, statuses = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=LEAKED_REASONING),
        _mock_response(content="Found tests under /app/tests.", finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    assert result["final_response"] == "Found tests under /app/tests."
    nudge = _sent(loop_agent, 1)[-1]
    assert nudge["role"] == "user"
    assert "search_files" in nudge["content"] and "NOT executed" in nudge["content"]
    assert any("search_files" in s for s in statuses)
    # Ephemeral: the nudge pair never survives into the returned transcript.
    assert not [m for m in result["messages"] if m.get("_dropped_toolcall_nudge")]
    assert not any(m.get("role") == "user" and "NOT executed" in str(m.get("content"))
                   for m in result["messages"])


def test_prose_content_with_call_in_reasoning_gets_the_nudge(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    result, _ = _run(loop_agent, [
        _mock_response(content="Let me look for the test files.", finish_reason="stop",
                       reasoning_content=LEAKED_REASONING),
        _mock_response(content="The tests are in /app/tests.", finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    assert result["final_response"] == "The tests are in /app/tests."
    assert "search_files" in _sent(loop_agent, 1)[-1]["content"]


def test_call_markup_in_content_gets_the_nudge(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    leaked = "<function=read_file>\n<parameter=path>\n/app/setup.py\n</parameter>\n"  # dangling
    result, _ = _run(loop_agent, [
        _mock_response(content=leaked, finish_reason="stop"),
        _mock_response(content="setup.py declares no tests.", finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    nudge = _sent(loop_agent, 1)[-1]["content"]
    assert "read_file" in nudge and "incomplete" in nudge


@pytest.mark.parametrize("same_tail_in_content", [False, True])
def test_fragment_only_reasoning_stop_is_not_promoted_and_gets_the_nudge(loop_agent, same_tail_in_content):
    from tests.agent.test_run_agent import _mock_response

    tail = FRAGMENT_TAILS[0]
    result, statuses = _run(loop_agent, [
        _mock_response(content=tail if same_tail_in_content else "", finish_reason="stop",
                       reasoning_content=tail),
        _mock_response(content="The script printed nothing because main() is never called.",
                       finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    assert result["final_response"] == "The script printed nothing because main() is never called."
    nudge = _sent(loop_agent, 1)[-1]["content"]
    assert "cut off" in nudge and "NOT executed" in nudge and "``" not in nudge
    assert any("cut-off tool call" in s for s in statuses)


def test_budget_exhaustion_falls_through_with_a_warning(loop_agent, caplog):
    from tests.agent.test_run_agent import _mock_response

    # No announced action (that would trip the separate stall guard) and some prose (markup alone
    # strips to empty and takes the empty-response ladder instead of promotion).
    text = ('The tests are probably under /app.\n'
            '<tool_call>{"name": "search_files", "arguments": {"path": "/app"}}</tool_call>')
    leaked = [_mock_response(content="", finish_reason="stop", reasoning_content=text) for _ in range(3)]
    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        result, statuses = _run(loop_agent, [*leaked, _mock_response(content="NEVER REACHED")])

    # Two nudges, then the third leak takes the pre-existing path (promotion on this route).
    assert result["api_calls"] == 3
    assert result["final_response"] == "The tests are probably under /app."
    assert any("NOT executed" in r.getMessage() for r in caplog.records)
    assert any("not executed" in s for s in statuses)


def test_budget_resets_on_a_clean_final_answer(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    leak = _mock_response(content="", finish_reason="stop", reasoning_content=LEAKED_REASONING)
    _run(loop_agent, [leak, leak, _mock_response(content="done", finish_reason="stop")])
    assert loop_agent._leaked_toolcall_retries == 0


def test_plain_reasoning_only_answer_is_still_promoted(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    answer = "The search_files tool is not needed here: the answer is 42."
    result, statuses = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=answer),
        _mock_response(content="NEVER REACHED", finish_reason="stop"),
    ])

    assert result["api_calls"] == 1
    assert result["final_response"] == answer
    assert not any("tool call as text" in s for s in statuses)


def test_non_chat_completions_routes_are_unchanged(loop_agent):
    from agent.turn_final_response import _reprompt_leaked_tool_call
    from tests.agent.test_run_agent import _mock_assistant_msg

    msg = _mock_assistant_msg(content="", reasoning_content=LEAKED_REASONING)
    for mode in ("codex_responses", "anthropic_messages", "bedrock_converse"):
        loop_agent.api_mode = mode
        messages = []
        assert _reprompt_leaked_tool_call(loop_agent, msg, "stop", messages) is False
        assert messages == []
