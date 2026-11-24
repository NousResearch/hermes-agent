"""Tool calls written as text instead of structured ``tool_calls``.

Local llama.cpp / vLLM routes serving Nemotron sometimes end a reply with
``finish_reason="stop"``, no ``tool_calls`` and the intended call left as markup in
``reasoning``/``content``. Before the fix the reasoning was promoted to the final answer (or the
prose accepted) and the turn ended "complete" with the action never run. The detector fires only
on markup naming an offered tool; the final-response path re-prompts (bounded) and never executes
the recovered text.
"""

from __future__ import annotations

import json
import logging
import time
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


@pytest.mark.parametrize("arguments", [
    "[" * 100_000 + "]" * 100_000,  # nested inline: the call body's own decode recurses
    json.dumps("[" * 100_000 + "]" * 100_000),  # nested inside a string: the arguments decode recurses
], ids=["inline", "in-string"])
def test_deeply_nested_arguments_are_an_incomplete_call_not_an_error(arguments):
    leak = find_leaked_tool_call(f'<tool_call>{{"name": "terminal", "arguments": {arguments}}}</tool_call>', TOOLS)
    assert (leak.name, leak.complete) == ("terminal", False)


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
    # A closer in prose, then a code block that ends the reply.
    "Close each argument with\n</parameter>\n```python\nprint('hello')\n```",
    "Close the block with `</tool_call>`",
    # An opener is present (unknown tool): not a fragment, and not a named leak either.
    "<function=launch_rockets><parameter=n>3</parameter></function>",
])
def test_closers_that_are_not_a_trailing_fragment_are_ignored(text):
    assert find_leaked_tool_call(text, TOOLS) is None


def test_long_code_block_reply_is_scanned_in_linear_time():
    # _mask_code blanks a fenced block into one long run of spaces; a tail check that backtracks
    # over that run took ~8 s on this 120 KB reply, and it runs on every final reply.
    text = "Here is the file:\n```python\n" + "x = 1\n" * 20_000 + "```\nDone."
    started = time.perf_counter()
    assert find_leaked_tool_call(text, TOOLS) is None
    assert time.perf_counter() - started < 2.0


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


THINK_WITH_CALL = ('<think>\nI should list the files first.\n'
                   '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>\n</think>\n')


@pytest.mark.parametrize("reply, extra, nudged", [
    # A transcript example in a fence: the reply's fence masking covers its <think> block too.
    (f"Qwen writes it like this:\n```text\n{THINK_WITH_CALL}```\nThat is the format.", {}, False),
    (THINK_WITH_CALL + "Listing them now.", {}, True),
    ("Let me look.", {"reasoning": LEAKED_REASONING}, True),
    ("Let me look.", {"reasoning_details": [{"type": "reasoning.summary", "summary": LEAKED_REASONING}]}, True),
], ids=["fenced-example", "bare-think", "reasoning", "reasoning-details"])
def test_inline_think_blocks_are_judged_with_the_reply(loop_agent, reply, extra, nudged):
    from tests.agent.test_run_agent import _mock_response

    result, statuses = _run(loop_agent, [_mock_response(content=reply, finish_reason="stop", **extra),
                                         _mock_response(content="Done.", finish_reason="stop")])

    assert sum("as text — re-prompting" in s for s in statuses) == int(nudged)
    assert result["api_calls"] == 1 + int(nudged)


def test_typed_thinking_blocks_are_still_scanned(loop_agent):
    from agent.turn_final_response import _reprompt_leaked_tool_call
    from tests.agent.test_run_agent import _mock_assistant_msg

    # List content carries its thinking as typed blocks that the reply scan can't read.
    msg = _mock_assistant_msg(content=[{"type": "thinking", "thinking": LEAKED_REASONING}])
    assert _reprompt_leaked_tool_call(loop_agent, msg, "stop", []) is True


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


def test_stop_gate_follow_up_keeps_the_spent_budget(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    text = ('The tests are probably under /app.\n'
            '<tool_call>{"name": "search_files", "arguments": {"path": "/app"}}</tool_call>')
    leaked = [_mock_response(content="", finish_reason="stop", reasoning_content=text) for _ in range(4)]
    notes = iter(["[System note: the language server reported problems in files you edited.]"])
    # The third leak ends on the normal path and a stop gate sends the turn back once: the fourth
    # leak finds the budget still spent instead of drawing a third nudge.
    with patch("agent.turn_stop_gates._late_lsp_verdicts", side_effect=lambda agent: next(notes, None)):
        result, statuses = _run(loop_agent, [*leaked, _mock_response(content="NEVER REACHED")])

    assert sum("as text — re-prompting" in s for s in statuses) == 2
    assert result["api_calls"] == 4


def test_not_executed_status_fires_once_per_turn(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    text = ('The tests are probably under /app.\n'
            '<tool_call>{"name": "search_files", "arguments": {"path": "/app"}}</tool_call>')

    def leaks(n):
        return [_mock_response(content="", finish_reason="stop", reasoning_content=text) for _ in range(n)]

    # The third and fourth leaks both find the budget spent: one warning for the turn, not two.
    notes = iter(["[System note: the language server reported problems in files you edited.]"])
    with patch("agent.turn_stop_gates._late_lsp_verdicts", side_effect=lambda agent: next(notes, None)):
        _, statuses = _run(loop_agent, leaks(4))
    assert sum("not executed" in s for s in statuses) == 1

    # A new turn warns again.
    _, statuses = _run(loop_agent, leaks(3))
    assert sum("not executed" in s for s in statuses) == 1


def test_budget_resets_on_a_clean_final_answer(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    leak = _mock_response(content="", finish_reason="stop", reasoning_content=LEAKED_REASONING)
    _run(loop_agent, [leak, leak, _mock_response(content="done", finish_reason="stop")])
    assert loop_agent._leaked_toolcall_retries == 0


def test_budget_resets_on_a_real_tool_round(loop_agent):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    leak = _mock_response(content="", finish_reason="stop", reasoning_content=LEAKED_REASONING)
    rounds = [_mock_response(content="", finish_reason="tool_calls",
                             tool_calls=[_mock_tool_call(name="search_files", arguments='{"path": "/app"}')])
              for _ in range(3)]
    # Three leaks in one turn, each followed by the real call: every tool round restores the
    # budget, so the third leak is still re-prompted instead of ending the turn.
    with patch("model_tools.handle_function_call", return_value="ok"):
        result, statuses = _run(loop_agent, [leak, rounds[0], leak, rounds[1], leak, rounds[2],
                                             _mock_response(content="Done.", finish_reason="stop")])

    assert result["final_response"] == "Done."
    assert sum("as text — re-prompting" in s for s in statuses) == 3
    assert not any("not executed" in s for s in statuses)


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


def test_call_drafted_then_rejected_in_reasoning_does_not_override_a_full_answer(loop_agent):
    # The model considered a call in its reasoning, decided against it, and answered. The call is
    # not the tail of the reasoning and the content is a complete answer: nothing was left undone.
    from tests.agent.test_run_agent import _mock_response

    reasoning = ('I could run <tool_call>{"name": "search_files", "arguments": {"path": "/app"}}'
                 '</tool_call> but the user already pasted the tree, so I can answer directly.')
    answer = "The tests are in /app/tests (from the tree you pasted)."
    result, statuses = _run(loop_agent, [
        _mock_response(content=answer, finish_reason="stop", reasoning_content=reasoning),
        _mock_response(content="SECOND ANSWER", finish_reason="stop"),
    ])

    assert result["api_calls"] == 1
    assert result["final_response"] == answer
    assert not any("as text" in s for s in statuses)


def test_call_drafted_then_rejected_in_reasoning_still_counts_without_an_answer(loop_agent):
    # Blank content: nothing reached the user, so any call in the reasoning still gets the nudge.
    from tests.agent.test_run_agent import _mock_response

    reasoning = ('I could run <tool_call>{"name": "search_files", "arguments": {"path": "/app"}}'
                 '</tool_call> but maybe not.')
    result, statuses = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=reasoning),
        _mock_response(content="The tests are in /app/tests.", finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    assert sum("as text — re-prompting" in s for s in statuses) == 1


_SEARCH_CALL = '<tool_call>{"name": "search_files", "arguments": {"path": "/app"}}</tool_call>'


@pytest.mark.parametrize("text, name", [
    (f"Searching now.\n{_SEARCH_CALL}\n", "search_files"),
    # Parallel calls at the end: the run of calls is the tail.
    (f'{_SEARCH_CALL}\n<tool_call>{{"name": "terminal", "arguments": {{}}}}</tool_call>', "terminal"),
    # A draft set aside, then the real call at the end.
    (f"Maybe {_SEARCH_CALL}? No, list first.\n"
     '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>', "terminal"),
    # Wrapper and end tokens that belong to the call.
    ("Look.\n<tool_call>\n<function=read_file>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>",
     "read_file"),
    ('commentary to=functions.terminal <|constrain|>json<|message|>{"command": "ls"}<|call|>', "terminal"),
    # Unclosed: the call runs to the end.
    ('Then <tool_call>{"name": "search_files", "arguments": {"path": "/a', "search_files"),
    (FRAGMENT_TAILS[1], ""),
    # The last tool name in the text sits inside the call's arguments: the call around it counts.
    ('{"name": "terminal", "arguments": {"spec": {"name": "read_file"}}}', "terminal"),
    ('<tool_call>{"name": "terminal", "arguments": {"spec": {"name": "read_file"}}}</tool_call>', "terminal"),
])
def test_trailing_scan_finds_a_call_the_text_ends_on(text, name):
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    assert find_trailing_leaked_tool_call(text, TOOLS).name == name


@pytest.mark.parametrize("text", [
    f"I could run {_SEARCH_CALL} but the tree is already here.",
    f"{_SEARCH_CALL}\n```text\nexample\n```",
    'Example: {"name": "read_file", "arguments": {"path": "/x"}} is how a call looks; not needed.',
    "No call at all.",
    None,
    f"{_SEARCH_CALL}\n<function=launch_rockets><parameter=n>3</parameter></function>",
])
def test_trailing_scan_ignores_calls_followed_by_more_text(text):
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    assert find_trailing_leaked_tool_call(text, TOOLS) is None


_XML_CALL = ("<tool_call>\n<function=terminal>\n<parameter=command>\ncat x\n</parameter>\n</function>\n"
             "</tool_call>\n")


@pytest.mark.parametrize("partial", [
    "<", "<tool", "<tool_call", "<tool_call>", "<tool_call>\n<function", "<tool_call>\n<function=",
    "<function=", "<function=termi", "<|tool_call", "<|tool_call>call:ter", "to=functions.", "</tool",
])
def test_trailing_scan_counts_the_start_of_one_more_call_as_the_tail(partial):
    # Several calls back to back, then the next one cut off after its first few characters.
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    text = "Reading the files.\n" + _XML_CALL * 3 + partial
    leak = find_trailing_leaked_tool_call(text, TOOLS)
    assert (leak.name, leak.fmt) == ("terminal", "xml_function")


@pytest.mark.parametrize("tail", ["<b>done</b>", "to be safe I answer directly.", "tool output was empty."])
def test_trailing_scan_still_treats_other_text_after_the_calls_as_prose(tail):
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    assert find_trailing_leaked_tool_call(_XML_CALL * 3 + tail, TOOLS) is None


def test_trailing_scan_judges_many_calls_by_the_last_one_in_linear_time():
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    drafts = (f"Maybe {_SEARCH_CALL}? Not yet.\n") * 2_000
    started = time.perf_counter()
    leak = find_trailing_leaked_tool_call(drafts + _XML_CALL, TOOLS)
    assert time.perf_counter() - started < 0.1
    assert (leak.name, leak.fmt) == ("terminal", "xml_function")

    started = time.perf_counter()
    assert find_trailing_leaked_tool_call(drafts + "None of these are needed; here is the answer.", TOOLS) is None
    assert time.perf_counter() - started < 0.1


_ADVERSARIAL = {
    "xml_openers": "<function=terminal>" * 20_000,
    "xml_openers_spaced": " x ".join(["<function=terminal>"] * 20_000),
    "hermes_unclosed": '<tool_call>{"name": "terminal" ' * 10_000,
    "gemma_unclosed": "<|tool_call>call:terminal{" * 10_000,
    "bare_json_unclosed": '{"name": "terminal", "arguments": ' * 10_000,
    "harmony": "to=functions.terminal " * 20_000,
}


@pytest.mark.parametrize("suffix", ["", " That was all just thinking; the answer is 42."])
@pytest.mark.parametrize("key", sorted(_ADVERSARIAL))
def test_scans_stay_fast_on_many_unclosed_openers(key, suffix):
    # Each unclosed opener searches ahead for its closer: walking every one of them is quadratic.
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    text = _ADVERSARIAL[key] + suffix
    for scan in (find_leaked_tool_call, find_trailing_leaked_tool_call):
        started = time.perf_counter()
        scan(text, TOOLS)
        assert time.perf_counter() - started < 0.2, scan.__name__


# Many openers naming a tool that is NOT offered, with an offered tool's name elsewhere in the text.
_UNOFFERED = {
    "gemma": "<|tool_call>call:launch{" * 10_000,
    "bare_json": '{"name": "launch", "arguments": ' * 10_000,
    "xml_openers": "<function=launch>" * 20_000,
    "hermes": '<tool_call>{"name": "launch" ' * 10_000,
    "harmony": "to=functions.launch " * 20_000,
    "xml_closed": "<function=launch><parameter=a>1</parameter></function>" * 10_000,
}
_REAL_CALL = '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>'


@pytest.mark.parametrize("real_call", [False, True])
@pytest.mark.parametrize("key", sorted(_UNOFFERED))
def test_unoffered_tool_openers_are_skipped_before_any_lookahead(key, real_call):
    # Each unknown-tool match searched ahead to the end of the text before its name was checked.
    from agent.leaked_tool_call import find_trailing_leaked_tool_call

    text = _UNOFFERED[key] + " terminal" + (_REAL_CALL if real_call else "")
    for scan in (find_leaked_tool_call, find_trailing_leaked_tool_call):
        started = time.perf_counter()
        leak = scan(text, TOOLS)
        assert time.perf_counter() - started < 0.2, scan.__name__
        if real_call:
            assert (leak.name, leak.fmt, leak.complete) == ("terminal", "hermes_json", True), scan.__name__
