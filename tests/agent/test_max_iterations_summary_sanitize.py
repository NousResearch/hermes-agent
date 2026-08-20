"""Regression tests for max-iterations summary sanitization (#58220).

``handle_max_iterations`` asks the model for a closing summary with no tools
bound. Open-weight models (z.ai/GLM in particular) answer that by emitting
their native tool-call template as plain content, so the summary that reaches
the user carries raw ``<tool_call>`` / ``<function_call>`` XML.

The old inline regex only understood ``<think>``. Both summary branches now
route through the canonical ``agent._strip_think_blocks()``, which also strips
leaked tool-call XML.

Two branches exist and both are covered here — the review on #58220 asked
specifically for the retry path, which earlier attempts left untested:

* initial branch — the first summary call returns usable content
* retry branch   — the first call returns empty, so a second call is made

The retry test asserts a second call really happened, so it cannot silently
degrade into re-testing the initial branch.
"""

import types

import pytest


LEAKY_SUMMARY = (
    "<think>I should call the tool again</think>"
    "Here is what I found.\n"
    '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
)


def _make_agent(summary_contents):
    """Real AIAgent with only the network edge stubbed.

    ``summary_contents`` is consumed one entry per summary call, so passing
    two entries drives the empty-then-retry path.
    """
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cached_system_prompt = "SYS"
    agent.model = "glm-4.7-flash"
    agent.provider = "zai"
    agent.max_iterations = 60

    calls = {"n": 0}
    remaining = list(summary_contents)

    class _Completions:
        def create(self, **_kwargs):
            calls["n"] += 1
            return "RAW-RESPONSE"

    agent._ensure_primary_openai_client = lambda reason=None: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions())
    )
    agent._get_transport = lambda: types.SimpleNamespace(
        normalize_response=lambda _r: types.SimpleNamespace(
            content=remaining.pop(0) if remaining else ""
        )
    )
    return agent, calls


def _run(agent):
    from agent.chat_completion_helpers import handle_max_iterations

    return handle_max_iterations(agent, [], api_call_count=1)


def test_initial_summary_branch_strips_leaked_tool_call_xml():
    agent, calls = _make_agent([LEAKY_SUMMARY])

    result = _run(agent)

    assert "Here is what I found." in result
    assert "<tool_call>" not in result
    assert "<think>" not in result
    assert calls["n"] == 1, "expected the initial branch only"


def test_retry_summary_branch_strips_leaked_tool_call_xml():
    # The first attempt yields nothing usable, so the retry branch runs — that
    # branch's sanitizer is what this test pins.
    agent, calls = _make_agent(["", LEAKY_SUMMARY])

    result = _run(agent)

    assert calls["n"] == 2, "retry branch did not run; test would not cover it"
    assert "Here is what I found." in result
    assert "<tool_call>" not in result
    assert "<think>" not in result


@pytest.mark.parametrize("contents", [[LEAKY_SUMMARY], ["", LEAKY_SUMMARY]])
def test_summary_never_leaks_xml_on_either_branch(contents):
    agent, _calls = _make_agent(contents)

    result = _run(agent)

    for marker in ("<tool_call>", "</tool_call>", "<function_call>", "<think>"):
        assert marker not in result


def test_summary_that_is_only_leaked_xml_falls_back_to_plain_notice():
    # Sanitizing can empty the summary entirely; the user must still get a
    # sentence rather than an empty message.
    agent, _calls = _make_agent(['<tool_call>{"name": "x"}</tool_call>'])

    result = _run(agent)

    assert result.strip()
    assert "<tool_call>" not in result
