"""A ``pre_api_request`` plugin may rewrite sampling knobs — never the prompt.

Contract under test:

- A dict returned by a ``pre_api_request`` plugin rewrites ``reasoning_effort`` / ``max_tokens`` /
  ``temperature`` / ``top_p`` on the outgoing provider request.
- Prompt-bearing keys (``messages`` / ``system`` / ``tools``) are NOT mutable through this hook: the
  prompt prefix has to stay byte-stable or provider caching breaks.
- Results that are not dicts, and keys outside the whitelist, are ignored — the pre-existing
  "return correlation data" contract keeps working unchanged.
- The whitelisted mutation reaches the wire payload the transport is handed, not an internal copy.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_run_agent import _make_tool_defs, _mock_response


@pytest.fixture()
def agent():
    """Minimal AIAgent with a mocked OpenAI client and no tool loading."""
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a.client.chat.completions.create.return_value = _mock_response()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _wire_kwargs(agent, hook_result, message="hi"):
    """Run one turn with *hook_result* returned by the pre_api_request hook; return wire kwargs."""

    def _hook(name, **_kwargs):
        if name == "pre_api_request" and hook_result is not None:
            return [hook_result]
        return []

    with (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "pre_api_request"),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_hook),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(message)

    assert agent.client.chat.completions.create.called, "no provider request was issued"
    return agent.client.chat.completions.create.call_args.kwargs


def test_hook_rewrites_sampling_knobs_on_the_wire(agent):
    kwargs = _wire_kwargs(agent, {"reasoning_effort": "low", "max_tokens": 4242, "temperature": 0.2})

    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_tokens"] == 4242
    assert kwargs["temperature"] == 0.2


def test_hook_cannot_touch_the_prompt(agent):
    baseline = _wire_kwargs(agent, None)
    injected = _wire_kwargs(
        agent,
        {"messages": [{"role": "user", "content": "INJECTED"}], "tools": [], "model": "evil-model"},
    )

    # The prompt the provider sees is identical to the un-hooked request.
    assert injected["messages"] == baseline["messages"]
    assert "INJECTED" not in json.dumps(injected["messages"])
    assert injected["tools"] == baseline["tools"]
    assert injected["model"] == baseline["model"]


def test_non_dict_results_and_unknown_keys_are_ignored(agent):
    """A plugin returning correlation data (the old contract) must not break the request."""
    baseline = _wire_kwargs(agent, None)
    kwargs = _wire_kwargs(agent, {"seen": 2, "mc": 5, "tc": 3, "max_tokens": 100})

    assert kwargs["max_tokens"] == 100  # whitelisted key applied
    assert "seen" not in kwargs and "mc" not in kwargs  # unknown keys dropped
    assert kwargs["messages"] == baseline["messages"]


def test_apply_request_mutations_handles_non_dict_and_noop_results():
    from agent.turn_api_request import _apply_request_mutations

    api_kwargs = {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"}
    _apply_request_mutations(
        api_kwargs,
        [None, "nope", 7, {"reasoning_effort": "max"}, {"reasoning_effort": "low", "seen": 1}],
        api_call_count=3,
        session_id="s1",
    )

    assert api_kwargs["reasoning_effort"] == "low"  # last writer wins
    assert "seen" not in api_kwargs
    assert api_kwargs["messages"] == [{"role": "user", "content": "x"}]


def test_a_failing_mutation_is_logged_and_never_fatal(agent, caplog):
    """The mutation step has its own guard: it logs, and the request still goes out.

    Without it, a malformed plugin result would be swallowed by the observer body's bare except
    and leave no trace of why the knob it asked for was not applied.
    """
    import agent.turn_api_request as turn_api_request

    def _boom(*_args, **_kwargs):
        raise RuntimeError("plugin result exploded")

    with (
        caplog.at_level("WARNING", logger="agent.conversation_loop"),
        patch.object(turn_api_request, "_apply_request_mutations", side_effect=_boom),
    ):
        kwargs = _wire_kwargs(agent, {"reasoning_effort": "low"})

    assert agent.client.chat.completions.create.called, "the request must still be issued"
    # The key is absent rather than "low": the mutation never ran, so nothing set it.
    assert kwargs.get("reasoning_effort") != "low", "the failed mutation must not be applied"
    assert "plugin result exploded" in caplog.text, "the failure must be visible in the logs"
