"""ONE historical replay policy: normal and forced-summary requests render identical user content.

PR #63298's sidecar-glue guard lived in ``build_api_messages`` only:
``_iteration_summary_api_messages`` still called ``substitute_api_content`` unconditionally, so a
source-identified user row replayed its SOURCE text on the normal request and the transient
injection glue on the max-iteration summary request — two replay policies for one history row
(prefix divergence; stale plugin/onboarding text re-entering a queued-prompt boundary). Both
wire-boundary consumers now share one policy.
"""

import pytest

from agent.chat_completion_helpers import _iteration_summary_api_messages
from agent.turn_context import build_api_messages
from run_agent import AIAgent


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _make():
        agent = AIAgent(api_key="k", base_url="https://api.groq.com/openai/v1", provider="custom",
                        model="m", quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "SYS"
        agent._current_turn_timestamp = 1728000000.0
        return agent

    return _make


def _normal_request_user_contents(agent, history, current_turn_user_idx):
    api_messages, _system = build_api_messages(
        agent, history, current_turn_user_idx=current_turn_user_idx,
        ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
        active_system_prompt="SYS")
    return [m["content"] for m in api_messages if m.get("role") == "user"]


def _forced_summary_user_contents(agent, history):
    out = _iteration_summary_api_messages(agent, [dict(m) for m in history])
    return [m["content"] for m in out if m.get("role") == "user"]


def _history(api_row):
    return [
        api_row,
        {"role": "assistant", "content": "a1", "timestamp": 150.0},
        {"role": "user", "content": "q2", "timestamp": 200.0},
    ]


def test_source_identified_row_renders_identically_in_normal_and_forced_summary_requests(make_agent):
    """RED for the bug (PR #63298 review): the normal request sent ``q1`` while the forced-summary
    request sent ``q1\\n\\nPLUGIN-CTX`` for the same source-identified history row. Prefix parity:
    one historical replay policy for both requests."""
    agent = make_agent()
    history = _history({"role": "user", "content": "q1", "api_content": "q1\n\nPLUGIN-CTX",
                        "message_id": "m-1", "timestamp": 100.0})
    normal = _normal_request_user_contents(agent, history, current_turn_user_idx=2)
    summary = _forced_summary_user_contents(agent, history)
    assert normal == ["q1", "q2"]
    assert summary == ["q1", "q2"]
    assert normal == summary


def test_anonymous_row_still_replays_its_sidecar_bytes_on_both_paths(make_agent):
    """The policy must not over-suppress: a row without a source identity still replays the exact
    bytes its turn sent (the prompt-cache prefix), on both request shapes."""
    agent = make_agent()
    history = _history({"role": "user", "content": "q1", "api_content": "q1\n\nPLUGIN-CTX",
                        "timestamp": 100.0})
    normal = _normal_request_user_contents(agent, history, current_turn_user_idx=2)
    summary = _forced_summary_user_contents(agent, history)
    assert normal == ["q1\n\nPLUGIN-CTX", "q2"]
    assert summary == ["q1\n\nPLUGIN-CTX", "q2"]
    assert normal == summary
