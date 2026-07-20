"""End-to-end tests for the self-reinforcing trailing-artifact scrub (#66150).

These drive the real ``run_conversation`` loop (not just the pure
``_strip_trailing_artifact`` helper), with a mocked OpenAI SDK returning a
response that ends in the stray "course" token. They assert the invariant that
matters in production: the token is gone from BOTH the user-facing
``final_response`` (what the gateway/CLI delivers) AND the persisted assistant
message in ``messages`` (what enters history and gets replayed), and that the
two agree — which is exactly the second review point (delivered text was
derived separately and previously kept the artifact even when history was
cleaned).

Mirrors the harness in tests/run_agent/test_turn_completion_explainer.py: we
patch ``run_agent.OpenAI`` and drive ``agent.client``, so these pass
identically in CI and locally.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._fallback_chain = []
    # Trailing-artifact scrub attrs (init reads these from config; the mocked
    # load_config returns {} so set the shipped defaults explicitly).
    agent._strip_trailing_artifacts = True
    agent._trailing_artifact_max_len = 12
    agent._trailing_artifact_min_repeats = 2
    agent._trailing_artifact_window = 8
    return agent


def _contaminated_history(token="course", n=2):
    """n assistant turns that each already end with the stray token, separated
    by user turns — i.e. the self-reinforcement loop is already established when
    the current turn runs. Uses realistic user/assistant alternation (real
    conversations never place two assistant turns back-to-back; the loop's
    'contiguous assistant turns' are adjacent in assistant-order but separated
    by user turns in the raw list, which is exactly what the scrub walks)."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"request {i}"})
        msgs.append({"role": "assistant", "content": f"Handling step {i}.\n\n{token}"})
    return msgs


def _last_assistant(messages):
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
            return m
    return None


def test_e2e_delivered_and_persisted_both_scrubbed_and_agree():
    """The whole point: once the loop is established, a new turn ending in the
    stray token must ship clean to the user AND land clean in history, and the
    two must be identical. Runs the real conversation loop."""
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="I'll run the command now.\n\ncourse", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "do the thing",
            conversation_history=_contaminated_history(n=2),
        )

    # Delivered (user-facing) text is scrubbed.
    assert result["final_response"] == "I'll run the command now."
    # Persisted assistant message (enters history / replay) is scrubbed too.
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == "I'll run the command now."
    # Delivered and persisted agree — no drift between what the user sees and
    # what gets replayed on the next turn.
    assert result["final_response"] == persisted["content"]


def test_e2e_one_off_token_delivered_intact():
    """A one-off trailing token with NO established loop in history is a
    legitimate reply and must NOT be stripped — end-to-end, delivered intact."""
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="Here is the answer.\n\ncourse", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the thing")  # no prior history

    assert result["final_response"] == "Here is the answer.\n\ncourse"
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == "Here is the answer.\n\ncourse"


def test_e2e_noncontiguous_history_delivered_intact():
    """Two old matching turns broken by a normal reply do NOT form an active
    loop, so a new turn ending in the token ships intact — end-to-end guard for
    the contiguity fix (unrelated historical matches must not delete content)."""
    agent = _make_agent(max_iterations=10)
    history = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "First.\n\ncourse"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "Second.\n\ncourse"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "A perfectly normal reply."},
        {"role": "user", "content": "keep going"},
    ]
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="Latest answer.\n\ncourse", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the thing", conversation_history=history)

    assert result["final_response"] == "Latest answer.\n\ncourse"
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == "Latest answer.\n\ncourse"


def test_e2e_disabled_flag_ships_artifact():
    """With the feature disabled, an established loop still ships the token both
    delivered and persisted — proves the scrub (not something else) is what
    removes it in the enabled cases."""
    agent = _make_agent(max_iterations=10)
    agent._strip_trailing_artifacts = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="Doing it.\n\ncourse", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "do the thing",
            conversation_history=_contaminated_history(n=3),
        )

    assert result["final_response"] == "Doing it.\n\ncourse"
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == "Doing it.\n\ncourse"


# ---------------------------------------------------------------------------
# Scratch-block variant: a trailing block of leaked "<marker>:" reasoning lines
# (rather than a single stray word). Same self-reinforcement loop, driven
# through the real conversation loop. Ground truth: production session
# 20260720_221918_7f069b (claude-opus-4-8) leaked repeating "count:" blocks.
# ---------------------------------------------------------------------------


def _scratch_contaminated_history(marker="count", n=2):
    """n assistant turns each ending in a leaked scratch block, alternating with
    user turns — the loop is already established when the current turn runs."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"request {i}"})
        msgs.append(
            {"role": "assistant", "content": f"Handling step {i}.\n\n{marker}: private note {i}."}
        )
    return msgs


def test_e2e_scratch_block_delivered_and_persisted_both_scrubbed_and_agree():
    """Once the scratch-block loop is established, a new turn ending in a
    trailing 'count:' block must ship clean to the user AND land clean in
    history, and the two must be identical. Real conversation loop."""
    agent = _make_agent(max_iterations=10)
    raw = "Here is the real answer.\n\ncount: double-check X.\ncount: actually Y."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=raw, finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "do the thing",
            conversation_history=_scratch_contaminated_history(n=2),
        )

    assert result["final_response"] == "Here is the real answer."
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == "Here is the real answer."
    assert result["final_response"] == persisted["content"]


def test_e2e_scratch_block_one_off_delivered_intact():
    """A one-off trailing scratch block with NO established loop is left intact
    end-to-end (might be legitimate content)."""
    agent = _make_agent(max_iterations=10)
    raw = "Here is the answer.\n\nnote: remember to test."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=raw, finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the thing")  # no prior history

    assert result["final_response"] == raw
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == raw


def test_e2e_scratch_block_mixed_closing_sentence_preserved():
    """C1 safety guarantee end-to-end: when a normal sentence closes the turn
    AFTER the scratch block (a 'mixed' leak), nothing is stripped — the block is
    not trailing, so the closing sentence is never lost."""
    agent = _make_agent(max_iterations=10)
    raw = "Analysis done.\n\ncount: check the suffix.\n\n先并行查这三项。"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=raw, finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "do the thing",
            conversation_history=_scratch_contaminated_history(n=3),
        )

    assert result["final_response"] == raw
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == raw


def test_e2e_scratch_block_disabled_flag_ships_block():
    """With the feature disabled, an established scratch-block loop still ships
    the block both delivered and persisted — proves the scrub is what removes it."""
    agent = _make_agent(max_iterations=10)
    agent._strip_trailing_artifacts = False
    raw = "Doing it.\n\ncount: leaked reasoning."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=raw, finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "do the thing",
            conversation_history=_scratch_contaminated_history(n=3),
        )

    assert result["final_response"] == raw
    persisted = _last_assistant(result["messages"])
    assert persisted is not None
    assert persisted["content"] == raw
