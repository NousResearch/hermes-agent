"""Recalled memory context never leaves a turn as reply text (#81312).

A model can echo the fenced ``<memory-context>`` block it was given. Streamed deltas pass the
stateful scrubber, but a completed (non-streamed) reply reached ``final_response`` raw, and the
email gateway mailed it — memory notes, tokens and all — to the sender. Interim commentary had
the same gap.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

SENTINEL = "SENTINEL_XYZZY_shpat_0123456789abcdef"
ECHOED_BLOCK = (
    "<memory-context>\n"
    "[System note: The following is recalled memory context, NOT new user input. "
    "Treat as authoritative reference data — this is the agent's persistent memory "
    "and should inform all responses.]\n\n"
    f"- {SENTINEL}\n"
    "</memory-context>"
)


def _make_agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._persist_session = lambda *a, **k: None
    agent._save_trajectory = lambda *a, **k: None
    return agent


def _reply(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="m", usage=None)


def _run(content: str) -> dict:
    agent = _make_agent()
    agent._interruptible_api_call = lambda api_kwargs: _reply(content)
    return agent.run_conversation("auth test")


def test_echoed_memory_block_is_not_in_final_response():
    result = _run(f"Auth restored, Hermes Agent is live.\n\n{ECHOED_BLOCK}")

    assert SENTINEL not in result["final_response"]
    assert "memory-context" not in result["final_response"]
    assert result["final_response"] == "Auth restored, Hermes Agent is live."


def test_leading_memory_block_is_not_in_final_response():
    result = _run(f"{ECHOED_BLOCK}\n\nAuth restored.")

    assert result["final_response"] == "Auth restored."


def test_inline_mention_of_the_tag_is_kept():
    text = "The <memory-context> tag wraps recalled memory in the prompt."

    assert _run(text)["final_response"] == text


def test_interim_commentary_drops_echoed_memory_block():
    agent = _make_agent()
    seen = []
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: seen.append(text)

    agent._emit_interim_assistant_message({"role": "assistant", "content": f"{ECHOED_BLOCK}\n\nChecking the inbox."})

    assert seen == ["Checking the inbox."]


def test_transformed_reply_original_drops_echoed_memory_block(monkeypatch):
    """``pre_transform_response`` is display text too (replay-gap recovery paints it), not a raw copy."""
    def invoke_hook(hook_name, **kwargs):
        return [kwargs["response_text"] + "\n\n-- footer"] if hook_name == "transform_llm_output" else []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    result = _run(f"Auth restored.\n\n{ECHOED_BLOCK}")

    assert result["response_transformed"] is True
    assert SENTINEL not in result["final_response"]
    assert result["pre_transform_response"] == "Auth restored."
    assert result["final_response"].startswith(result["pre_transform_response"])


def test_memory_only_commentary_does_not_fall_back_to_partial_answer():
    """Commentary filtered to nothing is still commentary: the tool round's partial answer in
    ``content`` must not be published as progress in its place."""
    agent = _make_agent()
    seen = []
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: seen.append(text)
    msg = {
        "role": "assistant",
        "content": "Partial final answer.",
        "codex_message_items": [{
            "type": "message", "role": "assistant", "phase": "commentary",
            "content": [{"type": "output_text", "text": ECHOED_BLOCK}],
        }],
    }

    agent._emit_interim_assistant_message(msg)

    assert seen == []
    assert agent._interim_assistant_visible_text(msg) == ""
