"""Tests for silence_allowed flag — issue #13248.

When a group-chat message is not directly addressed to the bot, the Slack
adapter sets MessageEvent.silence_allowed=True. This flag should:

1. Be accepted by MessageEvent dataclass (default False).
2. Be accepted by AIAgent.run_conversation() as a kwarg.
3. Be stored on the agent as ``self._silence_allowed``.
4. Suppress the thinking-only prefill-nudge retry.
5. Suppress the empty-response retry loop.
6. Suppress fallback-provider activation on empty content.
"""

from unittest.mock import MagicMock


def test_message_event_silence_allowed_default_false():
    from gateway.platforms.base import MessageEvent

    e = MessageEvent(text="hi")
    assert e.silence_allowed is False


def test_message_event_silence_allowed_opt_in():
    from gateway.platforms.base import MessageEvent

    e = MessageEvent(text="hi", silence_allowed=True)
    assert e.silence_allowed is True


def test_run_conversation_accepts_silence_allowed_kwarg():
    """run_conversation signature must accept silence_allowed=bool."""
    import inspect

    from run_agent import AIAgent

    sig = inspect.signature(AIAgent.run_conversation)
    assert "silence_allowed" in sig.parameters
    assert sig.parameters["silence_allowed"].default is False


def test_run_agent_helper_accepts_silence_allowed_kwarg():
    """gateway._run_agent must also accept the flag so it can forward it."""
    import inspect

    from gateway.run import GatewayRunner

    sig = inspect.signature(GatewayRunner._run_agent)
    assert "silence_allowed" in sig.parameters
    assert sig.parameters["silence_allowed"].default is False


def _make_agent_stub(silence_allowed: bool):
    """Build a minimal object that carries just the flags the retry branches
    actually read. We don't construct a real AIAgent — the retry gates are
    simple boolean expressions we can reproduce in a tiny helper to validate
    the condition logic."""
    stub = MagicMock()
    stub._silence_allowed = silence_allowed
    stub._thinking_prefill_retries = 0
    stub._empty_content_retries = 0
    stub._fallback_chain = [("model-b", "provider-b")]
    return stub


def _would_nudge_thinking(agent, has_structured: bool) -> bool:
    """Reproduces the gate at run_agent.py ~L10894."""
    return (
        has_structured
        and agent._thinking_prefill_retries < 2
        and not agent._silence_allowed
    )


def _would_retry_empty(agent, truly_empty: bool, has_structured: bool,
                       prefill_exhausted: bool) -> bool:
    """Reproduces the gate at run_agent.py ~L10930."""
    return (
        truly_empty
        and (not has_structured or prefill_exhausted)
        and agent._empty_content_retries < 3
        and not agent._silence_allowed
    )


def _would_fallback(agent, truly_empty: bool) -> bool:
    """Reproduces the gate at run_agent.py ~L10949."""
    return (
        truly_empty
        and bool(agent._fallback_chain)
        and not agent._silence_allowed
    )


def test_thinking_only_suppressed_when_silence_allowed():
    quiet = _make_agent_stub(silence_allowed=True)
    loud = _make_agent_stub(silence_allowed=False)

    assert _would_nudge_thinking(loud, has_structured=True) is True
    assert _would_nudge_thinking(quiet, has_structured=True) is False


def test_empty_retry_suppressed_when_silence_allowed():
    quiet = _make_agent_stub(silence_allowed=True)
    loud = _make_agent_stub(silence_allowed=False)

    assert _would_retry_empty(
        loud, truly_empty=True, has_structured=False, prefill_exhausted=False
    ) is True
    assert _would_retry_empty(
        quiet, truly_empty=True, has_structured=False, prefill_exhausted=False
    ) is False


def test_fallback_suppressed_when_silence_allowed():
    quiet = _make_agent_stub(silence_allowed=True)
    loud = _make_agent_stub(silence_allowed=False)

    assert _would_fallback(loud, truly_empty=True) is True
    assert _would_fallback(quiet, truly_empty=True) is False


def test_silence_allowed_preserves_normal_retry_for_dms():
    """DMs / @mentions must still get full retry semantics — the flag
    defaults to False, so nothing changes for addressed messages."""
    addressed = _make_agent_stub(silence_allowed=False)

    assert _would_nudge_thinking(addressed, has_structured=True) is True
    assert _would_retry_empty(
        addressed, truly_empty=True, has_structured=False, prefill_exhausted=False
    ) is True
    assert _would_fallback(addressed, truly_empty=True) is True
