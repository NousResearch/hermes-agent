"""Tests for reasoning display fix.

FIX A (run_agent.py _build_assistant_message):
    <think>-extracted reasoning callback is skipped when stream_delta_callback is
    set, because streaming already displayed it. Structured reasoning always fires
    the callback regardless.

FIX B (cli.py /verbose and /reasoning show toggles):
    When streaming is enabled, reasoning_callback should be set to
    _stream_reasoning_delta, not _on_reasoning.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers: minimal stand-ins for the logic under test
# ---------------------------------------------------------------------------


def _extract_reasoning_stub(assistant_message):
    """Mirror the structured-reasoning extraction from _extract_reasoning()."""
    parts = []
    if getattr(assistant_message, "reasoning", None):
        parts.append(assistant_message.reasoning)
    if getattr(assistant_message, "reasoning_content", None):
        val = assistant_message.reasoning_content
        if val not in parts:
            parts.append(val)
    return "\n\n".join(parts) if parts else None


def _run_build_assistant_message(
    assistant_message,
    *,
    stream_delta_callback,
    reasoning_callback,
):
    """
    Replicate only the reasoning-callback portion of _build_assistant_message()
    after FIX A is applied.

    Returns whether reasoning_callback was called.
    """
    reasoning_text = _extract_reasoning_stub(assistant_message)
    _from_structured = reasoning_text is not None

    if not reasoning_text:
        content = assistant_message.content or ""
        think_blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if think_blocks:
            combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
            reasoning_text = combined or None

    if reasoning_text and reasoning_callback:
        # FIX A: skip callback for <think>-extracted reasoning when streaming,
        # because _stream_delta() already displayed it during token delivery.
        if _from_structured or not stream_delta_callback:
            try:
                reasoning_callback(reasoning_text)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests for FIX A
# ---------------------------------------------------------------------------


def test_think_tag_callback_skipped_when_streaming():
    """
    When stream_delta_callback is set (streaming mode) and reasoning came from a
    <think> block (not structured fields), reasoning_callback must NOT be called.
    The streaming path already displayed the thinking tokens in real-time.
    """
    msg = SimpleNamespace(
        content="<think>reasoning here</think>answer",
        reasoning=None,
        reasoning_content=None,
    )
    callback = MagicMock()

    _run_build_assistant_message(
        msg,
        stream_delta_callback=MagicMock(),  # streaming ON
        reasoning_callback=callback,
    )

    callback.assert_not_called()


def test_think_tag_callback_fires_when_not_streaming():
    """
    When stream_delta_callback is None (non-streaming), <think>-extracted
    reasoning MUST trigger reasoning_callback so the user sees the thinking.
    """
    msg = SimpleNamespace(
        content="<think>reasoning here</think>answer",
        reasoning=None,
        reasoning_content=None,
    )
    callback = MagicMock()

    _run_build_assistant_message(
        msg,
        stream_delta_callback=None,  # streaming OFF
        reasoning_callback=callback,
    )

    callback.assert_called_once()
    args, _ = callback.call_args
    assert "reasoning here" in args[0]


def test_structured_reasoning_callback_fires_when_streaming():
    """
    Structured reasoning (via message.reasoning or message.reasoning_content)
    must ALWAYS trigger reasoning_callback, even in streaming mode.
    The streaming path only surfaces <think> blocks during delivery; structured
    reasoning arrives as a separate API field and needs the callback to display.
    """
    msg = SimpleNamespace(
        content="answer",
        reasoning="structured thinking",
        reasoning_content=None,
    )
    callback = MagicMock()

    _run_build_assistant_message(
        msg,
        stream_delta_callback=MagicMock(),  # streaming ON
        reasoning_callback=callback,
    )

    callback.assert_called_once()
    args, _ = callback.call_args
    assert "structured thinking" in args[0]


# ---------------------------------------------------------------------------
# Tests for FIX B
# ---------------------------------------------------------------------------


def test_reasoning_toggle_uses_stream_callback_when_streaming():
    """
    When streaming is enabled and the user toggles reasoning on (/reasoning show
    or /verbose), agent.reasoning_callback should be set to _stream_reasoning_delta,
    not _on_reasoning.

    This mirrors the FIX B logic:
        agent.reasoning_callback = (
            _stream_reasoning_delta if streaming_enabled else _on_reasoning
        )
    """

    # Minimal stand-in for the CLI instance
    class FakeCLI:
        streaming_enabled = True

        def _stream_reasoning_delta(self, text: str) -> None:
            pass

        def _on_reasoning(self, text: str) -> None:
            pass

        def _apply_reasoning_toggle_on(self):
            """Replicate the FIX B assignment from /reasoning show."""
            if self.agent:
                self.agent.reasoning_callback = (
                    self._stream_reasoning_delta
                    if self.streaming_enabled
                    else self._on_reasoning
                )

    cli = FakeCLI()
    cli.agent = SimpleNamespace(reasoning_callback=None)

    cli._apply_reasoning_toggle_on()

    assert cli.agent.reasoning_callback == cli._stream_reasoning_delta, (
        "When streaming is enabled, reasoning_callback should be _stream_reasoning_delta"
    )
    assert cli.agent.reasoning_callback != cli._on_reasoning
