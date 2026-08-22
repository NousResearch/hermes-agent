"""Feishu streaming card — CardKit v2 template generator + state machine.

Phase 4 / L4 — skeleton (2026-08-12). The full streaming-card UX (long
agent responses rendered as an incrementally-updated Feishu interactive
card instead of N split messages) is a multi-week project that needs:
  - a sidecar process owning card lifecycle,
  - a cross-process message bus for token-delta events,
  - a card-template designer pass.

This module ships the *card payload generator* and *state machine* as
isolated, testable building blocks so the sidecar (or an in-process
adapter fallback) can be added later without rewriting the card schema.

What this module does today
---------------------------
- Build a CardKit v2 ``header`` + ``elements`` payload for the three
  card states: ``thinking``, ``streaming``, ``final``.
- Update an in-place card from one state to the next by replacing the
  ``elements`` array — this is the only Feishu-supported update path
  (the ``card_id`` is preserved by the Feishu client, but elements
  must be re-sent via ``card.update``).
- Provide ``render_streaming_card(chat_id, state, content)`` returning
  a payload that the adapter can send via ``interactive`` msg_type.

It does NOT yet
---------------
- Spawn or manage a sidecar process.
- Wire the model-streaming-token callback into this module.
- Send the initial ``card.create`` then track ``card_id`` across the
  session — those happen in the adapter or sidecar, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CardState(str, Enum):
    """Lifecycle states for a streaming response card.

    Transitions: THINKING -> STREAMING -> FINAL (terminal).
    Error path: any state -> ERROR (terminal).
    """
    THINKING = "thinking"
    STREAMING = "streaming"
    FINAL = "final"
    ERROR = "error"


@dataclass
class StreamingCardSession:
    """Per-message card session state.

    A session is created when the agent emits its first response token
    (state=THINKING), advances through STREAMING as tokens arrive, and
    lands on FINAL (or ERROR). The ``card_id`` is what the adapter
    stores so subsequent updates patch the existing card instead of
    posting a new one.
    """
    state: CardState = CardState.THINKING
    card_id: Optional[str] = None
    content_buffer: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    # Header color tint per state — light variant that survives mobile.
    # Avoids the unreadable-on-dark-bg pitfall of full saturation.
    state_label: Dict[CardState, str] = field(default_factory=lambda: {
        CardState.THINKING: "🤔 思考中…",
        CardState.STREAMING: "✍️ 生成中…",
        CardState.FINAL: "✓ 完成",
        CardState.ERROR: "✗ 出错",
    })

    def append_token(self, token: str) -> None:
        """Append a streamed token; transitions THINKING → STREAMING."""
        if self.state == CardState.THINKING:
            self.state = CardState.STREAMING
        self.content_buffer.append(token)

    def finalize(self) -> None:
        """Mark the response as complete."""
        self.state = CardState.FINAL

    def fail(self, message: str) -> None:
        """Mark the response as errored; freeze buffer."""
        self.state = CardState.ERROR
        self.error_message = message


def _build_body_element(state: CardState, content: str, error: Optional[str]) -> Dict[str, Any]:
    """Build the body element for the current state.

    CardKit v2 supports ``markdown`` element for the body — same parser
    as ``tag: md``, so existing content renders the same way once the
    card is up. Truncate very long content to avoid Feishu's 30KB
    payload cap (Phase 1's MAX_MESSAGE_LENGTH aligns but the card
    path has its own tighter limit).
    """
    MAX_CARD_BODY = 25000  # leaves 5KB buffer under 30KB cap
    if len(content) > MAX_CARD_BODY:
        content = content[:MAX_CARD_BODY - 50] + "\n\n…(内容过长,已截断)"

    text = content
    if state == CardState.THINKING:
        text = "_⏳ 正在思考…_"
    elif state == CardState.STREAMING and not content.strip():
        text = "_⏳ 正在生成第一条内容…_"
    elif state == CardState.ERROR:
        text = f"**⚠️ 错误**\n\n{error or '未知错误'}"
    elif state == CardState.FINAL and not content.strip():
        text = "_（响应为空）_"

    return {
        "tag": "markdown",
        "content": text,
    }


def _build_header(state: CardState) -> Dict[str, str]:
    """Build the CardKit v2 header.

    Title carries the state label so users see "完成" without scrolling.
    Template colors are LIGHT blue/green/orange/red — dark variants are
    unreadable on Feishu's mobile dark theme.
    """
    title_map = {
        CardState.THINKING: "💭 思考中",
        CardState.STREAMING: "✍️ 生成中",
        CardState.FINAL: "✅ 完成",
        CardState.ERROR: "❌ 出错",
    }
    template_map = {
        CardState.THINKING: "blue",
        CardState.STREAMING: "blue",
        CardState.FINAL: "green",
        CardState.ERROR: "red",
    }
    return {
        "title": {"tag": "plain_text", "content": title_map[state]},
        "template": template_map[state],
    }


def render_streaming_card(
    session: StreamingCardSession,
    *,
    footer: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a CardKit v2 payload for the current session state.

    Args:
        session: Per-message state — content buffer + state enum.
        footer: Optional plain-text line appended below the body (e.g.
            a citation link or a "(see also: …)" pointer).

    Returns:
        A Feishu ``interactive``-type payload. The adapter sends this
        via ``msg_type=interactive``; on subsequent updates the same
        payload is patched (only ``elements`` change).
    """
    content = "".join(session.content_buffer)
    elements: List[Dict[str, Any]] = [_build_body_element(session.state, content, session.error_message)]

    if footer:
        elements.append({
            "tag": "markdown",
            "content": f"---\n{footer}",
        })

    return {
        "config": {"wide_screen_mode": True},
        "header": _build_header(session.state),
        "elements": elements,
    }


def transition(session: StreamingCardSession, new_state: CardState, *, error: Optional[str] = None) -> StreamingCardSession:
    """Mutate ``session`` to ``new_state`` and return it (for chaining).

    Idempotent — re-applying the same state is a no-op. Refuses to go
    backwards from FINAL/ERROR (terminal states).
    """
    terminal = {CardState.FINAL, CardState.ERROR}
    if session.state in terminal and new_state != session.state:
        # Final/error are terminal — refuse to unstick. Caller must
        # allocate a new session for a new response.
        return session
    session.state = new_state
    if new_state == CardState.ERROR and error:
        session.error_message = error
    return session


__all__ = [
    "CardState",
    "StreamingCardSession",
    "render_streaming_card",
    "transition",
]