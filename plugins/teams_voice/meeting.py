"""Meeting transcript accumulation + minutes prompt building.

The voice handlers append each caller/assistant turn to a :class:`MeetingTranscript`.
At call end (opt-in ``meeting_recap``) or on the ``post_meeting_minutes`` tool, the
handler asks the Hermes agent to summarize the transcript into minutes and post
them to the Teams conversation via the agent's own ``send_message`` tool — so the
cross-process delivery and DOCX creation are the agent's job, not the bridge's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MeetingTranscript:
    """Ordered (speaker, text) turns for end-of-call minutes."""

    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, speaker: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.turns.append((speaker or "Caller", text))

    def is_empty(self) -> bool:
        return not self.turns

    def render(self, max_chars: int = 12_000) -> str:
        body = "\n".join(f"{sp}: {tx}" for sp, tx in self.turns)
        # Keep the tail if very long (recent context matters most for minutes).
        return body[-max_chars:] if len(body) > max_chars else body


def is_summary_request(text: str) -> bool:
    """True if the caller asked to summarize / send minutes of the meeting."""
    t = (text or "").lower()
    summary = any(w in t for w in ("summarize", "summarise", "minutes", "recap", "notes"))
    subject = any(w in t for w in ("meeting", "call", "conversation", "discussion"))
    return summary and subject


def summarize_prompt(transcript: str) -> str:
    """Prompt the agent to produce minutes text only (no posting)."""
    return (
        "Summarize the transcript of this Microsoft Teams meeting into concise "
        "minutes with three sections — **Key Points**, **Decisions**, and **Action "
        "Items** (name owners where stated). Output only the minutes, briefly and "
        f"factually.\n\nTranscript:\n{transcript}"
    )


async def _deliver_to_teams(conversation_id: str, text: str) -> bool:
    """Post text to a Teams conversation via the adapter's standalone REST sender
    (works without the gateway running; reads bot creds from env)."""
    try:
        from plugins.platforms.teams.adapter import _standalone_send
    except Exception:  # noqa: BLE001 — teams adapter unavailable
        return False
    pconfig = type("_PConfig", (), {"extra": {}})()
    try:
        result = await _standalone_send(pconfig, chat_id=conversation_id, message=text)
        return bool(result.get("success"))
    except Exception:  # noqa: BLE001
        logger.error("[teams_voice] recap delivery failed", exc_info=True)
        return False


async def post_minutes(consult, transcript: "MeetingTranscript", conversation_id: str) -> str:
    """Summarize the transcript via the agent, then post the minutes to Teams.

    Returns a short spoken-result string. No-op on an empty transcript.
    """
    if transcript.is_empty() or not conversation_id:
        return "There wasn't enough of a conversation to summarize."
    try:
        minutes = await consult.ask(summarize_prompt(transcript.render()), timeout_s=120.0)
    except Exception:  # noqa: BLE001 — recap must never crash teardown
        logger.error("[teams_voice] meeting summary failed", exc_info=True)
        return "I couldn't summarize the meeting."
    minutes = (minutes or "").strip()
    if not minutes:
        return "I couldn't summarize the meeting."
    ok = await _deliver_to_teams(conversation_id, f"📝 **Meeting minutes**\n\n{minutes}")
    return (
        "I've posted the minutes to your Teams chat."
        if ok
        else "I summarized the meeting but couldn't post it to the chat."
    )
