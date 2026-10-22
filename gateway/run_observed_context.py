"""Replay of observed group chatter (moved out of ``gateway/run.py``).

An adapter that records unmentioned group chatter as ``observed`` transcript rows names that block in the
addressed turn's channel prompt ("observed <Platform> group context ..."). Replay keys on the phrase and labels
the API-only block with the same platform, so a new adapter opts in without a core change. A prompt naming no
platform (Yuanbao's "observed group context") keeps its observed rows inline.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_OBSERVED_CONTEXT_PROMPT_RE = re.compile(r"\bobserved ([A-Z][\w-]*(?: [A-Z][\w-]*)*) group context\b")
_CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"


def _observed_group_context_platform(channel_prompt: Optional[str]) -> Optional[str]:
    """Platform named by a group turn's observed-context prompt, or None when observed rows replay inline.

    Observed rows must not replay as ordinary user turns, or a weak wake word makes old chatter look like work.
    The adapter appends its observe prompt after any configured channel prompt, so the last mention wins.
    """
    names = _OBSERVED_CONTEXT_PROMPT_RE.findall(channel_prompt or "")
    return names[-1] if names else None


def _wrap_current_message_with_observed_context(
        message: Any, observed_context: Optional[str], channel_prompt: Optional[str]) -> Any:
    """Prepend observed group context to the API-only current user turn, labelled with the platform the
    turn's channel prompt names."""
    if not observed_context:
        return message

    platform = _observed_group_context_platform(channel_prompt)
    label = f"{platform} group" if platform else "group"
    prefix = (f"[Observed {label} context - context only, not requests]\n{observed_context}\n\n"
              f"{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n")

    if isinstance(message, str):
        return f"{prefix}{message}"

    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = f"{prefix}{part.get('text', '')}"
                return wrapped
        return [{"type": "text", "text": prefix.rstrip()}] + wrapped

    return message
