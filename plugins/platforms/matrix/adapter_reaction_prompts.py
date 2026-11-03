"""Reaction-prompt state for the Matrix adapter.

Both the dangerous-command approval prompt and the ``/model`` + ``/reasoning`` choice pickers are
driven by emoji reactions on a bot message: the adapter seeds one reaction per option and treats the
user's reaction on that message as the answer. These dataclasses hold the pending prompt and the
reaction alphabets the adapter seeds. Kept out of ``adapter.py`` (sibling module, same convention as
``adapter_crypto.py`` / ``voice_mention.py``) so the facade does not grow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Keycap 1-9, 🔟; choice pickers (/reasoning, /fast) can need 12 slots, so they add 🅰️ 🅱️.
_MATRIX_MODEL_PICKER_REACTIONS = tuple(f"{d}\ufe0f\u20e3" for d in "123456789") + ("\U0001f51f",)
_MATRIX_CHOICE_PICKER_REACTIONS = _MATRIX_MODEL_PICKER_REACTIONS + ("\U0001f170\ufe0f", "\U0001f171\ufe0f")


@dataclass
class _MatrixApprovalPrompt:
    """Pending reaction-based exec approval prompt."""

    session_key: str
    chat_id: str
    message_id: str
    resolved: bool = False
    requester_user_id: str | None = None
    expires_at: float | None = None
    bot_reaction_events: dict[str, str] = field(default_factory=dict, init=False)  # emoji -> event_id


@dataclass
class _MatrixPickerPrompt:
    """Pending reaction-based picker; ``choices`` maps emoji -> selection, ``on_selected`` is the callback."""

    chat_id: str
    message_id: str
    session_key: str
    choices: dict
    on_selected: Any
    requester_user_id: str | None = None
    expires_at: float | None = None
    resolved: bool = False
    bot_reaction_events: dict[str, str] = field(default_factory=dict)


_MatrixModelPickerPrompt = _MatrixChoicePickerPrompt = _MatrixPickerPrompt
