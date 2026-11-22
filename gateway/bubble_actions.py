"""Typed semantic actions for discrete multi-bubble streaming.

Eliminates control marker injection by replacing string-prefix interpretation
with immutable, strongly typed action objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class TextBubble:
    """Ordinary unprivileged text bubble."""

    content: str
    reply_to_id: Optional[str] = None


@dataclass(frozen=True)
class PollAction:
    """Typed native poll intent produced from structured <poll> tags."""

    title: str
    options: tuple[str, ...]
    allow_multiple: bool = False


@dataclass(frozen=True)
class TapbackAction:
    """Typed native tapback reaction produced from structured <tapback> tags."""

    reaction: str
    target_message_id: Optional[str] = None


@dataclass(frozen=True)
class EffectAction:
    """Typed native screen or bubble effect produced from structured <effect> tags."""

    effect_type: str
    content: str


@dataclass(frozen=True)
class FileAttachmentAction:
    """Authorized file attachment produced by a trusted internal generator or explicit call."""

    staged_path: str
    file_name: Optional[str] = None
    caption: Optional[str] = None
    token: Optional[str] = None


@dataclass(frozen=True)
class ImageAttachmentAction:
    """Authorized image attachment produced by a trusted internal generator or explicit call."""

    staged_path: str
    caption: Optional[str] = None
    token: Optional[str] = None


BubbleAction = (
    TextBubble
    | PollAction
    | TapbackAction
    | EffectAction
    | FileAttachmentAction
    | ImageAttachmentAction
)
