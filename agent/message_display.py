"""Pure conversation display semantics shared by transcript and summary adapters.

Messages are already decoded by storage. Projection never mutates stored/model
content, reads configuration or resolves attachments. Consumer policy chooses
which visible message kinds are suitable for its surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from agent.compaction_display import project_compaction_message_for_display
from agent.conversation_compression import _extract_steer_text_from_message
from agent.first_task_prompt import visible_text
from agent.prompt_builder import STEER_DISPLAY_KIND
from agent.skill_commands import describe_skill_invocation


@dataclass(frozen=True)
class DisplayProjection:
    """Display content and semantics, with the projected row's detail fields."""

    message: Mapping[str, Any]
    content: Any
    kind: str | None
    visible: bool


def project_message_for_display(message: dict[str, Any]) -> DisplayProjection:
    """Project producer-defined envelopes; classify visibility only from metadata."""
    raw_metadata = message.get("display_metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    if metadata.get("model_only") is True:
        return DisplayProjection(MappingProxyType(message.copy()), None, message.get("display_kind"), False)
    if message.get("display_kind") == "hidden":
        # A compaction carrier may contain a live ask even when the handoff is hidden.
        if not message.get("_compressed_summary"):
            return DisplayProjection(MappingProxyType(message.copy()), None, message.get("display_kind"), False)
    projected = (project_compaction_message_for_display(message)
                 if message.get("_compressed_summary") else message.copy())
    if projected is None:
        return DisplayProjection(MappingProxyType(message.copy()), None, message.get("display_kind"), False)
    content = projected.get("content")
    kind = projected.get("display_kind")
    if projected.get("role") == "user":
        if kind == STEER_DISPLAY_KIND:
            extracted = _extract_steer_text_from_message(projected)
            if extracted is not None:
                content = extracted
        if isinstance(content, str) and not kind:
            content = visible_text(content)
            if invocation := describe_skill_invocation(content, separator=" "):
                content, kind = invocation, "skill_invocation"
    return DisplayProjection(MappingProxyType(projected), content, kind, True)


# Preview policy is narrower than transcript visibility: timeline notices and
# completions have their own activity presentation and cannot replace a conversation excerpt.
_PREVIEW_EXCLUDED_KINDS = frozenset({
    "model_switch", "personality_switch", "auto_continue",
    "async_delegation_complete", "process_complete",
})


def conversation_preview_text(projection: DisplayProjection) -> str:
    """Eligible conversation content, flattened before the roster truncates it."""
    if not projection.visible or projection.message.get("role") not in {"user", "assistant"}:
        return ""
    if projection.kind in _PREVIEW_EXCLUDED_KINDS:
        return ""
    return " ".join(render_message_content(projection.content, image_urls=False).split())


TEXT_CONTENT_KINDS = frozenset({"text", "input_text", "output_text"})
IMAGE_CONTENT_KINDS = frozenset({"image_url", "input_image", "image"})
AUDIO_CONTENT_KINDS = frozenset({"input_audio", "audio"})


def content_image_url(part: dict) -> str:
    """The URL carried by an image part (``image_url`` dict or str), else ""."""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return image_url if isinstance(image_url, str) else ""


def _structured_content_text(content: dict, *, image_urls: bool) -> str:
    """Placeholder/text rendering of one structured content dict."""
    kind = content.get("type")
    if kind in TEXT_CONTENT_KINDS:
        return str(content.get("text") or content.get("content") or "")
    if kind in IMAGE_CONTENT_KINDS:
        return (content_image_url(content) if image_urls else "") or "[image]"
    if kind in AUDIO_CONTENT_KINDS:
        return "[audio]"
    if kind:
        return f"[{kind}]"
    if "text" in content:
        return str(content.get("text") or "")
    return "[structured content]"


def render_message_content(content: Any, *, image_urls: bool = True) -> str:
    """Render ``message['content']`` (str, parts list, or one structured dict) as a plain string. Image parts
    keep their URL inline so the desktop's ``extractEmbeddedImages`` and the resume payload agree with the
    cached message (else the inline image flashed, then vanished); other shapes become a placeholder.
    ``image_urls=False`` renders ``[image]`` instead — the ``inline_images=false`` read (#116511): a remote
    client reads a transcript in kilobytes instead of re-transmitting every stored attachment."""
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str) or (isinstance(part, dict) and isinstance(part.get("text"), str)):
                chunks.append(part if isinstance(part, str) else part["text"])
            elif isinstance(part, dict) and part.get("type"):
                rendered = _structured_content_text(part, image_urls=image_urls)
                chunks.append(rendered if part["type"] in TEXT_CONTENT_KINDS else f"\n{rendered}")
        return "".join(chunks)
    if isinstance(content, dict):
        return _structured_content_text(content, image_urls=image_urls)
    return "" if content is None else str(content)
