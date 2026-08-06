"""Input shaping for the Claude Agent SDK — text, images, and the capability probe.

Hermes carries user turns and auxiliary prompts as OpenAI-shaped content
(a string, or a list of ``{"type": "text"|"image_url", ...}`` parts).  The SDK
takes either a plain string or an async iterable of *stream-json* frames — the
same envelope the Claude Code CLI reads on stdin — whose ``content`` is
Anthropic-shaped blocks.

This module owns that translation, and the honest answer to "can the installed
SDK carry an image at all?".  Both matter because the alternative behaviours are
unacceptable: silently dropping the image, or falling back to the pre-SDK direct
HTTP path this runtime exists to remove
(``docs/design/claude-subscription-via-agent-sdk.md`` § 3).

Deliberately free of any dependency on :mod:`agent.anthropic_adapter`: that
module is the legacy direct-OAuth path and is scheduled for deletion, so the
SDK runtime must not grow a link to it.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ClaudeImageInputUnsupported(RuntimeError):
    """Raised when an image cannot be delivered through the installed SDK."""


# OpenAI-shaped part types that carry an image.
IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<params>;[^,]*)?,(?P<data>.*)$",
    re.DOTALL,
)

# What a caller is told when the runtime cannot carry the image itself. Points
# at the one supported alternative rather than at a provider we would pick for
# the user: the vision task's provider is an explicit config decision because it
# is a second billing source.
VISION_PROVIDER_REQUIRED_MESSAGE = (
    "The installed claude-agent-sdk cannot carry image input, so the Claude "
    "subscription runtime cannot see this image. Configure an explicit vision "
    "backend under `auxiliary.vision` in config.yaml (provider + model) and "
    "Hermes will describe the image through it instead. The image was NOT sent."
)


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


def _accepts_async_iterable(fn: Any) -> bool:
    """True when *fn*'s ``prompt`` parameter is annotated to take a stream.

    Signature introspection rather than a version comparison: the pin in
    ``pyproject.toml`` is a range, and what matters is what the installed
    package can actually be handed.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    param = parameters.get("prompt")
    if param is None:
        return False
    annotation = param.annotation
    if annotation is inspect.Parameter.empty:
        return False
    return "asynciterable" in str(annotation).lower().replace(" ", "")


def sdk_supports_streaming_input() -> bool:
    """True when this SDK build accepts structured (image-capable) input.

    Checks both entry points Hermes uses — the one-shot ``query()`` for
    auxiliary work and ``ClaudeSDKClient.query()`` for the conversation — because
    a build that only supports one of them supports neither for our purposes.
    Never raises: a missing extra is simply "no".
    """
    try:
        from claude_agent_sdk import ClaudeSDKClient, query
    except Exception:
        return False
    return _accepts_async_iterable(query) and _accepts_async_iterable(
        ClaudeSDKClient.query
    )


# ---------------------------------------------------------------------------
# Content translation
# ---------------------------------------------------------------------------


def _part_type(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("type") or "")
    return ""


def content_has_images(content: Any) -> bool:
    """True when OpenAI-shaped *content* carries at least one image part."""
    if not isinstance(content, list):
        return False
    return any(_part_type(part) in IMAGE_PART_TYPES for part in content)


def messages_have_images(messages: Any) -> bool:
    """True when any message in an OpenAI-shaped list carries an image."""
    if not isinstance(messages, list):
        return False
    for message in messages:
        if isinstance(message, dict) and content_has_images(message.get("content")):
            return True
    return False


def _image_source_from_url(url: str) -> Optional[Dict[str, Any]]:
    """Build an Anthropic image ``source`` from an OpenAI image URL.

    Handles the two forms Hermes actually produces: a base64 data URL (the
    common case — screenshots, pasted images, tool results) and a remote
    ``http(s)`` URL.  Anything else returns None so the caller can drop the part
    rather than send a block the CLI will reject.
    """
    url = str(url or "").strip()
    if not url:
        return None
    if url.startswith("data:"):
        match = _DATA_URL_RE.match(url)
        if not match:
            return None
        params = match.group("params") or ""
        if "base64" not in params.lower():
            # Only base64 payloads are representable as an Anthropic source.
            return None
        data = (match.group("data") or "").strip()
        if not data:
            return None
        return {
            "type": "base64",
            "media_type": match.group("mime") or "image/png",
            "data": data,
        }
    if url.startswith(("http://", "https://")):
        return {"type": "url", "url": url}
    return None


def content_to_sdk_blocks(content: Any) -> List[Dict[str, Any]]:
    """Translate OpenAI-shaped content into Anthropic-shaped SDK blocks.

    A bare string becomes a single text block.  Unrepresentable image parts are
    replaced with a visible text note instead of being dropped silently — a
    turn that quietly loses the user's screenshot reads to the user as the model
    ignoring them.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    blocks: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue
        ptype = _part_type(part)
        if ptype in {"text", "input_text"}:
            text = str(part.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif ptype in IMAGE_PART_TYPES:
            raw = part.get("image_url")
            url = raw.get("url", "") if isinstance(raw, dict) else str(raw or "")
            source = _image_source_from_url(url)
            if source is not None:
                blocks.append({"type": "image", "source": source})
            else:
                logger.debug("claude sdk input: unsupported image url form")
                blocks.append(
                    {
                        "type": "text",
                        "text": "[an image was attached but could not be encoded]",
                    }
                )
        elif ptype:
            # Already-Anthropic blocks (thinking, tool_result, ...) pass through.
            blocks.append(dict(part))
    return blocks


def blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Flatten SDK blocks back to plain text (images become a placeholder)."""
    parts: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            parts.append("[image]")
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Stream-json prompt
# ---------------------------------------------------------------------------


def build_sdk_user_frame(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One stream-json user frame carrying *blocks*.

    ``session_id`` is deliberately absent: both ``ClaudeSDKClient.query`` and
    the one-shot ``query()`` fill it in for us, and hard-coding one here is how
    an auxiliary call would end up writing into the conversation's session.
    """
    return {
        "type": "user",
        "message": {"role": "user", "content": blocks},
        "parent_tool_use_id": None,
    }


class ReplayableStreamPrompt:
    """An async-iterable prompt over materialized frames, safe to re-send.

    A bare async generator is single-use: the stale-session recovery path
    re-submits the prompt after a failed attempt, and a consumed generator
    would silently send an empty turn. This object returns a fresh iterator
    from every ``__aiter__`` call, so a retry replays the same frames.

    Frames are built eagerly on the calling thread but only iterated inside
    the SDK's own event loop, which keeps it safe to hand across the session
    facade's thread boundary.
    """

    def __init__(self, frames: List[Dict[str, Any]]) -> None:
        self.frames = list(frames)

    def __aiter__(self):
        async def _iterate():
            for frame in self.frames:
                yield frame

        return _iterate()

    def with_prefixed_text(self, text: str) -> "ReplayableStreamPrompt":
        """A copy whose first user frame carries *text* as its leading block.

        This is how the one-time context/history bootstrap rides an image
        turn: string prompts are concatenated, frame prompts get the same
        text as the first content block of the same user message — one turn
        either way, so role alternation is untouched.
        """
        if not text:
            return self
        frames = [dict(f) for f in self.frames]
        first = frames[0]
        message = dict(first.get("message") or {})
        content = [{"type": "text", "text": text}, *(message.get("content") or [])]
        message["content"] = content
        first["message"] = message
        frames[0] = first
        return ReplayableStreamPrompt(frames)


def make_sdk_stream_prompt(blocks: List[Dict[str, Any]]) -> Any:
    """Return a replayable async-iterable prompt yielding one user frame."""
    return ReplayableStreamPrompt([build_sdk_user_frame(blocks)])


def prepend_text_to_prompt(prompt: Any, text: str) -> Any:
    """Attach bootstrap *text* ahead of *prompt*, whatever shape it has.

    The Claude runtime's one-time bootstrap (Hermes context + prior history)
    used to do ``text + prompt`` unconditionally, which raises on the frame
    shape image turns produce — and bootstrap runs on the first turn of every
    session, so an image in the opening message crashed the turn.
    """
    if not text:
        return prompt
    if isinstance(prompt, str):
        return text + prompt
    if isinstance(prompt, ReplayableStreamPrompt):
        return prompt.with_prefixed_text(text)
    raise TypeError(
        f"cannot prepend bootstrap text to prompt of type {type(prompt).__name__}"
    )


def prompt_for_content(content: Any) -> Any:
    """Return the prompt to hand the SDK for OpenAI-shaped *content*.

    A plain string stays a plain string — the overwhelmingly common case, and
    the cheapest thing to send.  Structured content becomes a stream-json frame.

    Raises ``ClaudeImageInputUnsupported`` when the content carries an image and
    the installed SDK cannot carry one.
    """
    if isinstance(content, str):
        return content
    if not content_has_images(content):
        blocks = content_to_sdk_blocks(content)
        return blocks_to_text(blocks)
    if not sdk_supports_streaming_input():
        raise ClaudeImageInputUnsupported(VISION_PROVIDER_REQUIRED_MESSAGE)
    return make_sdk_stream_prompt(content_to_sdk_blocks(content))


__all__ = [
    "ClaudeImageInputUnsupported",
    "IMAGE_PART_TYPES",
    "VISION_PROVIDER_REQUIRED_MESSAGE",
    "blocks_to_text",
    "build_sdk_user_frame",
    "content_has_images",
    "content_to_sdk_blocks",
    "make_sdk_stream_prompt",
    "prepend_text_to_prompt",
    "ReplayableStreamPrompt",
    "messages_have_images",
    "prompt_for_content",
    "sdk_supports_streaming_input",
]
