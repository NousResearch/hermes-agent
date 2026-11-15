"""Did the assistant message carry reasoning? Decided once, for every caller.

The empty-response ladder has to answer "did the model actually say nothing?"
before it spends a retry on the model. It cannot answer that from ``content``
alone: several providers split the chain-of-thought out of the content and into
a dedicated field, so the content arrives EMPTY while the model in fact reasoned
in full. Treating that as silence costs a wasted round-trip per tool call — the
pathology reported in #21811.

Two shapes have to be recognised:

* **An inline tag.** The reasoning stayed in the content, wrapped in a think
  tag. MiniMax M2 is an interleaved-thinking model whose chat template
  PREFILLS the opening ``<think>``, so the completion arrives as
  ``…reasoning…</think>`` — a bare CLOSING tag with no opener. A detector
  that matches only opening tags misses it, the content strips to empty, and
  the ladder believes the model went silent.

  The tag names come from ``agent.think_scrubber.THINK_TAG_NAMES`` — the one
  list every surface that hides reasoning binds to — so a name added there is
  covered here too, and the OPENING and CLOSING forms are both matched.

* **A separate channel.** ``reasoning_content`` (Ollama qwen3.x PARSER,
  DeepSeek-R1, Moonshot, Novita), ``reasoning`` (Ollama's OpenAI-compatible
  endpoint maps its internal thinking field here — ollama/ollama#15288; also
  OpenRouter's flat field), ``reasoning_details`` (OpenRouter's unified
  format, and MiniMax M2's native OpenAI-compatible API, which return the
  chain-of-thought as a structured array) and ``thinking`` (Ollama's native
  ``/api/chat`` field, and some OpenAI-compatible proxies). Read as an
  attribute, as a dict key, or through the OpenAI SDK's ``model_extra``
  fall-through, which is where provider-specific fields live.

The two shapes answer DIFFERENT questions, and the ladder uses them
differently — see ``agent/turn_empty_response.py`` for why the inline shape
gates the post-tool nudge while the separate-channel shape gates the
thinking-only reminder.
"""

from __future__ import annotations

import re
from typing import Any

from agent.think_scrubber import THINK_TAG_NAMES

__all__ = [
    "REASONING_CHANNEL_FIELDS",
    "has_inline_reasoning_tag",
    "has_separate_reasoning_channel",
]

# The dedicated reasoning fields, in the order extract_reasoning reads them.
# ``reasoning_details`` is a list of dicts; ``reasoning`` may be a string.
REASONING_CHANNEL_FIELDS: tuple[str, ...] = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "thinking",
)

# An inline think tag, opening OR closing, in any of the canonical names.
# The closing form matters: MiniMax M2 prefills the opening <think>, so the
# completion carries only the closer, and an opener-only match misses it
# (#21811 residual). The names come from THINK_TAG_NAMES — the one list every
# reasoning-hiding surface binds to — so a name added there is covered here.
_INLINE_REASONING_TAG_RE = re.compile(
    r"</?(?:" + "|".join(re.escape(name) for name in THINK_TAG_NAMES) + r")\b",
    re.IGNORECASE,
)


def has_inline_reasoning_tag(content: Any) -> bool:
    """True when ``content`` carries a reasoning tag — opening or closing.

    A bare closing tag counts: MiniMax M2 prefills the opening ``<think>``,
    so the completion arrives as ``…reasoning…</think>`` and an opener-only
    match would miss it (#21811 residual).
    """
    if not content or not isinstance(content, str):
        # Typed content blocks ({"type": "thinking", ...}) are reasoning too,
        # but they carry no tag — extract_reasoning reads those.
        return False
    return bool(_INLINE_REASONING_TAG_RE.search(content))


def _field(message: Any, name: str) -> Any:
    """Read ``name`` off an assistant message, however the provider sent it.

    Accepts a pydantic-style attribute object (the OpenAI SDK message), a raw
    dict payload, and the SDK's ``model_extra`` fall-through, which is where
    provider-specific fields such as Ollama's ``thinking`` live.
    """
    if message is None:
        return None
    if isinstance(message, dict):
        value = message.get(name)
        if value:
            return value
        extra = message.get("model_extra")
        if isinstance(extra, dict):
            return extra.get(name)
        return None
    value = getattr(message, name, None)
    if value:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def has_separate_reasoning_channel(message: Any) -> bool:
    """True when the message carries reasoning in a field of its own.

    Covers ``reasoning``, ``reasoning_content``, ``reasoning_details`` and
    ``thinking``, read as an attribute, as a dict key, or through the OpenAI
    SDK's ``model_extra``. An empty value is not reasoning: an empty
    ``reasoning_details`` list must not count.
    """
    if message is None:
        return False
    for name in REASONING_CHANNEL_FIELDS:
        if _field(message, name):
            return True
    return False
