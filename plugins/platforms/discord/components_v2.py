"""Components V2 rendering for agent replies.

Discord's v2 component system lets one message carry structured layout (containers, text
blocks, separators) instead of a wall of markdown split across several messages. Hermes
replies are long and mixed prose/code, so v2 turns a multi-message spill into one scannable
message.

Three server-side limits drive every decision here, all verified against the API source
rather than the developer docs:

* ``MESSAGE_MAX_TOTAL_COMPONENTS_V2 = 40`` — total components anywhere in the tree, with no
  separate top-level cap.
* ``MAX_DISPLAYABLE_TEXT_SIZE = 4000`` — the **sum** of text across the whole tree, not a
  per-component budget. This is the important one: it is smaller than what the legacy
  chunker can deliver (8 x 2000), so v2 cannot be a blanket replacement for long replies.
* ``MAX_COMPONENT_DEPTH = 3``.

Because of the 4000-char total, :func:`build_layout_view` is deliberately partial: it
returns ``None`` whenever the content does not fit, and the caller falls back to the
existing chunked path. Rendering is therefore always an optimisation, never a constraint on
what Hermes can say.

One-way door: ``MessageFlags.IS_COMPONENTS_V2`` cannot be removed from a message once set,
and a v2 message rejects the legacy ``content`` field entirely. A message that starts life
as v2 must stay v2 for every subsequent edit, which is why the streaming edit path does not
use this module.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Server-side budgets (discord_api message_components/constants.py). Kept slightly under the
# real limits so format inflation on Discord's side cannot push a valid view over.
MAX_TOTAL_COMPONENTS = 40
MAX_DISPLAYABLE_TEXT = 4000
_TEXT_BUDGET = MAX_DISPLAYABLE_TEXT - 64
_COMPONENT_BUDGET = MAX_TOTAL_COMPONENTS - 4

# A single TextDisplay tops out at the same 4000 chars as the whole tree.
MAX_TEXT_DISPLAY = 4000

_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)", re.MULTILINE)


def _split_fenced(text: str) -> List[Tuple[str, str]]:
    """Split into ``(kind, chunk)`` segments where kind is ``code`` or ``prose``.

    Fenced code is kept intact so a block is never divided across components, which would
    break syntax highlighting and leave a dangling fence.
    """
    segments: List[Tuple[str, str]] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > cursor:
            prose = text[cursor:match.start()]
            if prose.strip():
                segments.append(("prose", prose))
        block = match.group()
        if block.strip():
            segments.append(("code", block))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        segments.append(("prose", tail))
    return segments


def plan_segments(content: str) -> Optional[List[Tuple[str, str]]]:
    """Plan the component tree for *content*, or ``None`` when v2 is not a good fit.

    Returns ``None`` (meaning "use the legacy chunked path") when the content is empty,
    exceeds the 4000-char total text budget, or would need more components than the budget
    allows. Callers must treat ``None`` as normal, not as an error.

    Exactly two conditions reject a plan: the summed text budget and the component count.
    Nothing splits prose by length, because ``_TEXT_BUDGET`` sits below the 4000-char
    single-component ceiling — any run of prose long enough to overflow one TextDisplay has
    already overflowed the whole-tree budget and been rejected. Segmentation therefore comes
    only from fenced code blocks, which is also what makes the output readable.
    """
    text = (content or "").strip()
    segments: List[Tuple[str, str]] = [
        (kind, chunk) for kind, chunk in _split_fenced(text) if chunk.strip()]

    if not segments:
        return None
    if sum(len(chunk) for _kind, chunk in segments) > _TEXT_BUDGET:
        return None
    # Each segment is a TextDisplay; separators sit between them, plus the Container itself.
    separators = max(0, len(segments) - 1)
    if len(segments) + separators + 1 > _COMPONENT_BUDGET:
        return None
    return segments


def build_layout_view(content: str, *, accent_colour: Optional[int] = None):
    """Build a ``LayoutView`` for *content*, or ``None`` to fall back to chunked text.

    Import is deferred so this module stays importable (and unit-testable) without discord.py.
    """
    segments = plan_segments(content)
    if segments is None:
        return None
    try:
        import discord
    except ImportError:
        return None

    separator_cls = getattr(discord.ui, "Separator", None)
    try:
        container = discord.ui.Container(accent_colour=accent_colour)
        for index, (_kind, chunk) in enumerate(segments):
            if index and separator_cls is not None:
                container.add_item(separator_cls())
            container.add_item(discord.ui.TextDisplay(chunk))
        view = discord.ui.LayoutView(timeout=None)
        view.add_item(container)
        return view
    except Exception:
        # Any library-side rejection (budget maths, version drift) degrades to legacy text
        # rather than dropping the reply.
        logger.debug("components v2 view construction failed; using legacy text", exc_info=True)
        return None
