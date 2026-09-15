"""DAG compaction: turn a message range into a lineage-bearing summary node.

Compaction here is *referential*, not lossy.  The raw messages stay in the
store; the node that replaces them carries (a) a deterministic extractive
digest and (b) a lineage edge to every message it absorbed.  That is what lets
``lcm_expand`` reconstruct the compacted range in bounded pages.

No model call is made.  The digest is deterministic so compaction is
reproducible and testable, and so a compaction pass can never fail on a
provider error (the host's built-in compressor remains the rollback path).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# In-process metadata markers.  The wire sanitizers strip underscore-prefixed
# keys before an API call, so these never reach a provider.
LCM_SUMMARY_METADATA_KEY = "_lcm_summary"
LCM_NODE_METADATA_KEY = "_lcm_node_id"
LCM_LEVEL_METADATA_KEY = "_lcm_level"

# Recognised by the built-in compressor's own summary detection
# (``agent.context_compressor.COMPRESSED_SUMMARY_METADATA_KEY``).  Setting it
# keeps LCM output classified as a context summary if another engine ever
# consumes this transcript.
COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"

_BULLET_EXCERPT_CHARS = 160


def is_lcm_summary(message: Any) -> bool:
    """True for a summary node rendered by this engine."""
    return isinstance(message, dict) and bool(message.get(LCM_SUMMARY_METADATA_KEY))


def summary_node_id(message: Any) -> Optional[str]:
    """Return the node id carried by an LCM summary message, if any."""
    if not is_lcm_summary(message):
        return None
    node_id = message.get(LCM_NODE_METADATA_KEY)
    return str(node_id) if node_id else None


def _visible_role(message: Any) -> Optional[str]:
    """Return the role a template counts for alternation, or ``None``.

    System and tool rows are skipped by strict chat templates, so they cannot
    collide with the marker; only user/assistant matter.
    """
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    return role if role in ("user", "assistant") else None


def choose_summary_role(
    head: Sequence[Dict[str, Any]], tail: Sequence[Dict[str, Any]]
) -> str:
    """Pick a marker role that alternates against the protected neighbours.

    The common shapes (``user → marker → user`` and
    ``assistant → marker → assistant``) both come out valid.  The one shape
    where no single role works — head ending in ``assistant`` and tail opening
    with ``user`` — is resolved by preferring to preserve the *head* (the
    prompt-cache prefix) and leaving the collision to the host's
    ``repair_message_sequence`` pass, exactly as the built-in compressor does.
    """
    last_head = next(
        (role for role in (_visible_role(m) for m in reversed(list(head))) if role),
        None,
    )
    first_tail = next(
        (role for role in (_visible_role(m) for m in list(tail)) if role),
        None,
    )
    if last_head == "assistant" and first_tail != "user":
        return "user"
    return "assistant"


def plan_compaction(
    messages: Sequence[Dict[str, Any]],
    *,
    protect_first_n: int,
    protect_last_n: int,
) -> Optional[Tuple[int, int]]:
    """Return ``(head_end, tail_start)`` for the compactable middle, or ``None``.

    Leading system messages plus the first ``protect_first_n`` non-system
    messages are protected verbatim, as are the last ``protect_last_n``
    messages.  Nothing to compact returns ``None`` rather than a degenerate
    empty range.
    """
    total = len(messages)
    if total == 0:
        return None

    head_end = 0
    protected_non_system = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            break
        role = message.get("role")
        if role == "system" and protected_non_system == 0:
            head_end = index + 1
            continue
        if protected_non_system < max(0, int(protect_first_n)):
            protected_non_system += 1
            head_end = index + 1
            continue
        break

    tail_start = max(head_end, total - max(0, int(protect_last_n)))
    if tail_start <= head_end:
        return None
    return head_end, tail_start


def build_summary(
    messages: Sequence[Dict[str, Any]],
    *,
    node_id: str,
    level: int,
    max_chars: int,
    page_size: int,
    prior_node_ids: Sequence[str] = (),
) -> str:
    """Render a bounded, deterministic digest for a compacted range.

    The recovery pointer is emitted first and never truncated; the bullet list
    is what gets trimmed when the budget runs out, and any bullets dropped are
    accounted for explicitly rather than silently.
    """
    count = len(messages)
    header = (
        f"[LCM summary node {node_id}] "
        f"Compacted {count} message(s) at DAG level {level}. "
        f"The raw messages are retained unchanged and recoverable in bounded "
        f"pages: call lcm_expand with node_id=\"{node_id}\" "
        f"(page_size <= {page_size})."
    )
    if prior_node_ids:
        header += " Superseded summary node(s): " + ", ".join(prior_node_ids) + "."

    if count == 0:
        return header[:max_chars]

    budget = max(0, max_chars - len(header) - 1)
    bullets: List[str] = []
    used = 0
    omitted = 0
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown") if isinstance(message, dict) else "unknown"
        text = _excerpt(message)
        bullet = f"- [{index}] ({role}) {text}"
        if used + len(bullet) + 1 > budget:
            omitted = count - index + 1
            break
        bullets.append(bullet)
        used += len(bullet) + 1

    body = header
    if bullets:
        body += "\n" + "\n".join(bullets)
    if omitted:
        body += (
            f"\n- ...{omitted} further message(s) omitted from this digest; "
            f"use lcm_expand to page them."
        )
    return body[:max_chars]


def _excerpt(message: Any) -> str:
    """One-line, whitespace-collapsed excerpt of a message."""
    text = ""
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                chunk.get("text", "")
                for chunk in content
                if isinstance(chunk, dict) and isinstance(chunk.get("text"), str)
            )
        elif content is not None:
            text = str(content)
        if not text and message.get("tool_calls"):
            text = "[tool call]"
    rendered = " ".join(str(text).split())
    if len(rendered) > _BULLET_EXCERPT_CHARS:
        rendered = rendered[: _BULLET_EXCERPT_CHARS - 1] + "…"
    return rendered


def build_summary_message(
    *,
    node_id: str,
    level: int,
    summary: str,
    role: str,
) -> Dict[str, Any]:
    """Wrap a summary digest as an OpenAI-format marker message."""
    return {
        "role": role,
        "content": summary,
        LCM_SUMMARY_METADATA_KEY: True,
        LCM_NODE_METADATA_KEY: node_id,
        LCM_LEVEL_METADATA_KEY: int(level),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
