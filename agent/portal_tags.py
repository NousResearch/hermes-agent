"""Centralized Nous Portal request tags.

Every Hermes request that hits the Nous Portal — main agent loop, auxiliary
client (compression / titles / vision / web_extract / session_search / etc.),
and any future code path — must carry the same product-attribution tags so
Nous can attribute usage to Hermes Agent and bucket it by client release.

Tag shape (sent in OpenAI-compatible ``extra_body['tags']``):

    [
        "product=hermes-agent",
        "client=hermes-client-v<__version__>",
    ]

The version is sourced live from ``hermes_cli.__version__`` so it auto-aligns
to whatever release is installed; the release script
(``scripts/release.py``) regex-bumps that single string, and every Portal
request picks up the new tag on the next process start.

Why one helper instead of inlining the literal at each site:
* Four call sites (main loop profile, aux client, run_agent compression
  fallback, web_tools fallback) used to drift apart — see PR #24194 which
  only got the aux site, leaving the main loop sending a different tag set.
* Tests should assert the same tag list everywhere; centralizing makes that
  assertion a one-liner against this module.

Do NOT pre-compute these as module-level constants in the consumers. The
version can change at runtime (editable installs, hot-reload tooling), and
``hermes_cli.__version__`` is the canonical source of truth.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import List, Optional

# ── Ambient conversation context ─────────────────────────────────────────────
#
# The main agent loop knows its ``session_id``; the dozens of auxiliary call
# sites (compression, title generation, vision, web_extract, session_search,
# MoA reference/aggregator slots, curator, kanban helpers, ...) do not — they
# funnel through ``agent.auxiliary_client.call_llm`` which has no session
# handle. Rather than threading a ``session_id`` parameter through every one
# of those call sites (and every future one), the agent loop publishes the
# active conversation id here and ``nous_portal_tags()`` picks it up as a
# fallback whenever no explicit ``session_id`` is passed.
#
# ContextVar (not a module global) so concurrent agents in one process —
# gateway sessions, delegate_task subagents, batch runners — never see each
# other's conversation id. Worker threads spawned via
# ``tools.thread_context.propagate_context_to_thread`` (background review,
# MoA fan-out, tool executor) inherit it through the copied Context; bare
# threads (title generator) capture it explicitly at spawn time.
_conversation_id: ContextVar[Optional[str]] = ContextVar(
    "nous_portal_conversation_id", default=None
)


def set_conversation_context(conversation_id: Optional[str]):
    """Publish the active conversation id for ambient Portal tagging.

    Called by the agent loop at turn entry with the conversation's stable
    id (the session-lineage ROOT id, so the tag survives context-compression
    session rotation). Pass ``None`` to clear. Returns the ContextVar token
    so callers can ``reset_conversation_context(token)`` on turn exit.
    """
    return _conversation_id.set(conversation_id or None)


def reset_conversation_context(token) -> None:
    """Restore the previous conversation context (pair with ``set_...``)."""
    try:
        _conversation_id.reset(token)
    except Exception:
        # Token from another Context (e.g. reset on a different thread) —
        # fall back to clearing rather than raising in cleanup paths.
        _conversation_id.set(None)


def get_conversation_context() -> Optional[str]:
    """Return the ambient conversation id, or ``None`` when unset."""
    return _conversation_id.get()


def _hermes_version() -> str:
    """Return the current Hermes release version, e.g. ``"0.13.0"``.

    Falls back to ``"unknown"`` if ``hermes_cli`` cannot be imported (should
    never happen in a real install — guarded for defensive testing).
    """
    try:
        from hermes_cli import __version__
        return __version__
    except Exception:
        return "unknown"


def hermes_client_tag() -> str:
    """Return the ``client=...`` tag for Nous Portal requests.

    Format: ``client=hermes-client-v<MAJOR>.<MINOR>.<PATCH>``.
    """
    return f"client=hermes-client-v{_hermes_version()}"


def conversation_tag(session_id: str) -> str:
    """Return the ``conversation=...`` tag for a Hermes session/conversation.

    Format: ``conversation=<session_id>``. ``session_id`` is the canonical
    Hermes conversation identifier (``AIAgent.session_id``) — the same value
    used for ``~/.hermes/sessions/`` storage, session logs, and lineage.

    Unlike the product/client tags this is high-cardinality (one value per
    conversation), so it is only appended when a session id is actually
    available — never as part of the always-on base tag set.
    """
    return f"conversation={session_id}"


def nous_portal_tags(session_id: str | None = None) -> List[str]:
    """Return the canonical list of Nous Portal product tags.

    Always returns a fresh list so callers can mutate it freely
    (e.g. ``merged_extra.setdefault("tags", []).extend(nous_portal_tags())``).

    When ``session_id`` is provided, a ``conversation=<session_id>`` tag is
    appended so Portal usage can be attributed to a specific Hermes
    conversation. When it is omitted, the ambient conversation context
    (``set_conversation_context``, published by the agent loop at turn
    entry) is used instead — this is how auxiliary calls (compression,
    titles, vision, MoA slots, ...) inherit the conversation tag without
    per-call-site plumbing. Callers outside any conversation (e.g. the
    auxiliary client's import-time base tags) get the canonical two-tag set.
    """
    tags = ["product=hermes-agent", hermes_client_tag()]
    # Ambient context first: the agent loop publishes the lineage ROOT id
    # (stable across context-compression rotation and delegate subagent
    # trees), which is the better conversation key than a per-segment
    # session_id passed explicitly. The explicit argument remains as a
    # fallback for callers running outside any agent turn.
    effective = get_conversation_context() or session_id
    if effective:
        tags.append(conversation_tag(effective))
    return tags
