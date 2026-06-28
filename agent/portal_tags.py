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

from typing import List


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
    conversation. Callers without a session id (e.g. the auxiliary client's
    always-on base tags) omit it and get the canonical two-tag set.
    """
    tags = ["product=hermes-agent", hermes_client_tag()]
    if session_id:
        tags.append(conversation_tag(session_id))
    return tags
