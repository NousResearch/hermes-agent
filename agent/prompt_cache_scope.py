"""Rotation-stable logical cache scope for prompt_cache_key derivation.

Context-compression rotation (legacy ``compression.in_place: false`` mode)
mints a new physical ``session_id`` mid-conversation to segment the
transcript. The prompt-cache scope introduced by #79161 was derived from that
physical id, so every rotation moved the conversation into a fresh cache
bucket even though it is logically the same conversation continuing
(issue #79017).

``resolve_prompt_cache_scope()`` maps the physical session id to the ROOT of
its *compression lineage* — the pre-rotation session id — using
``SessionDB.get_compression_lineage()``, whose fork-aware semantics
(hardened in #79193) give exactly the scope boundaries the cache key needs:

- compression-rotation children walk back to the original segment
  (rotation-stable scope — the fix);
- ``/new`` starts a lineage-less session (fresh scope);
- ``/branch`` children (``_branched_from``), delegate subagents
  (``_delegate_from``), and tool-tagged children (``source="tool"``) are
  explicit fork children and keep their own isolated scope, preserving the
  sibling/subagent isolation #79161 established;
- cron fires keep their physical ``cron_<job>_<ts>`` id here — the per-fire
  timestamp is stripped later by ``_cache_scope_from_session_id`` exactly as
  before.

The resolution is memoized per (agent, session_id): the lineage walk runs
once per transcript segment — NOT per API call — and re-runs only when
rotation actually changes ``agent.session_id`` (per the no-DB-on-the-hot-path
constraint recorded on #79017). Default installs compact in place and never
rotate, so they hit the memo forever and behave byte-identically to before.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MEMO_ATTR = "_prompt_cache_scope_memo"


def _lineage_root(session_id: str, session_db: Any) -> Optional[str]:
    """Return the compression-lineage root of *session_id*, or None.

    Defensive about the DB handle: test doubles and partially constructed
    agents can hand back non-list results — anything that is not a non-empty
    list/tuple whose first element is a non-empty string is ignored.
    """
    if session_db is None:
        return None
    try:
        lineage = session_db.get_compression_lineage(session_id)
    except Exception:
        logger.debug("prompt-cache scope lineage walk failed", exc_info=True)
        return None
    if isinstance(lineage, (list, tuple)) and lineage:
        root = lineage[0]
        if isinstance(root, str) and root:
            return root
    return None


def resolve_prompt_cache_scope(agent: Any) -> str:
    """Resolve the rotation-stable cache-scope id for *agent*'s conversation.

    Returns the compression-lineage ROOT of ``agent.session_id`` (the
    physical id itself when the session has no compression ancestry, no DB
    is attached, or the walk fails). The result is memoized on the agent
    keyed by the current session id, so the DB walk happens once per
    transcript segment rather than once per API call.
    """
    sid = str(getattr(agent, "session_id", None) or "")
    if not sid:
        return ""
    memo = getattr(agent, _MEMO_ATTR, None)
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == sid:
        return memo[1]
    db = getattr(agent, "_session_db", None)
    root = _lineage_root(sid, db) if db is not None else None
    scope = root or sid
    # Memoize on a successful walk, or when there is no DB to consult at all.
    # A failed/empty walk (row not persisted yet, transient DB error) is NOT
    # memoized: falling back to the physical id is the correct degraded
    # answer right now, but pinning it for the whole segment would keep the
    # scope wrong after the session row lands.
    if root is not None or db is None:
        try:
            setattr(agent, _MEMO_ATTR, (sid, scope))
        except Exception:
            # Frozen/slotted test doubles — resolution still works, just
            # unmemoized.
            pass
    return scope
