"""Rotation-stable logical cache scope for prompt_cache_key derivation.

Legacy compression rotation mints a new physical ``session_id`` mid-conversation, moving it
into a fresh cache bucket. ``resolve_prompt_cache_scope()`` maps the physical id to the ROOT
of its compression lineage — NOT ``get_conversation_root`` (the Portal-attribution walk),
which would collapse /branch children and delegate trees into one id. ``/new`` starts a
fresh scope; fork children (branch, delegate, tool-tagged) are isolated. Hosts minting one id
per RESPONSE declare the conversation via ``gateway_session_key``, which wins over the lineage
walk and is hashed to ``gwk_<sha256[:24]>`` (it embeds platform/chat/user identifiers).
"""

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MEMO_ATTR = "_prompt_cache_scope_memo"
_DECLARED_SCOPE_PREFIX = "gwk_"


def _lineage_root(session_id: str, session_db: Any) -> Optional[str]:
    """Compression-lineage root of *session_id*, or None (tolerates test-double results)."""
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


def _agent_source(
    agent: Any, session_id: str, session_db: Any, row_source: Optional[str] = None
) -> str:
    """The ``sessions.source`` this agent's conversation is recorded under.

    ``row_source``: the row's value if already read (``""`` = read, none; ``None`` = look it
    up). Before the row lands, use the SAME resolver persistence uses, not ``agent.platform``:
    they diverge under ``HERMES_SESSION_SOURCE`` and the declared scope is memoized at once,
    so both sides of a ``/new`` would otherwise hash the same scope.
    """
    if row_source is None and session_id and session_db is not None:
        try:
            row = session_db.get_session(session_id)
        except Exception:
            logger.debug("declared-scope source lookup failed", exc_info=True)
            row = None
        row_source = str(row.get("source") or "").strip() if row else ""
    if row_source:
        return row_source
    platform = getattr(agent, "platform", None)
    try:
        # Lazy: run_agent imports this module.
        from run_agent import _session_source_for_agent

        source = str(_session_source_for_agent(platform) or "").strip()
        if source:
            return source
    except Exception:
        logger.debug("declared-scope source authority unavailable", exc_info=True)
    return str(platform or "").strip()


def _conversation_generation(session_key: str, source: str, session_db: Any) -> str:
    """Durable generation for *session_key*'s current conversation (``""`` if none).

    The declared key survives ``/new``, so hashing it alone would reuse one scope across
    conversations. The counter advances with each reset boundary, independent of prunable
    rows and wall-clock; compression does not advance it.

    The declared key names a chat and deliberately survives `/new` and policy resets. See #79017, #86733.
    """
    reader = getattr(session_db, "latest_conversation_boundary", None)
    if not callable(reader):
        return ""
    generation = reader(session_key, source)
    return "" if generation is None else str(int(generation))


def declared_conversation_scope(agent: Any) -> Optional[str]:
    """Host-declared logical conversation scope (``gwk_<sha256[:24]>``), or None.

    Hashes ``(source, gateway_session_key, generation)``. None (fall back to the physical id)
    when no key is declared, for a background-review fork (``_persist_disabled``), for an
    explicit fork child, and on any DB error (fail closed rather than merge a fork onto its
    parent's key).
    """
    key = str(getattr(agent, "_gateway_session_key", "") or "").strip()
    if not key or getattr(agent, "_persist_disabled", False):
        return None
    sid = str(getattr(agent, "session_id", None) or "")
    db = getattr(agent, "_session_db", None)
    generation = ""
    row_source: Optional[str] = None
    if sid and db is not None:
        try:
            # One read for both halves of the row identity (fork verdict + source).
            # One read for both halves of the row's identity: the fork verdict and the source the peer
            # queries match on live on the same ``sessions`` row, and asking for them separately read it
            # twice per resolution (@teknium1 on #98811). A SessionDB without the combined view keeps the
            # original call, so nothing that predates it — including the doubles that certify the
            # fail-closed contract below — changes behaviour.
            identity = getattr(db, "declared_scope_identity", None)
            if callable(identity):
                is_fork, row_source = identity(sid)
            else:
                is_fork = db.is_explicit_fork_child(sid)
            if is_fork:
                return None
        except Exception:
            logger.debug("declared-scope fork check failed", exc_info=True)
            return None
    source = _agent_source(agent, sid, db, row_source)
    if db is not None:
        try:
            generation = _conversation_generation(key, source, db)
        except Exception:
            logger.debug("declared-scope generation read failed", exc_info=True)
            return None
    # Same identity tuple the peer queries use: same key under different sources must not collapse.
    carrier = f"{source}|{key}|{generation}"
    digest = hashlib.sha256(carrier.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{_DECLARED_SCOPE_PREFIX}{digest}"


def resolve_prompt_cache_scope(agent: Any) -> str:
    """Rotation-stable cache-scope id: declared scope, else the compression-lineage root of
    ``agent.session_id`` (the physical id without ancestry/DB). Memoized on the agent."""
    sid = str(getattr(agent, "session_id", None) or "")
    if not sid:
        return ""
    db = getattr(agent, "_session_db", None)
    # DB presence is part of the key: an agent that gains a DB handle later must re-resolve.
    key = (sid, db is not None)
    memo = getattr(agent, _MEMO_ATTR, None)
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == key:
        return memo[1]
    root = declared_conversation_scope(agent) or _lineage_root(sid, db)
    scope = root or sid
    # Memoize on success, with no DB, or when the agent never persists a row. A failed/empty
    # walk on a persisting agent is NOT memoized: the physical id is right for now (row not
    # yet persisted) but would stay wrong for the whole segment once it lands.
    if root is not None or db is None or getattr(agent, "_persist_disabled", False):
        try:
            setattr(agent, _MEMO_ATTR, (key, scope))
        except Exception:
            pass  # frozen/slotted doubles: resolution works, just unmemoized
    return scope


def declared_conversation_scope_safe(agent: Any) -> Optional[str]:
    """Never-raising variant of :func:`declared_conversation_scope`."""
    try:
        return declared_conversation_scope(agent)
    except Exception:
        logger.debug("declared conversation scope resolution failed", exc_info=True)
        return None


def resolve_prompt_cache_scope_safe(agent: Any) -> Optional[str]:
    """Never-raising variant of :func:`resolve_prompt_cache_scope` (None = use the physical id).
    At turn_context an exception inside ``set_runtime_main(...)`` would skip the whole binding.

    Returns None on any failure (or when there is no scope). Consumers treat None/empty as "fall back to the
    physical session_id", so a resolution failure degrades to pre-#79017 behavior instead of blocking the
    caller — important at turn_context's call site, where an exception raised inside the
    ``set_runtime_main(...)`` argument list would otherwise skip the whole runtime binding, not just the
    cache scope.
    """
    try:
        return resolve_prompt_cache_scope(agent) or None
    except Exception:
        logger.debug("prompt-cache scope resolution failed", exc_info=True)
        return None
