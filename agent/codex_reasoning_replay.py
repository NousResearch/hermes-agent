"""Replay policy for Responses ``reasoning.encrypted_content``: a durable, issuer-scoped
deny-list plus a bounded replay window for proxy issuers.

``encrypted_content`` is sealed to the backend identity that minted it, and the provider answers a
blob replayed from anywhere else with HTTP 400 ``invalid_encrypted_content`` ("... was not issued
to this caller"). A proxy/aggregator route (``other:*``) can rotate that identity between turns or
across a gateway restart.

The recovery in ``agent/turn_recovery.py`` used to answer with a process-wide bool
(``AIAgent._codex_reasoning_replay_enabled = False``): a gateway restart resurrected replay and
lost the same history again, and a different issuer on the same session lost replay it never
needed to lose. Two rules replace it:

- Replay is denied per ``(issuer_kind, issuer_model)`` pair. The pair whose blob was rejected is
  recorded in the session DB ``state_meta`` store (the same key/value mechanism the goal, loop and
  heartbeat bookkeeping uses), so the deny survives a process restart, and only that pair stops
  replaying.
- For proxy issuers only the ``agent.codex_proxy_replay_turns`` most recent assistant turns carry
  encrypted reasoning on the wire (default 2), which bounds the damage a single rejected blob can
  do. First-party issuers (``codex_backend``, ``xai_responses``, ``github_responses``) replay in
  full, as before.

Owner of both rules: callers read :func:`denied_pairs` / :func:`proxy_replay_max_turns` and thread
them into the converter; the durable store and the pair identity live here, nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import suppress
from typing import Any, Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ``state_meta`` key prefix for the deny-list rows (one row per issuer pair).
DENY_META_PREFIX = "codex_replay_deny:"

# Replay window for proxy/aggregator issuers (``agent.codex_proxy_replay_turns``).
DEFAULT_PROXY_REPLAY_TURNS = 2

# (issuer_kind, issuer_model) — the identity that sealed the blob. ``issuer_model`` is None on
# items persisted before model stamping, and when the wire model resolves to nothing.
IssuerPair = Tuple[str, Optional[str]]


def _pair_key(issuer_kind: str, issuer_model: Optional[str]) -> str:
    """``state_meta`` key for one issuer pair. Hashed because an ``other:<url>`` kind is long and
    carries ``:`` / ``/`` (and a raw pair would make the LIKE-prefix read ambiguous)."""
    digest = hashlib.sha1(f"{issuer_kind}\x00{issuer_model or ''}".encode("utf-8")).hexdigest()
    return f"{DENY_META_PREFIX}{digest[:16]}"


def _encode_pair(issuer_kind: str, issuer_model: Optional[str]) -> str:
    return json.dumps({"issuer_kind": issuer_kind, "issuer_model": issuer_model})


def _decode_pair(raw: Any) -> Optional[IssuerPair]:
    """Invert :func:`_encode_pair`; None for a row that is missing or unreadable."""
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    kind = payload.get("issuer_kind")
    if not isinstance(kind, str) or not kind:
        return None
    model = payload.get("issuer_model")
    return kind, model if isinstance(model, str) else None


def load_denied_pairs(session_db: Any) -> Set[IssuerPair]:
    """Every recorded pair, or an empty set when no store is bound or it cannot be read.

    A failed read must never break the request path: replay then simply looks enabled, which is
    exactly the state before the feature existed.
    """
    if session_db is None:
        return set()
    try:
        rows = session_db.list_meta_prefix(DENY_META_PREFIX)
    except Exception:
        logger.debug("Recovered reasoning replay: deny-list read failed", exc_info=True)
        return set()
    pairs: Set[IssuerPair] = set()
    for _key, raw in rows or ():
        pair = _decode_pair(raw)
        if pair is not None:
            pairs.add(pair)
    return pairs


def record_denied_pair(session_db: Any, issuer_kind: str, issuer_model: Optional[str]) -> bool:
    """Persist ``(issuer_kind, issuer_model)`` as denied. True when the row was durably stored.

    One row per pair (hashed key, upsert) rather than a rewritten list: no read-modify-write race
    between two agents in one process, and each recovery costs a single small write.
    """
    if session_db is None or not issuer_kind:
        return False
    try:
        session_db.set_meta(_pair_key(issuer_kind, issuer_model), _encode_pair(issuer_kind, issuer_model))
        return True
    except Exception:
        logger.warning(
            "Recovered reasoning replay: could not persist the deny for %s/%s; "
            "this process still skips it, a restart will replay it once",
            issuer_kind, issuer_model, exc_info=True,
        )
        return False


def _looks_like_store(db: Any) -> bool:
    """The two ``state_meta`` calls this module makes — a stub or a wrong object is not a store."""
    return db is not None and hasattr(db, "list_meta_prefix") and hasattr(db, "set_meta")


def _agent_session_db(agent: Any) -> Any:
    """The agent's session DB, opening the default one when a frontend did not supply it.

    Only ever called behind the once-per-agent cache in :func:`denied_pairs`, so this is one lazy
    attach per agent — the same pattern the session_search tool uses (``_get_session_db_for_recall``,
    which also refuses to open the canonical state DB for a persistence-isolated fork).
    """
    db = getattr(agent, "_session_db", None)
    if _looks_like_store(db):
        return db
    getter = getattr(agent, "_get_session_db_for_recall", None)
    if callable(getter):
        with suppress(Exception):
            candidate = getter()
            if _looks_like_store(candidate):
                return candidate
    return None


def denied_pairs(agent: Any) -> Set[IssuerPair]:
    """Denied issuer pairs for ``agent``, loaded from the durable store once and cached on it.

    The cached set is the process-local truth: :func:`deny_pair` adds to it directly, so the retry
    inside the current process skips the pair even if it could not be persisted.
    """
    cached = getattr(agent, "_codex_replay_denied_pairs", None)
    if isinstance(cached, set):
        return cached
    pairs = load_denied_pairs(_agent_session_db(agent))
    with suppress(Exception):
        agent._codex_replay_denied_pairs = pairs
    return pairs


def deny_pair(agent: Any, issuer_kind: Optional[str], issuer_model: Optional[str]) -> bool:
    """Deny replay for one issuer pair: always in memory, durably when a store is reachable.

    Returns True when the deny was persisted (False = in-process only, e.g. no session DB).
    """
    if not issuer_kind:
        return False
    denied_pairs(agent).add((issuer_kind, issuer_model))
    return record_denied_pair(_agent_session_db(agent), issuer_kind, issuer_model)


def pair_is_denied(
    issuer_kind: Optional[str], issuer_model: Optional[str], denied: Optional[Iterable[IssuerPair]],
) -> bool:
    """True when ``(issuer_kind, issuer_model)`` must not replay under ``denied``.

    A pair matches when it is recorded exactly, when the entry carries no model (deny the whole
    endpoint), or when the probe carries no model (an item persisted before model stamping — the
    endpoint is denied, so its unstamped blobs are too).
    """
    if not issuer_kind or not denied:
        return False
    for entry_kind, entry_model in denied:
        if entry_kind == issuer_kind and (entry_model == issuer_model or entry_model is None or issuer_model is None):
            return True
    return False


def replay_denied_for(
    denied: Optional[Iterable[IssuerPair]], issuer_kind: Optional[str], issuer_model: Optional[str],
) -> bool:
    """Replay gate for one request's issuer pair (the ``replay_encrypted_reasoning`` side)."""
    return pair_is_denied(issuer_kind, issuer_model, denied)


def current_issuer_pair(agent: Any) -> IssuerPair:
    """``(issuer_kind, issuer_model)`` of the request that just failed.

    The transport stamps exactly what it put on the wire (including a fast-mode model rewrite), so
    its stamp wins; a route/model re-derivation is the fallback for callers that never built a
    request through the transport (tests, ad-hoc conversions).
    """
    from agent.codex_responses_adapter import (
        _classify_responses_issuer, _wire_model_identity, classify_responses_route,
    )

    kind = model = None
    with suppress(Exception):
        transport = agent._get_transport()
        kind = getattr(transport, "_last_issuer_kind", None)
        model = getattr(transport, "_last_issuer_model", None)
    if not isinstance(kind, str) or not kind:
        kind = None
        with suppress(Exception):
            kind = _classify_responses_issuer(
                base_url=getattr(agent, "base_url", None), **classify_responses_route(agent)._asdict(),
            )
    if not isinstance(model, str) or not model:
        model = None
        with suppress(Exception):
            model = _wire_model_identity(getattr(agent, "model", None))
    return kind or "other", model


def proxy_replay_turns_from_config(raw: Any) -> int:
    """``agent.codex_proxy_replay_turns``: non-negative int (0 = replay nothing for proxy issuers);
    absent/invalid falls back to :data:`DEFAULT_PROXY_REPLAY_TURNS` with a warning."""
    if raw is None:
        return DEFAULT_PROXY_REPLAY_TURNS
    value: Optional[int] = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif not isinstance(raw, bool):
        with suppress(TypeError, ValueError):
            value = int(str(raw).strip())
    if value is None or value < 0:
        logger.warning(
            "agent.codex_proxy_replay_turns=%r is not a non-negative integer; using %d",
            raw, DEFAULT_PROXY_REPLAY_TURNS,
        )
        return DEFAULT_PROXY_REPLAY_TURNS
    return value


def proxy_replay_max_turns(agent: Any) -> int:
    """Replay window (in assistant turns) for proxy issuers on ``agent``'s route."""
    raw = getattr(agent, "codex_proxy_replay_turns", None)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return DEFAULT_PROXY_REPLAY_TURNS
    return raw
