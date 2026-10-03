"""Canonical production factory for the EvidencePackEngine.

This module is the single composition seam for the
``EvidencePackEngine`` in the classic Hermes CLI. It owns:

* ``build_evidence_pack_engine`` — the actual factory callable used by
  ``HermesCLI._get_goal_manager`` and ``build_objective_services``. It
  is NOT a higher-order factory builder: it constructs and returns a
  fully composed engine per call.

Composition invariants (production):

* Storage is *borrowed* from the CLI (``HermesCLI._session_db``); the
  factory never constructs a ``SessionDB`` and never opens a database.
  The factory never closes the borrowed storage.
* The state loader is a private closure over the borrowed storage
  reading exactly the SessionDB key ``goal:<session_id>``.
  ``storage.get_meta`` failures are NOT swallowed — they propagate so
  the engine's source-failure accounting observes them. Missing / falsy
  values return ``None``. Stored JSON is parsed; malformed JSON and
  non-Mapping top-level values raise.
* The ``contract`` source is always installed by the factory — any
  caller-supplied ``"contract"`` source is replaced. Other supplied
  sources are preserved by identity. ``None`` means an empty source
  map; non-Mapping raises ``TypeError``.
* The audit sink defaults to a fresh
  ``EvidencePackMonitoringAuditSink`` bound to the process monitoring
  emitter. The monitoring subsystem owns the emitter singleton; the
  factory, engine, and adapter only *borrow* the emitter and never
  close it.

Default-off:

* This module is only invoked when ``goals.evidence_pack.enabled`` is
  truthy. When disabled, ``build_objective_services`` returns without
  invoking the factory, the emitter is not resolved, the contract
  provider is not constructed, no engine is built, and no storage read
  occurs.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional

from agent.executive.services import EvidencePackStorageUnavailable
from agent.executive.knowledge_discovery import EvidencePackEngine
from agent.executive.knowledge_discovery.adapters import (
    EvidencePackMonitoringAuditSink,
)
from agent.executive.knowledge_discovery.adapters.contract_provider import (
    make_contract_provider,
)


# Canonical SessionDB state_meta key for the active goal / contract.
# Mirrors ``STATE_META_GOAL_KEY`` from the contract adapter — duplicated
# here to keep the factory self-contained and avoid leaking adapter
# internals into the engine composition contract.
_GOAL_STATE_META_KEY = "goal:{session_id}"


def _has_callable_attr(obj: Any, name: str) -> bool:
    """True iff ``obj`` exposes a callable attribute named ``name``."""
    if obj is None:
        return False
    value = getattr(obj, name, None)
    return callable(value)


def _validate_session_id(session_id: Any) -> str:
    """Validate and return ``session_id`` as a non-empty string."""
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    return session_id


def _validate_storage(storage: Any) -> Any:
    """Validate borrowed storage or raise the typed unavailable signal.

    The borrowed storage must structurally expose callable ``get_meta``
    and ``set_meta``. We never construct ``SessionDB``, never open a
    database, and never close storage — the caller's lifecycle owns it.
    """
    if storage is None:
        raise EvidencePackStorageUnavailable(
            "evidence_pack storage is unavailable (None)"
        )
    if not _has_callable_attr(storage, "get_meta"):
        raise EvidencePackStorageUnavailable(
            "evidence_pack storage is unavailable (no get_meta)"
        )
    if not _has_callable_attr(storage, "set_meta"):
        raise EvidencePackStorageUnavailable(
            "evidence_pack storage is unavailable (no set_meta)"
        )
    return storage


def _make_state_loader(
    *,
    storage: Any,
    session_id: str,
) -> Callable[..., Optional[Mapping[str, Any]]]:
    """Build a private read-only closure over the borrowed storage.

    The loader reads ``storage.get_meta("goal:<session_id>")`` and
    parses the stored JSON. ``storage.get_meta`` exceptions propagate;
    missing/falsy values return ``None``; malformed JSON raises;
    non-Mapping top-level values raise ``ValueError``. The loader does
    not import ``hermes_cli`` and never mutates storage-owned data.

    The loader closes over ``session_id`` and ignores any argument the
    contract adapter passes back to it — the adapter protocol requires
    a ``(session_id: str) -> ...`` signature, but the canonical
    production loader is bound to a single session at factory time.
    """

    key = _GOAL_STATE_META_KEY.format(session_id=session_id)

    def _loader(_sid: Any = None) -> Optional[Mapping[str, Any]]:
        # storage.get_meta failure MUST propagate (per spec).
        raw = storage.get_meta(key)
        if not raw:
            return None
        # raw may be a JSON string (canonical SessionDB shape) or
        # already a mapping. Normalize to a string for json.loads so
        # we always go through the canonical parser.
        if isinstance(raw, Mapping):
            text = json.dumps(dict(raw))
        elif isinstance(raw, (str, bytes, bytearray)):
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        else:
            raise ValueError(
                f"storage.get_meta returned unsupported type {type(raw).__name__}"
            )
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            # Malformed JSON propagates (do not convert to missing state).
            raise
        if not isinstance(parsed, Mapping):
            raise ValueError(
                "stored state_meta value is not a JSON object"
            )
        # Defensive shallow copy so callers cannot mutate storage-owned
        # state by accident.
        return dict(parsed)

    return _loader


def _coerce_sources(sources: Any) -> dict[str, Any]:
    """Normalize the caller's ``sources`` argument.

    ``None`` becomes an empty mapping. A Mapping is shallow-copied so
    the engine composition cannot mutate caller-supplied state by
    reference. Anything else raises ``TypeError``.
    """
    if sources is None:
        return {}
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a Mapping[str, callable] or None")
    # Shallow copy preserves callables by identity while isolating the
    # engine composition from caller-owned structure.
    return dict(sources)


def _resolve_audit_sink(audit_sink: Any) -> Any:
    """Return the audit sink the engine should be composed with.

    If the caller supplied one, use it directly. Otherwise resolve the
    process monitoring emitter (via ``agent.monitoring.emitter``) and
    wrap it in ``EvidencePackMonitoringAuditSink``. The monitoring
    subsystem owns the emitter; the factory borrows it and never
    closes it.
    """
    if audit_sink is not None:
        return audit_sink
    # Import locally so the module is only resolved on the audit path;
    # production default-off short-circuits before reaching this point.
    from agent.monitoring.emitter import get_emitter

    emitter = get_emitter()
    return EvidencePackMonitoringAuditSink(emitter=emitter)


def build_evidence_pack_engine(
    *,
    session_id: str,
    config: Mapping[str, Any],
    sources: Any | None = None,
    storage: Any | None = None,
    audit_sink: Any | None = None,
) -> EvidencePackEngine:
    """Construct the canonical production EvidencePackEngine.

    Parameters
    ----------
    session_id
        The active session identifier. Used to bind the contract state
        loader to ``SessionDB.state_meta["goal:<session_id>"]``. Must
        be a non-empty string.
    config
        The full CLI config mapping. Currently accepted for forward
        compatibility; the canonical engine does not consult it.
    sources
        Optional caller-supplied source map. ``None`` is treated as an
        empty mapping. The canonical factory installs the
        ``make_contract_provider`` callable under ``"contract"``,
        replacing any caller-supplied ``"contract"`` entry. All other
        entries are preserved by identity. Non-Mapping raises
        ``TypeError``.
    storage
        Borrowed storage (typically ``HermesCLI._session_db``). Must
        expose callable ``get_meta`` and ``set_meta``. The factory
        never constructs ``SessionDB``, never opens a database, and
        never closes storage. ``None`` or structurally invalid storage
        raises :class:`EvidencePackStorageUnavailable`.
    audit_sink
        Optional caller-supplied audit sink. When ``None`` the factory
        resolves the process monitoring emitter via
        ``agent.monitoring.emitter.get_emitter`` and wraps it in
        :class:`EvidencePackMonitoringAuditSink`. The emitter is
        borrowed; neither the factory nor the adapter closes it.

    Returns
    -------
    EvidencePackEngine
        A fully composed engine. The factory does NOT call
        ``dry_run`` / ``discover`` / ``rollback`` and does NOT cache a
        global engine — one engine is returned per call and lives for
        the duration of the borrower's lifecycle.

    Raises
    ------
    ValueError
        If ``session_id`` is not a non-empty string.
    TypeError
        If ``sources`` is not ``None`` or a Mapping.
    EvidencePackStorageUnavailable
        If ``storage`` is ``None`` or does not structurally expose
        callable ``get_meta`` / ``set_meta``. This typed exception is
        caught by ``build_objective_services`` and surfaced as
        ``evidence_pack_status="degraded"`` /
        ``evidence_pack_degrade_reason="storage_unavailable"`` /
        ``evidence_pack_error_type="EvidencePackStorageUnavailable"``.
    """
    sid = _validate_session_id(session_id)
    storage_obj = _validate_storage(storage)

    # Compose sources — always install the contract provider; preserve
    # all other caller-supplied sources by identity.
    sources_map = _coerce_sources(sources)
    state_loader = _make_state_loader(storage=storage_obj, session_id=sid)
    sources_map["contract"] = make_contract_provider(
        sid, state_loader=state_loader
    )

    # Compose the audit sink — caller's wins; otherwise resolve the
    # monitoring emitter exactly once and wrap it.
    sink = _resolve_audit_sink(audit_sink)

    engine = EvidencePackEngine(
        sources=sources_map,
        storage=storage_obj,
        audit_sink=sink,
    )
    return engine


__all__ = [
    "build_evidence_pack_engine",
]