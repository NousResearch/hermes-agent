"""Runtime override support for ``pre_llm_call`` plugin hooks.

A plugin may return ``{"runtime_override": {...}}`` from ``pre_llm_call`` to
proactively override the LLM API call parameters for the current turn:

    {"context": "recalled text...",            # existing behavior, unchanged
     "runtime_override": {
         "model": "gpt-5.6",
     }}

Contract (mirrors the ``pre_failover_decision`` redirect contract):

* ``redirect`` is *error-driven*: it is applied by the retry/failover machinery
  only after an API call has failed.  ``runtime_override`` is *proactive*: it is
  applied before the first API call of the turn.  The two do not conflict —
  redirect rewrites identity on the failover path, runtime_override rewrites
  identity on the primary path.
* The override is ephemeral and turn-scoped: it lives on ``agent._runtime_override``,
  is re-resolved on every turn prologue, is never persisted to the session DB and
  is never injected into the user message / session history.
* Unsupported keys are logged with a one-line warning and ignored (never crash).
* The override switches the model only: ``provider``, ``api_mode``, ``api_key``
  and ``base_url`` are intentionally unsupported (an earlier contract allowed
  them; they were removed) — credentials never flow through the hook return and
  the endpoint/wire are resolved only from the provider's existing settings, so
  a plugin cannot pick a network destination or a different wire for the
  session.
* TRUST IMPLICATION: a model switch stays inside the existing provider route —
  it never touches credentials or the endpoint, so it cannot redirect the
  session elsewhere.  Installing a plugin therefore grants it this power; only
  install plugins you trust.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Keys a plugin may override.  Anything else is logged and ignored.
#: ``system_prompt`` is intentionally NOT supported: it is the prompt-cache
#: prefix (byte-stable for the life of a conversation), so overriding it would
#: invalidate the cache and drop the core instructions. Model routing does not
#: need it; persona switching needs a separate cache-safe design.
RUNTIME_OVERRIDE_KEYS = frozenset({"model"})

#: All supported keys are plain non-empty strings.
_STRING_KEYS = frozenset({"model"})

#: Identity keys whose change makes the effective route different from the
#: pre-override route.  A route change refreshes derived route state the
#: same way ``switch_model`` / ``_try_activate_fallback`` do.
_ROUTE_KEYS = frozenset({"model"})


def validate_runtime_override(overrides: Any) -> Dict[str, str]:
    """Type-check + filter a plugin-provided ``runtime_override`` dict.

    Returns only the supported, correctly-typed keys.  Unsupported keys and
    wrong-typed values are logged with a one-line warning and dropped, so a
    misbehaving plugin can never crash the turn.
    """
    if not isinstance(overrides, dict):
        logger.warning(
            "pre_llm_call runtime_override ignored: expected dict, got %s",
            type(overrides).__name__,
        )
        return {}
    valid: Dict[str, str] = {}
    for key, value in overrides.items():
        if key not in RUNTIME_OVERRIDE_KEYS:
            logger.warning(
                "pre_llm_call runtime_override: unsupported key %r ignored "
                "(supported: %s)",
                key,
                ", ".join(sorted(RUNTIME_OVERRIDE_KEYS)),
            )
            continue
        if key in _STRING_KEYS and (not isinstance(value, str) or not value.strip()):
            logger.warning(
                "pre_llm_call runtime_override: key %r must be a non-empty "
                "string, ignored",
                key,
            )
            continue
        valid[key] = value.strip() if isinstance(value, str) else value
    return valid


def _refresh_derived_route_state(agent: Any, overrides: Dict[str, str]) -> None:
    """Refresh provider-derived state for the overridden route.

    Mirrors what the canonical route switch (``switch_model`` /
    ``_try_activate_fallback``) does when the active route changes: the
    switched-to provider's ``request_overrides`` (``extra_body``) replaces the
    previous provider's, and ``runtime_capabilities`` is re-resolved for the
    new model/endpoint.  The model-owned state the canonical switch projects
    (prompt-cache flags, context compressor, reasoning config) is refreshed by
    ``_project_override_model_state`` — the activation step this calls.  All of
    it is best-effort — a resolution failure must never crash the turn; the
    scope snapshot still restores the original values on exit.
    """
    try:
        from agent.agent_runtime_helpers import (
            _apply_switched_provider_request_overrides,
        )

        _apply_switched_provider_request_overrides(
            agent, str(overrides.get("provider") or agent.provider)
        )
    except Exception as _ro_exc:  # noqa: BLE001
        logger.warning(
            "runtime_override: request_overrides refresh failed (%s); "
            "keeping previous value for the scope",
            _ro_exc,
            exc_info=True,
        )
    try:
        from agent.native_compaction import resolve_native_compaction_capabilities

        agent.runtime_capabilities = resolve_native_compaction_capabilities(
            model=getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            provider=getattr(agent, "provider", "") or "",
            is_codex_backend=(
                (getattr(agent, "provider", "") or "").strip().lower()
                == "openai-codex"
            ),
        )
    except Exception as _cap_exc:  # noqa: BLE001
        logger.warning(
            "runtime_override: runtime_capabilities refresh failed (%s); "
            "keeping previous value for the scope",
            _cap_exc,
            exc_info=True,
        )
    _project_override_model_state(agent, overrides)


def _isolated_context_compressor(compressor: Any) -> Any:
    """Scope-owned copy of ``compressor`` the canonical projection may re-point.

    The canonical model-owned projection re-points ``agent.context_compressor``
    through ``update_model``, which ASSIGNS the model-owned fields (model,
    context length, thresholds, calibration state) on the instance — a shallow
    copy isolates those assignments from the session compressor.  Never mutate
    the pre-override compressor in place: its reference is restored on scope
    exit, so an in-place re-point would leave the override's context length on
    the session compressor.  ``update_model`` only assigns scalars (no
    nested-container mutation), so a shallow copy fully isolates the
    projection.  The copy keeps the durable session handles: a compression that
    actually fires inside the scope (or a fallback that supersedes it) is a
    real session event and its bookkeeping must persist.
    """
    import copy as _copy

    return _copy.copy(compressor)


class _ModelProjectionFailed(RuntimeError):
    """The model-owned projection could not be completed.

    Raised instead of silently keeping a half-projected agent: the override owns
    the route only if the whole projection lands.  The scope catches it and fails
    open to the configured route.
    """


def _project_override_model_state(agent: Any, overrides: Dict[str, str]) -> None:
    """Point the model-owned derived state at the overridden model (P1-1).

    Reuses the canonical model-owned projection from ``switch_model``
    (``_apply_model_owned_state`` in agent_runtime_helpers) instead of
    re-implementing it: prompt-cache flags, the per-model context-length
    re-point of ``agent.context_compressor``, and ``reasoning_config`` are
    projected exactly as a real switch projects them, so the request/preflight
    path (which reads ``context_compressor.threshold_tokens``,
    ``_use_prompt_caching`` and ``reasoning_config``) never sees the pre-override
    model's values while ``agent.model`` is the override model.

    The canonical projection re-points ``agent.context_compressor`` in place, so
    a scope-owned copy is swapped in first (``_isolated_context_compressor``);
    the pre-override reference is already in the scope snapshot (it is one of
    ``_DERIVED_ATTRS``) and is restored on exit.  All-or-nothing: a resolution
    failure raises ``_ModelProjectionFailed`` so the scope fails open to the
    configured route — a half-projected agent would serve one request that mixes
    two models' context length, cache policy and reasoning state.
    """
    _cc = getattr(agent, "context_compressor", None)
    if _cc is None:
        # Minimal/bare agent: it carries none of the model-owned state this
        # projection owns (no compressor to re-point, no derived context length),
        # so there is no second model's truth to contradict — nothing to project.
        return
    try:
        agent.context_compressor = _isolated_context_compressor(_cc)
    except Exception as _iso_exc:  # noqa: BLE001
        # Never project onto the session compressor in place, and never stand on
        # a half-applied route: fail open to the configured route.
        logger.warning(
            "runtime_override: compressor isolation failed; the override is "
            "not applied for this turn",
            exc_info=True,
        )
        raise _ModelProjectionFailed("compressor isolation failed") from _iso_exc
    try:
        from agent.agent_runtime_helpers import _apply_model_owned_state

        _apply_model_owned_state(
            agent,
            str(overrides.get("model") or getattr(agent, "model", "") or ""),
            snapshot=None,  # the scope owns rollback; no provider-switch snapshot
        )
        agent._cached_system_prompt = None
        # The cached system prompt's context-file caps are scaled by the model's
        # context window (build_system_prompt_parts reads
        # context_compressor.context_length). Invalidate so the next build
        # re-scales them — mirrors switch_model's `_cached_system_prompt = None`.
        # Snapshot/restore via _DERIVED_ATTRS makes this a no-leak transaction.
    except Exception as _moe_exc:  # noqa: BLE001
        # The projection may have partially mutated the copy AND the agent's
        # model-owned attributes: go back to the session compressor and let the
        # scope roll the whole transaction back (fail open to the configured
        # route) rather than serve a request that mixes two models' state.
        agent.context_compressor = _cc
        logger.warning(
            "runtime_override: model-owned state projection failed; the override "
            "is not applied for this turn",
            exc_info=True,
        )
        raise _ModelProjectionFailed("model-owned state projection failed") from _moe_exc


#: Sentinel meaning "the durable key/row did not exist".  Kept distinct from
#: ``None`` (a real stored NULL) so an exact restore re-creates absence.
_ABSENT = object()


def _snapshot_durable_protection(
    compressor: Any,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Capture the durable compression-protection rows a model projection clears.

    The canonical model-owned projection (``_apply_model_owned_state`` ->
    ``ContextCompressor.update_model``) clears the per-session fallback streak,
    failure cooldown, ineffective-strike count and proactive-prune runway as part
    of a REAL model switch.  A runtime_override reuses that projection but is
    ephemeral, so the scope rolls those durable clears back exactly on exit.
    Returns ``(None, True)`` when the compressor has no bound session row (nothing
    durable to roll back).

    Returns ``(snapshot, complete)``: ``complete`` is False when any getter
    failed.  A partial snapshot cannot restore what it never read, so the caller
    must treat an incomplete snapshot as "this override cannot be established"
    and fail open — never let the destructive clears run behind it.
    """
    db = getattr(compressor, "_session_db", None)
    session_id = getattr(compressor, "_session_id", "") or ""
    if db is None or not session_id:
        return None, True
    snapshot: Dict[str, Any] = {}
    complete = True
    for name, getter_name in (
        ("fallback_streak", "get_compression_fallback_streak"),
        ("ineffective_count", "get_compression_ineffective_count"),
    ):
        getter = getattr(db, getter_name, None)
        if callable(getter):
            try:
                snapshot[name] = getter(session_id)
            except Exception:  # noqa: BLE001
                complete = False
                logger.warning(
                    "runtime_override: %s snapshot failed", getter_name, exc_info=True
                )
    getter = getattr(db, "get_compression_failure_cooldown_row", None)
    if callable(getter):
        try:
            snapshot["cooldown_row"] = getter(session_id)
        except Exception:  # noqa: BLE001
            complete = False
            logger.warning(
                "runtime_override: cooldown snapshot failed", exc_info=True
            )
    getter = getattr(db, "get_session_model_config_value", None)
    if callable(getter):
        try:
            from agent.context_compressor import PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY

            snapshot["prune_rearm"] = getter(
                session_id, PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY, _ABSENT
            )
        except Exception:  # noqa: BLE001
            complete = False
            logger.warning(
                "runtime_override: prune-rearm snapshot failed", exc_info=True
            )
    return snapshot, complete


def _restore_durable_protection(
    compressor: Any, snapshot: Optional[Dict[str, Any]]
) -> bool:
    """Best-effort exact rollback of the projection's durable clears.

    Called on every non-superseded scope exit, including the exception path, so no
    DB row stays cleared permanently.  A superseded scope skips it: the fallback
    chain owns the route and its own ``update_model`` writes must stand.

    Returns True when every row covered by ``snapshot`` was written back.  False
    means a required setter failed, so the caller must treat the rollback as
    incomplete rather than report a finished recovery (see ``_abort``).
    """
    if not snapshot:
        return True
    db = getattr(compressor, "_session_db", None)
    session_id = getattr(compressor, "_session_id", "") or ""
    if db is None or not session_id:
        return True

    ok = True

    def _write(method: str, *args: Any) -> None:
        nonlocal ok
        setter = getattr(db, method, None)
        if callable(setter):
            try:
                setter(session_id, *args)
            except Exception:  # noqa: BLE001
                ok = False
                logger.warning(
                    "runtime_override: %s restore failed", method, exc_info=True
                )

    if "fallback_streak" in snapshot:
        _write("set_compression_fallback_streak", snapshot["fallback_streak"])
    if "ineffective_count" in snapshot:
        _write("set_compression_ineffective_count", snapshot["ineffective_count"])
    if "cooldown_row" in snapshot:
        restorer = getattr(db, "restore_compression_failure_cooldown_row", None)
        if callable(restorer):
            try:
                restorer(session_id, snapshot["cooldown_row"])
            except Exception:  # noqa: BLE001
                ok = False
                logger.warning(
                    "runtime_override: cooldown restore failed", exc_info=True
                )
    if "prune_rearm" in snapshot:
        try:
            from agent.context_compressor import PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY

            value = snapshot["prune_rearm"]
            _write(
                "patch_session_model_config",
                {PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: None if value is _ABSENT else value},
            )
        except Exception:  # noqa: BLE001
            ok = False
            logger.warning(
                "runtime_override: prune-rearm restore failed", exc_info=True
            )
    return ok


def _durable_protection_changed(
    compressor: Any, after: Optional[Dict[str, Any]]
) -> bool:
    """True when a newer durable write landed after ``after`` was captured.

    The scope captures the durable state right after its own projection; a
    nested/concurrent override, a real compression, or a fallback can then write
    legitimate newer values.  Restoring the pre-override snapshot over those
    would lose them, so the rollback is skipped in that case (only a snapshot
    that is still current is restored).  Compares only the keys captured in BOTH
    reads, so a getter that starts failing can never by itself suppress the
    rollback — absence of evidence is not evidence of a newer write.
    """
    if after is None:
        return False
    current, _complete = _snapshot_durable_protection(compressor)
    if not current:
        return False
    return any(current.get(key, value) != value for key, value in after.items())


class _RuntimeOverrideScope:
    """Context manager that temporarily applies an override to ``agent``.

    The override is an atomic route transaction, not a second, narrower route
    mutation primitive: it snapshots and restores every route-owned datum the
    canonical switch (``switch_model`` / ``_try_activate_fallback``) manages,
    so the effective route stays consistent through request construction,
    request middleware, ``pre_api_request``, wire execution, and response
    handling.

    Precedence with the error-driven failover path: a proactive override owns
    only the primary attempt.  If ``_try_activate_fallback`` succeeds while the
    scope is open, the fallback supersedes the override.  Supersession is an
    EXPLICIT handoff, never inferred: the fallback call site invokes
    ``consume_runtime_override(agent)``, which finds this scope (registered as
    ``agent._active_runtime_override_scope`` in ``__enter__``) and calls
    ``supersede()``.  ``__exit__`` then sees the ``_superseded`` flag and skips
    the route-identity restore (which would clobber the freshly activated
    fallback); the caller has already cleared ``agent._runtime_override`` so
    retries stay on the fallback route.
    """

    _ATTRS = ("model",)
    _MISSING = object()
    # Route-owned derived state the wire path reads and the route refresh
    # (``_refresh_derived_route_state``) rewrites for the new model.
    # Snapshot/restore the set unconditionally so untouched fields are no-ops
    # and changed fields revert atomically on exit.
    #
    # P1-1: this includes the MODEL-OWNED state the canonical switch
    # (``switch_model``) projects and request/preflight code reads, so the scope
    # never leaves two models' truths in play:
    #   * ``context_compressor`` is an OBJECT — the snapshot holds the reference,
    #     activation swaps in a scope-owned copy (see
    #     ``_project_override_model_state``), and exit restores the reference.
    #     The pre-override compressor is never mutated in place, so the
    #     override's context length cannot leak into the session compressor.
    #   * ``reasoning_config`` / ``_use_prompt_caching`` /
    #     ``_use_native_cache_layout`` are plain values — snapshot/restore as-is.
    #   * ``_config_context_length`` and ``_custom_providers`` are written by the
    #     shared projection (``_resolve_switch_context_length`` / the custom-provider
    #     refresh); restoring them keeps the override a no-leak transaction.
    _DERIVED_ATTRS = (
        "request_overrides",
        "runtime_capabilities",
        "context_compressor",
        "reasoning_config",
        "_use_prompt_caching",
        "_use_native_cache_layout",
        "_config_context_length",
        "_custom_providers",
        "_cached_system_prompt",
    )

    def __init__(self, agent: Any, overrides: Dict[str, str]) -> None:
        self.agent = agent
        self.overrides = overrides
        self._snapshot: Dict[str, Any] = {}
        self._client_kwargs_snapshot: Optional[Dict[str, Any]] = None
        self._transport_cache_snapshot: Optional[Dict[str, Any]] = None
        # Durable compression-protection rows the model projection clears; held
        # on the pre-override compressor and rolled back exactly on exit.
        self._session_compressor: Any = None
        self._compressor_durable_snapshot: Optional[Dict[str, Any]] = None
        # Durable state right after this scope's own projection: the rollback is
        # skipped when a newer write (nested/concurrent override, real
        # compression, fallback) changed it, so the snapshot cannot clobber it.
        self._compressor_durable_after: Optional[Dict[str, Any]] = None
        self._superseded = False
        # Set once a restore ran (a failed projection aborts inside __enter__),
        # so the real __exit__ cannot restore a second time.
        self._ended = False
        # Set when restoring a route-owned value failed.  A failed override may
        # only fall open when the rollback actually completed — a request issued
        # against a partially restored route is exactly the mixed state this
        # scope must never produce.
        self._restore_failed = False
        # Nesting: the registered outermost scope is this scope's parent, and a
        # supersession propagates to every live child (a child restoring its own
        # snapshot would otherwise resurrect the route the fallback replaced).
        self._parent_scope: Optional["_RuntimeOverrideScope"] = None
        self._children: list["_RuntimeOverrideScope"] = []

    def __enter__(self) -> "_RuntimeOverrideScope":
        agent = self.agent
        # ── Snapshot phase ─────────────────────────────────────────────
        # NOTE: agent._runtime_override is the canonical source.
        # TurnContext.runtime_override is derived from it at construction
        # time.  The call path in turn_api_call.py reads only the agent
        # attribute; keep that as the single point of truth.
        ov = self.overrides
        for name in self._ATTRS:
            if name in ov:
                self._snapshot[name] = getattr(agent, name, self._MISSING)
        for name in self._DERIVED_ATTRS:
            self._snapshot[name] = getattr(agent, name, self._MISSING)
        # request_overrides is replaced wholesale on activation, never mutated
        # in place — a shallow copy is enough to make the restore exact.
        if isinstance(self._snapshot.get("request_overrides"), dict):
            self._snapshot["request_overrides"] = dict(
                self._snapshot["request_overrides"]
            )
        # _client_kwargs feeds the per-request OpenAI-wire client.  Snapshot a
        # shallow copy so in-place mutation is reversible.
        ck = getattr(agent, "_client_kwargs", None)
        if isinstance(ck, dict):
            self._client_kwargs_snapshot = dict(ck)
            self._snapshot["_client_kwargs"] = ck
        # Transport cache: snapshot the content so exit can restore the
        # pre-override cache instead of leaving an override-mode transport in
        # the agent's per-mode cache.
        _tc = getattr(agent, "_transport_cache", self._MISSING)
        if isinstance(_tc, dict):
            self._transport_cache_snapshot = dict(_tc)
            self._snapshot["_transport_cache"] = _tc

        # ── Activation phase ───────────────────────────────────────────
        # A raise here would skip ``__exit__`` entirely and leave the durable
        # rows the projection just cleared cleared, so roll the scope back
        # (best-effort) before the original exception propagates.
        try:
            for name in self._ATTRS:
                if name in ov:
                    # Normalize before storing: the emptiness check below strips,
                    # but the stored value must be stripped too or " gpt-5.6 "
                    # flows into agent.model and onto the wire.
                    val = str(ov[name]).strip()
                    if not val:
                        logger.warning("runtime_override: empty value for %r ignored", name)
                        # Drop the rejected key so the route-refresh check below
                        # does not act on a value that was just declared ignored.
                        # In the canonical flow ov is a validated dict (never
                        # holds empties); this only guards direct callers.
                        ov.pop(name, None)
                        continue
                    ov[name] = val
                    setattr(agent, name, ov[name])
            if _ROUTE_KEYS.intersection(ov):
                # The canonical projection clears durable per-session compression
                # protection as part of a real model switch; snapshot those rows from
                # the still-bound session compressor BEFORE the projection so the
                # ephemeral override can write them back exactly on exit (see
                # ``_restore_durable_protection``).
                self._session_compressor = getattr(agent, "context_compressor", None)
                (
                    self._compressor_durable_snapshot,
                    _durable_complete,
                ) = _snapshot_durable_protection(self._session_compressor)
                if not _durable_complete:
                    # A partial snapshot cannot be rolled back exactly, and the
                    # projection erases exactly those rows: refuse the override
                    # instead of erasing protection this scope cannot restore.
                    raise _ModelProjectionFailed(
                        "durable protection snapshot is incomplete"
                    )
                _refresh_derived_route_state(agent, ov)
                # Fingerprint the post-projection durable state so exit can tell
                # our own clears apart from a newer legitimate write.
                _after, _complete_after = _snapshot_durable_protection(
                    self._session_compressor
                )
                self._compressor_durable_after = _after
        except _ModelProjectionFailed:
            # Fail open to the configured route: the override is NOT applied, so
            # the request runs on the route the agent already had — never on a
            # half-projected mixture of the two.
            logger.warning(
                "runtime_override: override not applied for this turn "
                "(model projection could not be completed); the configured route stands",
                exc_info=True,
            )
            self._abort()
            return self
        except BaseException:
            logger.warning(
                "runtime_override: activation failed; rolling the scope back",
                exc_info=True,
            )
            try:
                self.__exit__(*sys.exc_info())
            except Exception:  # noqa: BLE001 — rollback must not mask the cause
                logger.warning(
                    "runtime_override: rollback after activation failure failed",
                    exc_info=True,
                )
            raise

        # Register as the agent's active scope so the fallback handoff
        # (consume_runtime_override) can find and supersede this scope.
        # First-registered-wins: a nested scope (middleware/retry re-entry)
        # must not steal the registration from the outermost attempt scope —
        # the fallback sites all run outside it, so superseding the outermost
        # scope is always correct.  An inner scope therefore registers only
        # when no scope is registered, and unregisters only when it holds the
        # registration.
        if getattr(agent, "_active_runtime_override_scope", None) is None:
            agent._active_runtime_override_scope = self
        else:
            # Nested scope (middleware/retry re-entry): remember the parent so a
            # fallback handoff supersedes this scope too — a superseded parent
            # must not leave a live child free to restore the override route.
            self._parent_scope = agent._active_runtime_override_scope
            self._parent_scope._children.append(self)
        return self

    def _abort(self) -> None:
        """The one recovery path for an override that must not stand.

        Both projection failures (P1-A: the projection raised part-way; P1-B: the
        durable snapshot could not be read, so the destructive clears must not
        run) raise ``_ModelProjectionFailed`` and land here, so there is exactly
        one rollback implementation — ``__exit__``'s snapshot restore — and one
        set of recovery semantics:

        * rollback completed -> fall open to the configured route; the caller's
          ``with`` body runs on the route the agent already had;
        * rollback incomplete -> raise, because issuing the request against a
          partially restored route is the mixed-model state this scope exists to
          prevent.  A failed turn is recoverable; a silently mixed route is not.

        ``_ended`` makes the real ``__exit__`` a no-op so the restore cannot run
        twice, and ``agent._runtime_override`` is cleared so no retry reads the
        override as applied.
        """
        self.agent._runtime_override = {}
        try:
            self.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 — rollback must not mask the cause
            self._restore_failed = True
            logger.warning(
                "runtime_override: rollback after a failed override failed",
                exc_info=True,
            )
        if self._restore_failed:
            raise _ModelProjectionFailed(
                "override rollback could not be completed; refusing to issue a "
                "request against a partially restored route"
            )

    def supersede(self) -> None:
        """Explicitly hand the route to the fallback chain.

        Marks the scope superseded so ``__exit__`` skips the route-identity
        restore (which would clobber the freshly activated fallback), and
        clears ``agent._runtime_override`` so no retry iteration re-applies
        the failed override.  Called by ``consume_runtime_override`` from the
        ``_try_activate_fallback`` success sites.

        Propagates to every live nested scope: the fallback sites all run
        outside the attempt scopes, so an inner scope still open at that moment
        would otherwise restore its own snapshot — the override route — over the
        fallback identity on its way out.
        """
        self._superseded = True
        self.agent._runtime_override = {}
        for child in list(self._children):
            child.supersede()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        agent = self.agent
        try:
            if self._superseded:
                # The fallback chain took ownership of the route mid-scope and
                # rebuilt every route-owned datum for the fallback route.
                # Restoring the snapshotted pre-override state here would
                # clobber the fallback — intentional: the fallback path owns
                # the route now.  The caller has already cleared
                # ``agent._runtime_override`` via ``consume_runtime_override``,
                # so retries stay on the fallback route.
                return
            if self._ended:
                # A failed projection already restored this scope inside
                # __enter__; restoring twice is not harmless (it would rewrite
                # values the turn has moved on from).
                return
            self._ended = True
            for name, value in self._snapshot.items():
                if name in ("_client_kwargs", "_transport_cache"):
                    continue  # restored from the snapshots below
                if value is self._MISSING:
                    # Attribute did not exist before the override — don't
                    # fabricate it (tests build bare agents via __new__).
                    try:
                        delattr(agent, name)
                    except Exception:  # noqa: BLE001
                        self._restore_failed = True
                        logger.warning(
                            "runtime_override: could not remove %s during restore",
                            name,
                            exc_info=True,
                        )
                    continue
                try:
                    setattr(agent, name, value)
                except Exception:  # noqa: BLE001 — restore must never raise
                    self._restore_failed = True
                    logger.warning(
                        "runtime_override: could not restore %s", name, exc_info=True
                    )
            if self._client_kwargs_snapshot is not None:
                ck = getattr(agent, "_client_kwargs", None)
                if isinstance(ck, dict):
                    ck.clear()
                    ck.update(self._client_kwargs_snapshot)
                else:
                    self._restore_failed = True
                    logger.warning(
                        "runtime_override: _client_kwargs is no longer a dict; the "
                        "pre-override client state could not be restored"
                    )
            if self._transport_cache_snapshot is not None:
                tc = getattr(agent, "_transport_cache", None)
                if isinstance(tc, dict):
                    tc.clear()
                    tc.update(self._transport_cache_snapshot)
                else:
                    self._restore_failed = True
                    logger.warning(
                        "runtime_override: _transport_cache is no longer a dict; the "
                        "pre-override transport cache could not be restored"
                    )
            # The projection's durable compression-protection clears are rolled
            # back exactly, so a temporary model switch leaves the session's
            # persisted protection rows byte-identical (P1).  Skip the rollback
            # when a newer write landed after our own projection (nested or
            # concurrent override, real compression, fallback): restoring the
            # pre-override snapshot over it would lose that legitimate write.
            if _durable_protection_changed(
                self._session_compressor, self._compressor_durable_after
            ):
                logger.warning(
                    "runtime_override: skipping durable protection rollback for "
                    "session %r: a newer write landed after the override projection",
                    getattr(self._session_compressor, "_session_id", "") or "",
                )
                return
            if not _restore_durable_protection(
                self._session_compressor, self._compressor_durable_snapshot
            ):
                # A required durable setter failed: the rollback is incomplete, so
                # a caller that failed the override must not report a finished
                # recovery (see _abort).
                self._restore_failed = True
        finally:
            # Unregister on every exit path (normal restore AND superseded
            # skip), and only when this scope holds the registration.
            if self._parent_scope is not None:
                try:
                    self._parent_scope._children.remove(self)
                except ValueError:
                    logger.debug(
                        "runtime_override: nested scope was not registered with its parent"
                    )
            if getattr(agent, "_active_runtime_override_scope", None) is self:
                agent._active_runtime_override_scope = None


def apply_runtime_override(agent: Any, overrides: Dict[str, str]) -> "_RuntimeOverrideScope":
    """Return a context manager that applies ``overrides`` to ``agent``."""
    return _RuntimeOverrideScope(agent, overrides)


def consume_runtime_override(agent: Any) -> None:
    """Explicit supersede handoff: the fallback chain took ownership of the route.

    A proactive override owns only the primary attempt: once
    ``_try_activate_fallback`` succeeds, the fallback route supersedes it for
    the remainder of the logical request, so the next retry iteration must not
    re-enter the route that just failed.  Every ``_try_activate_fallback``
    success site calls this on success.

    When an override scope is active (``agent._active_runtime_override_scope``),
    this marks it superseded and clears ``agent._runtime_override``; when no
    scope is active (exception-driven fallbacks run after the attempt's scope
    already restored the agent) it clears the turn-scoped override directly so
    the current request stays on the fallback route.  None-safe: a bare agent
    without the registration attribute never raises.
    """
    scope = getattr(agent, "_active_runtime_override_scope", None)
    if scope is not None:
        scope.supersede()
        return
    try:
        agent._runtime_override = {}
    except AttributeError:
        pass  # bare test agent without the attribute
