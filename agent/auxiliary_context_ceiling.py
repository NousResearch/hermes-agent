"""Bounded owner for the auxiliary context-ceiling authority.

This shard is the SINGLE owner of the auxiliary context-ceiling
enforcement surface that used to live inline in
``agent/auxiliary_client.py``:

* ``_aux_relay_gate`` -- the relay-helper entry ceiling gate (final
  physical payload budget enforcement for the auxiliary physical owner).
* ``_aux_provider_callback`` -- the provider-boundary ceiling wrapper
  (the true physical seam, enforcing the ceiling on the FINAL payload at
  the moment it is handed to the provider).
* ``_rebind_aux_ceiling_for_fallback`` -- sync fallback ceiling rebinding
  (scopes the effective ceiling to a fallback destination's own
  authority).
* ``_rebind_aux_ceiling_for_fallback_async`` -- async counterpart.

These are the coherent "auxiliary context-ceiling authority" units
requested by the #78635 fracture contract.  They are byte-verbatim
relocations (docstrings and inline imports preserved); they read the
ceiling ContextVar owned by ``agent.model_metadata`` and the output-
reservation policy there -- they never duplicate that state.

``agent.auxiliary_client`` re-imports the four names below so that:

* historical import names (``from agent.auxiliary_client import
  _aux_provider_callback`` etc.) keep working;
* ``monkeypatch.setattr(agent.auxiliary_client, "_aux_provider_callback",
  ...)`` still affects the live relay path, because the relay helpers
  (``_relay_sync_completion`` / ``_relay_async_completion`` /
  ``_relay_sync_stream``) and the fallback dispatchers that CONSUME these
  names stay in ``auxiliary_client`` and resolve them through that
  module's globals.

There is exactly one implementation of each -- here.  The monolith keeps
only the narrow compatibility re-imports (see the re-export block in
``auxiliary_client.py``); it holds no second copy of any ceiling-authority
logic.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "_aux_relay_gate",
    "_aux_provider_callback",
    "_rebind_aux_ceiling_for_fallback",
    "_rebind_aux_ceiling_for_fallback_async",
]


def _aux_relay_gate(
    kwargs: dict[str, Any],
    task: str = "auxiliary",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Terminal ceiling gate for the auxiliary physical owner (relay helpers).

    The three relay helpers below are the SINGLE physical I/O owner for all
    auxiliary dispatch (sync / async / stream, primary + fallback + retry all
    converge on them — 20+ call sites).  ``call_llm`` sets the auxiliary
    effective ceiling (model context clamped by the profile ceiling) in a
    contextvar before dispatching; this reads it and enforces the transport-
    normalized budget on the FINAL payload.  A refusal raises the TYPED
    ``ContextCeilingExceeded`` (NOT wrapped in ``RuntimeError``) so the
    auxiliary catch-all chain classifies it as a local ceiling refusal, not a
    provider/auth error — no fallback, no credential-refresh retry, no bypass.

    ``provider`` / ``model`` feed the shared output-reservation policy so the
    auxiliary gate reserves the same allowance the compressor and the main
    gate do (final request cap → provider/profile implicit cap → default).
    """
    from agent.model_metadata import (
        get_aux_ceiling as _get_aux_ceiling,
        build_final_context_budget as _build_budget,
        enforce_final_context_budget as _enforce,
    )
    ceiling = _get_aux_ceiling()
    if not (isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0):
        return
    budget = _build_budget(kwargs, provider=provider, model=model)
    _enforce(budget, ceiling=ceiling, reason=task)


def _aux_provider_callback(
    callback: Callable[[dict[str, Any]], Any],
    *,
    task: str = "auxiliary",
    provider: str | None = None,
    model: str | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """Provider-boundary ceiling wrapper (the TRUE physical seam).

    The relay-helper entry gate (:func:`_aux_relay_gate`) enforces on the
    request the CALLER built.  If Relay (``relay_llm.execute`` /
    ``execute_async`` / ``stream_current``) or a middleware layer ENLARGES
    that request before invoking the provider callback — adding tokens,
    expanding tools, appending system content, or rewriting the payload in
    any way that increases its size — that enlargement is NOT checked by the
    entry gate.  This wrapper enforces the ceiling on the FINAL payload at
    the exact moment it is about to be handed to the provider, i.e. at the
    physical provider seam.  A refusal raises the TYPED
    ``ContextCeilingExceeded`` BEFORE ``callback`` runs, so the provider is
    never called with an oversized request.

    The ceiling is read from the same contextvar the entry gate reads, so
    both see the identical effective limit for the invocation.  The output
    reservation is resolved from the shared policy with the same
    ``provider`` / ``model`` so both seams agree.
    """
    def _gated(request: dict[str, Any]) -> Any:
        _aux_relay_gate(
            request, task=task, provider=provider, model=model
        )
        return callback(request)
    _gated.__name__ = getattr(callback, "__name__", "provider_callback")
    _gated.__doc__ = callback.__doc__
    return _gated



def _rebind_aux_ceiling_for_fallback(
    destination: "_FallbackDestination",
    *,
    api_key: str = "",
):
    """Scope the auxiliary effective ceiling to a fallback destination.

    ``call_llm`` / ``async_call_llm`` publish the ceiling for the INITIAL
    destination once (see ``_call_llm_impl`` / ``_async_call_llm_impl``).  The
    relay-helper physical gates read that ambient ContextVar.  When the
    fallback candidates (``_call_fallback_candidate_sync`` /
    ``_call_fallback_candidate_async``) dispatch to a DIFFERENT provider /
    model / base_url, they must rebind the ceiling to the fallback's own
    ``effective = min(raw_capability, profile_ceiling)`` — otherwise the
    initial destination's stale ceiling is inherited (P1 from review
    5049973836: a 900K initial ceiling lets a ~200K request through to a 128K
    fallback model; a 128K initial ceiling falsely refuses a ~300K request on
    a 900K fallback).

    The ceiling is set via ``set_aux_ceiling`` and reset via the returned
    token in a ``finally`` — so the prior value (the initial destination's
    ceiling) is restored after the fallback attempt completes, whether it
    succeeds, is ceiling-refused, or fails for another reason.  This keeps
    the initial destination's ceiling available for the NEXT fallback layer
    in the chain (which is itself a different destination and will rebind
    again).

    ``api_key`` is threaded into the resolver so the raw capability lookup is
    credential-aware (the existing ``_candidate_context_window()`` path
    already carries ``api_key`` for authenticated endpoint probing; the
    initial resolver in ``_call_llm_impl`` was omitting it).

    Usage (sync)::

        with _rebind_aux_ceiling_for_fallback(destination, api_key=...) as _ce:
            ...relay call...

    Usage (async)::

        async with _rebind_aux_ceiling_for_fallback_async(destination, api_key=...) as _ce:
            ...await relay call...

    ``as _ce`` is the ceiling value (or ``None`` if resolution failed) so
    tests can assert the exact limit used by the gate.
    """
    import contextlib
    from agent.model_metadata import (
        ContextCeilingExceeded as _cce,
        effective_context_length as _ecl,
        set_aux_ceiling as _set,
        reset_aux_ceiling as _reset,
    )
    _model = destination.model or ""
    _base = destination.base_url or ""
    try:
        _ceiling = _ecl(
            model=_model,
            base_url=_base,
            api_key=api_key or "",
            provider=destination.provider or "",
        )
    except Exception:
        _ceiling = None
    if not (isinstance(_ceiling, int) and not isinstance(_ceiling, bool) and _ceiling > 0):
        # FAIL CLOSED before provider I/O (Round 5 finding C, re-review of
        # 8e109d375): the fallback destination's context authority could not
        # be resolved (the resolver raised, or it returned a value that is not
        # a usable positive int).  Context authority is destination-scoped —
        # once the destination projection changed (provider / model / base_url
        # / credential), the SOURCE destination's ambient ceiling must NOT
        # authorize this new physical provider attempt.  Refusing with the
        # TYPED local ceiling refusal keeps it terminal by type: the
        # fallback / transient-retry / credential-refresh chains let
        # ``ContextCeilingExceeded`` propagate (no provider I/O, no fallback,
        # no retry, no credential refresh), so the refusal is NOT converted
        # into an ordinary provider error that would trigger an unauthorized
        # retry/fallback cycle.  (Deliberately NOT a new exception type: the
        # catch chains key on ``ContextCeilingExceeded``; a new type would
        # fall into ``except Exception`` and be treated as a provider error.)
        raise _cce(
            0, 0,
            reason=(
                "fallback destination's context authority could not be "
                "resolved; refusing to authorize the fallback provider under "
                "the source destination's ceiling"
            ),
        )
    _token = _set(_ceiling)

    @contextlib.contextmanager
    def _scoped():
        try:
            yield _ceiling
        finally:
            _reset(_token)

    return _scoped()


def _rebind_aux_ceiling_for_fallback_async(
    destination: "_FallbackDestination",
    *,
    api_key: str = "",
):
    """Async variant of :func:`_rebind_aux_ceiling_for_fallback`.

    Returns an ``asynccontextmanager`` that resolves the destination's ceiling
    in a background thread (``asyncio.to_thread``) so blocking HTTP probes do
    not freeze the asyncio event loop, then sets/resets the ContextVar token.
    Same scoped semantics as the sync variant: the prior value is restored in
    the ``finally`` after the fallback attempt, whether it succeeds, is
    ceiling-refused, or fails for another reason.

    Usage::

        async with _rebind_aux_ceiling_for_fallback_async(destination, api_key=...) as _ce:
            ...await relay call...
    """
    import asyncio
    import contextlib
    from agent.model_metadata import (
        ContextCeilingExceeded as _cce,
        effective_context_length as _ecl,
        set_aux_ceiling as _set,
        reset_aux_ceiling as _reset,
    )
    _model = destination.model or ""
    _base = destination.base_url or ""

    @contextlib.asynccontextmanager
    async def _scoped():
        # Resolve the ceiling off the event loop.  ``_ecl`` is a sync
        # function; ``asyncio.to_thread`` runs it in a worker thread.
        try:
            _ceiling = await asyncio.to_thread(
                _ecl,
                model=_model,
                base_url=_base,
                api_key=api_key or "",
                provider=destination.provider or "",
            )
        except Exception:
            _ceiling = None
        if not (isinstance(_ceiling, int) and not isinstance(_ceiling, bool) and _ceiling > 0):
            # FAIL CLOSED before provider I/O (Round 5 finding C, re-review of
            # 8e109d375): the fallback destination's context authority could
            # not be resolved (the resolver raised, or it returned a value
            # that is not a usable positive int).  Context authority is
            # destination-scoped — once the destination projection changed
            # (provider / model / base_url / credential), the SOURCE
            # destination's ambient ceiling must NOT authorize this new
            # physical provider attempt.  Raising the TYPED local ceiling
            # refusal keeps it terminal by type: the async fallback /
            # transient-retry / credential-refresh chains let
            # ``ContextCeilingExceeded`` propagate (no provider I/O, no
            # fallback, no retry, no credential refresh), so the refusal is
            # NOT converted into an ordinary provider error that would
            # trigger an unauthorized retry/fallback cycle.  (Deliberately NOT
            # a new exception type: the catch chains key on
            # ``ContextCeilingExceeded``; a new type would fall into
            # ``except Exception`` and be treated as a provider error.)
            raise _cce(
                0, 0,
                reason=(
                    "fallback destination's context authority could not be "
                    "resolved; refusing to authorize the fallback provider "
                    "under the source destination's ceiling"
                ),
            )
        _token = _set(_ceiling)
        try:
            yield _ceiling
        finally:
            _reset(_token)

    return _scoped()
