"""Per-session runtime options (model, reasoning, /fast tier) for GatewayRunner.

One durable-first commit is shared by ``/model``, ``/reasoning`` and ``/fast`` (typed and
picker): the store write lands first and live ``SessionState`` follows only on success, so a
failed save leaves memory, the queued model note and any one-turn restore untouched.

Bound onto ``GatewayRunner`` via the MRO. Lock order: ``_model_switch_lock`` (runner-wide), then
the per-session admission lock, then the store's threading lock. Nothing takes
``_model_switch_lock`` while holding an admission lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from contextvars import copy_context
from functools import partial
from typing import Any, Dict, Mapping, Optional

from gateway.session_persistence import UNSET
from gateway.session_state import SERVICE_TIER_UNSET

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

__all__ = [
    "UNSET", "SessionBusy", "SessionMissing", "SessionConflict", "GatewaySessionOptionsMixin",
]

_RUNTIME_FIELDS = ("model_override", "reasoning_override", "service_tier_override")

# apply_session_options keys. ``initial`` suppresses the next-turn model note (a host restoring
# its saved choice at session start); ``confirm_model_selection`` answers a selection guard.
_API_OPTION_KEYS = frozenset({
    "model", "provider", "reasoning_effort", "fast", "confirm_model_selection", "initial",
})


class SessionBusy(Exception):
    """A turn owns the session; runtime options only change while it is idle."""


class SessionMissing(Exception):
    """The routing entry vanished or was replaced (a boundary won) before the write landed."""


class SessionConflict(Exception):
    """The live options moved between the host API's validation and its commit."""


class _Rejected(Exception):
    """A structured ``apply_session_options`` rejection (``code`` + human-readable ``error``)."""

    def __init__(self, code: str, error: str) -> None:
        super().__init__(error)
        self.code, self.error = code, error


def runtime_options_signature(conversation: Any) -> tuple:
    """Comparable snapshot of the live options (credentials stripped) for the API's CAS."""
    from gateway.session import sanitize_model_override

    reasoning = conversation.reasoning_override
    tier = conversation.service_tier_override
    return (
        sanitize_model_override(conversation.model_override),
        dict(reasoning) if isinstance(reasoning, dict) else None,
        "<inherit>" if tier is SERVICE_TIER_UNSET else tier,
    )


def _rejected(code: str, error: str) -> Dict[str, Any]:
    return {"status": "rejected", "code": code, "error": error}


def _reasoning_label(value: Optional[dict]) -> Optional[str]:
    if value is None:
        return None
    if value.get("enabled") is False:
        return "none"
    return str(value.get("effort") or "medium")


def _same_option(name: str, live: Any, new: Any) -> bool:
    """Whether writing *new* over *live* changes nothing (models compare without credentials)."""
    if name == "model_override":
        from gateway.session import sanitize_model_override
        return sanitize_model_override(live) == sanitize_model_override(new)
    if name == "service_tier_override":
        return live is new or (live is not SERVICE_TIER_UNSET and new is not SERVICE_TIER_UNSET
                               and live == new)
    return live == new


def _durable_tier(live: Any) -> Optional[str]:
    """Live tier encoding -> store encoding: unset = inherit (None), None = explicit "normal"."""
    if live is SERVICE_TIER_UNSET:
        return None
    return "normal" if live is None else str(live)


def _live_tier(durable: Optional[str]) -> Any:
    """Store tier encoding -> live encoding (inverse of ``_durable_tier``)."""
    if durable is None:
        return SERVICE_TIER_UNSET
    return None if durable == "normal" else durable


def session_busy_reply(command: str) -> str:
    """The generic mid-turn reject text (``_dispatch_busy_slash_command``'s catch-all)."""
    return (
        f"⏳ Agent is running — `/{command}` can't run "
        f"mid-turn. Wait for the current response or `/stop` first."
    )


class GatewaySessionOptionsMixin:
    """Admission lock and the durable-first runtime-options commit."""

    # ------------------------------------------------------------- admission lock

    def _session_admission_lock(self, session_key: str) -> asyncio.Lock:
        """The per-session lock shared by turn admission, option commits, boot-resume and the
        /login clear. Weakly held: a lock lives only while someone holds or waits on it, so the
        map never grows with idle sessions. ``__dict__`` access keeps bare ``object.__new__``
        test runners working."""
        locks = self.__dict__.get("_session_admission_locks")
        if locks is None:
            locks = self.__dict__["_session_admission_locks"] = weakref.WeakValueDictionary()
        lock = locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            locks[session_key] = lock
        return lock

    def _session_admission_lock_held(self, session_key: str) -> bool:
        """Synchronous probe for claimers that cannot await (boot-resume). Never creates a lock."""
        locks = self.__dict__.get("_session_admission_locks")
        lock = locks.get(session_key) if locks is not None else None
        return lock is not None and lock.locked()

    @contextlib.asynccontextmanager
    async def _session_admission(self, session_key: str):
        """Hold the admission lock for a commit or a /login clear. On release, a boot-resume that
        deferred behind it is rescheduled once (after any turn already parked on the lock)."""
        try:
            async with self._session_admission_lock(session_key):
                yield
        finally:
            deferred = self.__dict__.get("_resume_deferred_keys")
            if deferred and session_key in deferred:
                platform = deferred.pop(session_key)
                asyncio.get_running_loop().call_soon(self._schedule_resume_pending_sessions, platform)

    def _defer_resume_until_admission_free(self, session_key: str, platform: Any) -> None:
        """Remember a boot-resume skipped because a commit held the admission lock."""
        self.__dict__.setdefault("_resume_deferred_keys", {})[session_key] = platform

    # ------------------------------------------------------------- durable commit

    def _submit_store_call(self, name: str, *args: Any, **kwargs: Any) -> "asyncio.Future[Any]":
        """Run ``session_store.<name>`` on the loop's default executor (where ``AsyncSessionStore``
        already runs every store call) with this task's contextvars. Not the turn pool: ten busy
        turns would queue this write for minutes while the admission lock parks the session.

        Returns the executor Future, not a Task: the live assignment is a done-callback on it, and
        nothing that cancels tasks (a cancelled caller, loop teardown's cancel-all) can cancel a
        running executor Future's callback away from its write. A closing executor is a storage
        failure (``OSError``), not a programming error."""
        method = getattr(self.session_store, name)
        loop = asyncio.get_running_loop()
        try:
            return loop.run_in_executor(None, copy_context().run, partial(method, *args, **kwargs))
        except RuntimeError as exc:
            raise OSError(f"session store unavailable: {exc}") from exc

    async def _commit_session_runtime_options(
        self, source: Any, patch: Dict[str, Any], *, session_key: Optional[str] = None,
    ) -> bool:
        """Durable-first write of the patched runtime options for one session.

        ``patch`` names only the fields to change, in the LIVE encoding: ``model_override`` (the
        full live dict; the store keeps model/provider/base_url only), ``reasoning_override`` and
        ``service_tier_override`` (``SERVICE_TIER_UNSET`` = inherit, ``None`` = explicit normal).
        Under the admission lock: re-check busy (``SessionBusy``), resolve the route and consume an
        auto-reset boundary (#48031, #58403), then persist with a session_id compare-and-swap.
        Live state is assigned by a done-callback on the write itself, only for the patched fields
        and only while the conversation epoch is unchanged; a model commit pops an armed one-turn
        restore at that moment (parity with typed ``/model``). Raises ``OSError`` on a failed save
        and ``SessionMissing`` when the entry vanished or a boundary replaced it; memory is
        untouched in every failure."""
        unknown = set(patch) - set(_RUNTIME_FIELDS)
        if unknown:
            raise ValueError(f"unknown runtime option(s): {sorted(unknown)}")
        key = session_key or self._session_key_for_source(source)
        if not key:
            raise SessionMissing("source did not resolve to a session key")
        async with self._session_admission(key):
            return await self._commit_session_runtime_options_locked(source, key, patch)

    async def _commit_session_runtime_options_locked(
        self, source: Any, session_key: str, patch: Dict[str, Any], *,
        expected: Optional[tuple] = None, refine=None,
    ) -> bool:
        """Body of ``_commit_session_runtime_options``; the caller holds the admission lock.

        ``expected`` (host API): the live-option signature its validation ran against; when a
        slash command or /login moved it since, raise ``SessionConflict`` (later user commands
        win). Skipped when this call consumed an auto-reset boundary: the old options are gone.
        ``refine(conversation, boundary) -> patch``: recompute the patch against the live state
        just before the write (the API commits only fields that change); an empty patch writes
        nothing and returns True."""
        if refine is None:
            unknown = set(patch) - set(_RUNTIME_FIELDS)
            if unknown:
                raise ValueError(f"unknown runtime option(s): {sorted(unknown)}")
        if self._is_session_running(session_key):
            raise SessionBusy(session_key)
        if getattr(self, "session_store", None) is None:
            # No routing store at all (bare harness): nothing durable to disagree with.
            conversation = self._session_state(session_key).conversation
            if refine is not None:
                patch = refine(conversation, False)
            self._assign_runtime_patch(conversation, patch)
            return True
        if source is not None:
            entry = await self.async_session_store.get_or_create_session(source)
        else:
            entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None:
            raise SessionMissing(session_key)
        boundary = bool(getattr(entry, "was_auto_reset", False))
        if boundary:
            # The route crossed an idle/daily boundary: drop the old conversation's scope NOW so it
            # cannot leak into the fresh session, and consume the flag so the next message's
            # cleanup does not wipe what we are about to store (#48031, #58403).
            self._clear_conversation_scope(session_key, reason="auto_reset")
            self._evict_cached_agent(session_key)
            entry.was_auto_reset = False
        if self._is_session_running(session_key):
            raise SessionBusy(session_key)
        self._rehydrate_session_runtime_options(session_key)
        state = self._session_state(session_key)
        if expected is not None and not boundary and runtime_options_signature(state.conversation) != expected:
            raise SessionConflict(session_key)
        if refine is not None:
            patch = refine(state.conversation, boundary)
            if not patch:
                return True
            unknown = set(patch) - set(_RUNTIME_FIELDS)
            if unknown:
                raise ValueError(f"unknown runtime option(s): {sorted(unknown)}")
        epoch = state.persistent.conversation_epoch
        write: Dict[str, Any] = {}
        if "model_override" in patch:
            write["model_override"] = patch["model_override"]
        if "reasoning_override" in patch:
            write["reasoning_override"] = patch["reasoning_override"]
        if "service_tier_override" in patch:
            write["service_tier_override"] = _durable_tier(patch["service_tier_override"])
        unit = self._submit_store_call(
            "set_runtime_options", session_key, expected_session_id=entry.session_id, **write)
        loop = asyncio.get_running_loop()
        settled: "asyncio.Future[bool]" = loop.create_future()

        def _assign_then_settle(done: "asyncio.Future[Any]") -> None:
            # Runs on the loop once the write is terminal. ``settled`` must always resolve: the
            # caller is parked on it under the admission lock.
            try:
                if done.cancelled():
                    # The executor dropped the job before it ran (shutdown): nothing was written.
                    settled.set_exception(OSError("session store write was cancelled"))
                    return
                failure = done.exception()
                if failure is not None:
                    settled.set_exception(failure)
                    return
                if not done.result():
                    settled.set_exception(SessionMissing(session_key))
                    return
                live = self._session_state(session_key)
                if live.persistent.conversation_epoch != epoch:
                    # /new, /resume or a reset cleared this conversation while the write was in
                    # flight; the value belongs to the old one and must not leak into the new.
                    settled.set_exception(SessionMissing(session_key))
                    return
                self._assign_runtime_patch(live.conversation, patch)
                settled.set_result(True)
            except BaseException as exc:  # noqa: BLE001 - never leave the caller parked
                if not settled.done():
                    settled.set_exception(exc)

        unit.add_done_callback(_assign_then_settle)
        return await self._settle_runtime_options_write(settled, session_key)

    @staticmethod
    def _assign_runtime_patch(conversation: Any, patch: Dict[str, Any]) -> None:
        """Assign ONLY the patched fields to live state."""
        if "model_override" in patch:
            value = patch["model_override"]
            conversation.model_override = dict(value) if value is not None else None
            # Decided at assignment, not at submit: a /moa or --once armed while the write was in
            # flight is superseded by the durable model, exactly as typed /model does.
            conversation.one_turn_restore = None
        if "reasoning_override" in patch:
            value = patch["reasoning_override"]
            conversation.reasoning_override = dict(value) if value is not None else None
        if "service_tier_override" in patch:
            conversation.service_tier_override = patch["service_tier_override"]

    @staticmethod
    async def _settle_runtime_options_write(settled: "asyncio.Future[bool]", session_key: str) -> bool:
        """Wait for the write + live assignment across ARBITRARILY repeated cancellation, then
        surface the last cancel. Never ``uncancel()``: this loop requested none of those cancels,
        so an enclosing ``asyncio.timeout()`` or TaskGroup must still see them."""
        cancelled: Optional[asyncio.CancelledError] = None
        while not settled.done():
            try:
                await asyncio.shield(settled)
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception:  # noqa: BLE001 - settled is terminal; retrieved below
                break
        if cancelled is not None:
            # Retrieve the terminal result so a failed write behind the cancel is never an
            # unobserved exception.
            failure = None if settled.cancelled() else settled.exception()
            if failure is not None:
                logger.warning(
                    "Durable runtime-options write for %s failed while the caller was cancelled",
                    session_key, exc_info=failure,
                )
            raise cancelled
        return settled.result()

    # ------------------------------------------------------------- structured host API

    async def apply_session_options(self, source: Any, options: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate and apply per-session model / reasoning / fast options without a chat turn.

        For hosts that drive a session's runtime options from their own UI instead of injecting
        visible slash commands (#92185). Keys: ``model`` (``""`` = inherit the configured model),
        ``provider`` (with ``model``), ``reasoning_effort`` (``""`` = inherit, ``none`` or a
        ``VALID_REASONING_EFFORTS`` level), ``fast`` (bool), ``confirm_model_selection`` (answers a
        cost / data-policy selection guard) and ``initial`` (no next-turn model note).

        The whole patch is validated before any state moves; the write is the same durable-first
        commit the slash commands use, so the API and the user never disagree. A slash command or
        /login that lands between validation and commit wins (``conflict``). Returns
        ``{"status": "accepted", "session_key", "applied", "effective": {model, provider,
        reasoning_effort, fast}, "warning"?}``, ``{"status": "confirmation_required", "code",
        "title", "message", "error"}``, or ``{"status": "rejected", "code", "error"}`` with code
        ``invalid_options``, ``invalid_session``, ``session_busy``, ``model_rejected``,
        ``reasoning_rejected``, ``fast_rejected``, ``fast_unsupported``, ``conflict``,
        ``session_missing`` or ``durable_write_failed``."""
        if not isinstance(options, Mapping):
            return _rejected("invalid_options", "session options must be a mapping")
        unknown = sorted(str(key) for key in set(options) - _API_OPTION_KEYS)
        if unknown:
            return _rejected("invalid_options", f"unknown session option(s): {', '.join(unknown)}")
        if "provider" in options and "model" not in options:
            return _rejected("invalid_options", "provider requires model")
        # The identity seam every ingress path uses first: a route that targets an unserved
        # profile is refused, never folded into the active profile's namespace.
        self._canonicalize(source)
        if getattr(source, "profile_route_rejected", False) is True:
            return _rejected("invalid_session", "session source's profile route targets an unserved profile")
        normalized = await asyncio.to_thread(self._normalize_source_for_session_key, source)
        session_key = self._session_key_for_source(normalized)
        if not session_key:
            return _rejected("invalid_session", "session source did not resolve to a session key")
        if self._is_session_running(session_key):  # early out; the commit re-checks under the lock
            return _rejected("session_busy", "session options can only change while the session is idle")
        # The source's profile scope (multiplexed), or the launch profile's scope once a hosted room
        # made unscoped credential reads fail closed (#112878).
        with self._profile_scope_for_source(normalized):
            if "model" in options:
                # The runner-wide /model lock (#115818): lock order is switch lock, then admission.
                async with self._model_switch_lock():
                    return await self._apply_session_options_scoped(normalized, session_key, options)
            return await self._apply_session_options_scoped(normalized, session_key, options)

    async def _apply_session_options_scoped(
        self, source: Any, session_key: str, options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._rehydrate_session_runtime_options(session_key)
        state = self._session_state(session_key)
        expected = runtime_options_signature(state.conversation)
        current_model, runtime = self._resolve_session_agent_runtime(source=source, session_key=session_key)
        current = {
            "model": current_model, "provider": str(runtime.get("provider") or "openrouter"),
            "base_url": str(runtime.get("base_url") or ""),
        }
        effective = dict(current, warning="")
        try:
            patch = await self._validate_session_options(source, session_key, options, runtime, effective)
        except _Rejected as exc:
            return _rejected(exc.code, exc.error)
        if isinstance(patch, dict) and patch.get("status") == "confirmation_required":
            return patch
        applied: list = []
        if patch:
            try:
                await self._commit_validated_session_options(
                    source, session_key, patch, expected, effective, applied)
            except _Rejected as exc:
                return _rejected(exc.code, exc.error)
            except SessionBusy:
                return _rejected("session_busy", "session options can only change while the session is idle")
            except SessionConflict:
                return _rejected(
                    "conflict",
                    "session runtime options changed while the request was being validated; re-read and retry")
            except SessionMissing:
                return _rejected("session_missing", "session disappeared while applying runtime options")
            except OSError as exc:
                logger.warning("session runtime options durable write failed for %s: %s", session_key, exc)
                return _rejected("durable_write_failed", f"could not persist session runtime options: {exc}")
        if "model" in applied:
            if not bool(options.get("initial")) and (
                effective["model"], effective["provider"]) != (current["model"], current["provider"]):
                self._queue_structured_model_note(session_key, current["model"], effective)
            await self._mirror_structured_model_to_session_db(session_key, effective)
        conversation = state.conversation
        result = {
            "status": "accepted", "session_key": session_key, "applied": applied,
            "effective": {
                "model": effective["model"], "provider": effective["provider"],
                "reasoning_effort": _reasoning_label(conversation.reasoning_override),
                "fast": conversation.service_tier_override == "priority",
            },
        }
        if effective["warning"]:
            result["warning"] = effective["warning"]
        return result

    async def _validate_session_options(
        self, source: Any, session_key: str, options: Mapping[str, Any], runtime: dict,
        effective: Dict[str, Any],
    ) -> Any:
        """Validate every requested field (no state moves); returns the patch in the live
        encoding, or a ``confirmation_required`` result. Raises ``_Rejected``."""
        from hermes_cli.models import resolve_fast_mode_overrides
        from hermes_constants import parse_reasoning_effort

        patch: Dict[str, Any] = {}
        if "model" in options:
            model_patch = await self._validate_session_model_option(
                source, session_key, options, runtime, effective)
            if isinstance(model_patch, dict) and model_patch.get("status") == "confirmation_required":
                return model_patch
            patch["model_override"] = model_patch
        if "reasoning_effort" in options:
            raw = options.get("reasoning_effort")
            if raw is not None and not isinstance(raw, str):
                raise _Rejected("reasoning_rejected", "reasoning_effort must be a string")
            effort = (raw or "").strip().lower()
            parsed = parse_reasoning_effort(effort) if effort else None
            if effort and parsed is None:
                raise _Rejected("reasoning_rejected", f"unsupported reasoning effort: {effort}")
            patch["reasoning_override"] = parsed
        if "fast" in options:
            fast = options.get("fast")
            if not isinstance(fast, bool):
                raise _Rejected("fast_rejected", "fast must be a boolean")
            if fast and resolve_fast_mode_overrides(
                    effective["model"], provider=effective["provider"] or None,
                    base_url=effective["base_url"] or None) is None:
                raise _Rejected("fast_unsupported", "fast mode is not available for this model")
            # fast:false is explicit normal (live None), so it also replaces an auto or cold tier.
            patch["service_tier_override"] = "priority" if fast else None
        return patch

    async def _validate_session_model_option(
        self, source: Any, session_key: str, options: Mapping[str, Any], runtime: dict,
        effective: Dict[str, Any],
    ) -> Any:
        """Resolve ``model``/``provider`` exactly as ``/model`` does (same switch resolution, same
        selection guards); returns the live override (None = inherit) or a confirmation result."""
        from gateway.run import _hermes_home, _load_gateway_config, _resolve_gateway_model
        from gateway.slash_commands_model import _ModelSwitchContext

        requested_model = str(options.get("model") or "").strip()
        requested_provider = str(options.get("provider") or "").strip()
        if not requested_model:
            if requested_provider:
                raise _Rejected("invalid_options", "provider requires model")
            user_config = _load_gateway_config()
            model_cfg = user_config.get("model") if isinstance(user_config.get("model"), dict) else {}
            effective.update(
                model=_resolve_gateway_model(user_config),
                provider=str(model_cfg.get("provider") or effective["provider"]),
                base_url=str(model_cfg.get("base_url") or ""),
                api_mode=str(model_cfg.get("api_mode") or ""),
            )
            return None
        profile_home = self._profile_scope_key_for_source(source)
        ctx = _ModelSwitchContext(
            session_key=session_key, source=source,
            config_path=(profile_home or _hermes_home) / "config.yaml", persist_global=False,
        )
        ctx.read_config()
        ctx.apply_override(self._session_state(session_key).conversation.model_override or {})
        result, error = await self._perform_model_switch(ctx, requested_model, requested_provider, source)
        if error is not None:
            raise _Rejected("model_rejected", error)
        try:
            from hermes_cli.model_selection_guards import (
                combined_selection_warning, selection_context_for_agent)
            warning = await asyncio.to_thread(
                combined_selection_warning, result.new_model, provider=result.target_provider,
                base_url=result.base_url or ctx.current_base_url or "",
                api_key=result.api_key or ctx.current_api_key or "", model_info=result.model_info,
                selection_context=selection_context_for_agent(self._cached_agent_for(session_key)),
            )
        except Exception:
            warning = None
        if warning is not None and not bool(options.get("confirm_model_selection")):
            return {
                "status": "confirmation_required", "code": "model_confirmation_required",
                "title": warning.title, "message": warning.message,
                "error": f"{warning.title}: {warning.message}",
            }
        effective.update(
            model=result.new_model, provider=result.target_provider, base_url=result.base_url or "",
            api_mode=result.api_mode or "", warning=str(result.warning_message or ""),
        )
        return {
            "model": result.new_model, "provider": result.target_provider, "api_key": result.api_key,
            "base_url": result.base_url, "api_mode": result.api_mode,
            "request_overrides": dict(result.request_overrides or {}),
            "capabilities": dict(result.runtime_capabilities or {}),
        }

    async def _commit_validated_session_options(
        self, source: Any, session_key: str, patch: Dict[str, Any], expected: tuple,
        effective: Dict[str, Any], applied: list,
    ) -> None:
        """The shared commit, plus the API's CAS and "only fields that change" refinement."""
        from hermes_cli.models import resolve_fast_mode_overrides

        def _refine(conversation: Any, boundary: bool) -> Dict[str, Any]:
            if boundary and "model_override" not in patch:
                # The old conversation's model override is gone: describe (and fast-check) the
                # model the fresh session will actually run.
                model, rt = self._resolve_session_agent_runtime(source=source, session_key=session_key)
                effective.update(model=model, provider=str(rt.get("provider") or "openrouter"),
                                 base_url=str(rt.get("base_url") or ""))
                if patch.get("service_tier_override") == "priority" and resolve_fast_mode_overrides(
                        model, provider=effective["provider"] or None,
                        base_url=effective["base_url"] or None) is None:
                    raise _Rejected("fast_unsupported", "fast mode is not available for this model")
            changed = {
                name: value for name, value in patch.items()
                if not _same_option(name, getattr(conversation, name), value)
            }
            applied[:] = [label for label, name in (
                ("model", "model_override"), ("reasoning_effort", "reasoning_override"),
                ("fast", "service_tier_override"),
            ) if name in changed]
            return changed

        async with self._session_admission(session_key):
            await self._commit_session_runtime_options_locked(
                source, session_key, patch, expected=expected, refine=_refine)
            if applied:
                self._evict_cached_agent(session_key)

    def _queue_structured_model_note(self, session_key: str, previous_model: str, effective: dict) -> None:
        """Same next-turn note ``/model`` queues, so the model's self-identification follows."""
        from hermes_cli.model_switch import format_model_for_display
        if not hasattr(self, "_pending_model_notes"):
            self._pending_model_notes = {}
        self._pending_model_notes[session_key] = (
            f"[Note: model was just switched from {format_model_for_display(previous_model)} to "
            f"{format_model_for_display(effective['model'])} via {effective['provider']}. "
            f"Adjust your self-identification accordingly.]"
        )

    async def _mirror_structured_model_to_session_db(self, session_key: str, effective: dict) -> None:
        """Best-effort: record the model on the session row for the dashboard (#34850)."""
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return
        try:
            entry = await self.async_session_store.lookup_by_session_key(session_key)
            if entry is not None:
                await session_db.update_session_model(
                    entry.session_id, effective["model"], provider=effective["provider"],
                    base_url=effective["base_url"] or None, api_mode=effective.get("api_mode") or None)
        except Exception:
            logger.debug("Failed to mirror structured model option", exc_info=True)
