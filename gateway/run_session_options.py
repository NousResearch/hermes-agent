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
from typing import Any, Dict, Optional

from gateway.session_persistence import UNSET
from gateway.session_state import SERVICE_TIER_UNSET

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

__all__ = ["UNSET", "SessionBusy", "SessionMissing", "GatewaySessionOptionsMixin"]

_RUNTIME_FIELDS = ("model_override", "reasoning_override", "service_tier_override")


class SessionBusy(Exception):
    """A turn owns the session; runtime options only change while it is idle."""


class SessionMissing(Exception):
    """The routing entry vanished or was replaced (a boundary won) before the write landed."""


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
        self, source: Any, session_key: str, patch: Dict[str, Any],
    ) -> bool:
        if self._is_session_running(session_key):
            raise SessionBusy(session_key)
        if getattr(self, "session_store", None) is None:
            # No routing store at all (bare harness): nothing durable to disagree with.
            self._assign_runtime_patch(self._session_state(session_key).conversation, patch)
            return True
        if source is not None:
            entry = await self.async_session_store.get_or_create_session(source)
        else:
            entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None:
            raise SessionMissing(session_key)
        if getattr(entry, "was_auto_reset", False):
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
