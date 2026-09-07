"""The off-turn, paired-DM-only ``/signin`` command."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import logging
import threading
from contextvars import copy_context
from typing import Any
from uuid import uuid4

from gateway.run_agent_cache import _first_agent
from gateway.slash_access import policy_for_source
from hermes_cli import anon_auth

logger = logging.getLogger("gateway.run")

_SIGNIN_SINGLE = "signin"
_LOCK_TYPE = type(threading.Lock())


@dataclasses.dataclass
class _SignInAttempt:
    attempt_id: str
    key: tuple
    source: Any
    cancelled: bool = False


# These adapters use "dm" for a broadcast topic, a channel, or an agent peer. Sending a consent
# link and sign-in code there would publish them rather than deliver them to one person.
_SIGNIN_BLOCKED_PLATFORMS = frozenset({"ntfy", "raft", "a2a"})


class GatewaySignInCommandsMixin:
    _SIGNIN_SINGLE = _SIGNIN_SINGLE

    def _signin_registry(self):
        """Return the lazily created registry used by normal and bare test runners."""
        lock = getattr(self, "_signin_lock", None)
        if not isinstance(lock, _LOCK_TYPE):
            lock = self._signin_lock = threading.Lock()
        attempts = getattr(self, "_signin_attempts", None)
        if not isinstance(attempts, dict):
            attempts = self._signin_attempts = {}
        return lock, attempts

    def _signin_executor(self):
        executor = getattr(self, "_signin_exec", None)
        if executor is None or getattr(executor, "_shutdown", False):
            executor = self._signin_exec = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hermes-signin")
        return executor

    async def _run_signin_blocking(self, func):
        ctx = copy_context()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._signin_executor(), ctx.run, func)

    async def _handle_signin_command(self, event) -> str:
        src = event.source
        platform = str(getattr(src.platform, "value", src.platform)).lower()
        paired_dm = (
            getattr(src, "chat_type", None) in {"dm", "private"}
            and bool(getattr(src, "chat_id", None))
            and platform not in _SIGNIN_BLOCKED_PLATFORMS
        )
        if not paired_dm:
            return anon_auth.SIGNIN_DM_ONLY

        policy = policy_for_source(self.config, src)
        if policy.enabled and not policy.is_admin(getattr(src, "user_id", None)):
            return anon_auth.SIGNIN_NOT_ALLOWED

        key = (str(getattr(src.platform, "value", src.platform)), str(src.chat_id),
               str(getattr(src, "user_id", "") or ""))
        lock, attempts = self._signin_registry()
        # The private executor has one worker and the live attempt may be polling on it for the
        # code's full lifetime. Trip its hook before queueing the local state read, otherwise a
        # replacement command would wait behind the very attempt it needs to stop.
        with lock:
            live = attempts.get(_SIGNIN_SINGLE)
            if live is not None and live.key != key:
                return anon_auth.SIGNIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True

        state = await self._run_signin_blocking(anon_auth.current_nous_state)
        if state and not anon_auth.is_guest_state(state):
            return anon_auth.UPGRADE_ALREADY_SIGNED_IN

        with lock:
            live = attempts.get(_SIGNIN_SINGLE)
            if live is not None and live.key != key:
                return anon_auth.SIGNIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True
            attempt = _SignInAttempt(
                attempt_id=uuid4().hex[:8], key=key, source=src)
            attempts[_SIGNIN_SINGLE] = attempt

        self._retain_background_task(asyncio.create_task(self._run_signin(attempt)))
        return anon_auth.UPGRADE_START

    async def _run_signin(self, attempt: _SignInAttempt) -> None:
        lock, attempts = self._signin_registry()

        def _cancelled() -> bool:
            with lock:
                return attempt.cancelled

        gen = anon_auth.run_sign_in(
            timeout_seconds=15.0,
            cancelled=_cancelled,
            cancel_wins_after_promotion=False,
        )
        pushed_terminal = False
        try:
            while True:
                try:
                    state = await self._run_signin_blocking(lambda: next(gen, None))
                except RuntimeError as exc:
                    logger.warning(
                        "/signin %s: executor unavailable, stopping: %s", attempt.attempt_id, exc)
                    return
                if state is None:
                    return
                await self._render_signin_state(attempt, state)
                if state.terminal:
                    pushed_terminal = True
                    return
        except asyncio.CancelledError:
            with lock:
                attempt.cancelled = True
            raise
        except Exception:
            logger.warning("/signin attempt %s failed", attempt.attempt_id, exc_info=True)
            if not pushed_terminal:
                with contextlib.suppress(Exception):
                    await self._push_signin(attempt, anon_auth.UPGRADE_NOT_COMPLETED)
        finally:
            with lock:
                if attempts.get(_SIGNIN_SINGLE) is attempt:
                    attempts.pop(_SIGNIN_SINGLE, None)
            with contextlib.suppress(Exception):
                gen.close()

    async def _push_signin(self, attempt: _SignInAttempt, text: str) -> None:
        """Push one notice without allowing a transport failure to abort the state drain."""
        try:
            await self._deliver_platform_notice(attempt.source, text)
        except Exception:
            logger.warning("/signin %s: push failed", attempt.attempt_id, exc_info=True)

    async def _render_signin_state(self, attempt: _SignInAttempt, state) -> None:
        if isinstance(state, anon_auth.Code):
            await self._push_signin(attempt, state.link)
            await self._push_signin(attempt, state.code)
            await self._push_signin(attempt, state.copy_with_wait)
            return
        if isinstance(state, anon_auth.Waiting):
            return
        if isinstance(state, anon_auth.Completed):
            if state.model_changed and state.model:
                await self._sweep_sessions_off_welcome()
            await self._push_signin(attempt, state.copy)
            return
        await self._push_signin(attempt, state.copy)

    async def _sweep_sessions_off_welcome(self) -> None:
        """Evict cached free-tier routes and clear overrides pinned to that route."""
        lock = getattr(self, "_agent_cache_lock", None)
        cache = getattr(self, "_agent_cache", None) or {}
        with (lock or contextlib.nullcontext()):
            entries = list(cache.items())
        keys = [
            key for key, entry in entries
            if (agent := _first_agent(entry)) is not None
            and str(getattr(agent, "provider", "")) == "nous"
            and str(getattr(agent, "model", "")) == anon_auth.GUEST_MODEL
        ]
        for key in keys:
            try:
                self._evict_cached_agent(key)
            except Exception:
                logger.warning("/signin: failed to evict free-tier session %s", key, exc_info=True)
            try:
                overrides = self._session_model_overrides
                override = overrides.get(key) or {}
                if str(override.get("model") or "") == anon_auth.GUEST_MODEL:
                    overrides.pop(key, None)
                    await self.async_session_store.set_model_override(key, None)
            except Exception:
                logger.warning("/signin: failed to clear free-tier override for %s", key, exc_info=True)
