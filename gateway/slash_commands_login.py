"""The off-turn, paired-DM-only ``/login`` command."""

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

_LOGIN_SINGLE = "login"
_LOCK_TYPE = type(threading.Lock())


@dataclasses.dataclass
class _SignInAttempt:
    attempt_id: str
    key: tuple
    source: Any
    cancelled: bool = False


# These adapters use "dm" for a broadcast topic, a channel, or an agent peer. Sending a consent
# link and sign-in code there would publish them rather than deliver them to one person.
_LOGIN_BLOCKED_PLATFORMS = frozenset({"ntfy", "raft", "a2a"})


class GatewayLoginCommandsMixin:
    _LOGIN_SINGLE = _LOGIN_SINGLE

    def _login_registry(self):
        """Return the lazily created registry used by normal and bare test runners."""
        lock = getattr(self, "_login_lock", None)
        if not isinstance(lock, _LOCK_TYPE):
            lock = self._login_lock = threading.Lock()
        attempts = getattr(self, "_login_attempts", None)
        if not isinstance(attempts, dict):
            attempts = self._login_attempts = {}
        return lock, attempts

    def _login_executor(self):
        executor = getattr(self, "_login_exec", None)
        if executor is None or getattr(executor, "_shutdown", False):
            executor = self._login_exec = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hermes-login")
        return executor

    async def _run_login_blocking(self, func):
        ctx = copy_context()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._login_executor(), ctx.run, func)

    async def _handle_login_command(self, event) -> str:
        src = event.source
        platform = str(getattr(src.platform, "value", src.platform)).lower()
        paired_dm = (
            getattr(src, "chat_type", None) in {"dm", "private"}
            and bool(getattr(src, "chat_id", None))
            and platform not in _LOGIN_BLOCKED_PLATFORMS
        )
        if not paired_dm:
            return anon_auth.LOGIN_DM_ONLY

        policy = policy_for_source(self.config, src)
        if policy.enabled and not policy.is_admin(getattr(src, "user_id", None)):
            return anon_auth.LOGIN_NOT_ALLOWED

        key = (str(getattr(src.platform, "value", src.platform)), str(src.chat_id),
               str(getattr(src, "user_id", "") or ""))
        lock, attempts = self._login_registry()
        # The private executor has one worker and the live attempt may be polling on it for the
        # code's full lifetime. Trip its hook before queueing the local state read, otherwise a
        # replacement command would wait behind the very attempt it needs to stop.
        with lock:
            live = attempts.get(_LOGIN_SINGLE)
            if live is not None and live.key != key:
                return anon_auth.LOGIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True

        state = await self._run_login_blocking(anon_auth.current_nous_state)
        if state and not anon_auth.is_guest_state(state):
            return anon_auth.UPGRADE_ALREADY_SIGNED_IN

        with lock:
            live = attempts.get(_LOGIN_SINGLE)
            if live is not None and live.key != key:
                return anon_auth.LOGIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True
            attempt = _SignInAttempt(
                attempt_id=uuid4().hex[:8], key=key, source=src)
            attempts[_LOGIN_SINGLE] = attempt

        self._retain_background_task(asyncio.create_task(self._run_login(attempt)))
        return anon_auth.UPGRADE_START

    async def _run_login(self, attempt: _SignInAttempt) -> None:
        lock, attempts = self._login_registry()

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
                    state = await self._run_login_blocking(lambda: next(gen, None))
                except RuntimeError as exc:
                    logger.warning(
                        "/login %s: executor unavailable, stopping: %s", attempt.attempt_id, exc)
                    return
                if state is None:
                    return
                await self._render_login_state(attempt, state)
                if state.terminal:
                    pushed_terminal = True
                    return
        except asyncio.CancelledError:
            with lock:
                attempt.cancelled = True
            raise
        except Exception:
            logger.warning("/login attempt %s failed", attempt.attempt_id, exc_info=True)
            if not pushed_terminal:
                with contextlib.suppress(Exception):
                    await self._push_login(attempt, anon_auth.UPGRADE_NOT_COMPLETED)
        finally:
            with lock:
                if attempts.get(_LOGIN_SINGLE) is attempt:
                    attempts.pop(_LOGIN_SINGLE, None)
            with contextlib.suppress(Exception):
                gen.close()

    async def _push_login(self, attempt: _SignInAttempt, text: str) -> None:
        """Push one notice without allowing a transport failure to abort the state drain."""
        try:
            await self._deliver_platform_notice(attempt.source, text)
        except Exception:
            logger.warning("/login %s: push failed", attempt.attempt_id, exc_info=True)

    async def _render_login_state(self, attempt: _SignInAttempt, state) -> None:
        if isinstance(state, anon_auth.Code):
            await self._push_login(attempt, state.link)
            await self._push_login(attempt, state.code)
            await self._push_login(attempt, state.copy_with_wait)
            return
        if isinstance(state, anon_auth.Waiting):
            return
        if isinstance(state, anon_auth.Completed):
            copy = state.copy
            if state.model_changed and state.model:
                failed = await self._sweep_sessions_off_welcome()
                if failed:
                    copy += "\nSome chats are still on the free tier; use /model in those chats to switch."
            await self._push_login(attempt, copy)
            return
        await self._push_login(attempt, state.copy)

    async def _sweep_sessions_off_welcome(self) -> int:
        """Clear durable overrides before eviction; return the number of failed clears."""
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
        failed = 0
        for key in keys:
            overrides = self._session_model_overrides
            override = overrides.get(key) or {}
            if str(override.get("model") or "") == anon_auth.GUEST_MODEL:
                for attempt in range(2):
                    try:
                        await self.async_session_store.set_model_override(key, None)
                        break
                    except Exception:
                        if attempt == 1:
                            logger.warning(
                                "/login: failed to clear free-tier override for %s", key, exc_info=True)
                else:
                    # Retain the live route while disk still pins it, including on a rebuild.
                    failed += 1
                    continue
                overrides.pop(key, None)
            try:
                self._evict_cached_agent(key)
            except Exception:
                logger.warning("/login: failed to evict free-tier session %s", key, exc_info=True)
        return failed
