"""The off-turn, paired-DM-only ``/login`` command: bare, the Nous sign-in; ``/login <plugin>``, the device-code
sign-in of an ``oauth:`` client that plugin declares (:mod:`hermes_cli.plugin_oauth`). Either way the gateway,
not the agent, posts the link and code to this DM and saves the grant; nothing goes through model context."""

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

from agent.i18n import t
from gateway.run_agent_cache import _first_agent
from gateway.slash_access import policy_for_runner_source
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
    saving: bool = False   # past the final cancel check: the save is committed, a replacement waits


_PLUGIN_LOGIN_START = "Signing in {plugin} ({client}): the link and code follow. Only you should open them."
_PLUGIN_LOGIN_DONE = "Signed in to {client} for plugin '{plugin}'. The grant is stored in this Hermes profile."
_PLUGIN_LOGIN_COPY = {
    "cancelled": anon_auth.UPGRADE_CANCELLED.strip(),
    "oauth_device_code_expired": anon_auth.UPGRADE_TIMED_OUT,
    "oauth_authorization_denied": "The sign-in was refused, so nothing was saved.",
    "oauth_scope_not_granted": "Not every permission was granted, so nothing was saved. Start again and allow them all.",
    "oauth_device_verification_rejected": ("The sign-in was stopped: the plugin's sign-in server pointed to a page "
                                           "outside the hosts the plugin declares."),
    "oauth_client_id_missing": "The plugin's OAuth client is not set in this Hermes profile's .env.",
    "oauth_client_secret_missing": "The plugin's OAuth client is not set in this Hermes profile's .env.",
}


# These adapters use "dm" for a broadcast topic, a channel, or an agent peer. Sending a sign-in
# link and code there would publish them rather than deliver them to one person.
_LOGIN_BLOCKED_PLATFORMS = frozenset({"ntfy", "raft", "a2a"})


def _publish_local_notice(runner, source, text: str) -> bool:
    """A LOCAL session's adapter ``send`` is a no-op (viewers read the event stream), so the
    sign-in link/code/result reach its viewers as a sticky ``notification.show``. False when
    *source* is not a live local session (the platform push applies)."""
    from gateway.config import Platform
    if getattr(source, "platform", None) != Platform.LOCAL:
        return False
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    live = authority.sessions.get(source.chat_id) if authority is not None else None
    if live is None:
        return False
    live.event_stream.publish(source.chat_id, {"text": text, "level": "info", "kind": "sticky",
                                               "key": "login:" + uuid4().hex[:8]},
                              event_type="notification.show")
    return True


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

        policy = policy_for_runner_source(self, src)
        if policy.enabled and not policy.is_admin(getattr(src, "user_id", None)):
            return anon_auth.LOGIN_NOT_ALLOWED

        key = (str(getattr(src.platform, "value", src.platform)), str(src.chat_id),
               str(getattr(src, "user_id", "") or ""))
        get_args = getattr(event, "get_command_args", None)
        target = str(get_args() or "").strip() if callable(get_args) else ""
        if target:
            return await self._start_plugin_login(src, key, target)
        lock, attempts = self._login_registry()
        # The private executor has one worker and the live attempt may be polling on it for the
        # code's full lifetime. Trip its hook before queueing the local state read, otherwise a
        # replacement command would wait behind the very attempt it needs to stop.
        with lock:
            live = attempts.get(_LOGIN_SINGLE)
            if live is not None and (live.key != key or live.saving):
                return anon_auth.LOGIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True

        state = await self._run_login_blocking(anon_auth.current_nous_state)
        if state and not anon_auth.is_guest_state(state):
            return anon_auth.UPGRADE_ALREADY_SIGNED_IN

        with lock:
            live = attempts.get(_LOGIN_SINGLE)
            if live is not None and (live.key != key or live.saving):
                return anon_auth.LOGIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True
            attempt = _SignInAttempt(
                attempt_id=uuid4().hex[:8], key=key, source=src)
            attempts[_LOGIN_SINGLE] = attempt

        self._retain_background_task(asyncio.create_task(self._run_login(attempt)))
        return anon_auth.UPGRADE_START

    async def _start_plugin_login(self, src, key: tuple, target: str) -> str:
        """``/login <plugin>[:<id>]``: resolve the plugin's device-capable ``oauth:`` entry, then run its
        device sign-in in the background on the same single-attempt registry as the Nous sign-in."""
        from hermes_cli import plugin_oauth

        lock, attempts = self._login_registry()
        with lock:  # trip a live attempt of ours first: the one-worker executor may be polling for it
            live = attempts.get(_LOGIN_SINGLE)
            if live is not None and (live.key != key or live.saving):
                return anon_auth.LOGIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True
        try:
            client, grant_key = await self._run_login_blocking(lambda: plugin_oauth.device_client(target))
        except LookupError as exc:
            return str(exc)
        except Exception:
            logger.warning("/login %s: could not resolve the plugin", target, exc_info=True)
            return anon_auth.UPGRADE_NOT_COMPLETED
        with lock:
            live = attempts.get(_LOGIN_SINGLE)
            if live is not None and (live.key != key or live.saving):
                return anon_auth.LOGIN_BUSY_ELSEWHERE
            if live is not None:
                live.cancelled = True
            attempt = _SignInAttempt(attempt_id=uuid4().hex[:8], key=key, source=src)
            attempts[_LOGIN_SINGLE] = attempt
        plugin = grant_key.rsplit(":", 1)[0]
        self._retain_background_task(asyncio.create_task(self._run_plugin_login(attempt, client, grant_key)))
        return _PLUGIN_LOGIN_START.format(plugin=plugin, client=client.id)

    def _login_hooks(self, attempt: _SignInAttempt):
        """``(cancelled, persist_guard)`` for one attempt."""
        lock, _ = self._login_registry()

        def _cancelled() -> bool:
            with lock:
                return attempt.cancelled

        @contextlib.contextmanager
        def _persist_guard():
            # The final cancel check claims the save under the registry lock, so a replacing /login
            # either cancels first or is told a sign-in is finishing. The save itself runs outside
            # the lock: the event loop takes that lock, and a save can wait on the auth store.
            with lock:
                if not attempt.cancelled:
                    attempt.saving = True
                may_persist = attempt.saving
            yield may_persist

        return _cancelled, _persist_guard

    async def _run_login(self, attempt: _SignInAttempt) -> None:
        cancelled, persist_guard = self._login_hooks(attempt)
        gen = anon_auth.run_sign_in(timeout_seconds=15.0, cancelled=cancelled, persist_guard=persist_guard)

        async def render(state) -> bool:
            await self._render_login_state(attempt, state)
            return state.terminal

        await self._drain_login(attempt, gen, render)

    async def _run_plugin_login(self, attempt: _SignInAttempt, client, grant_key: str) -> None:
        from hermes_cli import plugin_oauth

        cancelled, persist_guard = self._login_hooks(attempt)
        gen = plugin_oauth.device_sign_in(client, grant_key, cancelled=cancelled, may_save=persist_guard)
        plugin = grant_key.rsplit(":", 1)[0]

        async def render(state) -> bool:
            if isinstance(state, plugin_oauth.DeviceCode):
                await self._push_login(attempt, state.link)
                await self._push_login(attempt, state.user_code)
                await self._push_login(
                    attempt, f"{anon_auth.UPGRADE_DO_NOT_SHARE} {anon_auth.format_wait_line(state.expires_in)}")
                return False
            if state == "signed_in":
                copy = _PLUGIN_LOGIN_DONE.format(client=client.id, plugin=plugin)
            else:
                copy = _PLUGIN_LOGIN_COPY.get(str(state), anon_auth.UPGRADE_NOT_COMPLETED)
            await self._push_login(attempt, copy)
            return True

        await self._drain_login(attempt, gen, render)

    async def _drain_login(self, attempt: _SignInAttempt, gen, render) -> None:
        """Advance a sign-in generator on the login executor and render each state, until a terminal one."""
        lock, attempts = self._login_registry()
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
                if await render(state):
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
            if _publish_local_notice(self, attempt.source, text):
                return
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
                    copy += t("gateway.login.free_tier_remaining")
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
