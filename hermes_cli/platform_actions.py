"""Capability-gated platform action facade for plugins (#64176, action half).

Every verb returns a structured result dict — ``{"ok": True, ...}`` on success, ``{"ok": False,
"error": <code>, "detail": <str>}`` on failure — and never raises into hook dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)

ACTIONS_CONTRACT_VERSION = 1

CAPABILITY_ID = "gateway.platform_actions"
# Consent/audit gate for plugin status lines (ctx.emit_status / PlatformActions.send_status).
# Like every capability this is consent + audit visibility, NOT a security sandbox — plugins
# remain full-trust code and the raw gateway/adapter surface stays intentionally ungated.
CAPABILITY_STATUS_ID = "gateway.plugin_status"

# Plain-send fallback suppression (adapters without send_or_update_status). Process-local,
# bounded, and never load-bearing for correctness: a lost entry only re-enables a duplicate
# transient bubble, never a missing state transition.
_MAX_STATUS_THROTTLE = 256
_STATUS_THROTTLE: Dict[tuple, str] = {}
_STATUS_THROTTLE_LOCK = threading.Lock()
# Bound for the cross-loop await in the async primitive: adapter sends are sub-2s in normal
# operation (Telegram/Slack API round-trip), so 10s of a queued-but-never-started delivery
# signals a wedged gateway loop, not a slow API. Same order as the plugin await bound
# (plugins.hook_callback_timeout, resolve_plugin_command_result).
_STATUS_DELIVERY_TIMEOUT_SECS = 10.0


def _err(code: str, detail: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "error": code}
    if detail:
        result["detail"] = detail
    return result


def _ok(**fields: Any) -> Dict[str, Any]:
    return {"ok": True, **fields}


def _clean_or_none(value: Any) -> Optional[str]:
    return str(value).strip() if value is not None and str(value).strip() else None


def _consume_detached(fut: Any) -> None:
    """Done-callback for fire-and-forget deliveries: observe (swallow) the outcome so neither
    the asyncio task nor the concurrent Future logs 'exception was never retrieved'. Delivery
    failures are best-effort by design and must never propagate into the caller."""
    try:
        fut.exception()
    except (asyncio.CancelledError, Exception):
        pass


# -- per-platform verb implementations (adapter, *args) -> result ------------


async def _telegram_add_reaction(adapter, chat_id, message_id, emoji):
    if await adapter._set_reaction(chat_id, message_id, emoji):
        return _ok(action="add_reaction")
    return _err("action_failed", "telegram set_message_reaction failed")


async def _discord_add_reaction(adapter: Any, chat_id: str, message_id: str, emoji: str) -> Dict[str, Any]:
    client = getattr(adapter, "_client", None)
    if client is None:
        return _err("adapter_disconnected", "discord client unavailable")
    try:
        channel_id = int(str(chat_id))
        msg_id = int(str(message_id))
    except (TypeError, ValueError):
        return _err("invalid_argument", "discord ids must be numeric")
    channel = client.get_channel(channel_id)
    if channel is None:
        channel = await client.fetch_channel(channel_id)
    message = await channel.fetch_message(msg_id)
    await message.add_reaction(emoji)
    return _ok(action="add_reaction")


async def _telegram_set_thread_title(adapter, chat_id, thread_id, title):
    await adapter.rename_dm_topic(chat_id, int(thread_id), title)
    return _ok(action="set_thread_title")


async def _discord_set_thread_title(adapter, chat_id, thread_id, title):
    if await adapter.rename_thread(thread_id, title):
        return _ok(action="set_thread_title")
    return _err("action_failed", "discord thread rename failed")


_VERBS = {
    "add_reaction": {"telegram": _telegram_add_reaction, "discord": _discord_add_reaction},
    "set_thread_title": {"telegram": _telegram_set_thread_title, "discord": _discord_set_thread_title},
}


class _StatusJob(NamedTuple):
    """A fully-resolved, ready-to-schedule status delivery (identity frozen at prepare time)."""
    platform: str
    chat_id: str
    status_key: str
    text: str
    metadata: Optional[Dict[str, Any]]
    mode_hint: str          # "edit" (adapter dedupes) | "send" (plain fallback)
    throttle_key: tuple
    adapter: Any
    loop: Optional[object]  # runner._gateway_loop at prepare time


class PlatformActions:
    """Per-plugin facade over the live gateway adapter registry.

    Instances are cheap and hold only the owning plugin id; the gateway runner and adapters are
    resolved at call time so a facade created before the gateway starts (plugin ``register()`` runs
    first) still works once adapters connect.
    """

    def __init__(self, plugin_id: str):
        self._plugin_id = plugin_id

    # -- shared plumbing ----------------------------------------------------

    def _capability_granted(self, capability_id: str = CAPABILITY_ID) -> bool:
        try:
            from hermes_cli.plugin_capabilities import plugin_capability_granted

            return plugin_capability_granted(self._plugin_id, capability_id)
        except Exception:
            # Ground rule: failure to read consent state = not granted.
            logger.debug("platform_actions capability check failed for %s", self._plugin_id, exc_info=True)
            return False

    def _resolve_adapter(self, platform: str, profile_name: Optional[str] = None):
        """Return ``(adapter, error_dict)``; exactly one is non-None. ``profile_name`` overrides
        the active-profile read for callers that already resolved a session-scoped profile
        (e.g. plugin status routed by ``session_id``); ``None`` keeps the historical behaviour."""
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
        except Exception:
            runner = None
        if runner is None:
            return None, _err("gateway_unavailable", "no gateway runner is active in this process")
        try:
            from gateway.config import Platform

            platform_enum = Platform(str(platform).strip().lower())
        except Exception:
            return None, _err("unknown_platform", f"unknown platform {platform!r}")
        # Multiplex/Team-Gateway: a secondary profile's adapters live in runner._profile_adapters,
        # not runner.adapters. Every adapter-resolution path goes through the same profile-aware,
        # fail-closed lookup so a plugin scoped to one profile can never act through another
        # profile's bot identity. The bare default-profile lookup is only for a runner predating
        # _authorization_adapter (defensive, not expected).
        resolve_fn = getattr(runner, "_authorization_adapter", None)
        if callable(resolve_fn):
            if profile_name is None:
                try:
                    from hermes_cli.profiles import get_active_profile_name

                    profile_name = get_active_profile_name()
                except Exception:
                    # Fail closed: an unresolvable profile must not degrade to the default profile's bot.
                    logger.debug(
                        "platform_actions: profile resolution failed for %s",
                        self._plugin_id, exc_info=True,
                    )
                    return None, _err(
                        "adapter_not_registered",
                        f"no {platform_enum.value} adapter is registered "
                        "(active profile could not be resolved)",
                    )
            adapter = resolve_fn(platform_enum, profile_name)
        else:
            adapter = getattr(runner, "adapters", {}).get(platform_enum)
        if adapter is None:
            return None, _err("adapter_not_registered", f"no {platform_enum.value} adapter is registered")
        try:
            connected = bool(adapter.is_connected)
        except Exception:
            connected = False
        if not connected:
            return None, _err("adapter_disconnected", f"the {platform_enum.value} adapter is not connected")
        return adapter, None

    def _gate(self, platform: str, **required: Any):
        """Run the shared gate chain. Returns ``(adapter, error_dict)``."""
        if not self._capability_granted():
            return None, _err(
                "capability_not_granted",
                f"plugin {self._plugin_id!r} lacks the {CAPABILITY_ID!r} "
                "capability (grant via consent flow or "
                f"plugins.entries.{self._plugin_id}.allow_platform_actions)",
            )
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                return None, _err("invalid_argument", f"{name} must be a non-empty string")
        return self._resolve_adapter(platform)

    async def _run(self, verb: str, platform: str, *args: str, **required: Any) -> Dict[str, Any]:
        """Gate, dispatch *verb* to the adapter's platform implementation, audit, return."""
        adapter, error = self._gate(platform, **required)
        if error is None and adapter is not None:
            try:
                impl = _VERBS[verb].get(getattr(adapter.platform, "value", None))
                if impl is None:
                    result = _err("unsupported_platform_action", f"{verb} is not implemented for {platform}")
                else:
                    result = await impl(adapter, *args)
            except Exception as exc:
                result = _err("action_failed", str(exc)[:512])
        else:
            result = error or _err("gateway_unavailable")
        self._audit(verb, platform, result)
        return result

    # -- v1 verbs -----------------------------------------------------------

    async def add_reaction(self, platform: str, chat_id: str, message_id: str, emoji: str) -> Dict[str, Any]:
        """Add/set an emoji reaction on a platform message."""
        return await self._run(
            "add_reaction", platform, chat_id, message_id, emoji,
            chat_id=chat_id, message_id=message_id, emoji=emoji,
        )

    async def set_thread_title(self, platform: str, chat_id: str, thread_id: str, title: str) -> Dict[str, Any]:
        """Rename a thread / forum topic."""
        return await self._run(
            "set_thread_title", platform, chat_id, thread_id, title,
            chat_id=chat_id, thread_id=thread_id, title=title,
        )

    # -- status (ctx.emit_status / await ctx.platform_actions.send_status) ------
    #
    # Best-effort transient status for the CURRENT conversation: never starts a model turn,
    # never touches SessionDB conversation history, never raises into the caller. Same-loop
    # calls await the delivery coroutine directly; cross-thread/cross-loop calls schedule it
    # onto the gateway loop and await the wrapped future (async) or never wait (sync).
    # Nothing here may block on Future.result() — from the gateway loop thread that self-deadlocks,
    # and inside pre_tool_call a blocked callback would trip the fail-closed timeout policy.

    @staticmethod
    def _runner_ref():
        try:
            from gateway.run import _gateway_runner_ref

            return _gateway_runner_ref()
        except Exception:
            return None

    def _resolve_status_target(self, session_id, platform, chat_id):
        """→ ``(platform_str, chat_id, thread_id, profile, error)`` in the spec'd priority:
        explicit ``session_id`` (FAIL-CLOSED — an unknown id must not silently reroute to the
        ambient ContextVar chat), then task-local ContextVars (strict read, no os.environ
        fallback), then an explicit ``platform``+``chat_id`` pair for background callers.
        All absent → ``invalid_argument``; never a default chat."""
        if session_id is not None and str(session_id).strip():
            runner = self._runner_ref()
            store = getattr(runner, "session_store", None) if runner is not None else None
            entry = None
            if store is not None and hasattr(store, "lookup_by_session_id"):
                try:
                    entry = store.lookup_by_session_id(str(session_id))
                except Exception:
                    logger.debug("platform_actions: session lookup failed for %s", self._plugin_id, exc_info=True)
            if entry is None:
                return None, None, None, None, _err("invalid_argument", f"unknown session_id {str(session_id)!r}")
            source = getattr(entry, "origin", None) or entry
            plat = getattr(source, "platform", None) or getattr(entry, "platform", None)
            raw_chat = getattr(source, "chat_id", "") or ""
            if plat is None or not str(raw_chat).strip():
                return None, None, None, None, _err("invalid_argument", f"session {str(session_id)!r} has no origin routing")
            return (
                getattr(plat, "value", str(plat)), str(raw_chat),
                _clean_or_none(getattr(source, "thread_id", None)),
                _clean_or_none(getattr(source, "profile", None)), None,
            )
        try:
            from gateway.session_context import get_current_session_identity

            ident = get_current_session_identity()
        except Exception:
            ident = None
        if ident is not None:
            return ident["platform"], ident["chat_id"], ident.get("thread_id"), ident.get("profile"), None
        if platform is not None and chat_id is not None:
            return (str(platform).strip().lower(), str(chat_id).strip(), None, None, None)
        return None, None, None, None, _err(
            "invalid_argument",
            "no session_id, no current-conversation context, and no explicit platform+chat_id "
            "— refusing to route a status message",
        )

    def _prepare_status(self, text, key, session_id, platform, chat_id, metadata):
        """Synchronous (caller-thread) validation + identity + adapter resolution + throttle
        check-and-reserve. Returns ``(job, None)`` when delivery should proceed, ``(None, result)``
        when a structured result must be returned instead (error or throttle-skip)."""
        import hashlib

        if not isinstance(text, str) or not text.strip():
            return None, _err("invalid_argument", "text must be a non-empty string")
        if not isinstance(key, str) or not key.strip():
            return None, _err("invalid_argument", "key must be a non-empty string")
        if not self._capability_granted(CAPABILITY_STATUS_ID):
            return None, _err(
                "capability_not_granted",
                f"plugin {self._plugin_id!r} lacks the {CAPABILITY_STATUS_ID!r} capability "
                f"(grant via consent flow or plugins.entries.{self._plugin_id}.allow_plugin_status)",
            )
        plat, chat, thread, profile, err = self._resolve_status_target(session_id, platform, chat_id)
        if err is not None:
            return None, err
        adapter, err = self._resolve_adapter(plat, profile)
        if err is not None:
            return None, err
        # Thread/topic routing rides the metadata the adapters actually consume
        # (telegram _metadata_thread_id / slack _resolve_thread_ts both read "thread_id");
        # a caller-supplied thread in metadata is never overwritten.
        resolved_metadata = dict(metadata) if isinstance(metadata, dict) else None
        if thread and resolved_metadata is not None and not (
            {"thread_id", "message_thread_id"} & set(resolved_metadata)
        ):
            resolved_metadata["thread_id"] = thread
        elif thread and resolved_metadata is None:
            resolved_metadata = {"thread_id": thread}
        # Namespace the key so a plugin can never collide with core status lines (keyed by
        # event_type) or with another plugin; the plugin identity comes from the facade, never
        # from caller-passed text.
        status_key = f"plugin:{self._plugin_id}:{key}"
        # profile None = the resolver's active/launch-profile fallback (same semantics as the
        # other platform-action verbs); keyed explicitly so fallback and a real profile never
        # collapse onto one throttle entry.
        throttle_key = (self._plugin_id, plat, profile or "active", chat, thread or "", status_key)
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        with _STATUS_THROTTLE_LOCK:
            if _STATUS_THROTTLE.get(throttle_key) == digest:
                return None, _ok(action="send_status", mode="skipped")
            # Reserve at check time (single critical section) so concurrent identical emits
            # cannot both pass the check and double-send; _deliver_status releases the
            # reservation on failure so a transient error is not silently swallowed forever.
            if len(_STATUS_THROTTLE) >= _MAX_STATUS_THROTTLE:
                for stale in list(_STATUS_THROTTLE)[: _MAX_STATUS_THROTTLE // 2]:
                    _STATUS_THROTTLE.pop(stale, None)
            _STATUS_THROTTLE[throttle_key] = digest
        has_native = callable(getattr(adapter, "send_or_update_status", None))
        runner = self._runner_ref()
        return (
            _StatusJob(
                platform=plat, chat_id=chat, status_key=status_key, text=text,
                metadata=resolved_metadata,
                mode_hint="edit" if has_native else "send",
                throttle_key=throttle_key, adapter=adapter,
                loop=getattr(runner, "_gateway_loop", None) if runner is not None else None,
            ),
            None,
        )

    def _release_status_reservation(self, throttle_key: tuple) -> None:
        with _STATUS_THROTTLE_LOCK:
            _STATUS_THROTTLE.pop(throttle_key, None)

    async def _deliver_status(self, job: _StatusJob) -> Dict[str, Any]:
        """Run the actual send on the gateway loop (wherever this coroutine executes)."""
        try:
            from gateway.status_delivery import send_or_update_status

            result = await send_or_update_status(
                job.adapter, job.chat_id, job.status_key, job.text, job.metadata,
            )
            if getattr(result, "success", False):
                message_id = getattr(result, "message_id", None)
                # Reservation from _prepare_status already holds the digest; nothing to write.
                return _ok(action="send_status", mode=job.mode_hint,
                           message_id=(str(message_id) if message_id is not None else None))
            self._release_status_reservation(job.throttle_key)
            return _err("action_failed", str(getattr(result, "error", "status send failed"))[:512])
        except asyncio.CancelledError:
            self._release_status_reservation(job.throttle_key)
            raise
        except Exception as exc:
            self._release_status_reservation(job.throttle_key)
            return _err("action_failed", str(exc)[:512])

    @staticmethod
    def _schedule_detached(coro, target_loop, *, on_current_thread: bool):
        """One scheduling primitive for the sync path: hand ``coro`` to ``target_loop`` without
        ever waiting; every failure path CLOSES the coroutine (no 'never awaited' leaks)."""
        try:
            if on_current_thread:
                task = target_loop.create_task(coro)  # raises RuntimeError if the loop stopped
                task.add_done_callback(_consume_detached)
                return None
            from agent.async_utils import safe_schedule_threadsafe

            fut = safe_schedule_threadsafe(
                coro, target_loop, logger=logger, log_message="plugin status scheduling failed",
            )  # safe_schedule_threadsafe closes the coroutine itself when it returns None
            return None if fut is not None else _err("gateway_unavailable", "loop rejected the coroutine")
        except Exception as exc:  # create_task raced a shutdown, etc.
            coro.close()
            return _err("gateway_unavailable", f"scheduling failed: {str(exc)[:200]}")

    async def send_status(self, text: str, key: str, *, session_id: Optional[str] = None,
                          platform: Optional[str] = None, chat_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async delivery primitive: post (or edit) this conversation's transient status line
        and await the structured delivery result. Structured for every runtime failure (never
        raises into hook dispatch); ``asyncio.CancelledError`` propagates (cancellation is not
        a failure). Same-gateway-loop calls await directly; other loops schedule onto it and
        await the wrapped future bounded by ``_STATUS_DELIVERY_TIMEOUT_SECS``
        (``{"ok": False, "error": "status_timeout"}`` on expiry)."""
        result: Optional[Dict[str, Any]] = None
        label = str(platform or "unknown")
        try:
            job, early = self._prepare_status(text, key, session_id, platform, chat_id, metadata)
            if early is not None:
                result = early
            else:
                label = job.platform
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:  # async def always has one; defensive
                    running = None
                if job.loop is not None and getattr(job.loop, "is_closed", lambda: False)():
                    self._release_status_reservation(job.throttle_key)
                    result = _err("gateway_unavailable", "gateway event loop is closed")
                elif job.loop is None or running is job.loop:
                    # No runner loop published (test doubles / pre-startup) or we ARE on it: run here.
                    result = await self._deliver_status(job)
                else:
                    from agent.async_utils import safe_schedule_threadsafe

                    fut = safe_schedule_threadsafe(
                        self._deliver_status(job), job.loop, logger=logger,
                        log_message=f"plugin status scheduling failed for {self._plugin_id}",
                    )
                    if fut is None:
                        self._release_status_reservation(job.throttle_key)
                        result = _err("gateway_unavailable", "gateway loop refused the coroutine (shutting down?)")
                    else:
                        try:
                            result = await asyncio.wait_for(asyncio.wrap_future(fut),
                                                            timeout=_STATUS_DELIVERY_TIMEOUT_SECS)
                        except asyncio.TimeoutError:
                            fut.cancel()
                            self._release_status_reservation(job.throttle_key)
                            result = _err(
                                "status_timeout",
                                f"gateway loop did not complete the status within "
                                f"{_STATUS_DELIVERY_TIMEOUT_SECS:.0f}s",
                            )
                        except asyncio.CancelledError:
                            # Cancelling the wrapped future before the delivery task starts means
                            # _deliver_status's own release never runs — release here (idempotent
                            # pop; if the task DID start, its handler releases too).
                            fut.cancel()
                            self._release_status_reservation(job.throttle_key)
                            raise
        except asyncio.CancelledError:
            raise  # caller-initiated cancellation is not a runtime failure
        except Exception as exc:
            # Structured, never-raising is the contract — but keep the traceback observable so a
            # programmer error inside the facade is not silently reclassified as an adapter failure.
            logger.debug("platform_actions: send_status internal error for %s",
                         self._plugin_id, exc_info=True)
            result = _err("action_failed", f"status emission failed: {str(exc)[:200]}")
        self._audit("send_status", label, result or _err("action_failed", "no result"))
        return result

    def emit_status(self, text: str, key: str, *, session_id: Optional[str] = None,
                    platform: Optional[str] = None, chat_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sync fire-and-forget status emission — ACCEPTANCE semantics only.

        Returns ``{"accepted": bool, "error": str | None}`` reflecting SCHEDULING, not delivery;
        never waits, never calls ``Future.result()``, never raises into hook dispatch. Delivery
        outcome lands in the audit log. Scheduling (same-loop detached task or cross-thread
        ``safe_schedule_threadsafe``) runs through ONE primitive with no-leak cleanup on every
        failure path."""
        job = None
        try:
            job, early = self._prepare_status(text, key, session_id, platform, chat_id, metadata)
            if early is not None:
                self._audit("emit_status", str(platform or "unknown"), early)
                if early.get("ok"):  # throttle-skip before scheduling: accepted, nothing to send
                    return {"accepted": True, "error": None}
                return {"accepted": False, "error": early.get("error", "failed"), "detail": early.get("detail", "")}
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            loop = job.loop
            if loop is not None and (getattr(loop, "is_closed", lambda: False)()):
                self._release_status_reservation(job.throttle_key)
                return {"accepted": False, "error": "gateway_unavailable", "detail": "gateway loop is closed"}
            if loop is None and running is None:
                self._release_status_reservation(job.throttle_key)
                return {"accepted": False, "error": "gateway_unavailable", "detail": "no live gateway loop"}
            target = loop if loop is not None else running  # caller loop only when runner loop unpublished
            sched_err = self._schedule_detached(
                self._deliver_status(job), target, on_current_thread=(running is target),
            )
            if sched_err is not None:
                self._release_status_reservation(job.throttle_key)
                return {"accepted": False, "error": sched_err["error"], "detail": sched_err.get("detail", "")}
            self._audit("emit_status", job.platform, _ok(action="send_status", mode=job.mode_hint, scheduled=True))
            return {"accepted": True, "error": None}
        except Exception as exc:  # acceptance must never take down a hook callback
            if job is not None:
                self._release_status_reservation(job.throttle_key)
            logger.debug("platform_actions: emit_status failed for %s", self._plugin_id, exc_info=True)
            return {"accepted": False, "error": "emit_failed", "detail": str(exc)[:200]}

    def _audit(self, verb: str, platform: str, result: Dict[str, Any]) -> None:
        """Every platform action is logged (the #64176 'all actions logged' rule)."""
        logger.info(
            "platform_action plugin=%s verb=%s platform=%s ok=%s%s",
            self._plugin_id,
            verb,
            platform,
            result.get("ok"),
            "" if result.get("ok") else f" error={result.get('error')}",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
