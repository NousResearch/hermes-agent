"""hc-384 Feishu self-reconnect supervisor + hc-385 heartbeat — zero-in-place seam.

What this replaces
==================
The lark SDK's built-in websocket auto-reconnect was observed in prod to try
exactly once and give up (0 successful reconnects across ~188 daily drops),
leaving Feishu bots dead for 5–15h until the next gateway restart (hc-384).
The fix: the adapter **owns** reconnection — disable the SDK's retry, watch the
socket from a supervisor task, rebuild it on a backoff ladder, verify liveness
with a ``/bot/v3/info`` probe, and escalate to a *retryable* fatal error once
the ladder is exhausted so the gateway's reconnect watcher recreates the
adapter (re-acquiring the app lock). This mirrors the Telegram resilience
pattern; measured ~39s self-heal in prod.

This module also carries the hc-385 **long-task heartbeat** (opt-in, zh status
message edited in place during long agent turns). It used to live inline in our
fork's ``gateway/platforms/feishu.py``; upstream v0.18 deleted that file when it
migrated Feishu to a bundled plugin (``plugins/platforms/feishu/adapter.py``,
upstream 5600105478) and has no equivalent (``send_typing`` is a documented
no-op for Feishu), so the heartbeat now rides this seam to keep the upstream
adapter file byte-for-byte upstream.

How the seam attaches (v0.18 retarget)
======================================
Upstream v0.18 loads platform adapters **lazily** through
``gateway.platform_registry``: bundled platform plugins register a deferred
loader, and the adapter module is imported (as ``hermes_plugins.<slug>.adapter``,
a synthetic module name) only when the registry is first asked for that
platform. That means there is no stable importable module path to patch at
plugin-discovery time, and our old extraction-point stubs
(``_start_ws_supervisor`` / ``_cancel_ws_supervisor`` in the deleted
``gateway/platforms/feishu.py``) are gone.

So the seam now attaches one level up, at the single stable choke point every
Feishu adapter instance passes through:

    ``gateway.platform_registry.PlatformRegistry.create_adapter``

``apply()`` wraps that method. When it returns a ``feishu`` adapter, the
wrapper instruments the adapter's **class** once (idempotent):

* wraps ``connect``    — on success: arm supervisor state, disable the lark
  SDK's auto-reconnect on the live ws client (supervisor owns reconnection),
  and start the watcher task;
* wraps ``disconnect`` — flags the teardown as intentional and cancels the
  supervisor + heartbeats before upstream's own teardown runs;
* wraps ``on_processing_start`` / ``on_processing_complete`` — hc-385
  heartbeat start/stop around a turn;
* binds the reconnect-ladder + heartbeat helpers (pure additions; upstream's
  class has none of these names).

Because the gateway's reconnect watcher recreates adapters through the same
``create_adapter`` path, a recreated adapter is instrumented too.

Upstream v0.18 coverage check (why this seam still exists)
==========================================================
Reviewed upstream work in the v2026.6.19..v2026.7.1 window touching Feishu:
``b296915c82`` routes blocking SDK calls through an adapter-owned executor,
``7ee0b68973`` prevents executor resurrection during real shutdown, and the
adapter tunes the SDK's own retry (``_reconnect_nonce``/``_reconnect_interval``)
— but upstream still **delegates reconnection to the lark SDK** (the mechanism
that failed in prod) and has no liveness watchdog, no relaunch ladder, and no
escalation to an adapter recreate. So the supervisor is kept, retargeted.

Revert lever
============
``FEISHU_WS_SELF_RECONNECT=false`` (env, or ``ws_self_reconnect: false`` under
``platforms.feishu.extra``) keeps the wrapper passive: the SDK keeps its own
auto-reconnect and no supervisor task starts — pure upstream behavior with the
overlay loaded. The hc-385 heartbeat stays opt-in via ``FEISHU_HEARTBEAT`` /
``heartbeat_enabled`` (default off).

``apply()`` is idempotent and fail-safe: if the registry method is missing it
logs an error and returns ``False`` (the plugin warns; platform creation keeps
working on stock upstream behavior) but never raises.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Upstream symbols this seam binds against. Centralized so the seam-test can
# assert they still exist with a compatible shape. If upstream renames/moves
# any of these, both the patch AND the seam-test break loudly.
_TARGET_REGISTRY_MODULE = "gateway.platform_registry"
_TARGET_REGISTRY_CLS = "PlatformRegistry"
_TARGET_FACTORY_METHOD = "create_adapter"
_FEISHU_PLATFORM_NAME = "feishu"

# Adapter methods the class instrumentation wraps. All four must exist on the
# upstream adapter class; a missing one means upstream reshaped the adapter and
# the seam refuses to arm (logs, stays stock).
_WRAPPED_METHODS = ("connect", "disconnect", "on_processing_start", "on_processing_complete")

# The reconnect-ladder + heartbeat helpers bound onto the adapter class. Pure
# additions (upstream has none of these names); the seam-test asserts they are
# all present after instrumentation.
_BOUND_HELPERS = (
    "_start_ws_supervisor",
    "_cancel_ws_supervisor",
    "_supervise_websocket",
    "_websocket_appears_dead",
    "_reconnect_websocket_with_backoff",
    "_verify_ws_alive",
    "_teardown_ws_thread",
    "_maybe_start_heartbeat",
    "_stop_heartbeat",
    "_cancel_all_heartbeats",
    "_heartbeat_text",
    "_run_heartbeat",
)

# ---------------------------------------------------------------------------
# Reconnect tuning. The hc-384 behavior tests monkeypatch these on THIS module
# (``apex_overlay.feishu_supervisor``) to make the ladder fast.
# ---------------------------------------------------------------------------
_FEISHU_WS_HEALTH_CHECK_INTERVAL = 45        # seconds between liveness checks
_FEISHU_WS_RECONNECT_MAX_ATTEMPTS = 6        # ladder length before escalating to a gateway-level adapter recreate
_FEISHU_WS_RECONNECT_BASE_DELAY = 5          # seconds (5 → 10 → 20 → 40 → 60 → 60)
_FEISHU_WS_RECONNECT_MAX_DELAY = 60          # seconds cap, matches Telegram's ladder
_FEISHU_WS_RECONNECT_VERIFY_DELAY = 5        # seconds to let a relaunched socket establish before probing
_FEISHU_WS_RECONNECT_CONNECT_TIMEOUT = 30    # seconds bound on a single relaunch (guards against a hung hydrate)
_FEISHU_WS_PROBE_TIMEOUT = 10                # seconds for the /bot/v3/info liveness probe

# hc-385: long-task heartbeat templates (product copy — zh, matches the fork's
# original inline implementation).
_FEISHU_HEARTBEAT_RUNNING_TEMPLATE = "🔄 仍在执行,已运行约 {minutes} 分钟…"
_FEISHU_HEARTBEAT_FINAL_TEMPLATE = "☑️ 本轮处理结束(用时约 {minutes} 分钟)"

# Guard so apply() is idempotent even if called from multiple boot paths.
_APPLIED = False
_MARK = "_apex_overlay_feishu_supervisor"

# Class-level marker: set once the adapter class has been instrumented. Also
# what ``apex_overlay_active`` reports.
_ACTIVE_FLAG = "_apex_ws_supervisor_active"


def apex_overlay_active(adapter) -> bool:
    """True when this overlay has instrumented the adapter's class."""
    return bool(getattr(type(adapter), _ACTIVE_FLAG, False))


# ---------------------------------------------------------------------------
# Config knobs (config.yaml ``platforms.feishu.extra`` first, env fallback —
# the exact precedence the fork's old in-tree FeishuAdapterSettings used).
# ---------------------------------------------------------------------------

def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    try:
        s = str(value).strip().lower()
    except Exception:
        return default
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any, default: int, min_value: int) -> int:
    try:
        n = int(str(value).strip())
    except Exception:
        return default
    return max(n, min_value)


def _load_overlay_settings(adapter: Any) -> None:
    """Populate the overlay's per-instance knobs on the adapter.

    Reads ``platforms.feishu.extra`` (the same dict upstream's
    ``FeishuAdapterSettings.from_config`` parses) with env-var fallback:

    * ``ws_self_reconnect`` / ``FEISHU_WS_SELF_RECONNECT`` — default True
      (hc-384 revert lever).
    * ``heartbeat_enabled`` / ``FEISHU_HEARTBEAT`` — default False (hc-385 is
      opt-in).
    * ``heartbeat_interval_seconds`` (min 10) and
      ``heartbeat_initial_delay_seconds`` (min 5).

    hc-493 note — heartbeat_enabled overlaps a SEPARATE mechanism: the generic
    gateway-level "still working" notifier (``gateway/run.py``'s
    ``_notify_long_running``, gated by the unrelated
    ``display.platforms.feishu.long_running_notifications`` — Hermes Cloud's
    provisioned config defaults this ON for every Feishu agent). That loop is
    scheduled from inside the same message-processing call chain this
    overlay's ``on_processing_start`` hook wraps, so the two run concurrently
    on their own independent timers/wording. Turning hc-385's heartbeat on
    WITHOUT also turning the generic notifier off for this platform produces
    two overlapping "still running" messages per long task, not one — see the
    warning below.
    """
    extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
    adapter._ws_self_reconnect = _to_bool(
        extra.get("ws_self_reconnect", os.getenv("FEISHU_WS_SELF_RECONNECT", "true")),
        default=True,
    )
    adapter._heartbeat_enabled = _to_bool(
        extra.get("heartbeat_enabled", os.getenv("FEISHU_HEARTBEAT", "false")),
        default=False,
    )
    if adapter._heartbeat_enabled:
        logger.warning(
            "[Feishu] apex-overlay hc-385 heartbeat is enabled. This fires "
            "independently of the generic gateway 'long_running_notifications' "
            "notifier, which Hermes Cloud provisions ON by default for Feishu "
            "(display.platforms.feishu.long_running_notifications). If that "
            "setting is also on, users will see two separate 'still working' "
            "messages per long task. Set it to false for this agent if only "
            "the Feishu-native heartbeat is wanted (hc-493)."
        )
    adapter._heartbeat_interval = _to_int(
        extra.get("heartbeat_interval_seconds", os.getenv("FEISHU_HEARTBEAT_INTERVAL_SECONDS", 60)),
        default=60, min_value=10,
    )
    adapter._heartbeat_initial_delay = _to_int(
        extra.get("heartbeat_initial_delay_seconds", os.getenv("FEISHU_HEARTBEAT_INITIAL_DELAY_SECONDS", 60)),
        default=60, min_value=5,
    )


def _disable_sdk_auto_reconnect(adapter: Any) -> None:
    """Turn off the lark SDK's own retry on the CURRENT ws client.

    Called only while the supervisor owns reconnection. Unlike upstream's
    ``_disable_websocket_auto_reconnect`` (a disconnect-path helper that also
    drops the ``_ws_client`` reference), this keeps the client referenced so
    the supervisor can watch ``_conn`` for the death signal.
    """
    client = getattr(adapter, "_ws_client", None)
    if client is None:
        return
    try:
        setattr(client, "_auto_reconnect", False)
    except Exception:
        logger.debug("[Feishu] apex-overlay: could not disable SDK auto-reconnect", exc_info=True)


# ---------------------------------------------------------------------------
# Supervisor lifecycle (bound onto the adapter class).
#
# These take ``self`` explicitly because they become unbound functions
# assigned as class attributes.
# ---------------------------------------------------------------------------

def _start_ws_supervisor(self) -> None:
    """Start the background task that watches the websocket and reconnects.

    No-op in webhook mode or when self-reconnect is disabled (revert flag).
    """
    if getattr(self, "_connection_mode", None) != "websocket":
        return
    if not getattr(self, "_ws_self_reconnect", True):
        return
    loop = getattr(self, "_loop", None)
    if loop is None or loop.is_closed():
        return
    # A real asyncio loop always has create_task; guarding lets duck-typed
    # loop doubles in tests connect without a supervisor.
    create_task = getattr(loop, "create_task", None)
    if create_task is None:
        return
    existing = getattr(self, "_ws_supervisor_task", None)
    if existing is not None and not existing.done():
        return
    self._ws_supervisor_task = create_task(self._supervise_websocket())


async def _cancel_ws_supervisor(self) -> None:
    task = getattr(self, "_ws_supervisor_task", None)
    self._ws_supervisor_task = None
    if task is None or task.done():
        return
    if task is asyncio.current_task():
        # Reached from inside the supervisor itself: the ladder escalated to
        # a fatal error, which drives disconnect(). Don't await ourselves.
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ---------------------------------------------------------------------------
# Reconnect ladder + liveness probe (pure additions bound onto the class).
# ---------------------------------------------------------------------------

async def _supervise_websocket(self) -> None:
    """Periodically verify the websocket is alive; reconnect when it dies.

    With the SDK's auto-reconnect disabled, a silent drop makes the SDK's
    receive loop clear ``_conn`` and stop — so ``_conn is None`` is the death
    signal.
    """
    try:
        while self._running and not getattr(self, "_intentional_disconnect", False):
            await asyncio.sleep(_FEISHU_WS_HEALTH_CHECK_INTERVAL)
            if not self._running or getattr(self, "_intentional_disconnect", False):
                return
            if getattr(self, "_ws_reconnecting", False):
                continue
            if self._websocket_appears_dead():
                logger.warning(
                    "[Feishu] Websocket appears dead (no live connection); "
                    "starting reconnect ladder"
                )
                await self._reconnect_websocket_with_backoff()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("[Feishu] Websocket supervisor error", exc_info=True)


def _websocket_appears_dead(self) -> bool:
    client = getattr(self, "_ws_client", None)
    if client is None:
        return True
    future = getattr(self, "_ws_future", None)
    if future is not None and future.done():
        return True
    # lark's _disconnect() sets _conn to None when the receive loop exits.
    return getattr(client, "_conn", None) is None


async def _teardown_ws_thread(self) -> None:
    """Stop the running websocket client thread and its event loop.

    Used by the reconnect ladder, which rebuilds the socket without releasing
    the app lock or marking the adapter disconnected (upstream's ``disconnect``
    keeps its own inline teardown for the intentional-shutdown path). Safe to
    call when no thread runs.

    This also unwedges the zombie thread the lark SDK leaves behind on a
    silent drop: ``Client.start`` blocks forever on an internal keep-alive
    loop even after the receive loop has exited, so stopping the thread's
    event loop is what lets ``start`` return and the thread die before we
    relaunch.
    """
    client = getattr(self, "_ws_client", None)
    if client is not None:
        try:
            setattr(client, "_auto_reconnect", False)
        except Exception:
            pass
    self._ws_client = None

    ws_thread_loop = getattr(self, "_ws_thread_loop", None)
    if ws_thread_loop is not None and not ws_thread_loop.is_closed():
        logger.debug("[Feishu] Cancelling websocket thread tasks and stopping loop")

        def cancel_all_tasks() -> None:
            tasks = [t for t in asyncio.all_tasks(ws_thread_loop) if not t.done()]
            logger.debug("[Feishu] Found %d pending tasks in websocket thread", len(tasks))
            for task in tasks:
                task.cancel()
            ws_thread_loop.call_later(0.1, ws_thread_loop.stop)

        try:
            ws_thread_loop.call_soon_threadsafe(cancel_all_tasks)
        except RuntimeError:
            # Loop was already stopped/closed between the guard and here.
            pass

    ws_future = getattr(self, "_ws_future", None)
    if ws_future is not None:
        try:
            logger.debug("[Feishu] Waiting for websocket thread to exit (timeout=10s)")
            await asyncio.wait_for(asyncio.shield(ws_future), timeout=10.0)
            logger.debug("[Feishu] Websocket thread exited cleanly")
        except asyncio.TimeoutError:
            logger.warning("[Feishu] Websocket thread did not exit within 10s - may be stuck")
        except asyncio.CancelledError:
            logger.debug("[Feishu] Websocket thread cancelled during teardown")
        except Exception as exc:
            logger.debug("[Feishu] Websocket thread exited with error: %s", exc, exc_info=True)

    self._ws_future = None
    self._ws_thread_loop = None


async def _reconnect_websocket_with_backoff(self) -> None:
    """Exponential-backoff reconnect ladder (mirrors Telegram's pattern).

    Tears the dead socket down (without releasing the app lock), relaunches
    it via upstream's ``_connect_websocket``, re-disables the SDK's retry on
    the fresh client, and verifies liveness via the ``/bot/v3/info`` probe.
    Once the ladder is exhausted, escalate to a retryable fatal error so the
    gateway's reconnect watcher recreates the adapter — a full reconnect that
    re-acquires the app lock and resumes auto-recovery from there.
    """
    if getattr(self, "_ws_reconnecting", False):
        return
    self._ws_reconnecting = True
    try:
        for attempt in range(1, _FEISHU_WS_RECONNECT_MAX_ATTEMPTS + 1):
            if not self._running or getattr(self, "_intentional_disconnect", False):
                return
            delay = min(
                _FEISHU_WS_RECONNECT_BASE_DELAY * (2 ** (attempt - 1)),
                _FEISHU_WS_RECONNECT_MAX_DELAY,
            )
            logger.warning(
                "[Feishu] Websocket reconnect attempt %d/%d in %ds",
                attempt, _FEISHU_WS_RECONNECT_MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)
            if not self._running or getattr(self, "_intentional_disconnect", False):
                return
            try:
                await self._teardown_ws_thread()
                await asyncio.wait_for(
                    self._connect_websocket(), _FEISHU_WS_RECONNECT_CONNECT_TIMEOUT
                )
                _disable_sdk_auto_reconnect(self)
            except Exception as exc:
                logger.warning(
                    "[Feishu] Websocket reconnect attempt %d relaunch failed: %s",
                    attempt, exc,
                )
                continue
            if await self._verify_ws_alive():
                logger.info(
                    "[Feishu] Websocket reconnected and verified on attempt %d", attempt
                )
                return
            logger.warning(
                "[Feishu] Websocket reconnect attempt %d connected but failed "
                "liveness probe", attempt,
            )
        message = (
            "Feishu websocket could not reconnect after "
            f"{_FEISHU_WS_RECONNECT_MAX_ATTEMPTS} attempts"
        )
        logger.error("[Feishu] %s; escalating for gateway restart", message)
        self._set_fatal_error("feishu_ws_reconnect_exhausted", message, retryable=True)
        await self._notify_fatal_error()
    finally:
        self._ws_reconnecting = False


async def _verify_ws_alive(self) -> bool:
    """Confirm a freshly relaunched websocket actually works.

    Guards against "logs say reconnected but it's dead": a new socket object
    can exist while the bot endpoint is unreachable. Probes ``/bot/v3/info``
    (a real HTTP round-trip) through upstream's adapter-owned executor.

    Upstream v0.18's ``_hydrate_bot_identity`` swallows errors and returns
    ``None`` (it is a metadata hydrator, not a health check), so the probe is
    made here using the lark request symbols from the adapter's own module —
    resolved via ``type(self).__module__`` so the synthetic plugin module name
    (``hermes_plugins.<slug>.adapter``) never needs to be hardcoded.
    """
    await asyncio.sleep(_FEISHU_WS_RECONNECT_VERIFY_DELAY)
    if not self._running or getattr(self, "_intentional_disconnect", False):
        return False
    client = getattr(self, "_ws_client", None)
    if client is None or getattr(client, "_conn", None) is None:
        return False
    api_client = getattr(self, "_client", None)
    if api_client is None:
        return False
    mod = sys.modules.get(type(self).__module__)
    base_request = getattr(mod, "BaseRequest", None)
    http_method = getattr(mod, "HttpMethod", None)
    token_type = getattr(mod, "AccessTokenType", None)
    if base_request is None or http_method is None or token_type is None:
        # lark symbols unavailable (shouldn't happen on a connected adapter);
        # fall back to the socket-object check that already passed above.
        return True
    try:
        req = (
            base_request.builder()
            .http_method(http_method.GET)
            .uri("/open-apis/bot/v3/info")
            .token_types({token_type.TENANT})
            .build()
        )
        resp = await asyncio.wait_for(
            self._run_blocking(api_client.request, req), _FEISHU_WS_PROBE_TIMEOUT
        )
        content = getattr(getattr(resp, "raw", None), "content", None)
        return bool(content)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# hc-385 — long-task heartbeat (opt-in; bound onto the class).
# ---------------------------------------------------------------------------

def _maybe_start_heartbeat(self, event) -> None:
    if not getattr(self, "_heartbeat_enabled", False) or getattr(self, "_intentional_disconnect", False):
        return
    message_id = getattr(event, "message_id", None)
    chat_id = getattr(getattr(event, "source", None), "chat_id", None)
    if not message_id or not chat_id:
        return
    loop = getattr(self, "_loop", None)
    if loop is None or loop.is_closed():
        return
    create_task = getattr(loop, "create_task", None)
    if create_task is None:
        return
    if not hasattr(self, "_heartbeat_tasks"):
        self._heartbeat_tasks = {}
    existing = self._heartbeat_tasks.get(message_id)
    if existing is not None and not existing.done():
        return
    self._heartbeat_tasks[message_id] = create_task(
        self._run_heartbeat(chat_id, message_id)
    )


def _stop_heartbeat(self, message_id: Optional[str]) -> None:
    if not message_id:
        return
    tasks: Dict[str, asyncio.Task] = getattr(self, "_heartbeat_tasks", None) or {}
    task = tasks.get(message_id)
    if task is not None and not task.done() and task is not asyncio.current_task():
        task.cancel()


async def _cancel_all_heartbeats(self) -> None:
    tasks: Dict[str, asyncio.Task] = getattr(self, "_heartbeat_tasks", None) or {}
    current = asyncio.current_task()
    pending = [t for t in tasks.values() if t and not t.done() and t is not current]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()


def _heartbeat_text(self, started_at: float, *, final: bool) -> str:
    minutes = max(1, int((time.monotonic() - started_at) // 60))
    template = (
        _FEISHU_HEARTBEAT_FINAL_TEMPLATE if final else _FEISHU_HEARTBEAT_RUNNING_TEMPLATE
    )
    return template.format(minutes=minutes)


async def _run_heartbeat(self, chat_id: str, source_message_id: str) -> None:
    """Send and periodically edit one "still running" status message.

    Stays silent until ``_heartbeat_initial_delay`` so short tasks never see
    a heartbeat — this preserves Feishu's deliberately minimal feel. Runs as
    its own task and only edits an already-sent message, so it neither blocks
    the event loop nor resets the agent's inactivity clock (that clock is
    driven by agent tool activity in the gateway, not by adapter-side sends).
    """
    started_at = time.monotonic()
    heartbeat_message_id: Optional[str] = None
    try:
        await asyncio.sleep(getattr(self, "_heartbeat_initial_delay", 60))
        while self._running and not getattr(self, "_intentional_disconnect", False):
            text = self._heartbeat_text(started_at, final=False)
            if heartbeat_message_id is None:
                result = await self.send(chat_id, text, reply_to=source_message_id)
                if result and result.success and result.message_id:
                    heartbeat_message_id = result.message_id
            else:
                await self.edit_message(chat_id, heartbeat_message_id, text)
            await asyncio.sleep(getattr(self, "_heartbeat_interval", 60))
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("[Feishu] Heartbeat task error", exc_info=True)
    finally:
        tasks = getattr(self, "_heartbeat_tasks", None)
        if isinstance(tasks, dict):
            tasks.pop(source_message_id, None)
        # Leave one accurate final line so the channel doesn't keep showing
        # "still running" after the turn ends — only if we actually posted.
        if heartbeat_message_id is not None:
            try:
                await self.edit_message(
                    chat_id,
                    heartbeat_message_id,
                    self._heartbeat_text(started_at, final=True),
                )
            except Exception:
                logger.debug("[Feishu] Heartbeat finalize edit failed", exc_info=True)


# ---------------------------------------------------------------------------
# Method wrappers installed by the class instrumentation.
# ---------------------------------------------------------------------------

def _wrap_connect(orig):
    @functools.wraps(orig)
    async def connect(self, *args, **kwargs):
        ok = await orig(self, *args, **kwargs)
        if not ok:
            return ok
        try:
            self._intentional_disconnect = False
            self._ws_reconnecting = False
            if not hasattr(self, "_heartbeat_tasks"):
                self._heartbeat_tasks = {}
            if not hasattr(self, "_ws_supervisor_task"):
                self._ws_supervisor_task = None
            _load_overlay_settings(self)
            if (
                getattr(self, "_connection_mode", None) == "websocket"
                and self._ws_self_reconnect
            ):
                # Supervisor owns reconnection: turn off the SDK's broken
                # single-shot retry so a silent drop leaves a detectable dead
                # socket (_conn is None) instead of a half-alive zombie.
                _disable_sdk_auto_reconnect(self)
                self._start_ws_supervisor()
                logger.info("[Feishu] apex-overlay self-reconnect supervisor armed")
        except Exception:
            # The overlay must never turn a successful connect into a failure.
            logger.warning(
                "[Feishu] apex-overlay supervisor arming failed; adapter runs "
                "with the lark SDK's own reconnect", exc_info=True,
            )
        return ok

    setattr(connect, _MARK, True)
    return connect


def _wrap_disconnect(orig):
    @functools.wraps(orig)
    async def disconnect(self, *args, **kwargs):
        try:
            self._intentional_disconnect = True
            await self._cancel_ws_supervisor()
            await self._cancel_all_heartbeats()
        except Exception:
            logger.debug("[Feishu] apex-overlay pre-disconnect cleanup failed", exc_info=True)
        return await orig(self, *args, **kwargs)

    setattr(disconnect, _MARK, True)
    return disconnect


def _wrap_on_processing_start(orig):
    @functools.wraps(orig)
    async def on_processing_start(self, event) -> None:
        await orig(self, event)
        # hc-385: long-task heartbeat (opt-in, independent of reactions).
        try:
            self._maybe_start_heartbeat(event)
        except Exception:
            logger.debug("[Feishu] Heartbeat start failed", exc_info=True)

    setattr(on_processing_start, _MARK, True)
    return on_processing_start


def _wrap_on_processing_complete(orig):
    @functools.wraps(orig)
    async def on_processing_complete(self, event, outcome) -> None:
        # hc-385: stop the heartbeat regardless of reaction settings.
        try:
            self._stop_heartbeat(getattr(event, "message_id", None))
        except Exception:
            logger.debug("[Feishu] Heartbeat stop failed", exc_info=True)
        await orig(self, event, outcome)

    setattr(on_processing_complete, _MARK, True)
    return on_processing_complete


# ---------------------------------------------------------------------------
# Class instrumentation + registry interception
# ---------------------------------------------------------------------------

def _instrument_feishu_adapter_class(adapter_cls) -> bool:
    """Instrument the (lazily imported) FeishuAdapter class. Idempotent.

    Wraps the four lifecycle methods and binds the supervisor/heartbeat
    helpers. Returns False (and leaves the class stock) if any expected
    upstream method is missing — never raises.
    """
    if getattr(adapter_cls, _ACTIVE_FLAG, False):
        return True

    for name in _WRAPPED_METHODS:
        if getattr(adapter_cls, name, None) is None:
            logger.error(
                "apex_overlay: %s.%s is missing — upstream reshaped the Feishu "
                "adapter; hc-384 self-reconnect + hc-385 heartbeat are NOT active.",
                adapter_cls.__name__, name,
            )
            return False
    # The ladder rebuilds the socket through these upstream internals.
    for name in ("_connect_websocket", "_run_blocking", "_set_fatal_error", "_notify_fatal_error"):
        if getattr(adapter_cls, name, None) is None:
            logger.error(
                "apex_overlay: %s.%s is missing — the reconnect ladder has no "
                "relaunch primitive; hc-384 self-reconnect is NOT active.",
                adapter_cls.__name__, name,
            )
            return False

    adapter_cls.connect = _wrap_connect(adapter_cls.connect)
    adapter_cls.disconnect = _wrap_disconnect(adapter_cls.disconnect)
    adapter_cls.on_processing_start = _wrap_on_processing_start(adapter_cls.on_processing_start)
    adapter_cls.on_processing_complete = _wrap_on_processing_complete(adapter_cls.on_processing_complete)

    adapter_cls._start_ws_supervisor = _start_ws_supervisor
    adapter_cls._cancel_ws_supervisor = _cancel_ws_supervisor
    adapter_cls._supervise_websocket = _supervise_websocket
    adapter_cls._websocket_appears_dead = _websocket_appears_dead
    adapter_cls._reconnect_websocket_with_backoff = _reconnect_websocket_with_backoff
    adapter_cls._verify_ws_alive = _verify_ws_alive
    adapter_cls._teardown_ws_thread = _teardown_ws_thread
    adapter_cls._maybe_start_heartbeat = _maybe_start_heartbeat
    adapter_cls._stop_heartbeat = _stop_heartbeat
    adapter_cls._cancel_all_heartbeats = _cancel_all_heartbeats
    adapter_cls._heartbeat_text = _heartbeat_text
    adapter_cls._run_heartbeat = _run_heartbeat

    # hc-385: Feishu's create+connect is deferred off the gateway-ready path
    # (see apex_overlay.gateway_bootstrap). The class-level flag additionally
    # marks the adapter background-connectable for any path that creates it
    # directly (e.g. the reconnect watcher).
    adapter_cls.CONNECT_IN_BACKGROUND = True

    setattr(adapter_cls, _ACTIVE_FLAG, True)
    setattr(adapter_cls, _MARK, True)
    logger.debug("apex_overlay: hc-384/385 Feishu adapter class instrumented")
    return True


def _wrap_create_adapter(orig):
    @functools.wraps(orig)
    def create_adapter(self, name, config):
        adapter = orig(self, name, config)
        if adapter is not None and name == _FEISHU_PLATFORM_NAME:
            try:
                _instrument_feishu_adapter_class(type(adapter))
            except Exception:
                logger.warning(
                    "apex_overlay: Feishu adapter instrumentation failed; the "
                    "adapter runs with stock upstream reconnect behavior",
                    exc_info=True,
                )
        return adapter

    setattr(create_adapter, _MARK, True)
    return create_adapter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Install the hc-384/385 Feishu seam at the platform-registry choke point.

    Wraps ``PlatformRegistry.create_adapter`` so every Feishu adapter instance
    (initial startup, cron delivery, reconnect-watcher recreation) comes out
    with the self-reconnect supervisor + heartbeat instrumentation, without
    ever importing the lazily-loaded Feishu plugin module early (which would
    reintroduce the multi-second lark_oapi import upstream deferred on
    purpose).

    Idempotent and fail-safe: returns ``True`` if applied (or already present),
    ``False`` if the registry class/method is missing (the seam-test turns that
    into a hard CI failure; at runtime platform creation keeps working with
    stock upstream behavior and the plugin warns). Never raises.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    try:
        reg_mod = importlib.import_module(_TARGET_REGISTRY_MODULE)
        registry_cls = getattr(reg_mod, _TARGET_REGISTRY_CLS)
        orig = getattr(registry_cls, _TARGET_FACTORY_METHOD)
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not load %s.%s.%s — hc-384 Feishu "
            "self-reconnect + hc-385 heartbeat are NOT active (the lark SDK's "
            "reconnect stays in charge). Upstream may have renamed/moved it. (%s)",
            _TARGET_REGISTRY_MODULE, _TARGET_REGISTRY_CLS, _TARGET_FACTORY_METHOD, exc,
        )
        return False

    if getattr(orig, _MARK, False):
        _APPLIED = True
        return True

    setattr(registry_cls, _TARGET_FACTORY_METHOD, _wrap_create_adapter(orig))

    _APPLIED = True
    logger.debug("apex_overlay: hc-384/385 Feishu registry seam applied")
    return True
