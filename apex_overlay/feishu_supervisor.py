"""hc-384 Feishu self-reconnect supervisor — applied as a zero-in-place seam.

What this replaces
==================
The lark SDK's built-in websocket auto-reconnect was observed in prod to try
exactly once and give up (0 successful reconnects across ~188 daily drops),
leaving Feishu bots dead for 5–15h until the next gateway restart (hc-384).
The fix: the adapter **owns** reconnection — disable the SDK's single-shot
retry, watch the socket from a supervisor task, rebuild it on a backoff ladder,
verify liveness with a ``/bot/v3/info`` probe, and escalate to a *retryable*
fatal error once the ladder is exhausted so the gateway's reconnect watcher
recreates the adapter (re-acquiring the app lock). This mirrors the Telegram
resilience pattern (``telegram.py`` ``_handle_polling_network_error`` /
``_verify_polling_after_reconnect``); measured ~39s self-heal in prod.

The original fix lived as in-place lines inside ``gateway/platforms/feishu.py``
(+447 over upstream for hc-384/385 combined). This module moves the
**reconnect/supervisor** half into ``apex_overlay/`` so the hot upstream file
stays as close to upstream as possible. The **long-task heartbeat** (hc-385)
deliberately stays inline in ``feishu.py`` (it is product-surface behavior, not
resilience plumbing — see ``OVERLAY-SEAM-AUDIT.md`` Tier1 #2).

How the seam works
==================
``feishu.py`` keeps two lifecycle entry points as **upstream-faithful no-op
stubs**: ``FeishuAdapter._start_ws_supervisor`` (called at the end of
``connect()``) and ``FeishuAdapter._cancel_ws_supervisor`` (awaited in
``disconnect()``). With the stubs and nothing else, the adapter behaves exactly
like stock Hermes: no supervisor task, and the SDK keeps its own auto-reconnect
(the inline ``_apply_feishu_ws_runtime_overrides`` only disables the SDK retry
when this overlay has marked the class active — see ``apex_overlay_active``).

``apply()`` swaps those two stubs for the real supervisor implementations and
binds the four helper methods the ladder needs
(``_supervise_websocket``, ``_websocket_appears_dead``,
``_reconnect_websocket_with_backoff``, ``_verify_ws_alive``). It also sets a
class marker so the inline override knows to disable the SDK's retry.

That gives the cleanest seam tier (monkey-patch, zero behavioral debt in the
hot file): ``connect()``/``disconnect()`` keep calling two method names; whether
those start/cancel a real reconnect supervisor is decided entirely here. The
matching seam-test (``tests/apex_overlay/test_feishu_supervisor_seam.py``) pins
the two swapped methods plus every upstream attribute the ladder depends on, so
an upstream rename/move turns a *silently reverted-to-SDK-reconnect* adapter
into a *loud CI failure*.

``apply()`` is idempotent and fail-safe: if the target class/methods are missing
it logs an error and returns ``False`` (the plugin warns; the adapter keeps
running on the upstream-faithful stubs + SDK reconnect) but never raises.

The ``ws_self_reconnect`` revert flag (env ``FEISHU_WS_SELF_RECONNECT=false``)
still works after this overlay applies: the real ``_start_ws_supervisor``
returns early when it is False, and ``_apply_feishu_ws_runtime_overrides`` leaves
the SDK's retry on — so a revert is pure upstream behavior even with the overlay
loaded.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Upstream/overlay symbols this seam binds against. Centralized so the seam-test
# can assert they still exist with a compatible shape. If upstream renames/moves
# the adapter or the lifecycle entry points, both the patch AND the seam-test
# break loudly.
_TARGET_FEISHU_MODULE = "gateway.platforms.feishu"
_TARGET_ADAPTER_CLS = "FeishuAdapter"

# The two lifecycle entry points feishu.py keeps as upstream-faithful no-op
# stubs and this overlay swaps for the real implementations. start() and
# disconnect() call these by name.
_LIFECYCLE_METHODS = (
    "_start_ws_supervisor",
    "_cancel_ws_supervisor",
)

# The reconnect-ladder helpers. Pure additions (upstream has none); only ever
# called from the supervisor/ladder itself, so they need no in-tree stub — but
# the seam-test and the hc-384 behavior tests call them on an adapter instance
# after apply(), so apply() must bind them.
_LADDER_METHODS = (
    "_supervise_websocket",
    "_websocket_appears_dead",
    "_reconnect_websocket_with_backoff",
    "_verify_ws_alive",
)

# ---------------------------------------------------------------------------
# Reconnect tuning. These lived in feishu.py as module constants; moved here so
# the hot file sheds them. The hc-384 behavior tests monkeypatch these on THIS
# module (``apex_overlay.feishu_supervisor``) to make the ladder fast.
# ---------------------------------------------------------------------------
_FEISHU_WS_HEALTH_CHECK_INTERVAL = 45        # seconds between liveness checks
_FEISHU_WS_RECONNECT_MAX_ATTEMPTS = 6        # in-adapter ladder length before escalating to a gateway restart
_FEISHU_WS_RECONNECT_BASE_DELAY = 5          # seconds (5 → 10 → 20 → 40 → 60 → 60)
_FEISHU_WS_RECONNECT_MAX_DELAY = 60          # seconds cap, matches Telegram's ladder
_FEISHU_WS_RECONNECT_VERIFY_DELAY = 5        # seconds to let a relaunched socket establish before probing
_FEISHU_WS_RECONNECT_CONNECT_TIMEOUT = 30    # seconds bound on a single relaunch (guards against a hung hydrate)
_FEISHU_WS_PROBE_TIMEOUT = 10                # seconds for the /bot/v3/info liveness probe

# Guard so apply() is idempotent even if called from multiple boot paths.
_APPLIED = False
_MARK = "_apex_overlay_feishu_supervisor"

# Class-level marker the inline _apply_feishu_ws_runtime_overrides reads to know
# this overlay is active (so it disables the SDK's broken auto-reconnect only
# when the supervisor is actually installed; otherwise behavior stays upstream).
_ACTIVE_FLAG = "_apex_ws_supervisor_active"


def apex_overlay_active(adapter) -> bool:
    """True when this overlay is installed for the adapter's class.

    ``feishu.py``'s inline ``_apply_feishu_ws_runtime_overrides`` calls this to
    decide whether to disable the lark SDK's auto-reconnect. Without the overlay
    the answer is False, so the SDK keeps its own (upstream) reconnect and the
    adapter never ends up with *no* reconnection at all.
    """
    return bool(getattr(type(adapter), _ACTIVE_FLAG, False))


# ---------------------------------------------------------------------------
# Lifecycle entry points (swap the feishu.py upstream-faithful no-op stubs).
#
# These take ``self`` explicitly because they become unbound functions assigned
# as class attributes.
# ---------------------------------------------------------------------------

def _start_ws_supervisor(self) -> None:
    """Start the background task that watches the websocket and reconnects.

    No-op in webhook mode or when self-reconnect is disabled (revert flag).
    """
    if self._connection_mode != "websocket" or not self._ws_self_reconnect:
        return
    loop = self._loop
    if loop is None or loop.is_closed():
        return
    # A real asyncio loop always has create_task; guarding lets duck-typed
    # loop doubles in tests connect without a supervisor.
    create_task = getattr(loop, "create_task", None)
    if create_task is None:
        return
    existing = self._ws_supervisor_task
    if existing is not None and not existing.done():
        return
    self._ws_supervisor_task = create_task(self._supervise_websocket())


async def _cancel_ws_supervisor(self) -> None:
    task = self._ws_supervisor_task
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

    On a silent drop the SDK's receive loop clears ``_conn`` and stops
    (auto-reconnect disabled), so ``_conn is None`` is the death signal.
    """
    try:
        while self._running and not self._intentional_disconnect:
            await asyncio.sleep(_FEISHU_WS_HEALTH_CHECK_INTERVAL)
            if not self._running or self._intentional_disconnect:
                return
            if self._ws_reconnecting:
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
    client = self._ws_client
    if client is None:
        return True
    future = self._ws_future
    if future is not None and future.done():
        return True
    # lark's _disconnect() sets _conn to None when the receive loop exits.
    return getattr(client, "_conn", None) is None


async def _reconnect_websocket_with_backoff(self) -> None:
    """Exponential-backoff reconnect ladder (mirrors Telegram's pattern).

    Tears the dead socket down (without releasing the app lock), relaunches
    it, and verifies liveness via the ``/bot/v3/info`` probe. Once the ladder
    is exhausted, escalate to a retryable fatal error so the gateway's
    reconnect watcher recreates the adapter — a full reconnect that
    re-acquires the app lock and resumes auto-recovery from there.
    """
    if self._ws_reconnecting:
        return
    self._ws_reconnecting = True
    try:
        for attempt in range(1, _FEISHU_WS_RECONNECT_MAX_ATTEMPTS + 1):
            if not self._running or self._intentional_disconnect:
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
            if not self._running or self._intentional_disconnect:
                return
            try:
                await self._teardown_ws_thread()
                await asyncio.wait_for(
                    self._connect_websocket(), _FEISHU_WS_RECONNECT_CONNECT_TIMEOUT
                )
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
    can exist while the bot endpoint is unreachable. Reuses
    ``_hydrate_bot_identity`` (a real /bot/v3/info round-trip) as the probe.
    """
    await asyncio.sleep(_FEISHU_WS_RECONNECT_VERIFY_DELAY)
    if not self._running or self._intentional_disconnect:
        return False
    client = self._ws_client
    if client is None or getattr(client, "_conn", None) is None:
        return False
    try:
        return bool(
            await asyncio.wait_for(self._hydrate_bot_identity(), _FEISHU_WS_PROBE_TIMEOUT)
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Install the hc-384 Feishu self-reconnect supervisor onto FeishuAdapter.

    Swaps the two upstream-faithful no-op lifecycle stubs
    (``_start_ws_supervisor`` / ``_cancel_ws_supervisor``) for the real
    implementations, binds the four ladder helpers, and marks the class active
    so the inline websocket-override hook disables the lark SDK's broken
    auto-reconnect.

    Idempotent and fail-safe: returns ``True`` if applied (or already present),
    ``False`` if the target class or a lifecycle stub is missing (the seam-test
    turns that into a hard CI failure; at runtime the adapter keeps the no-op
    stubs and the lark SDK's own reconnect, and the plugin warns). Safe to call
    from any boot path; repeat calls are a no-op.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    try:
        feishu_mod = importlib.import_module(_TARGET_FEISHU_MODULE)
        adapter_cls = getattr(feishu_mod, _TARGET_ADAPTER_CLS)
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not load %s.%s — hc-384 Feishu self-reconnect "
            "is NOT active (the lark SDK's broken single-shot auto-reconnect "
            "stays in charge). Upstream may have renamed/moved it. (%s)",
            _TARGET_FEISHU_MODULE, _TARGET_ADAPTER_CLS, exc,
        )
        return False

    # The lifecycle stubs must already exist on the class (feishu.py provides
    # the upstream-faithful no-op versions we replace). If they're gone,
    # upstream removed our extraction points — refuse to silently change shape.
    for name in _LIFECYCLE_METHODS:
        if getattr(adapter_cls, name, None) is None:
            logger.error(
                "apex_overlay: %s.%s is missing — the Feishu supervisor seam "
                "has no extraction point to patch. hc-384 self-reconnect is "
                "NOT active.",
                _TARGET_ADAPTER_CLS, name,
            )
            return False

    if getattr(adapter_cls, _ACTIVE_FLAG, False):
        _APPLIED = True
        return True

    # Swap the lifecycle stubs and bind the ladder helpers. The supervisor
    # methods (pure additions) never shadow an upstream method.
    setattr(adapter_cls, "_start_ws_supervisor", _start_ws_supervisor)
    setattr(adapter_cls, "_cancel_ws_supervisor", _cancel_ws_supervisor)
    setattr(adapter_cls, "_supervise_websocket", _supervise_websocket)
    setattr(adapter_cls, "_websocket_appears_dead", _websocket_appears_dead)
    setattr(adapter_cls, "_reconnect_websocket_with_backoff", _reconnect_websocket_with_backoff)
    setattr(adapter_cls, "_verify_ws_alive", _verify_ws_alive)
    setattr(adapter_cls, _ACTIVE_FLAG, True)
    setattr(adapter_cls, _MARK, True)

    _APPLIED = True
    logger.debug("apex_overlay: hc-384 Feishu self-reconnect supervisor seam applied")
    return True
