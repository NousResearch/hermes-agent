"""hc-384/385 non-blocking platform startup — applied as a zero-in-place seam.

What this replaces
==================
Runtime containers expose the **API server** as the conversation-ready
surface. Feishu's adapter imports the official ``lark_oapi`` SDK at module
import time and its websocket attach is slow; on a cold start that import +
connect used to block the gateway's "ready" signal for seconds, and a single
flaky Feishu connect could wedge or restart-loop the whole gateway
(hc-384/385). The fix: let Feishu (and any adapter that advertises
``CONNECT_IN_BACKGROUND``) **create + connect in the background** so the API
conversation path declares ready first, with status tracking + retry queueing
so the reconnect watcher still heals a failed attach (measured ~39s self-heal
in prod).

The original fix lived as ``+276`` in-place lines inside
``gateway/run.py`` (the single hottest overlay file — 77 upstream commits
since our fork point). This module moves **all** of that behavior into
``apex_overlay/`` so the upstream file stays as close to upstream as possible,
and re-applies it at gateway boot by monkey-patching the runner.

How the seam works
==================
``gateway/run.py`` was refactored so the per-platform connect loop inside
``GatewayRunner.start()`` is a single method,
``_connect_configured_platforms()``, whose **in-tree body is upstream's
original sequential loop** (behavior-identical to stock Hermes — Feishu would
block ready). ``apply()`` monkey-patches that method with the background-connect
version below, and binds eight small helper methods the loop needs.

v0.18 contract update: upstream's loop can now abort startup mid-way when a
shutdown/restart request arrives (``_abort_startup_if_shutdown_requested``),
and wires a per-adapter authorization check (``set_authorization_check``).
Both are mirrored here; the method's return grew from 5 counters to a 6-tuple
``(aborted, connected, background, enabled, nonretryable, retryable)``.

That gives us the cleanest possible seam tier (monkey-patch, zero behavioral
debt left in the hot file): ``start()`` keeps calling one method; whether that
method blocks-or-backgrounds is decided entirely here. The matching seam-test
(``tests/apex_overlay/test_gateway_bootstrap_seam.py``) pins
``_connect_configured_platforms`` plus every upstream attribute the loop and
helpers depend on, so an upstream rename/move turns a *silently reverted-to-
blocking* gateway into a *loud CI failure*.

``apply()`` is idempotent and fail-safe: if the target method is missing it
logs an error and returns ``False`` (the plugin warns; the gateway keeps
running on upstream's blocking loop) but never raises.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Upstream symbols this seam binds against. Centralized so the seam-test can
# assert they still exist with a compatible shape. If upstream renames/moves
# any of these, both the patch AND the seam-test break loudly.
_TARGET_RUN_MODULE = "gateway.run"
_TARGET_RUNNER_CLS = "GatewayRunner"
_TARGET_LOOP_METHOD = "_connect_configured_platforms"

# Guard so apply() is idempotent even if called from multiple boot paths.
_APPLIED = False
_MARK = "_apex_overlay_gateway_bootstrap"

# The helper methods this seam installs onto GatewayRunner. The loop body
# (overlay _connect_configured_platforms) calls these by name; the seam-test
# asserts they are all present after apply().
_HELPER_NAMES = (
    "_adapter_connects_in_background",
    "_platform_creation_connects_in_background",
    "_prepare_adapter",
    "_register_connected_adapter",
    "_queue_platform_retry",
    "_track_background_task",
    "_connect_adapter_in_background",
    "_create_and_connect_adapter_in_background",
)


# ---------------------------------------------------------------------------
# Helper methods (bound onto GatewayRunner by apply()).
#
# These are new (upstream has none of them); they are pure additions, so
# binding them never shadows an upstream method. They take ``self`` explicitly
# because they become unbound functions assigned as class attributes.
# ---------------------------------------------------------------------------

def _adapter_connects_in_background(self, adapter: Any) -> bool:
    """Return True when an adapter should not block gateway ready.

    Runtime containers expose the API server as the conversation-ready
    surface. Platform websocket attachment can continue in the background
    as long as the API server path is already accepting turns.
    """
    return bool(getattr(adapter, "CONNECT_IN_BACKGROUND", False))


def _platform_creation_connects_in_background(self, platform: Any) -> bool:
    """Return True when adapter import/construction should also be deferred.

    Feishu's adapter module imports the official lark_oapi SDK at module
    import time. Even when baked into the image, that first import can take
    long enough to delay the API conversation-ready signal. Defer the whole
    create+connect path for known background-connect platforms so the API
    server can declare ready first.
    """
    from gateway.config import Platform

    return platform == Platform.FEISHU


def _prepare_adapter(self, adapter: Any) -> None:
    """Wire gateway callbacks into an adapter before connect.

    Mirrors upstream's inline wiring block, including the v0.18 fail-closed
    per-adapter authorization check (upstream bb304b4914).
    """
    adapter.set_message_handler(self._handle_message)
    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
    adapter.set_session_store(self.session_store)
    adapter.set_busy_session_handler(self._handle_active_session_busy_message)
    adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
    adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
    adapter._busy_text_mode = self._busy_text_mode


def _register_connected_adapter(self, platform, adapter) -> None:
    """Record a connected adapter and refresh dependent routers/state."""
    self.adapters[platform] = adapter
    self._sync_voice_mode_state_to_adapter(adapter)
    self.delivery_router.adapters = self.adapters
    self._wire_teams_pipeline_runtime()
    self._update_platform_runtime_status(
        platform.value,
        platform_state="connected",
        error_code=None,
        error_message=None,
    )


def _queue_platform_retry(
    self,
    platform,
    platform_config,
    *,
    attempts: int,
    error_message: str,
    error_code: Optional[str] = None,
    retryable: bool = True,
) -> None:
    """Update runtime status and queue a failed platform for reconnect."""
    self._update_platform_runtime_status(
        platform.value,
        platform_state="retrying" if retryable else "fatal",
        error_code=error_code,
        error_message=error_message,
    )
    if retryable:
        self._failed_platforms[platform] = {
            "config": platform_config,
            "attempts": attempts,
            "next_retry": time.monotonic() + 30,
        }


def _track_background_task(self, task: "asyncio.Task") -> None:
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)


async def _connect_adapter_in_background(
    self,
    adapter,
    platform,
    platform_config,
) -> None:
    """Connect a platform after the gateway's conversation path is ready."""
    try:
        connect_started = time.monotonic()
        success = await self._connect_adapter_with_timeout(adapter, platform)
        connect_elapsed_ms = int((time.monotonic() - connect_started) * 1000)
        if success:
            self._register_connected_adapter(platform, adapter)
            logger.info(
                "✓ %s connected in background in %dms",
                platform.value,
                connect_elapsed_ms,
            )
            try:
                from gateway.channel_directory import build_channel_directory
                directory_started = time.monotonic()
                await build_channel_directory(self.adapters)
                logger.info(
                    "%s background channel directory refresh finished in %dms",
                    platform.value,
                    int((time.monotonic() - directory_started) * 1000),
                )
            except Exception:
                logger.debug(
                    "%s background channel directory refresh failed",
                    platform.value,
                    exc_info=True,
                )
            try:
                self._schedule_resume_pending_sessions(platform=platform)
            except Exception:
                logger.debug(
                    "resume-pending reschedule after %s background connect failed",
                    platform.value,
                    exc_info=True,
                )
            return

        logger.warning(
            "✗ %s failed to connect in background after %dms",
            platform.value,
            connect_elapsed_ms,
        )
        await self._safe_adapter_disconnect(adapter, platform)
        if adapter.has_fatal_error:
            self._queue_platform_retry(
                platform,
                platform_config,
                attempts=1,
                error_code=adapter.fatal_error_code,
                error_message=adapter.fatal_error_message or "failed to connect",
                retryable=adapter.fatal_error_retryable,
            )
        else:
            self._queue_platform_retry(
                platform,
                platform_config,
                attempts=1,
                error_message="failed to connect",
            )
    except asyncio.CancelledError:
        await self._safe_adapter_disconnect(adapter, platform)
        raise
    except Exception as e:
        logger.error("✗ %s background connect error: %s", platform.value, e)
        await self._safe_adapter_disconnect(adapter, platform)
        self._queue_platform_retry(
            platform,
            platform_config,
            attempts=1,
            error_message=str(e),
        )


async def _create_and_connect_adapter_in_background(
    self,
    platform,
    platform_config,
) -> None:
    """Create and connect a platform after the API conversation path is ready."""
    create_started = time.monotonic()
    logger.info("Creating %s adapter in background...", platform.value)
    try:
        adapter = await asyncio.to_thread(self._create_adapter, platform, platform_config)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - create_started) * 1000)
        logger.error(
            "✗ %s background adapter creation failed after %dms: %s",
            platform.value,
            elapsed_ms,
            exc,
        )
        self._update_platform_runtime_status(
            platform.value,
            platform_state="fatal",
            error_code="adapter_create_failed",
            error_message=str(exc),
        )
        return

    elapsed_ms = int((time.monotonic() - create_started) * 1000)
    if not adapter:
        logger.warning(
            "No adapter available for %s after %dms background creation",
            platform.value,
            elapsed_ms,
        )
        self._update_platform_runtime_status(
            platform.value,
            platform_state="fatal",
            error_code="adapter_unavailable",
            error_message="adapter unavailable",
        )
        return

    logger.info("%s adapter created in background in %dms", platform.value, elapsed_ms)
    self._prepare_adapter(adapter)
    logger.info("Connecting to %s in background...", platform.value)
    await self._connect_adapter_in_background(adapter, platform, platform_config)


# ---------------------------------------------------------------------------
# The overlaid per-platform connect loop. This REPLACES the upstream-faithful
# body of GatewayRunner._connect_configured_platforms (which run.py keeps as
# stock Hermes' sequential, blocking loop). Same return contract: the 6-tuple
# start() unpacks.
# ---------------------------------------------------------------------------

async def _connect_configured_platforms(self):
    """Background-connect aware per-platform startup loop (hc-384/385).

    Returns ``(aborted, connected_count, background_connect_count,
    enabled_platform_count, startup_nonretryable_errors,
    startup_retryable_errors)`` — the exact tuple ``start()`` unpacks
    (``aborted`` mirrors upstream's v0.18 shutdown-during-startup early
    return). Feishu (and any ``CONNECT_IN_BACKGROUND`` adapter) is scheduled
    off the critical path so the API conversation surface declares ready
    first.
    """
    from gateway.config import Platform

    connected_count = 0
    background_connect_count = 0
    enabled_platform_count = 0
    startup_nonretryable_errors: list[str] = []
    startup_retryable_errors: list[str] = []

    def _result(aborted: bool):
        return (
            aborted,
            connected_count,
            background_connect_count,
            enabled_platform_count,
            startup_nonretryable_errors,
            startup_retryable_errors,
        )

    # Initialize and connect each configured platform
    for platform, platform_config in self.config.platforms.items():
        if await self._abort_startup_if_shutdown_requested():
            return _result(True)
        if not platform_config.enabled:
            continue
        enabled_platform_count += 1

        if self._platform_creation_connects_in_background(platform):
            logger.info(
                "Scheduling %s adapter creation/connect in background...",
                platform.value,
            )
            self._update_platform_runtime_status(
                platform.value,
                platform_state="connecting",
                error_code=None,
                error_message=None,
            )
            task = asyncio.create_task(
                self._create_and_connect_adapter_in_background(
                    platform,
                    platform_config,
                ),
                name=f"gateway-{platform.value}-background-create-connect",
            )
            self._track_background_task(task)
            background_connect_count += 1
            continue

        create_started = time.monotonic()
        adapter = self._create_adapter(platform, platform_config)
        create_elapsed_ms = int((time.monotonic() - create_started) * 1000)
        if not adapter:
            # Distinguish between missing builtin deps and missing plugin
            _pval = platform.value
            _builtin_names = {m.value for m in Platform.__members__.values()}
            if _pval not in _builtin_names:
                logger.warning(
                    "No adapter for '%s' — is the plugin installed? "
                    "(platform is enabled in config.yaml but no plugin registered it)",
                    _pval,
                )
            else:
                logger.warning("No adapter available for %s", _pval)
            continue
        logger.info("%s adapter created in %dms", platform.value, create_elapsed_ms)

        # Set up message + fatal error handlers
        self._prepare_adapter(adapter)

        # Try to connect. Adapters such as Feishu can attach in the
        # background so the API conversation path becomes ready first.
        background_connect = self._adapter_connects_in_background(adapter)
        logger.info(
            "Connecting to %s%s...",
            platform.value,
            " in background" if background_connect else "",
        )
        self._update_platform_runtime_status(
            platform.value,
            platform_state="connecting",
            error_code=None,
            error_message=None,
        )
        if background_connect:
            task = asyncio.create_task(
                self._connect_adapter_in_background(
                    adapter,
                    platform,
                    platform_config,
                ),
                name=f"gateway-{platform.value}-background-connect",
            )
            self._track_background_task(task)
            background_connect_count += 1
            continue
        try:
            success = await self._connect_adapter_with_timeout(adapter, platform)
            if await self._abort_startup_if_shutdown_requested(adapter, platform):
                return _result(True)
            if success:
                self._register_connected_adapter(platform, adapter)
                connected_count += 1
                logger.info("✓ %s connected", platform.value)
            else:
                logger.warning("✗ %s failed to connect", platform.value)
                # Defensive cleanup: a failed connect() may have
                # allocated resources (aiohttp.ClientSession, poll
                # tasks, bridge subprocesses) before giving up.
                # Without this call, those resources are orphaned
                # and Python logs "Unclosed client session" at
                # process exit. Adapter disconnect() implementations
                # are expected to be idempotent and tolerate
                # partial-init state.
                await self._safe_adapter_disconnect(adapter, platform)
                if adapter.has_fatal_error:
                    self._queue_platform_retry(
                        platform,
                        platform_config,
                        attempts=1,
                        error_code=adapter.fatal_error_code,
                        error_message=adapter.fatal_error_message or "failed to connect",
                        retryable=adapter.fatal_error_retryable,
                    )
                    target = (
                        startup_retryable_errors
                        if adapter.fatal_error_retryable
                        else startup_nonretryable_errors
                    )
                    target.append(
                        f"{platform.value}: {adapter.fatal_error_message}"
                    )
                else:
                    self._queue_platform_retry(
                        platform,
                        platform_config,
                        attempts=1,
                        error_message="failed to connect",
                    )
                    startup_retryable_errors.append(
                        f"{platform.value}: failed to connect"
                    )
        except Exception as e:
            logger.error("✗ %s error: %s", platform.value, e)
            # Same defensive cleanup path for exceptions — an adapter
            # that raised mid-connect may still have a live
            # aiohttp.ClientSession or child subprocess.
            await self._safe_adapter_disconnect(adapter, platform)
            self._queue_platform_retry(
                platform,
                platform_config,
                attempts=1,
                error_message=str(e),
            )
            startup_retryable_errors.append(f"{platform.value}: {e}")
        if await self._abort_startup_if_shutdown_requested():
            return _result(True)

    return _result(False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Install the hc-384/385 background-startup seam onto GatewayRunner.

    Binds the eight helper methods and swaps the per-platform connect loop for
    the background-connect version. Idempotent and fail-safe: returns ``True``
    if applied (or already present), ``False`` if the target class/method is
    missing (the seam-test turns that into a hard CI failure; at runtime the
    gateway falls back to upstream's blocking loop and the plugin warns). Safe
    to call from any boot path; repeat calls are a no-op.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    try:
        run_mod = importlib.import_module(_TARGET_RUN_MODULE)
        runner_cls = getattr(run_mod, _TARGET_RUNNER_CLS)
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not load %s.%s — hc-384/385 non-blocking "
            "platform startup is NOT active (Feishu will block gateway ready). "
            "Upstream may have renamed/moved it. (%s)",
            _TARGET_RUN_MODULE, _TARGET_RUNNER_CLS, exc,
        )
        return False

    # The loop method must already exist on the class (run.py provides the
    # upstream-faithful version we replace). If it's gone, upstream removed our
    # extraction point — refuse silently reverting to a different shape.
    orig_loop = getattr(runner_cls, _TARGET_LOOP_METHOD, None)
    if orig_loop is None:
        logger.error(
            "apex_overlay: %s.%s is missing — the gateway bootstrap seam has "
            "no extraction point to patch. hc-384/385 background startup is "
            "NOT active.",
            _TARGET_RUNNER_CLS, _TARGET_LOOP_METHOD,
        )
        return False

    if getattr(orig_loop, _MARK, False):
        _APPLIED = True
        return True

    # Bind the helper methods (pure additions; never shadow upstream).
    _connect_configured_platforms.__dict__[_MARK] = True
    setattr(runner_cls, _TARGET_LOOP_METHOD, _connect_configured_platforms)
    setattr(runner_cls, "_adapter_connects_in_background", _adapter_connects_in_background)
    setattr(runner_cls, "_platform_creation_connects_in_background", _platform_creation_connects_in_background)
    setattr(runner_cls, "_prepare_adapter", _prepare_adapter)
    setattr(runner_cls, "_register_connected_adapter", _register_connected_adapter)
    setattr(runner_cls, "_queue_platform_retry", _queue_platform_retry)
    setattr(runner_cls, "_track_background_task", _track_background_task)
    setattr(runner_cls, "_connect_adapter_in_background", _connect_adapter_in_background)
    setattr(runner_cls, "_create_and_connect_adapter_in_background", _create_and_connect_adapter_in_background)

    _APPLIED = True
    logger.debug("apex_overlay: hc-384/385 gateway background-startup seam applied")
    return True
