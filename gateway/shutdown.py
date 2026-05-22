"""Gateway shutdown and drain management — extracted from gateway/run.py.

Graceful stop, drain, and restart logic for the GatewayRunner.  These
functions take a ``runner`` parameter (the GatewayRunner instance)
instead of ``self``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_restart_drain_timeout() -> float:
    """Load graceful gateway restart/stop drain timeout in seconds."""
    from gateway.restart import (
        DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
        parse_restart_drain_timeout,
    )

    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        try:
            import yaml as _y

            from hermes_cli.config import _hermes_home, cfg_get

            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
        except Exception:
            pass
    value = parse_restart_drain_timeout(raw)
    if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
        try:
            float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid restart_drain_timeout '%s', using default %.0fs",
                raw,
                DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
            )
    return value


async def drain_active_agents(runner: Any, timeout: float) -> tuple[Dict[str, Any], bool]:
    snapshot = snapshot_running_agents(runner)
    last_active_count = running_agent_count(runner)
    last_status_at = 0.0

    def _maybe_update_status(force: bool = False) -> None:
        nonlocal last_active_count, last_status_at
        now = asyncio.get_running_loop().time()
        active_count = running_agent_count(runner)
        if force or active_count != last_active_count or (now - last_status_at) >= 1.0:
            runner._update_runtime_status("draining")
            last_active_count = active_count
            last_status_at = now

    if not runner._running_agents:
        _maybe_update_status(force=True)
        return snapshot, False

    _maybe_update_status(force=True)
    if timeout <= 0:
        return snapshot, True

    deadline = asyncio.get_running_loop().time() + timeout
    while runner._running_agents and asyncio.get_running_loop().time() < deadline:
        _maybe_update_status()
        await asyncio.sleep(0.1)
    timed_out = bool(runner._running_agents)
    _maybe_update_status(force=True)
    return snapshot, timed_out


async def notify_active_sessions_of_shutdown(runner: Any) -> None:
    """Send shutdown/restart notifications to active chats and home channels.

    Called at the very start of stop() — adapters are still connected so
    messages can be delivered. Best-effort: individual send failures are
    logged and swallowed so they never block the shutdown sequence.
    """
    from gateway.platforms.base import Platform

    active = snapshot_running_agents(runner)

    action = "restarting" if runner._restart_requested else "shutting down"
    hint = (
        "Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
        if runner._restart_requested
        else "Your current task will be interrupted."
    )
    msg = f"\u26a0\ufe0f Gateway {action} \u2014 {hint}"

    notified: set[tuple[str, str, Optional[str]]] = set()
    for session_key in active:
        source = None
        try:
            if getattr(runner, "session_store", None) is not None:
                runner.session_store._ensure_loaded()
                entry = runner.session_store._entries.get(session_key)
                source = getattr(entry, "origin", None) if entry else None
        except Exception as e:
            logger.debug(
                "Failed to load session origin for shutdown notification %s: %s",
                session_key,
                e,
            )

        if source is None:
            source = runner._get_cached_session_source(session_key)

        if source is not None:
            platform_str = source.platform.value
            chat_id = str(source.chat_id)
            thread_id = source.thread_id
        else:
            # Fall back to parsing the session key when no persisted
            # origin is available (legacy sessions/tests).
            from gateway.run import _parse_session_key

            _parsed = _parse_session_key(session_key)
            if not _parsed:
                continue
            platform_str = _parsed["platform"]
            chat_id = _parsed["chat_id"]
            thread_id = _parsed.get("thread_id")

        # Deduplicate only identical delivery targets. Thread/topic-aware
        # platforms can share a parent chat while still routing to distinct
        # destinations via metadata.
        dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
        if dedup_key in notified:
            continue

        try:
            platform = Platform(platform_str)
            adapter = runner.adapters.get(platform)
            if not adapter:
                continue

            platform_cfg = runner.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                    platform_str,
                )
                continue

            # Include thread_id if present so the message lands in the
            # correct forum topic / thread.
            metadata = {"thread_id": thread_id} if thread_id else None

            result = await adapter.send(chat_id, msg, metadata=metadata)
            if result is not None and getattr(result, "success", True) is False:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str,
                    chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                continue

            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to active chat %s:%s",
                platform_str, chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to %s:%s: %s",
                platform_str, chat_id, e,
            )

    # Snapshot adapters up front: adapter.send() can hit a fatal error
    # path that pops the adapter from self.adapters (see _handle_fatal
    # elsewhere), which would otherwise trigger
    # ``RuntimeError: dictionary changed size during iteration`` —
    # observed in a user report during gateway shutdown.
    for platform, adapter in list(runner.adapters.items()):
        home = runner.config.get_home_channel(platform)
        if not home or not home.chat_id:
            continue

        platform_cfg = runner.config.platforms.get(platform)
        if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
            logger.info(
                "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                platform.value,
            )
            continue

        dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
        if dedup_key in notified:
            continue

        try:
            metadata = {"thread_id": home.thread_id} if home.thread_id else None
            if metadata:
                result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
            else:
                result = await adapter.send(str(home.chat_id), msg)
            if result is not None and getattr(result, "success", True) is False:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                continue

            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to home channel %s:%s",
                platform.value,
                home.chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to home channel %s:%s: %s",
                platform.value,
                home.chat_id,
                e,
            )


def finalize_shutdown_agents(runner: Any, active_agents: Dict[str, Any]) -> None:
    for agent in active_agents.values():
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook

            _invoke_hook(
                "on_session_finalize",
                session_id=getattr(agent, "session_id", None),
                platform="gateway",
            )
        except Exception:
            pass
        cleanup_agent_resources(runner, agent)


def cleanup_agent_resources(runner: Any, agent: Any) -> None:
    """Best-effort cleanup for temporary or cached agent instances."""
    if agent is None:
        return
    try:
        if hasattr(agent, "shutdown_memory_provider"):
            session_messages = getattr(agent, "_session_messages", None)
            if isinstance(session_messages, list):
                agent.shutdown_memory_provider(session_messages)
            else:
                agent.shutdown_memory_provider()
    except Exception:
        pass
    try:
        if hasattr(agent, "close"):
            agent.close()
    except Exception:
        pass
    try:
        from agent.auxiliary_client import cleanup_stale_async_clients

        cleanup_stale_async_clients()
    except Exception:
        pass


async def shutdown_stop(
    runner: Any,
    *,
    restart: bool = False,
    detached_restart: bool = False,
    service_restart: bool = False,
) -> None:
    """Stop the gateway and disconnect all adapters."""
    if restart:
        runner._restart_requested = True
        runner._restart_detached = detached_restart
        runner._restart_via_service = service_restart
    if runner._stop_task is not None:
        await runner._stop_task
        return

    _AGENT_PENDING_SENTINEL = object()

    from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
    from hermes_cli.config import _hermes_home

    async def _stop_impl() -> None:
        def _kill_tool_subprocesses(phase: str) -> None:
            """Kill tool subprocesses + tear down terminal envs + browsers.

            Called twice in the shutdown path: once eagerly after a
            drain timeout forces agent interrupt (so we reclaim bash/
            sleep children before systemd TimeoutStopSec escalates to
            SIGKILL on the cgroup — #8202), and once as a final
            catch-all at the end of _stop_impl() for the graceful
            path or anything respawned mid-teardown.

            All steps are best-effort; exceptions are swallowed so
            one subsystem's failure doesn't block the rest.
            """
            try:
                from tools.process_registry import process_registry

                _killed = process_registry.kill_all()
                if _killed:
                    logger.info(
                        "Shutdown (%s): killed %d tool subprocess(es)",
                        phase, _killed,
                    )
            except Exception as _e:
                logger.debug("process_registry.kill_all (%s) error: %s", phase, _e)
            try:
                from tools.terminal_tool import cleanup_all_environments

                cleanup_all_environments()
            except Exception as _e:
                logger.debug("cleanup_all_environments (%s) error: %s", phase, _e)
            try:
                from tools.browser_tool import cleanup_all_browsers

                cleanup_all_browsers()
            except Exception as _e:
                logger.debug("cleanup_all_browsers (%s) error: %s", phase, _e)

        logger.info(
            "Stopping gateway%s...",
            " for restart" if runner._restart_requested else "",
        )
        _stop_started_at = time.monotonic()

        def _phase_elapsed() -> float:
            return time.monotonic() - _stop_started_at

        runner._running = False
        runner._draining = True

        # Notify all chats with active agents BEFORE draining.
        # Adapters are still connected here, so messages can be sent.
        await notify_active_sessions_of_shutdown(runner)
        logger.info(
            "Shutdown phase: notify_active_sessions done at +%.2fs",
            _phase_elapsed(),
        )

        timeout = runner._restart_drain_timeout

        # Pre-mark sessions as resume_pending BEFORE the drain wait.
        # If the process is killed by the service manager during the
        # drain, the durable marker is already written so the next
        # gateway boot can recover in-flight sessions (#27856).
        _pre_drain_keys: list[str] = []
        for _sk, _agent in list(runner._running_agents.items()):
            if _agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                runner.session_store.mark_resume_pending(
                    _sk,
                    "restart_timeout" if runner._restart_requested else "shutdown_timeout",
                )
                _pre_drain_keys.append(_sk)
            except Exception as _e:
                logger.debug("pre-drain mark_resume_pending failed for %s: %s", _sk, _e)

        _drain_started_at = time.monotonic()
        active_agents, timed_out = await drain_active_agents(runner, timeout)
        logger.info(
            "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
            "timed_out=%s, active_at_start=%d, active_now=%d)",
            _phase_elapsed(),
            time.monotonic() - _drain_started_at,
            timed_out,
            len(active_agents),
            running_agent_count(runner),
        )

        if not timed_out:
            # Drain completed gracefully — all running sessions finished.
            # Clear the pre-drain resume_pending markers so sessions that
            # completed during the drain window don't carry a stale flag.
            for _sk in _pre_drain_keys:
                if _sk not in runner._running_agents:
                    try:
                        runner.session_store.clear_resume_pending(_sk)
                    except Exception as _e:
                        logger.debug(
                            "clear_resume_pending after drain failed for %s: %s",
                            _sk, _e,
                        )

        if timed_out:
            logger.warning(
                "Gateway drain timed out after %.1fs with %d active agent(s); interrupting remaining work.",
                timeout,
                running_agent_count(runner),
            )
            _resume_reason = (
                "restart_timeout" if runner._restart_requested else "shutdown_timeout"
            )
            for _sk, _agent in list(runner._running_agents.items()):
                if _agent is _AGENT_PENDING_SENTINEL:
                    continue
                try:
                    runner.session_store.mark_resume_pending(_sk, _resume_reason)
                except Exception as _e:
                    logger.debug(
                        "mark_resume_pending failed for %s: %s",
                        _sk, _e,
                    )
            interrupt_running_agents(
                runner,
                "Gateway restarting" if runner._restart_requested else "Gateway shutting down",
            )
            interrupt_deadline = asyncio.get_running_loop().time() + 5.0
            while runner._running_agents and asyncio.get_running_loop().time() < interrupt_deadline:
                runner._update_runtime_status("draining")
                await asyncio.sleep(0.1)

            # Kill lingering tool subprocesses NOW, before we spend more
            # budget on adapter disconnect / session DB close.
            _kill_tool_subprocesses("post-interrupt")
            logger.info(
                "Shutdown phase: post-interrupt tool kill done at +%.2fs",
                _phase_elapsed(),
            )

        if runner._restart_requested and runner._restart_detached:
            try:
                await runner._launch_detached_restart_command()
            except Exception as e:
                logger.error("Failed to launch detached gateway restart: %s", e)

        finalize_shutdown_agents(runner, active_agents)

        # Also shut down memory providers on idle cached agents.
        _cache_lock = getattr(runner, "_agent_cache_lock", None)
        _cache = getattr(runner, "_agent_cache", None)
        if _cache_lock is not None and _cache is not None:
            with _cache_lock:
                _idle_agents = list(_cache.values())
                _cache.clear()
            for _entry in _idle_agents:
                _agent = (
                    _entry[0] if isinstance(_entry, tuple) else _entry
                )
                cleanup_agent_resources(runner, _agent)

        for platform, adapter in list(runner.adapters.items()):
            _adapter_started_at = time.monotonic()
            try:
                await adapter.cancel_background_tasks()
            except Exception as e:
                logger.debug("\u2717 %s background-task cancel error: %s", platform.value, e)
            try:
                await adapter.disconnect()
                logger.info(
                    "\u2713 %s disconnected (%.2fs)",
                    platform.value,
                    time.monotonic() - _adapter_started_at,
                )
            except Exception as e:
                logger.error(
                    "\u2717 %s disconnect error after %.2fs: %s",
                    platform.value,
                    time.monotonic() - _adapter_started_at,
                    e,
                )
        logger.info(
            "Shutdown phase: all adapters disconnected at +%.2fs",
            _phase_elapsed(),
        )

        for _task in list(runner._background_tasks):
            if _task is runner._stop_task:
                continue
            _task.cancel()
        runner._background_tasks.clear()

        runner.adapters.clear()
        runner._running_agents.clear()
        runner._running_agents_ts.clear()
        runner._pending_messages.clear()
        runner._pending_approvals.clear()
        if hasattr(runner, '_busy_ack_ts'):
            runner._busy_ack_ts.clear()
        runner._shutdown_event.set()

        # Global cleanup: kill any remaining tool subprocesses not tied
        # to a specific agent (catch-all for zombie prevention).
        _kill_tool_subprocesses("final-cleanup")
        logger.info(
            "Shutdown phase: final-cleanup tool kill done at +%.2fs",
            _phase_elapsed(),
        )

        try:
            from agent.auxiliary_client import shutdown_cached_clients

            shutdown_cached_clients()
        except Exception as _e:
            logger.debug("shutdown_cached_clients error: %s", _e)

        # Close SQLite session DBs so the WAL write lock is released.
        for _db_holder in (runner, getattr(runner, "session_store", None)):
            _db = getattr(_db_holder, "_db", None) if _db_holder else None
            if _db is None or not hasattr(_db, "close"):
                continue
            try:
                _db.close()
            except Exception as _e:
                logger.debug("SessionDB close error: %s", _e)
        logger.info(
            "Shutdown phase: SessionDB close done at +%.2fs",
            _phase_elapsed(),
        )

        from gateway.status import remove_pid_file, release_gateway_runtime_lock

        remove_pid_file()
        release_gateway_runtime_lock()

        if not timed_out:
            try:
                (_hermes_home / ".clean_shutdown").touch()
            except Exception:
                pass
        else:
            logger.info(
                "Skipping .clean_shutdown marker \u2014 drain timed out with "
                "interrupted agents; next startup will suspend recently "
                "active sessions.",
            )

        if active_agents:
            increment_restart_failure_counts(runner, set(active_agents.keys()))

        if runner._restart_requested and runner._restart_via_service:
            runner._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
            runner._exit_reason = runner._exit_reason or "Gateway restart requested"

        runner._draining = False
        runner._update_runtime_status("stopped", runner._exit_reason)
        logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

    runner._stop_task = asyncio.create_task(_stop_impl())
    await runner._stop_task


async def wait_for_shutdown(runner: Any) -> None:
    """Wait for shutdown signal."""
    await runner._shutdown_event.wait()


# --- Helper functions used by drain/shutdown logic ---


def snapshot_running_agents(runner: Any) -> Dict[str, Any]:
    return {
        session_key: agent
        for session_key, agent in runner._running_agents.items()
        if agent is not object()  # _AGENT_PENDING_SENTINEL-compatible check
    }


def running_agent_count(runner: Any) -> int:
    return sum(
        1 for agent in runner._running_agents.values()
        if agent is not object()
    )


def interrupt_running_agents(runner: Any, reason: str) -> None:
    for session_key, agent in list(runner._running_agents.items()):
        if agent is object():
            continue
        try:
            agent.interrupt(reason)
            logger.debug("Interrupted running agent for session %s during shutdown", session_key)
        except Exception as e:
            logger.debug("Failed interrupting agent during shutdown: %s", e)


def increment_restart_failure_counts(runner: Any, active_keys: set) -> None:
    try:
        runner._increment_restart_failure_counts(active_keys)
    except Exception:
        pass
