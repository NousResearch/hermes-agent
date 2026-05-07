"""File-descriptor watchdog and resource observability for the gateway.

Periodically logs process-level resource metrics (open FDs, thread count,
cached async clients, socket states) and optionally triggers a clean
restart when resource usage exceeds safe thresholds.

This catches slow leaks (like the CLOSE_WAIT accumulation from
auxiliary_client cache races) before they hit EMFILE and crash the gateway.
"""

import asyncio
import logging
import os
import resource
import subprocess
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Metric collection ───────────────────────────────────────────────────

def get_fd_count() -> int:
    """Return the number of open file descriptors for this process."""
    pid = os.getpid()
    fd_dir = f"/dev/fd"
    try:
        return len(os.listdir(fd_dir))
    except OSError:
        # Fallback: try /proc on Linux
        try:
            return len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            return -1


def get_fd_limit() -> int:
    """Return the soft RLIMIT_NOFILE for this process."""
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft


def get_thread_count() -> int:
    """Return the number of active threads."""
    return threading.active_count()


def get_cached_client_count() -> int:
    """Return the number of entries in the auxiliary_client cache."""
    try:
        from agent.auxiliary_client import _client_cache, _client_cache_lock
        with _client_cache_lock:
            return len(_client_cache)
    except Exception:
        return -1


def get_socket_states() -> Dict[str, int]:
    """Return a dict of TCP socket states → count for this process.

    Uses lsof (macOS/Linux) to enumerate sockets.  Returns empty dict
    on failure rather than raising.
    """
    pid = str(os.getpid())
    states: Dict[str, int] = {}
    try:
        # -i = internet sockets, -a = AND, -p = pid, -n = no DNS
        result = subprocess.run(
            ["lsof", "-i", "-a", "-p", pid, "-n", "-P"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 10:
                # Last column often has state like "(ESTABLISHED)" or "(CLOSE_WAIT)"
                state_col = parts[-1]
                if state_col.startswith("(") and state_col.endswith(")"):
                    state = state_col[1:-1]
                    states[state] = states.get(state, 0) + 1
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return states


def collect_metrics() -> dict:
    """Collect all resource metrics in one snapshot."""
    fd_count = get_fd_count()
    fd_limit = get_fd_limit()
    sockets = get_socket_states()
    return {
        "fd_count": fd_count,
        "fd_limit": fd_limit,
        "fd_usage_pct": round(fd_count / fd_limit * 100, 1) if fd_limit > 0 else 0,
        "thread_count": get_thread_count(),
        "cached_clients": get_cached_client_count(),
        "sockets": sockets,
        "close_wait": sockets.get("CLOSE_WAIT", 0),
    }


# ── Logging ─────────────────────────────────────────────────────────────

def log_metrics(metrics: Optional[dict] = None) -> dict:
    """Log current resource metrics and return them."""
    if metrics is None:
        metrics = collect_metrics()

    socket_summary = ", ".join(
        f"{state}={count}" for state, count in sorted(metrics["sockets"].items())
    ) or "none"

    logger.info(
        "FD watchdog: fds=%d/%d (%.1f%%) threads=%d cached_clients=%d "
        "close_wait=%d sockets=[%s]",
        metrics["fd_count"], metrics["fd_limit"], metrics["fd_usage_pct"],
        metrics["thread_count"], metrics["cached_clients"],
        metrics["close_wait"], socket_summary,
    )
    return metrics


# ── Watchdog thresholds ─────────────────────────────────────────────────

DEFAULT_FD_THRESHOLD_PCT = 70   # restart if FDs > 70% of limit
# CLOSE_WAIT threshold: some accumulation is normal for HTTP keep-alive clients
# whose server-side connections time out before the client reuses them.  Only
# the old catastrophic leak (pre-fix, hit 60+ in an hour) needs to trigger this.
# Healthy steady-state is usually well under 50.
DEFAULT_CLOSE_WAIT_THRESHOLD = 150
DEFAULT_CHECK_INTERVAL = 300    # 5 minutes


def should_restart(metrics: dict,
                   fd_pct_threshold: float = DEFAULT_FD_THRESHOLD_PCT,
                   close_wait_threshold: int = DEFAULT_CLOSE_WAIT_THRESHOLD) -> Optional[str]:
    """Check if the gateway should restart based on resource metrics.

    Returns a reason string if restart is warranted, None otherwise.
    """
    if metrics["fd_usage_pct"] > fd_pct_threshold:
        return (f"FD usage {metrics['fd_count']}/{metrics['fd_limit']} "
                f"({metrics['fd_usage_pct']}%) exceeds {fd_pct_threshold}% threshold")

    if metrics["close_wait"] > close_wait_threshold:
        return (f"CLOSE_WAIT count {metrics['close_wait']} "
                f"exceeds threshold {close_wait_threshold}")

    return None


# ── Proactive cleanup ───────────────────────────────────────────────────

def _try_cleanup() -> int:
    """Run all available cleanup routines. Returns number of clients cleaned."""
    cleaned = 0
    try:
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, cleanup_stale_async_clients,
        )
        with _client_cache_lock:
            before = len(_client_cache)
        cleanup_stale_async_clients()
        with _client_cache_lock:
            after = len(_client_cache)
        cleaned = before - after
    except Exception as e:
        logger.debug("Watchdog cleanup error: %s", e)
    return cleaned


# ── Async watchdog task ─────────────────────────────────────────────────

async def fd_watchdog_loop(
    runner: object,
    interval: float = DEFAULT_CHECK_INTERVAL,
    fd_pct_threshold: float = DEFAULT_FD_THRESHOLD_PCT,
    close_wait_threshold: int = DEFAULT_CLOSE_WAIT_THRESHOLD,
) -> None:
    """Background task that periodically checks resource health.

    Runs inside the gateway's event loop. On threshold breach, attempts
    cleanup first, then signals the runner to restart if still unhealthy.

    Args:
        runner: GatewayRunner instance (must have a .stop() method).
        interval: Seconds between checks.
        fd_pct_threshold: FD usage percentage that triggers restart.
        close_wait_threshold: CLOSE_WAIT count that triggers restart.
    """
    logger.info(
        "FD watchdog started: interval=%ds fd_threshold=%.0f%% close_wait_threshold=%d",
        interval, fd_pct_threshold, close_wait_threshold,
    )
    while True:
        await asyncio.sleep(interval)
        try:
            metrics = collect_metrics()
            log_metrics(metrics)

            reason = should_restart(metrics, fd_pct_threshold, close_wait_threshold)
            if reason:
                logger.warning("FD watchdog: threshold breached — %s", reason)

                # Try cleanup first before escalating to restart.
                cleaned = _try_cleanup()
                if cleaned:
                    logger.info("FD watchdog: cleaned %d stale client(s), re-checking", cleaned)
                    metrics = collect_metrics()
                    log_metrics(metrics)
                    reason = should_restart(metrics, fd_pct_threshold, close_wait_threshold)

                if reason:
                    logger.error(
                        "FD watchdog: triggering clean restart — %s", reason
                    )
                    # Mark the runner as exiting with failure BEFORE stopping it.
                    # This causes start_gateway() to return False, which causes
                    # main() to sys.exit(1). The non-zero exit code is critical:
                    # launchd's KeepAlive.SuccessfulExit=false only restarts the
                    # service on unsuccessful (non-zero) exits.  A clean stop()
                    # without this flag would exit 0 and launchd would leave the
                    # gateway down.
                    try:
                        setattr(runner, "should_exit_with_failure", True)
                        setattr(runner, "exit_reason", f"FD watchdog: {reason}")
                    except Exception:
                        pass
                    try:
                        stop_fn = getattr(runner, "stop", None)
                        if stop_fn:
                            if asyncio.iscoroutinefunction(stop_fn):
                                await stop_fn()
                            else:
                                stop_fn()
                    except Exception as e:
                        logger.error("FD watchdog: failed to stop runner: %s", e)
                        # Last resort: SIGTERM ourselves (non-zero exit)
                        os.kill(os.getpid(), 15)
                    return  # Exit the watchdog loop after triggering restart

        except asyncio.CancelledError:
            logger.info("FD watchdog stopped")
            return
        except Exception as e:
            logger.warning("FD watchdog error: %s", e, exc_info=True)
