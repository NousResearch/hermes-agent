"""
Code-change watcher for the gateway.

Polls the mtime of every hermes-agent source file that was imported into
sys.modules from the same checkout the gateway was launched from.  When any
file's mtime is newer than the process start time, the watcher logs a warning
and calls ``os.execv`` to reload the gateway in-place, preserving the PID
(important for launchd / systemd supervisors that track the PID directly).

Configuration
-------------
config.yaml (or ~/.hermes/config.yaml):
    gateway:
        auto_restart_on_code_change: false   # default: true

Environment override (wins over config):
    HERMES_GATEWAY_NO_AUTO_RESTART=1         # disable auto-restart

Opt-out is intentional for tests and for operators who want to control restarts
themselves.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# How often to check mtimes (seconds).
_POLL_INTERVAL_SECS = 60

# How long to wait after detecting a change before re-exec'ing.
# Gives the supervisor loop a moment to finish any in-flight work.
_RESTART_DELAY_SECS = 5


def _is_disabled_by_env() -> bool:
    """Return True when the env-var opt-out is active."""
    val = os.environ.get("HERMES_GATEWAY_NO_AUTO_RESTART", "").strip()
    return val not in ("", "0", "false", "False", "no", "No")


def _is_disabled_by_config(cfg: Optional[Dict]) -> bool:
    """Return True when config.yaml opts out."""
    if cfg is None:
        return False
    try:
        from hermes_cli.config import cfg_get

        value = cfg_get(cfg, "gateway", "auto_restart_on_code_change", default=True)
        # Explicit False in config disables the watcher.
        if isinstance(value, bool):
            return not value
        # Truthy string "false" / "0" / "no" also disables.
        if isinstance(value, str):
            return value.strip().lower() in ("false", "0", "no", "off")
        return not bool(value)
    except Exception:
        return False


def _collect_watched_files(checkout_root: Path) -> List[Path]:
    """
    Walk sys.modules and collect source files under ``checkout_root``.

    Only .py files that actually exist on disk are returned — .pyc / compiled
    extensions are skipped because they don't have meaningful standalone mtimes
    for our purpose (the .py source is what git touches on a ``git pull``).
    """
    watched: Dict[str, Path] = {}
    root_str = str(checkout_root.resolve())
    for module in list(sys.modules.values()):
        try:
            file_attr = getattr(module, "__file__", None)
        except Exception:
            continue
        if not file_attr:
            continue
        try:
            p = Path(file_attr).resolve()
        except Exception:
            continue
        if not str(p).startswith(root_str):
            continue
        if p.suffix != ".py":
            continue
        if p.is_file():
            watched[str(p)] = p
    return list(watched.values())


def _detect_checkout_root() -> Optional[Path]:
    """
    Infer the hermes-agent checkout root from sys.modules.

    Uses the first module whose __file__ lives inside a directory that also
    contains a 'gateway' subdirectory — that's a reliable proxy for the repo
    root.
    """
    for module in list(sys.modules.values()):
        try:
            f = getattr(module, "__file__", None)
        except Exception:
            continue
        if not f:
            continue
        try:
            p = Path(f).resolve()
        except Exception:
            continue
        # Walk up until we find a directory containing 'gateway/'
        for ancestor in [p.parent] + list(p.parents):
            if (ancestor / "gateway").is_dir() and (ancestor / "hermes_cli").is_dir():
                return ancestor
    return None


class CodeWatcher:
    """
    Polls hermes-agent source mtimes and os.execv-restarts the gateway when
    any file is newer than the process start time.

    Runs as a daemon thread — it never outlives the gateway process.
    """

    def __init__(
        self,
        *,
        process_start_time: float,
        poll_interval: float = _POLL_INTERVAL_SECS,
        restart_delay: float = _RESTART_DELAY_SECS,
        checkout_root: Optional[Path] = None,
        cfg: Optional[Dict] = None,
    ) -> None:
        self._process_start_time = process_start_time
        self._poll_interval = poll_interval
        self._restart_delay = restart_delay
        self._checkout_root = checkout_root
        self._cfg = cfg
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gateway-code-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.debug("gateway code-watcher started (poll_interval=%ds)", int(self._poll_interval))

    def stop(self) -> None:
        """Signal the polling thread to stop and wait briefly for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._poll_interval * 0.1))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        root = self._checkout_root or _detect_checkout_root()
        if root is None:
            logger.warning(
                "gateway code-watcher: could not detect checkout root — watcher disabled"
            )
            return

        logger.debug("gateway code-watcher: watching %s", root)
        while not self._stop_event.wait(timeout=self._poll_interval):
            changed = self._scan(root)
            if changed:
                logger.warning(
                    "gateway: code change detected in %s, re-execing in %ds to pick up new code",
                    changed,
                    int(self._restart_delay),
                )
                self._stop_event.wait(timeout=self._restart_delay)
                # One final check — if stop was requested during the delay, skip re-exec.
                if self._stop_event.is_set():
                    logger.info("gateway code-watcher: stop requested during restart delay, aborting re-exec")
                    return
                self._do_execv()
                return  # Never reached unless os.execv raises (non-Linux edge case)

    def _scan(self, root: Path) -> Optional[str]:
        """
        Return the path of the first changed file, or None if nothing changed.
        """
        files = _collect_watched_files(root)
        for path in files:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > self._process_start_time:
                return str(path)
        return None

    def _do_execv(self) -> None:
        """Replace the current process image with a fresh copy of itself."""
        executable = sys.executable
        argv = [executable] + sys.argv
        logger.info(
            "gateway: re-execing as %s %s",
            executable,
            " ".join(sys.argv),
        )
        try:
            os.execv(executable, argv)
        except Exception as exc:  # pragma: no cover
            logger.error("gateway: os.execv failed: %s", exc)


def start_code_watcher(
    *,
    cfg: Optional[Dict] = None,
    checkout_root: Optional[Path] = None,
    poll_interval: float = _POLL_INTERVAL_SECS,
    restart_delay: float = _RESTART_DELAY_SECS,
) -> Optional[CodeWatcher]:
    """
    Create and start a CodeWatcher if the feature is enabled.

    Returns the running CodeWatcher instance so callers can stop() it on
    graceful shutdown, or None when disabled.

    Priority order for opt-out:
        1. HERMES_GATEWAY_NO_AUTO_RESTART=1  (env wins over everything)
        2. gateway.auto_restart_on_code_change: false  (config.yaml)
        3. Default: enabled
    """
    if _is_disabled_by_env():
        logger.debug("gateway code-watcher: disabled by HERMES_GATEWAY_NO_AUTO_RESTART")
        return None
    if _is_disabled_by_config(cfg):
        logger.debug("gateway code-watcher: disabled by config (gateway.auto_restart_on_code_change=false)")
        return None

    # Capture process start time from /proc on Linux, or psutil / platform
    # fallback on macOS.  We use the kernel-reported start time so a
    # recently git-pulled file (mtime = pull timestamp) that predates the
    # last gateway boot is correctly ignored.
    process_start_time = _get_process_start_time()

    watcher = CodeWatcher(
        process_start_time=process_start_time,
        poll_interval=poll_interval,
        restart_delay=restart_delay,
        checkout_root=checkout_root,
        cfg=cfg,
    )
    watcher.start()
    return watcher


def _get_process_start_time() -> float:
    """
    Return the current process's start time as a Unix timestamp (float).

    Tries several approaches in order of reliability:
    1. gateway.status.get_process_start_time (already used by the gateway)
    2. psutil.Process().create_time() if available
    3. /proc/<pid>/stat on Linux
    4. Fallback to time.time() - uptime estimate (conservative: assumes
       process started at most 1 second before this call; means the watcher
       will ignore changes that arrived in the last second — acceptable).
    """
    pid = os.getpid()

    # 1. gateway.status already has cross-platform logic
    try:
        from gateway.status import get_process_start_time as _gs_start

        t = _gs_start(pid)
        if t is not None:
            return float(t)
    except Exception:
        pass

    # 2. psutil
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        pass

    # 3. /proc on Linux
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            fields = f.read().split()
        # Field 22 (0-indexed: 21) is starttime in clock ticks since boot
        import math

        clk_tck = os.sysconf("SC_CLK_TCK")
        btime: Optional[int] = None
        with open("/proc/stat", encoding="utf-8") as f:
            for line in f:
                if line.startswith("btime"):
                    btime = int(line.split()[1])
                    break
        if btime is None:
            raise ValueError("btime not found in /proc/stat")
        starttime_ticks = int(fields[21])
        return float(btime + math.floor(starttime_ticks / clk_tck))
    except Exception:
        pass

    # 4. Conservative fallback: current time (process "just started")
    logger.debug("gateway code-watcher: could not determine process start time, using current time as baseline")
    return time.time()
