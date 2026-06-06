"""Background auto-update for Hermes Agent.

When ``updates.auto_update`` is enabled in config, a daemon thread
periodically checks for newer versions (git origin/main or PyPI) and
applies them with ``hermes update --yes``. After a successful update it
triggers a gateway restart so the new code takes effect.

The thread is started once during gateway initialisation.  CLI mode does
not auto-update (the user runs ``hermes update`` manually).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Marker file written after a successful auto-update so we don't
# restart-loop.  Cleared after the gateway restarts.
_RESTART_MARKER = Path(
    os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"),
) / ".auto_update_restarted"


def _fetch_remote() -> bool:
    """Try ``git fetch origin main``. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("auto-update: git fetch failed: %s", exc)
        return False


def _check_behind() -> int:
    """Return number of commits behind origin/main (0 = up-to-date)."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as exc:
        logger.debug("auto-update: rev-list failed: %s", exc)
    return 0


def _has_changed_pyproject() -> bool:
    """Check if pyproject.toml or requirements changed upstream."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD..origin/main", "--",
             "pyproject.toml", "uv.lock", "requirements*.txt"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True  # safer to assume yes on error


def _run_update() -> bool:
    """Run ``hermes update --yes``. Returns True if successful."""
    hermes_bin = _find_hermes_bin()
    if not hermes_bin:
        return False
    try:
        result = subprocess.run(
            [hermes_bin, "update", "--yes"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info("auto-update: update applied successfully")
            return True
        else:
            logger.warning(
                "auto-update: update failed (exit %d): %s",
                result.returncode,
                result.stderr[-300:],
            )
            return False
    except subprocess.TimeoutExpired:
        logger.warning("auto-update: update timed out after 300s")
        return False
    except Exception as exc:
        logger.warning("auto-update: update exception: %s", exc)
        return False


def _find_hermes_bin() -> str | None:
    """Locate the ``hermes`` binary."""
    import shutil
    return shutil.which("hermes") or os.environ.get("HERMES_BIN")


def _schedule_gateway_restart() -> None:
    """Write the restart marker so the gateway runner knows to restart."""
    try:
        _RESTART_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _RESTART_MARKER.write_text("1")
        logger.info("auto-update: restart marker written, gateway will restart")
    except Exception as exc:
        logger.warning("auto-update: failed to write restart marker: %s", exc)


def check_restart_marker() -> bool:
    """Check if the auto-update restart marker exists and clear it.

    Called early in gateway startup so it can act before accepting messages.
    Returns True if a restart is expected (i.e. we just updated).
    """
    if _RESTART_MARKER.exists():
        try:
            _RESTART_MARKER.unlink()
        except Exception:
            pass
        return True
    return False


def auto_update_cycle(config: dict) -> bool:
    """Run one auto-update check+apply cycle.

    Args:
        config: The loaded config dict (with ``updates`` section).

    Returns:
        True if an update was applied and a restart is needed.
    """
    updates_cfg = config.get("updates", {})
    if not updates_cfg.get("auto_update", False):
        return False

    project_root = Path(__file__).resolve().parent.parent
    git_dir = project_root / ".git"
    if not git_dir.is_dir():
        logger.debug("auto-update: not a git checkout, skipping")
        return False

    if not _fetch_remote():
        logger.debug("auto-update: fetch failed, will retry next cycle")
        return False

    behind = _check_behind()
    if behind <= 0:
        logger.debug("auto-update: already up to date")
        return False

    logger.info("auto-update: %d commit(s) behind, applying update", behind)
    if not _run_update():
        logger.warning("auto-update: update failed, will retry next cycle")
        return False

    # Update applied — schedule a gateway restart.
    _schedule_gateway_restart()
    return True


# ──────────────────────────────────────────────────────────────────────
# Background thread
# ──────────────────────────────────────────────────────────────────────


class AutoUpdateThread:
    """Daemon thread that periodically checks for updates.

    Started once during gateway init.  Runs forever in the background.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        updates_cfg = config.get("updates", {})
        self._enabled = bool(updates_cfg.get("auto_update", False))
        self._interval = int(updates_cfg.get("auto_update_interval", 86400))
        self._restart_requested = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if not self._enabled:
            logger.debug("auto-update: disabled by config")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="auto-update",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "auto-update: background thread started (interval=%ds)",
            self._interval,
        )

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def restart_requested(self) -> bool:
        return self._restart_requested

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                updated = auto_update_cycle(self._config)
                if updated:
                    self._restart_requested = True
                    return  # thread exits; caller handles restart
            except Exception as exc:
                logger.warning("auto-update: cycle failed: %s", exc)

            self._stop_event.wait(self._interval)


__all__ = [
    "auto_update_cycle",
    "AutoUpdateThread",
    "check_restart_marker",
]
