"""
Auto-pull watcher for the ~/.hermes config checkout.

``~/.hermes`` is a git checkout of the user's hermes-config repo.  When new
skills, SOULs, or config changes are merged via PR, the live checkout drifts
behind ``origin/main`` until someone manually ``git pull``.  This module
adds a background thread that fast-forwards the checkout once per hour when
the tree is clean and on ``main``.

Safety contract
---------------
- Only fast-forwards (``git pull --ff-only``).  Never force-pushes, never
  rebases, never touches an ahead-of-origin or diverged tree.
- Preserves uncommitted local edits — it never discards a dirty tree.
- Skips silently when the current branch is not ``main``.
- All subprocess calls are non-blocking relative to the event loop — the
  watcher runs in a daemon thread.

Wedged-state alerting
---------------------
The live ``~/.hermes`` tree is almost always dirty (continuous curator /
worker skill-edit churn).  A dirty tree blocks ``git pull --ff-only``, so a
silent dirty-skip would let incoming merges pile up on disk forever with no
operator signal — which is exactly what happened (PRs #54–#58 piled up
undelivered with zero alerting until a cron broke).

To fix that, ``_pull_once`` distinguishes "dirty-tree-can't-ff while
behind>0" (the WEDGED state) from "already up-to-date".  When wedged, it
invokes an optional ``notify`` callback so the gateway can page the operator
via the same home-channel surface used for shutdown/startup notifications.
The :class:`HermesHomePuller` throttles those alerts (default: at most one
per 6 hours) so the operator isn't paged every hourly tick.

Configuration
-------------
config.yaml:
    gateway:
        hermes_home_auto_pull: true   # default: true

Environment override (wins over config):
    HERMES_GATEWAY_NO_AUTO_PULL=1    # disable auto-pull
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)

# Interval between pull checks (seconds).  Once per hour is intentionally
# conservative — this is a "drift catch-up" tool, not a live-sync mechanism.
_CHECK_INTERVAL_SECS = 3600

# Timeout for each git subprocess call.
_GIT_TIMEOUT_SECS = 30

# Minimum seconds between wedged-state alerts.  The puller ticks hourly, so
# without a throttle a persistently-wedged tree (the common case) would page
# the operator every hour.  6h means at most ~4 pages/day for a tree that
# stays wedged, which is enough to stay visible without becoming noise.
_ALERT_THROTTLE_SECS = 6 * 3600

# Type alias for the wedged-state notifier.  Called with the number of
# commits the live checkout is behind origin/main.  The gateway supplies a
# callback that schedules a home-channel send on the event loop.
WedgedNotifier = Callable[[int], None]


def _is_disabled_by_env() -> bool:
    """Return True when the env-var opt-out is active."""
    val = os.environ.get("HERMES_GATEWAY_NO_AUTO_PULL", "").strip()
    return val not in ("", "0", "false", "False", "no", "No")


def _is_disabled_by_config(cfg: Optional[Dict]) -> bool:
    """Return True when config.yaml opts out."""
    if cfg is None:
        return False
    try:
        from hermes_cli.config import cfg_get

        value = cfg_get(cfg, "gateway", "hermes_home_auto_pull", default=True)
        if isinstance(value, bool):
            return not value
        if isinstance(value, str):
            return value.strip().lower() in ("false", "0", "no", "off")
        return not bool(value)
    except Exception:
        return False  # config unreadable — don't disable


def _run_git(hermes_home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *hermes_home*.  Returns the CompletedProcess."""
    cmd = ["git", "-C", str(hermes_home), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECS,
    )


def _current_branch(hermes_home: Path) -> str:
    """Return the current branch name, or '' on failure."""
    result = _run_git(hermes_home, "rev-parse", "--abbrev-ref", "HEAD")
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _is_clean(hermes_home: Path) -> bool:
    """Return True when there are no staged, unstaged, or untracked changes."""
    result = _run_git(hermes_home, "status", "--porcelain")
    if result.returncode != 0:
        return False
    return result.stdout.strip() == ""


def _commits_behind(hermes_home: Path) -> int:
    """
    Return how many commits HEAD is behind origin/main.

    Runs ``git fetch`` first (quiet, so a slow network doesn't cause noise),
    then counts commits reachable from origin/main but not from HEAD.
    Returns 0 if the fetch fails (conservative — don't pull if we can't
    verify remote state).
    """
    fetch = _run_git(hermes_home, "fetch", "origin", "--quiet")
    if fetch.returncode != 0:
        logger.debug(
            "hermes-home puller: git fetch failed (%s); skipping this check",
            fetch.stderr.strip(),
        )
        return 0

    rev_list = _run_git(
        hermes_home, "rev-list", "--count", "HEAD..origin/main"
    )
    if rev_list.returncode != 0:
        return 0
    try:
        return int(rev_list.stdout.strip())
    except ValueError:
        return 0


def _fast_forward(hermes_home: Path) -> bool:
    """
    Attempt a fast-forward pull from origin/main.

    Returns True on success, False on failure.  Logs outcome at INFO
    (success) or WARNING (failure).
    """
    result = _run_git(
        hermes_home, "pull", "--ff-only", "origin", "main", "--quiet"
    )
    if result.returncode == 0:
        logger.info(
            "hermes-home puller: fast-forwarded ~/.hermes to origin/main"
        )
        return True
    logger.warning(
        "hermes-home puller: pull --ff-only failed (%s)",
        result.stderr.strip(),
    )
    return False


def _pull_once(
    hermes_home: Path,
    *,
    notify: Optional[WedgedNotifier] = None,
) -> None:
    """
    Single pull attempt: check branch → count behind → fast-forward or alert.

    The check order matters: ``behind`` is computed *before* the clean check
    so the wedged state (dirty tree that can't fast-forward while behind>0)
    is distinguishable from "already up-to-date".  When wedged and a
    ``notify`` callback is supplied, it is invoked with the behind count so
    the caller can surface an operator alert.

    Branch states:
    - not ``main``  → silent DEBUG skip (a feature branch is operator intent).
    - ``main``, behind==0 → silent DEBUG skip (up-to-date).
    - ``main``, behind>0, clean → fast-forward.
    - ``main``, behind>0, dirty → WEDGED: log WARNING + call ``notify``.

    All failures are non-fatal: they log at DEBUG/WARNING and return.
    """
    try:
        branch = _current_branch(hermes_home)
        if branch != "main":
            logger.debug(
                "hermes-home puller: skipping pull — on branch %r (not main)",
                branch,
            )
            return

        behind = _commits_behind(hermes_home)
        if behind == 0:
            logger.debug(
                "hermes-home puller: ~/.hermes is up-to-date with origin/main"
            )
            return

        if _is_clean(hermes_home):
            logger.info(
                "hermes-home puller: ~/.hermes is %d commit(s) behind origin/main; pulling",
                behind,
            )
            _fast_forward(hermes_home)
            return

        # main + behind>0 + dirty = WEDGED.  The fast-forward can't land
        # because the tree is dirty; without an alert the incoming merges
        # would pile up silently.  Log loudly and notify the operator.
        logger.warning(
            "hermes-home puller: WEDGED — %s is %d commit(s) behind "
            "origin/main but the working tree is dirty; auto-pull blocked",
            display_hermes_home(),
            behind,
        )
        if notify is not None:
            try:
                notify(behind)
            except Exception:
                logger.exception(
                    "hermes-home puller: wedged-state notify callback failed"
                )

    except subprocess.TimeoutExpired:
        logger.warning("hermes-home puller: git timed out; will retry next cycle")
    except Exception:
        logger.exception("hermes-home puller: unexpected error during pull check")


class HermesHomePuller:
    """
    Background daemon thread that periodically fast-forwards ~/.hermes.

    Lifecycle mirrors :class:`gateway.code_watcher.CodeWatcher` — construct,
    then call :meth:`start` to launch the thread, :meth:`stop` to join it.

    When ``notifier`` is supplied, the puller pages the operator on the
    wedged state (dirty tree + behind>0 on main), throttled to at most one
    alert per ``alert_throttle_secs``.
    """

    def __init__(
        self,
        hermes_home: Path,
        interval: float = _CHECK_INTERVAL_SECS,
        *,
        notifier: Optional[Callable[[str], None]] = None,
        alert_throttle_secs: float = _ALERT_THROTTLE_SECS,
    ) -> None:
        self._hermes_home = hermes_home
        self._interval = interval
        self._notifier = notifier
        self._alert_throttle_secs = alert_throttle_secs
        # Monotonic timestamp of the last wedged-alert we actually emitted.
        # ``None`` means "never alerted" so the first wedged tick always fires.
        self._last_alert_ts: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Typed as Callable so tests can monkey-patch _run without a type error.
        self._run: Callable[[], None] = self._run_impl

    # ------------------------------------------------------------------
    # Wedged-state alerting
    # ------------------------------------------------------------------

    def _now(self) -> float:
        """Monotonic clock — isolated as a method so tests can patch it."""
        return time.monotonic()

    def _on_wedged(self, behind: int) -> None:
        """
        Wedged-state callback passed into :func:`_pull_once`.

        Builds the operator-facing message and forwards it to the configured
        notifier, subject to the per-instance throttle.
        """
        if self._notifier is None:
            return

        now = self._now()
        if (
            self._last_alert_ts is not None
            and (now - self._last_alert_ts) < self._alert_throttle_secs
        ):
            logger.debug(
                "hermes-home puller: wedged alert suppressed (throttled, "
                "%.0fs since last alert < %.0fs window)",
                now - self._last_alert_ts,
                self._alert_throttle_secs,
            )
            return

        message = (
            f"⚠️ hermes-home auto-pull is WEDGED: {display_hermes_home()} is {behind} "
            f"commit(s) behind origin/main, but the working tree is dirty so "
            f"`git pull --ff-only` is blocked. Incoming merges are piling up "
            f"undelivered on disk. Reconcile the live checkout manually "
            f"(commit/stash local edits, then `git pull --ff-only`)."
        )
        try:
            self._notifier(message)
        except Exception:
            logger.exception("hermes-home puller: notifier raised; alert dropped")
            return
        # Only advance the throttle clock once the alert was handed off.
        self._last_alert_ts = now

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="hermes-home-puller",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "hermes-home puller: started (interval=%ds, home=%s)",
            int(self._interval),
            self._hermes_home,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run_impl(self) -> None:
        # Run once at startup (after a short warm-up delay) so the first
        # check happens quickly — avoids waiting a full hour on gateway boot.
        time.sleep(10)
        _pull_once(self._hermes_home, notify=self._on_wedged)

        while not self._stop_event.wait(timeout=self._interval):
            _pull_once(self._hermes_home, notify=self._on_wedged)


def start_hermes_home_puller(
    cfg: Optional[Dict] = None,
    hermes_home: Optional[Path] = None,
    interval: float = _CHECK_INTERVAL_SECS,
    *,
    notifier: Optional[Callable[[str], None]] = None,
    alert_throttle_secs: float = _ALERT_THROTTLE_SECS,
) -> Optional[HermesHomePuller]:
    """
    Create and start a :class:`HermesHomePuller` if auto-pull is enabled.

    Returns the puller instance (so the caller can stop it cleanly), or
    ``None`` when auto-pull is disabled.

    ``hermes_home`` defaults to the value of :func:`hermes_constants.get_hermes_home`.

    ``notifier`` is an optional ``Callable[[str], None]`` used to page the
    operator when the auto-pull is wedged (dirty tree + behind on main). The
    gateway threads in a callback that schedules a home-channel send on the
    event loop. When omitted, the puller logs the wedged state at WARNING but
    does not page anyone.
    """
    if _is_disabled_by_env():
        logger.info("hermes-home puller: disabled via HERMES_GATEWAY_NO_AUTO_PULL env")
        return None

    if _is_disabled_by_config(cfg):
        logger.info("hermes-home puller: disabled via config gateway.hermes_home_auto_pull=false")
        return None

    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = get_hermes_home()
        except Exception:
            logger.warning(
                "hermes-home puller: cannot resolve hermes_home; disabled"
            )
            return None

    git_dir = hermes_home / ".git"
    if not git_dir.exists():
        logger.info(
            "hermes-home puller: %s has no .git directory; disabled (not a git checkout)",
            hermes_home,
        )
        return None

    puller = HermesHomePuller(
        hermes_home=hermes_home,
        interval=interval,
        notifier=notifier,
        alert_throttle_secs=alert_throttle_secs,
    )
    puller.start()
    return puller
