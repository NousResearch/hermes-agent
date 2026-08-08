"""Orphan goal recovery — resume standing goals whose owning process died.

A standing ``/goal`` (Ralph loop) only runs inside a live process. Goal
state is persisted in ``SessionDB.state_meta`` under ``goal:<session_id>``,
but nothing executes the loop when the owning process dies — a desktop chat
PTY reaped after detach, a closed terminal, or a gateway that was stopped
all leave the goal ``active`` but inert.

This module is the shared supervisor used by both long-lived surfaces that
can legitimately own a goal loop:

- ``gateway/run.py`` — scans the profiles the gateway serves at boot and on
  a periodic sweep;
- ``hermes_cli/web_server.py`` (``hermes serve`` / dashboard) — the same
  scan for desktop-scoped sessions.

Ownership model
---------------
Each goal row records ``owner_pid`` + ``last_owner_seen_at`` at set time
(and refreshes ``last_owner_seen_at`` on every evaluated turn, via
``GoalManager.evaluate_after_turn``). A goal is orphaned when:

- its status is ``active`` (paused/cleared/done are never touched), and
- no live owner remains, where "live owner" means:
  * the surface hosting the sweep has a live in-process session for the
    goal (``owns_goal`` callback — e.g. the desktop chat's tui_gateway
    session registry), or
  * the stored ``owner_pid`` belongs to a different, still-alive process
    (the classic double-fire guard: a surface whose goal loop is running
    must never have its continuation re-queued by a sibling supervisor).
  * A surface that sets goals in-process (``hermes serve``, whose
    tui_gateway owns desktop goals in attach mode) passes ``self_pid`` so
    its own pid is judged by the session registry instead — an alive pid
    proves nothing about a loop that may have been reaped.

Cross-process safety
--------------------
Gateway and ``hermes serve`` can both run on one machine, so the claim
step is serialized with a per-``HERMES_HOME`` lock file; only the winner
flips the goal to ``orphaned`` (the flag flip IS the claim), and a freshly
claimed goal is not re-claimed for a short window while the winner is
driving it. A supervisor that crashes mid-drive leaves ``orphaned=True``
with a stale timestamp, so the next sweep after the window heals it.

``/goal status`` reflects reality via the persisted flag:
``mark_goal_orphaned`` / ``clear_goal_orphaned`` in ``hermes_cli/goals.py``
produce the "active-orphaned" vs "active-running" distinction.

The sweep is pure: ``sweep()`` returns the ``(home, session_id, state)``
claims this process won. Each surface drives the continuation turn through
its own machinery (adapter FIFO for the gateway, ``tui_gateway`` dispatch
for ``hermes serve``) and calls :meth:`clear_claim` once the turn is
handed off, so message-role alternation and prompt caching stay untouched.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: A goal whose owner pid is dead may still be claimed immediately; the
#: silence window only gates goals with NO owner pid at all (rows written
#: before owner stamping existed, or a set-then-crash race).
_OWNER_SILENCE_SECONDS = 300.0

#: A claimed goal (``orphaned=True``) is not re-claimed within this window,
#: even by another supervisor process — the winner is still driving.
_RECENT_CLAIM_WINDOW_SECONDS = 30.0


def _owner_alive(state: object) -> bool:
    """True when the goal's stored owner process is still alive.

    Uses the same cross-platform, footgun-safe liveness check the goal
    wait-barrier uses (``gateway.status._pid_exists`` — avoids
    ``os.kill(pid, 0)`` which on Windows is NOT a no-op, bpo-14484).
    Any error resolves to False (treat unknown as dead) so a stale owner
    can never block recovery forever.
    """
    pid = getattr(state, "owner_pid", None)
    if not pid or int(pid) <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(int(pid)))
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _last_owner_seen(state: object) -> float:
    try:
        return float(getattr(state, "last_owner_seen_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _goal_is_orphaned(
    state: object,
    now: float,
    *,
    surface_owns: bool = False,
    self_pid: Optional[int] = None,
) -> bool:
    """True when an active goal should be claimed by this sweep.

    ``surface_owns`` — True when the sweeping surface hosts a live
    in-process session for the goal (its loop is running; never claim).

    ``self_pid`` — when set, a goal whose ``owner_pid`` equals this
    process is judged purely by the session registry (``surface_owns``):
    the owning surface and the sweeping surface are the same process, so
    an alive pid proves nothing about the loop. Used by ``hermes serve``,
    which sets desktop goals in-process. Surfaces that do not set goals
    in-process (the messaging gateway) leave it None, so a live owner pid
    always means "loop running elsewhere — do not touch".
    """
    if getattr(state, "status", None) != "active":
        return False
    if surface_owns:
        return False
    owner = 0
    try:
        owner = int(getattr(state, "owner_pid", 0) or 0)
    except (TypeError, ValueError):
        owner = 0
    if owner > 0 and owner != (self_pid or 0) and _owner_alive(state):
        return False
    if getattr(state, "orphaned", False):
        # Previously claimed — the freshness gate is re-checked under the
        # claim lock, so an orphaned flag alone means "claimable" here
        # (heals a supervisor that crashed mid-drive).
        return True
    if owner > 0:
        # Owner pid recorded but the process is gone → claim immediately.
        return True
    # No owner pid at all (rows written before owner stamping, or a
    # set-then-crash race) → require the silence window.
    return (now - _last_owner_seen(state)) > _OWNER_SILENCE_SECONDS


def _claim_lock_path(home: str) -> str:
    """Return the shared recovery claim-lock path for a HERMES_HOME.

    One lock per home across ALL surfaces, so a gateway and a
    ``hermes serve`` backing the same home can never both claim the same
    goal in the same sweep.
    """
    import hashlib

    digest = hashlib.sha1(str(home).encode("utf-8")).hexdigest()[:12]
    return os.path.join(str(home), "runtime", f"goal-recovery-{digest}.lock")


class _ClaimLock:
    """Cross-process claim lock (fcntl on POSIX, msvcrt on Windows).

    Mirrors ``hermes_cli.active_sessions._FileLock``: one process at a time
    owns the claim critical section.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def __enter__(self) -> "_ClaimLock":
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._fh = open(self.path, "a+b")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("goal recovery: claim lock open failed: %s", exc)
            self._fh = None
            return self
        if os.name == "nt":
            try:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            except Exception:
                self._fh.close()
                self._fh = None
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                self._fh.close()
                self._fh = None
        return self

    def __exit__(self, *_exc) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        finally:
            self._fh = None


class GoalRecoverySweeper:
    """Scans HERMES_HOMEs for orphaned active goals and claims them.

    ``sweep()`` is pure: it returns ``[(home, session_id, state), ...]``
    for the goals this process won the claim on. The surface then drives
    each continuation turn through its own turn machinery and calls
    :meth:`clear_claim`. Never raises — a broken DB or missing module
    degrades to an empty sweep.
    """

    def __init__(
        self,
        homes: List[str],
        *,
        owns_goal: Optional[Callable[[str], bool]] = None,
        self_pid: Optional[int] = None,
    ):
        self.homes = [str(h) for h in homes]
        # Surface-specific live-ownership check, e.g. "this session is
        # currently live in my in-process session registry". Called with
        # the goal's session_id.
        self._owns_goal = owns_goal
        # Process identity for the in-process-owner override (serve).
        self._self_pid = int(self_pid) if self_pid else None

    def sweep(self) -> List[Tuple[str, str, object]]:
        """Scan every configured HERMES_HOME; return the claimed orphans.

        The claim (persisted ``orphaned`` flag + timestamp) is what stops a
        sibling supervisor from double-firing. The caller drives each claim
        through its own turn machinery and then calls :meth:`clear_claim`
        so ``/goal status`` flips back to active-running.
        """
        claimed: List[Tuple[str, str, object]] = []
        for home in self.homes:
            claimed.extend(self._sweep_home(home))
        return claimed

    def clear_claim(self, home: str, session_id: str) -> None:
        """Clear the orphaned flag for a claim this process drove."""
        try:
            db = self._session_db_for_home(home)
            if db is None:
                return
            self._clear_orphaned(db, session_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "goal recovery: orphaned-clear failed for %s: %s", session_id, exc
            )

    def _sweep_home(self, home: str) -> List[Tuple[str, str, object]]:
        try:
            from hermes_cli.goals import GOAL_META_PREFIX
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("goal recovery: goals module unavailable: %s", exc)
            return []

        db = self._session_db_for_home(home)
        if db is None:
            return []

        now = time.time()
        try:
            keys = db.list_meta_keys(GOAL_META_PREFIX)
        except Exception as exc:
            logger.debug("goal recovery: meta enumeration failed for %s: %s", home, exc)
            return []

        claimed: List[Tuple[str, str, object]] = []
        for key in keys:
            session_id = key[len(GOAL_META_PREFIX):]
            if not session_id:
                continue
            state = self._load_goal_at(db, session_id)
            if state is None:
                continue
            surface_owns = bool(self._owns_goal(session_id)) if self._owns_goal else False
            if not _goal_is_orphaned(state, now, surface_owns=surface_owns, self_pid=self._self_pid):
                continue
            if not self._claim(db, session_id):
                continue
            claimed_state = self._load_goal_at(db, session_id)
            claimed.append((home, session_id, claimed_state or state))
        return claimed

    @staticmethod
    def _session_db_for_home(home: str):
        """Open a SessionDB bound to ``home`` (per-home, like goals.py)."""
        try:
            from pathlib import Path

            from hermes_state import SessionDB

            return SessionDB(db_path=Path(str(home)) / "state.db")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("goal recovery: SessionDB unavailable for %s: %s", home, exc)
            return None

    @staticmethod
    def _load_goal_at(db, session_id: str):
        """Load a goal row through the DB instance bound to its home.

        ``goals.load_goal`` resolves the process-default HERMES_HOME, which
        is wrong when sweeping a secondary profile. Parse directly instead.
        """
        from hermes_cli.goals import GoalState, _meta_key

        raw = db.get_meta(_meta_key(session_id))
        if not raw:
            return None
        try:
            return GoalState.from_json(raw)
        except Exception as exc:
            logger.warning(
                "GoalManager: could not parse stored goal for %s: %s", session_id, exc
            )
            return None

    def _claim(self, db, session_id: str) -> bool:
        """Atomically (per-home lock) claim this goal for this sweep.

        The flag flip (``orphaned`` True + claim timestamp) IS the claim:
        a goal claimed within the recent window is skipped even if a
        sibling supervisor is mid-drive; a goal claimed long ago (winner
        crashed) is re-claimed.
        """
        from hermes_cli.goals import load_goal, save_goal

        lock_path = _claim_lock_path(db.db_path)
        try:
            with _ClaimLock(lock_path):
                state = load_goal(session_id, db=db)
                if state is None or getattr(state, "status", None) != "active":
                    return False
                now = time.time()
                if getattr(state, "orphaned", False) and (
                    now - _last_owner_seen(state)
                ) <= _RECENT_CLAIM_WINDOW_SECONDS:
                    return False
                state.orphaned = True
                state.last_owner_seen_at = now
                save_goal(session_id, state, db=db)
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("goal recovery: claim failed for %s: %s", session_id, exc)
            return False

    @staticmethod
    def _clear_orphaned(db, session_id: str) -> None:
        """Clear the orphaned flag once the adopted surface is driving."""
        from hermes_cli.goals import clear_goal_orphaned

        clear_goal_orphaned(session_id, db=db)


__all__ = ["GoalRecoverySweeper"]
