"""Profile delete/rename: stop every process bound to a profile before its home is removed or moved.

``profiles.delete_profile`` and ``profiles.rename_profile`` call these once the retirement
tombstone is published: the messaging gateway named by its runtime-locked ``gateway.pid``, the
other spawn-ledger backends bound to the profile (Desktop-spawned ``serve``/``dashboard``/``webapp``),
and its Bot Desktop. A kill is only ever sent to a positively identified process incarnation, so
a recycled PID is never signalled.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _argv_profile_selectors(argv: list):
    """Yield every profile name selected via ``-p X`` / ``--profile X`` / ``--profile=X``."""
    for i, tok in enumerate(argv):
        if tok in {"--profile", "-p"} and i + 1 < len(argv):
            yield argv[i + 1]
        elif tok.startswith("--profile="):
            yield tok.split("=", 1)[1]


def _profile_bound_backend_pids(canon: str, profile_dir: Path) -> list[tuple[int, float]]:
    """PIDs of running Hermes *backends* bound to this profile (``gateway.pid`` only tracks
    the messaging gateway). Tightly scoped: current-user processes, backend subcommands only
    (never an interactive ``chat``/``tui``), never this process or its ancestors. Empty when
    ``psutil`` can't inspect anything."""
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    from hermes_cli.process_identity import reapable_ledger_identities
    from hermes_cli.profiles import normalize_profile_name

    identified = reapable_ledger_identities()
    if not identified:
        return []

    try:
        resolved_dir = profile_dir.resolve()
    except OSError:
        resolved_dir = profile_dir

    # Never terminate ourselves or a parent (`hermes -p <canon> profile delete` runs under
    # the very profile it's deleting).
    skip: set[int] = {os.getpid()}
    with contextlib.suppress(Exception):
        parent = psutil.Process(os.getpid()).parent()
        while parent is not None:
            skip.add(parent.pid)
            parent = parent.parent()
    try:
        current_user = psutil.Process(os.getpid()).username()
    except Exception:
        current_user = None
    identities: list[tuple[int, float]] = []
    for proc in psutil.process_iter(["pid", "name", "username", "cmdline", "create_time"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if not isinstance(pid, int) or pid in skip or pid not in identified:
                continue
            created = info.get("create_time")
            if not isinstance(created, (int, float)) or isinstance(created, bool):
                created = proc.create_time()  # a process gone or denied meanwhile is skipped below
            if abs(float(created) - identified[pid]) > 0.001:
                continue
            if current_user is not None and info.get("username") != current_user:
                continue
            argv = info.get("cmdline") or []
            if not argv:
                continue

            # Bound to THIS profile by selector flag, or by HERMES_HOME pointing at its dir.
            bound = any(normalize_profile_name(sel) == canon for sel in _argv_profile_selectors(argv))
            if not bound:
                with contextlib.suppress(Exception):  # environ() can raise AccessDenied even same-user
                    env_home = (proc.environ() or {}).get("HERMES_HOME", "")
                    bound = bool(env_home) and Path(env_home).resolve() == resolved_dir
            if bound:
                identities.append((pid, identified[pid]))
        except Exception:
            continue  # NoSuchProcess / AccessDenied / ZombieProcess and anything else
    return identities


def _wait_then_force_kill(
    pids: List[int], *, alive: Callable[[int], bool],
    start_times: Optional[Dict[int, Optional[float]]] = None, wait: float = 10.0,
) -> bool:
    """After a graceful ``terminate_pid``, poll ``alive(pid)`` (the caller's identity check, so a
    recycled PID never reads as a straggler) every 0.5s for up to *wait* seconds, then force-kill
    stragglers. True when every pid exited without one. ``start_times`` pins each force kill to
    the incarnation seen before the graceful signal; without it the start time is read at kill
    time and a pid whose identity is then unprovable is never force-killed."""
    from gateway.status import get_process_start_time, terminate_pid
    stragglers = [pid for pid in pids if alive(pid)]
    for _ in range(int(wait / 0.5)):
        if not stragglers:
            return True
        time.sleep(0.5)
        stragglers = [pid for pid in stragglers if alive(pid)]
    for pid in stragglers:
        with contextlib.suppress(OSError):  # includes ProcessLookupError / PermissionError
            if start_times is not None:
                expected_start_time = start_times.get(pid)
            elif (expected_start_time := get_process_start_time(pid)) is None or not alive(pid):
                continue
            terminate_pid(pid, force=True, expected_start_time=expected_start_time)
    return not stragglers


def _stop_profile_backends(canon: str, profile_dir: Path) -> None:
    """Terminate any Desktop-spawned / stray backends bound to this profile.

    Complements ``_stop_gateway_process`` (which only knows ``gateway.pid``):
    without this, a live ``serve``/``dashboard`` backend keeps creating files
    under the profile dir while ``rmtree`` walks it, so the final ``rmdir``
    fails with ``ENOTEMPTY`` and the delete doesn't converge.  Best-effort:
    any failure is reported and swallowed so it never makes delete worse.
    """
    identities = _profile_bound_backend_pids(canon, profile_dir)
    if not identities:
        return

    try:
        import psutil  # type: ignore
        from gateway.status import terminate_pid
    except ImportError:
        return
    ledger_created = dict(identities)

    def _identity_alive(pid: int) -> bool:
        try:
            actual_created = float(psutil.Process(pid).create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            return False
        return abs(actual_created - ledger_created[pid]) <= 0.001

    signaled: list[int] = []
    for pid in ledger_created:
        if not _identity_alive(pid):
            continue
        try:
            terminate_pid(pid)  # graceful first
            signaled.append(pid)
        except (ProcessLookupError, PermissionError, OSError):
            continue

    if signaled:
        _wait_then_force_kill(signaled, alive=_identity_alive)
        print(f"✓ Stopped {len(signaled)} profile backend process(es)")


def _stop_bot_desktop(profile_dir: Path) -> None:
    """Stop the profile's Bot Desktop (Xvnc + Xfce launcher) before its directory is removed or renamed;
    gateway shutdown does not reach it (its own session, its own pid file). Scoped through the hermes-home
    override so the runtime reads THIS profile's bot-desktop/ state, whichever profile invoked the op.
    A failure here is logged, never fatal: the profile op is what the user asked for."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.bot_desktop import runtime
    if not runtime.is_supported_host():
        return
    token = set_hermes_home_override(profile_dir)
    try:
        if runtime.stop():
            print("✓ Bot Desktop stopped")
            # The screen a human's exclusion protected is gone; on rename the directory (lease.json
            # included) moves with the profile, and a human lease for a dead viewer would fence the
            # agent out of the renamed profile's next screen until someone force-released it.
            from tools.bot_desktop import lease
            lease.release()
    except Exception as e:
        logger.warning("Could not stop the Bot Desktop of %s: %s", profile_dir, e)
    finally:
        reset_hermes_home_override(token)


def _stop_gateway_process(profile_dir: Path) -> None:
    """Stop the positively identified gateway named by its runtime-locked PID file."""
    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return
    try:
        from gateway.status import (
            get_process_start_time,
            get_running_pid,
            recorded_gateway_home_conflicts,
            terminate_pid as _terminate_pid,
        )

        raw = pid_file.read_text(encoding="utf-8-sig").strip()
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data["pid"])
        # Cross-profile kill refusal (#89315): the record's hermes_home stamp
        # names the gateway's TRUE owner. A contaminated/poisoned gateway.pid
        # inside this profile dir can point at another profile's live gateway
        # — killing it starts the mutual SIGTERM restart loop from the issue.
        if recorded_gateway_home_conflicts(data, expected_home=profile_dir):
            print(
                f"✗ Refusing to stop PID {pid}: its recorded HERMES_HOME "
                f"belongs to a different profile than {profile_dir} "
                "(stale/poisoned PID record, #89315)."
            )
            return
        pid = get_running_pid(pid_file, cleanup_stale=False)
        if pid is None:
            return
        expected_start_time = get_process_start_time(pid)
        # Route through terminate_pid so Windows uses the appropriate
        # primitive (taskkill / TerminateProcess) — raw os.kill with
        # _signal.SIGKILL raises AttributeError at import time on Windows,
        # and raw os.kill with SIGTERM doesn't cascade to child processes
        # the same way taskkill /T does.
        _terminate_pid(pid)  # graceful first
        # On Windows os.kill(pid, 0) is NOT a no-op: liveness is the runtime-locked record.
        if _wait_then_force_kill(
            [pid],
            alive=lambda p: get_running_pid(pid_file, cleanup_stale=False) == p,
            start_times={pid: expected_start_time},
        ):
            print(f"✓ Gateway stopped (PID {pid})")
        else:
            print(f"✓ Gateway force-stopped (PID {pid})")
    except (ProcessLookupError, PermissionError):
        print("✓ Gateway already stopped")
    except Exception as e:
        print(f"⚠ Could not stop gateway: {e}")
