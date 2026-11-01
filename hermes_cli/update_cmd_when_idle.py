"""``hermes update --when-idle``: update only once the gateway has nothing in flight.

A scheduled or unattended update that starts while a chat turn, cron run or subagent is live makes
the gateway refuse new work for up to its drain cap. ``--when-idle`` resolves the target first
(nothing to do → exit 0, no backup, no restart), then polls the gateway's ``active-work`` verb
through :func:`hermes_cli.web_server_idle_proof.idle_proof` until two polls in a row prove it idle,
and only then runs the normal update. Past the bound it gives up without touching anything.

A target that failed under ``--when-idle`` is recorded and skipped by later ``--when-idle`` runs
until a newer target appears; a plain ``hermes update`` (a person asking) always tries it, and
succeeding moves HEAD there, which retires the record.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Callable, Optional

#: Exit status for a run that gave up waiting (``EX_TEMPFAIL``): a scheduler retries later.
WHEN_IDLE_TIMEOUT_EXIT = 75
DEFAULT_MAX_WAIT_S = 3600.0
POLL_S = 5.0
_UNITS = {"s": 1, "m": 60, "h": 3600}
#: Flags whose run reads, prints or records something and exits instead of installing.
_NOT_AN_UPDATE = {"--check": "check", "--plan": "plan", "--list-venv-holders": "list_venv_holders",
                  "--set-channel": "set_channel", "--install-id": "install_id"}
#: Monotonic end of the running ``--when-idle`` bound while its update runs: the commit-point pause
#: asks each gateway to pause only if idle (``pause-if-idle``) and keeps waiting until then.
_pause_deadline: Optional[float] = None
_refused_busy = False
BUSY_AT_PAUSE = ("the gateway took new work just before the update could pause it and stayed busy past "
                 "the --when-idle bound; the checkout was not changed")
_WORK_NOUNS = {"chat": "chat turn", "cron": "cron run", "delegation": "subagent", "api": "API run",
               "managed": "managed turn", "side_agent": "side agent", "deferred": "deferred task"}


def parse_max_wait(value: str) -> float:
    """``90``/``90s``, ``30m``, ``2h`` → seconds (argparse ``type``)."""
    import argparse
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(value).lower())
    if not match or float(match.group(1)) <= 0:
        raise argparse.ArgumentTypeError(f"--when-idle expects a duration like 90s, 30m or 2h, not {value!r}")
    return float(match.group(1)) * _UNITS[match.group(2) or "s"]


def declined_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "logs" / "update_receipts" / "declined_target.json"


def read_declined() -> Optional[dict]:
    try:
        data = json.loads(declined_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("target"), str) else None


def record_declined(target: str, label: str, exit_code: int) -> None:
    from utils import atomic_json_write
    path = declined_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {"target": target, "label": label, "exit_code": exit_code,
                             "failed_at": datetime.now(UTC).isoformat()})


def _work_summary(units: list) -> str:
    kinds: dict[str, int] = {}
    for unit in units:
        kind = str(unit.get("kind") or "work") if isinstance(unit, dict) else "work"
        kinds[kind] = kinds.get(kind, 0) + 1
    return ", ".join(f"{n} {_WORK_NOUNS.get(k, k)}{'' if n == 1 else 's'}" for k, n in sorted(kinds.items()))


def gateway_busy() -> Optional[str]:
    """:func:`idle_proof`'s turn probe over the control socket: ``""`` when no running gateway holds
    work (or none runs), a summary like ``"1 chat turn, 1 cron run"`` when one does, ``None`` when a
    gateway runs but did not answer ``identify`` or ``active-work`` (cannot prove idle)."""
    from gateway.control_socket import identify_gateway, query_gateway_control
    from gateway.status import live_gateway_pid_for_home
    from hermes_cli.update_receipt import _profile_homes

    seen: set = set()
    units: list = []
    for _profile, home in _profile_homes():
        identity = identify_gateway(home, timeout=2.0)
        if not identity:
            # No answer is also a timeout, or a gateway without the socket: a live owner whose
            # work cannot be read is not idle.
            if live_gateway_pid_for_home(home) is not None:
                return None
            continue
        if identity.get("pid") in seen:
            continue
        seen.add(identity.get("pid"))
        answer = query_gateway_control(home, "active-work", timeout=8.0)
        if answer is None:
            return None
        units += list(answer.get("active_work") or [])
        if not answer.get("active_work") and answer.get("active_agents"):
            units += [{"kind": "work"}] * int(answer["active_agents"])
    return _work_summary(units)


def wait_until_idle(max_wait_s: float, *, probe: Callable[[], Optional[str]] = gateway_busy,
                    poll_s: float = POLL_S, clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep) -> bool:
    """True once two consecutive polls prove nothing is in flight; False past *max_wait_s*.

    Two polls, because a turn that ends as the next one is admitted would read idle once."""
    from hermes_cli.web_server_idle_proof import idle_proof

    deadline, idle_polls, last = clock() + max_wait_s, 0, None
    while True:
        # Parked approvals/clarifies are part of the turn holding them, so the gateway's work list is
        # the whole answer: no separate human-input ledger to consult from this process.
        verdict = idle_proof(turn_probe=probe, input_probe=lambda: 0)
        idle_polls = idle_polls + 1 if verdict["idle"] is True else 0
        if clock() >= deadline:
            return False  # a proof that lands at or past the bound (slow probes) is too late to act on
        if idle_polls >= 2:
            return True
        status = verdict.get("detail") or verdict.get("reason")
        if verdict["idle"] is not True and status != last:
            left = max(0, int(deadline - clock()))
            what = "the gateway to answer" if verdict["idle"] is None else str(status)
            print(f"→ Waiting for {what} before updating ({left // 60}m{left % 60:02d}s left)...", flush=True)
            last = status
        sleep(min(poll_s if verdict["idle"] is not True else min(poll_s, 1.0), deadline - clock()))


def pause_must_find_idle() -> bool:
    """True while a ``--when-idle`` update runs: its pause may only take an idle gateway."""
    return _pause_deadline is not None


def wait_again(busy: str) -> bool:
    """The gateway refused ``pause-if-idle`` (*busy*): wait for idle again within the bound's rest."""
    global _refused_busy
    print(f"→ The gateway took new work before the pause ({busy or 'work in flight'}); waiting for it...", flush=True)
    if wait_until_idle(max(0.0, (_pause_deadline or 0.0) - time.monotonic()), probe=gateway_busy):
        return True
    _refused_busy = True
    return False


def _resolve_target(args) -> Optional[tuple[str, str]]:
    """``(target_sha, label)`` the update would move to, or None when HEAD already has it."""
    from hermes_cli import update_cmd as uc
    from hermes_cli import update_cmd_check as check
    from hermes_cli.main import PROJECT_ROOT as root, _resolve_update_branch

    git_cmd = uc._base_git_cmd()
    uc._map_ssl_cert_file_for_git(git_cmd)
    head = uc._capture_head_sha(git_cmd, root)
    branch = _resolve_update_branch(args)
    if not getattr(args, "branch", None):
        from hermes_cli.config import get_config_path, require_readable_config_before_write
        from hermes_cli.source_releases import resolve_source_target
        from hermes_cli.update_channel import channel_record, rides_default_channel
        selected = uc._source_update_channel(args)
        record = channel_record(require_readable_config_before_write(get_config_path()), root)
        target = resolve_source_target(selected, git_cmd, root,
                                       forward_only=rides_default_channel(record, selected, root))
        if target.commit:
            return None if target.ahead or target.commit == head else (target.commit, target.label)
        branch = target.branch or branch
    shallow = check.is_shallow_repository(git_cmd, root)
    fetched, compare = check.fetch_compare_branch(git_cmd, root, branch, ["--depth", "1"] if shallow else [])
    if fetched.returncode != 0:
        raise OSError(f"could not fetch {branch}: {(fetched.stderr or '').strip()[:200]}")
    sha = check._git(git_cmd, root, ["rev-parse", compare]).stdout.strip()
    if not sha or sha == head or (not shallow and check._git(
            git_cmd, root, ["merge-base", "--is-ancestor", sha, "HEAD"]).returncode == 0):
        return None
    return sha, f"{compare} {sha[:12]}"


def _project_root() -> Path:
    from hermes_cli.main import PROJECT_ROOT
    return Path(PROJECT_ROOT)


def run_when_idle(args, cmd_update: Callable) -> None:
    """``hermes update --when-idle``: resolve, skip a declined target, wait for idle, then update."""
    if getattr(args, "gateway", False):
        print("✗ --when-idle cannot run from a chat /update: that turn is the work it would wait for.")
        sys.exit(2)
    if clash := [flag for flag, dest in _NOT_AN_UPDATE.items() if getattr(args, dest, None)]:
        print(f"✗ --when-idle waits to install an update; {clash[0]} does not install one. Run it without --when-idle.")
        sys.exit(2)
    if not (_project_root() / ".git").exists():
        return cmd_update(args)  # package/zip installs: the normal path refuses or routes them
    try:
        resolved = _resolve_target(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"✗ Could not resolve the update target: {exc}. Nothing was changed.")
        sys.exit(1)
    if resolved is None:
        from hermes_cli.update_cmd_commit import _obligation_paths
        if not any(path.exists() for path in _obligation_paths(_project_root())):
            print("✓ Already up to date; nothing to wait for.")
            return None
        # An earlier update moved HEAD but still owes its completion or restart: the normal path
        # repairs that, once nothing is in flight.
        resolved = ("", "the completion an earlier update still owes")
    target, label = resolved
    declined = read_declined()
    if target and declined and declined["target"] == target:
        print(f"→ Skipping {label}: an unattended update to it failed at {declined.get('failed_at', '?')}.")
        print("  It is retried automatically once a newer version ships; run `hermes update` to retry it now.")
        return None
    max_wait = float(args.when_idle)
    deadline = time.monotonic() + max_wait
    print(f"→ Update available: {label}. Waiting for the gateway to go idle (up to {int(max_wait // 60)}m).")
    if not wait_until_idle(max_wait):
        _give_up(max_wait)
    print("✓ Idle. Updating now.")
    _apply_when_idle(args, cmd_update, target, label, deadline, max_wait)


def _give_up(max_wait: float) -> None:
    print(f"✗ Still busy after {int(max_wait // 60)}m; not updating. Nothing was changed.")
    print("  Run `hermes update` to update now (in-flight turns finish first).")
    sys.exit(WHEN_IDLE_TIMEOUT_EXIT)


def _apply_when_idle(args, cmd_update: Callable, target: str, label: str, deadline: float, max_wait: float) -> None:
    """Run the update with its commit-point pause bound to idle gateways until *deadline*."""
    global _pause_deadline, _refused_busy
    from hermes_cli.update_cmd_commit import armed_move_targets
    already_armed = len(armed_move_targets())
    _pause_deadline, _refused_busy = deadline, False
    try:
        cmd_update(args)
    except SystemExit as exc:
        if _refused_busy:
            _give_up(max_wait)  # the pause refused before any checkout move: retryable, nothing declined
        code = exc.code if isinstance(exc.code, int) else 1
        # Only a run that reached a checkout move tried a release; a refusal before it (another
        # updater holds the lock, a fetch failed) stays retryable. The commit it moved to is the one
        # declined: the update resolves its own target, which may be newer than the one waited for.
        attempted = armed_move_targets()[already_armed:]
        if code and attempted:
            tried = attempted[-1]
            record_declined(tried, label if tried == target else tried[:12], code)
            print(f"  Recorded {label if tried == target else tried[:12]} as failed: "
                  "later --when-idle runs skip it until a newer version ships.")
        raise
    finally:
        _pause_deadline = None
