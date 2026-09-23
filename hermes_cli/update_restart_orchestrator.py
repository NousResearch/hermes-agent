"""Deterministic post-update gateway restart orchestration.

Design: ``build/gateway_restart/POST-UPDATE-GATEWAY-RESTART-DESIGN.md`` — an audit of two real
updates on this host on 2026-09-23. The restart machinery already exists
(``hermes_cli/update_cmd_fleet.py``, ``update_host_obligation.py``, ``gateway_launchd.py``); what
was missing is ordering, single-writer discipline, and a health predicate that includes the code
generation. This module owns exactly those, one gap per section:

* **G1/G9 — two actors, two signals, no attribution.** ``host-restart-lease.json`` in
  :func:`gateway.host_rendezvous.host_state_dir` (beside the host obligation) is the single-writer
  gate: it records the requesting actor (pid/argv/cwd/HERMES_HOME/trigger) and the drain deadline
  it opened, so a second actor's SIGTERM is *refused and named* instead of amputating the first
  actor's drain (today's 19:28:46 SIGTERM / 19:28:53 SIGUSR1 overlap).
* **G2 — verification windows smaller than the plist's respawn floor.** :func:`restart_budgets`
  derives the fresh-pid window from the *live* plist (``ThrottleInterval + 15``, floor 45 s), so a
  successor that appears at 31 s is waited for instead of declared ``stale``.
* **G3 — contradictory bookkeeping.** :func:`fleet_verdict` takes the fleet matrix's own predicate
  as its authority, so ``gateway_restart.incomplete`` and the printed row states cannot disagree
  (receipt B carried ``incomplete=false`` next to a ``stale`` row).
* **G5 — the served code generation is a conjunct.** :func:`health_predicate` adds
  ``served code_sha == pulled sha`` to launchd liveness + child command + heartbeat + arbiter.
  A gateway serving pre-update ``sys.modules`` is then not "healthy" by construction.
* **G6 — the drain budget is honest.** ``min(configured, live ExitTimeOut − 5)``: on this host the
  CLI's 1815 s is fiction under launchd's 60 s SIGKILL, so the budget is 55 s and both are logged.
* **G8 — idempotency keys.** :func:`restart_action_key` / :func:`lease_request_is_stale`: a repeat
  of the same ``(sha, runtime-identity)`` is a logged no-op, and a request for an older sha hands
  the fleet to the later update instead of restarting onto older code.

S8 catch-up is *scheduled* here (:func:`schedule_catch_up`) and :func:`run_due_catch_up` is the
self-firing entry point, but nothing yet calls it on a timer: whether the liveness watchdog may
*act* on staleness or only alert is an open decision (§5 item 2 of the design). Until that is
answered an armed obligation is still consumed by the next ``hermes update`` through the existing
``update_cmd_fleet._apply_pending_fleet_restart_catchup``.

Premise correction against live evidence: §3.1 of the design cited ``host-gateway.json.code_sha``
as the served generation. That record publishes role/home/pid/profiles only — no ``code_sha``
(verified live 2026-09-23). The served generation is stamped into the launching home's
``gateway_state.json`` by :func:`gateway.status._get_code_identity_fields`, which is the same
source the fleet matrix reads, so :func:`served_code_sha` resolves it there (falling back through
the host record's ``home`` to this profile's ``HERMES_HOME``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger("hermes_cli.update_cmd")

#: Single-writer gate for "this host is being restarted", beside ``host-update-restart.json``.
LEASE_NAME = "host-restart-lease.json"

#: §3.2 — how long a second actor waits on a held lease before reporting ``already-in-progress``.
LEASE_WAIT_S = 120.0
LEASE_POLL_S = 0.5
#: A lease whose holder stopped heartbeating is stale and may be taken over (total hard cap).
LEASE_TTL_S = 900.0

#: §3.2 — fresh-pid wait = ``ThrottleInterval + margin``, never under this floor.
FRESH_PID_MARGIN_S = 15.0
FRESH_PID_FLOOR_S = 45.0
#: launchd escalates SIGTERM → SIGKILL at its ``ExitTimeOut``; keep this much headroom inside it.
DRAIN_RESERVE_S = 5.0
DRAIN_FLOOR_S = 5.0
DRAIN_CEILING_S = 1815.0
#: §3.1 S5 — two 30 s heartbeat beats.
HEARTBEAT_MAX_AGE_S = 60.0
#: §3.1 S6 — fleet-matrix settle window before a verdict is computed.
FLEET_VERIFY_TIMEOUT_S = 90.0
FLEET_VERIFY_POLL_S = 5.0
#: §3.1 S8 — catch-up attempts, then escalation.
CATCH_UP_DELAYS_S = (120.0, 600.0, 1800.0)

#: The fleet matrix's own failure predicate (``update_receipt.print_fleet_version_matrix``).
FLEET_FAILING_STATES = ("stale", "down")
#: A gateway on the old code only because this updater runs inside it; expected, not a failure.
FLEET_OK_STATES = ("current", "restart_pending")

_INT_RE = re.compile(r"<key>\s*([A-Za-z]+)\s*</key>\s*<integer>\s*(-?\d+)\s*</integer>")


# --------------------------------------------------------------------------- #
# Actor identity (G9)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Requestor:
    """Who asked for the restart: the durable attribution the 19:27 update lacked."""

    pid: int
    argv: tuple[str, ...]
    cwd: str
    hermes_home: str
    trigger: str

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "hermes_home": self.hermes_home,
            "trigger": self.trigger,
        }

    def describe(self) -> str:
        argv = " ".join(self.argv) or "?"
        return f"pid={self.pid} trigger={self.trigger or '?'} argv=[{argv}] cwd={self.cwd or '?'}"


def current_requestor(trigger: str = "", *, pid: Optional[int] = None) -> Requestor:
    """This process's identity, as recorded in the lease. Never raises."""
    home = ""
    with suppress(Exception):
        from hermes_cli.update_cmd import get_hermes_home

        home = str(get_hermes_home())
    cwd = ""
    with suppress(Exception):
        cwd = os.getcwd()
    return Requestor(
        pid=int(pid if pid is not None else os.getpid()),
        argv=tuple(str(part) for part in sys.argv),
        cwd=cwd,
        hermes_home=home,
        trigger=str(trigger or ""),
    )


# --------------------------------------------------------------------------- #
# Supervisor-derived budgets (G2, G6)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RestartBudgets:
    """Waits derived from the live supervisor contract, not from constants.

    ``throttle_interval_s`` / ``exit_timeout_s`` are ``None`` when no launchd plist was readable —
    the configured drain then stands untouched (fail-open: an unreadable plist never shortens or
    extends a budget on a host that has no launchd job).
    """

    drain_s: float
    fresh_pid_s: float
    throttle_interval_s: Optional[float] = None
    exit_timeout_s: Optional[float] = None

    def describe(self) -> str:
        return (
            f"drain={self.drain_s:.0f}s fresh_pid={self.fresh_pid_s:.0f}s "
            f"throttle={_fmt(self.throttle_interval_s)} exit_timeout={_fmt(self.exit_timeout_s)}"
        )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}s"


def launchd_plist_path() -> Optional[Path]:
    """Installed gateway plist for THIS profile's label, or ``None`` off macOS. Never raises."""
    with suppress(Exception):
        from hermes_cli.gateway import get_launchd_plist_path

        return Path(get_launchd_plist_path())
    return None


def launchd_plist_paths() -> list[Path]:
    """Every plist that could supervise this host's gateway, most specific first.

    The active profile's derived label is not the whole story: a multiplexed host runs ONE gateway
    installed under the DEFAULT profile's label (``ai.hermes.gateway`` on this box) while a named
    profile's CLI derives ``ai.hermes.gateway-<profile>`` — a path that does not exist. Budgets must
    come from the plist launchd will actually enforce, so the install's other gateway labels are
    candidates too.
    """
    candidates: list[Path] = []
    derived = launchd_plist_path()
    if derived is not None:
        candidates.append(derived)
    with suppress(Exception):
        from hermes_cli.gateway import launchd_gateway_labels_for_install

        agents = Path.home() / "Library" / "LaunchAgents"
        candidates.extend(agents / f"{label}.plist" for label in launchd_gateway_labels_for_install())
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def launchd_plist_limits(path: Optional[Path] = None) -> dict[str, float]:
    """``{"ThrottleInterval": 30.0, "ExitTimeOut": 60.0}`` read from the installed plist(s).

    With no explicit ``path`` every candidate is read and the values are combined to the TIGHTEST
    contract (smallest ``ExitTimeOut``, largest ``ThrottleInterval``): any of those jobs may be the
    one supervising the gateway being restarted, and a budget must fit the fastest trap.

    Fail-open by contract: an absent/unreadable/unparseable plist contributes nothing, and no
    readable plist at all yields ``{}`` — every derived budget then falls back to its floor, never
    to a guess about a supervisor that may not exist.
    """
    targets = [path] if path is not None else launchd_plist_paths()
    limits: dict[str, float] = {}
    for target in targets:
        if target is None:
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for key, value in _INT_RE.findall(text):
            if key not in ("ThrottleInterval", "ExitTimeOut"):
                continue
            with suppress(Exception):
                number = float(int(value))
                if key == "ExitTimeOut":
                    limits[key] = min(limits.get(key, number), number)
                else:
                    limits[key] = max(limits.get(key, number), number)
    return limits


def restart_budgets(
    *,
    configured_drain_s: Optional[float] = None,
    plist_path: Optional[Path] = None,
    plist_limits: Optional[dict[str, float]] = None,
) -> RestartBudgets:
    """Drain + fresh-pid waits derived from the live plist (§3.2, closes G2/G6).

    ``configured_drain_s`` is the CLI's own budget (``_get_restart_exit_wait_budget()``, already
    floored at 45 s by its caller). Under launchd it is capped at ``live ExitTimeOut − 5`` because
    launchd SIGKILLs at ``ExitTimeOut`` regardless: waiting 1815 s for a process the supervisor
    kills at 60 s is not a budget, it is a stall.
    """
    configured = float(configured_drain_s) if configured_drain_s is not None else 45.0
    configured = max(configured, DRAIN_FLOOR_S)
    limits = launchd_plist_limits(plist_path) if plist_limits is None else dict(plist_limits)
    exit_timeout = limits.get("ExitTimeOut")
    throttle = limits.get("ThrottleInterval")

    drain = min(configured, DRAIN_CEILING_S)
    if exit_timeout is not None and exit_timeout > 0:
        drain = min(drain, max(exit_timeout - DRAIN_RESERVE_S, DRAIN_FLOOR_S))

    if throttle is not None and throttle >= 0:
        fresh_pid = max(throttle + FRESH_PID_MARGIN_S, FRESH_PID_FLOOR_S)
    else:
        fresh_pid = FRESH_PID_FLOOR_S
    return RestartBudgets(
        drain_s=drain, fresh_pid_s=fresh_pid, throttle_interval_s=throttle, exit_timeout_s=exit_timeout
    )


# --------------------------------------------------------------------------- #
# Restart lease: single writer, attribution, idempotency (G1, G8, G9)
# --------------------------------------------------------------------------- #

def lease_path() -> Optional[Path]:
    """``<host_state_dir>/host-restart-lease.json``, or ``None`` when unresolvable."""
    with suppress(Exception):
        from gateway.host_rendezvous import host_state_dir

        return host_state_dir() / LEASE_NAME
    return None


def read_restart_lease() -> Optional[dict]:
    """The published lease, or ``None`` when absent/corrupt/foreign-versioned."""
    path = lease_path()
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    return payload


def _pid_alive(pid: int) -> bool:
    """Liveness for a lease holder. Unprovable liveness counts as alive (a lease is never stolen
    on a probe failure — the same fail-closed rule the systemd unit collapsing applies to identity).
    """
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid))
    except Exception:
        return True


def lease_holder_pid(payload: Optional[dict]) -> int:
    if not isinstance(payload, dict):
        return 0
    requestor = payload.get("requestor")
    pid = requestor.get("pid") if isinstance(requestor, dict) else payload.get("pid")
    try:
        return int(pid or 0)
    except (TypeError, ValueError):
        return 0


def lease_is_stale(
    payload: Optional[dict], *, alive: Callable[[int], bool] = _pid_alive, now: Optional[float] = None
) -> bool:
    """True when the lease may be taken over: holder gone, or TTL passed."""
    if not isinstance(payload, dict):
        return True
    moment = float(now if now is not None else time.time())
    expires = payload.get("expires_at")
    if isinstance(expires, (int, float)) and moment >= float(expires):
        return True
    return not alive(lease_holder_pid(payload))


def restart_in_progress(
    *, alive: Callable[[int], bool] = _pid_alive, now: Optional[float] = None
) -> bool:
    """True while a live actor holds the restart lease (§3.1, suppresses the false boot warning)."""
    payload = read_restart_lease()
    return bool(payload) and not lease_is_stale(payload, alive=alive, now=now)


def restart_action_key(sha: str, runtime_ids: Sequence[str] = ()) -> str:
    """Idempotency key for one restart action: the pulled code × the runtimes it may touch (G8)."""
    runtimes = ",".join(sorted(str(r) for r in runtime_ids if str(r)))
    return f"{sha or 'unknown'}::{runtimes or 'all'}"


def lease_request_is_stale(sha: str, *, checkout_sha: Optional[str] = None) -> bool:
    """True when ``sha`` is an ancestor of the checkout: a later update owns the fleet.

    §3.3 guarantee 2. Identity that cannot be proved (no checkout sha, no git history, an
    unrelated sha) is *not* stale — acting is the fail-closed side of a restart, and the older
    obligation is settled by ``update_cmd_fleet._marker_only_restart_obsolete`` instead.
    """
    if not sha or not checkout_sha or sha == checkout_sha:
        return False
    with suppress(Exception):
        from hermes_cli.update_cmd_fleet_checkout import checkout_contains

        return bool(checkout_contains(sha))
    return False


@dataclass(frozen=True)
class LeaseOutcome:
    """Result of :func:`acquire_restart_lease`.

    ``state``: ``acquired`` · ``takeover`` · ``held`` (another live actor owns it; caller must
    defer, exit 0) · ``stale-request`` (§3.3 #2) · ``unavailable`` (no writable host state dir).
    """

    state: str
    acquired: bool
    key: str
    waited_s: float = 0.0
    holder: Optional[dict] = None
    path: Optional[str] = None

    def describe(self) -> str:
        holder = (self.holder or {}).get("requestor") or {}
        who = f"{holder.get('trigger') or '?'} pid={holder.get('pid')}" if holder else "unknown"
        return f"state={self.state} key={self.key} waited={self.waited_s:.1f}s holder={who}"


def _write_lease(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from utils import atomic_json_write

        atomic_json_write(path, payload, mode=0o600)
        return True
    except Exception as exc:  # defensive: a lease write failure must never abort the updater
        logger.debug("Could not write restart lease %s: %s", path, exc)
        return False


def acquire_restart_lease(
    *,
    sha: str,
    runtime_ids: Sequence[str] = (),
    trigger: str = "",
    requestor: Optional[Requestor] = None,
    checkout_sha: Optional[str] = None,
    wait_s: Optional[float] = None,
    poll_s: float = LEASE_POLL_S,
    alive: Callable[[int], bool] = _pid_alive,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> LeaseOutcome:
    """Take the host restart lease, waiting up to ``wait_s`` for a live holder (§3.1 S0).

    A held lease is never broken: the second actor waits, then reports ``held`` so its caller can
    exit ``rc=0 already-in-progress`` — exactly the outcome today's race lacked, where the second
    actor SIGTERMed the gateway seven seconds before the first actor's drain signal landed.
    A stale holder (pid gone, or TTL passed) is taken over with a ``lease_takeover`` log.
    """
    # Late-bound so a caller (or a fixture) can shorten the wait; the design's 120 s stands as the
    # default. A bound default would freeze the constant at import time and force every test of the
    # deferral path to sleep the full two minutes.
    wait_s = LEASE_WAIT_S if wait_s is None else wait_s
    key = restart_action_key(sha, runtime_ids)
    path = lease_path()
    if path is None:
        return LeaseOutcome(state="unavailable", acquired=False, key=key)
    if lease_request_is_stale(sha, checkout_sha=checkout_sha):
        logger.info("restart_lease state=stale-request key=%s checkout=%s", key, (checkout_sha or "")[:10])
        return LeaseOutcome(state="stale-request", acquired=False, key=key)

    actor = requestor if requestor is not None else current_requestor(trigger)
    started = float(now())
    deadline = started + max(float(wait_s), 0.0)
    while True:
        payload = read_restart_lease()
        if payload is None or lease_is_stale(payload, alive=alive, now=now()):
            takeover = bool(payload)
            moment = float(now())
            body = {
                "version": 1,
                "key": key,
                "sha": sha or "",
                "runtime_ids": [str(r) for r in runtime_ids],
                "requestor": actor.to_json(),
                "state": "draining",
                "acquired_at": moment,
                "expires_at": moment + LEASE_TTL_S,
                "drain": {},
                "restarts": [],
            }
            if not _write_lease(path, body):
                return LeaseOutcome(state="unavailable", acquired=False, key=key, path=str(path))
            if takeover:
                logger.warning(
                    "restart_lease_takeover key=%s actor=[%s] previous=%s",
                    key, actor.describe(), json.dumps(payload.get("requestor") or {}),
                )
            logger.info("restart_lease state=%s key=%s actor=[%s]",
                        "takeover" if takeover else "acquired", key, actor.describe())
            return LeaseOutcome(
                state="takeover" if takeover else "acquired", acquired=True, key=key,
                waited_s=float(now()) - started, path=str(path),
            )
        if float(now()) >= deadline:
            logger.warning(
                "restart_lease state=held key=%s actor=[%s] holder=%s waited=%.1fs — deferring",
                key, actor.describe(), json.dumps(payload.get("requestor") or {}), float(now()) - started,
            )
            return LeaseOutcome(
                state="held", acquired=False, key=key, waited_s=float(now()) - started,
                holder=payload, path=str(path),
            )
        sleep(min(float(poll_s), max(deadline - float(now()), 0.01)))


def release_restart_lease(*, key: Optional[str] = None) -> bool:
    """Drop the lease. ``key`` guards against releasing a successor's lease. Never raises."""
    path = lease_path()
    if path is None:
        return False
    payload = read_restart_lease()
    if key is not None and isinstance(payload, dict) and payload.get("key") != key:
        logger.debug("restart_lease release skipped: key mismatch (%s)", key)
        return False
    try:
        path.unlink(missing_ok=True)
        logger.info("restart_lease released key=%s", key or "any")
        return True
    except OSError as exc:
        logger.debug("Could not release restart lease: %s", exc)
        return False


def amend_restart_lease(**fields: Any) -> Optional[dict]:
    """Merge ``fields`` into the held lease (drain deadline, restart records). Never raises."""
    path = lease_path()
    payload = read_restart_lease()
    if path is None or payload is None:
        return None
    payload.update(fields)
    if not _write_lease(path, payload):
        return None
    return payload


# --------------------------------------------------------------------------- #
# One signal path + arbitration (G1)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SignalOutcome:
    """What a signal request did: ``sent`` / ``refused``, to whom, and under whose authority."""

    sent: bool
    signal_name: str
    pid: int
    actor: str
    reason: str
    deadline_ts: Optional[float] = None
    exited: Optional[bool] = None

    def describe(self) -> str:
        return (
            f"signal={self.signal_name} pid={self.pid} actor={self.actor} "
            f"reason={self.reason} sent={self.sent}"
        )


def mark_drain_started(
    pid: int, *, deadline_ts: float, actor: str, label: str = "", key: Optional[str] = None
) -> None:
    """Record the open drain deadline in the lease so arbitration can defend it (G1).

    Kept per-pid (``drains``) *and* as the most recent single entry (``drain``): one update drains
    several pids in sequence (every manual profile gateway), and defending only the last one would
    let a second actor amputate the earlier drains.
    """
    payload = read_restart_lease() or {}
    drains = payload.get("drains")
    if not isinstance(drains, dict):
        drains = {}
    entry = {"pid": int(pid), "deadline_ts": float(deadline_ts), "actor": actor,
             "label": label, "at": time.time()}
    drains[str(int(pid))] = entry
    amend_restart_lease(drain=entry, drains=drains)
    logger.info(
        "restart_drain_open pid=%s actor=%s label=%s deadline=%.1f budget=%.0fs",
        pid, actor, label or "?", deadline_ts, max(deadline_ts - time.time(), 0.0),
    )


def open_drain_for(pid: int, payload: Optional[dict] = None) -> Optional[dict]:
    """The open drain entry for ``pid``, or ``None``. Never raises."""
    lease = payload if payload is not None else (read_restart_lease() or {})
    if not isinstance(lease, dict):
        return None
    try:
        target = int(pid)
    except (TypeError, ValueError):
        return None
    drains = lease.get("drains")
    if isinstance(drains, dict):
        entry = drains.get(str(target))
        if isinstance(entry, dict):
            return entry
    candidate = lease.get("drain")
    if isinstance(candidate, dict):
        try:
            if int(candidate.get("pid") or 0) == target:
                return candidate
        except (TypeError, ValueError):
            return None
    return None


def restart_signal_gate(
    pid: int,
    signal_name: str,
    *,
    actor: str,
    payload: Optional[dict] = None,
    now: Optional[float] = None,
) -> SignalOutcome:
    """Arbitrate a signal request against the open drain (§3.1 S3).

    SIGUSR1 is the restart signal; a destructive SIGTERM/SIGKILL for a pid whose graceful drain is
    still inside its deadline is **refused**, and the offending actor is named in one structured
    line. That is the fix for the 19:28:46 SIGTERM: the second actor no longer shortens the first
    actor's deferral, and the durable record says who tried.
    """
    moment = float(now if now is not None else time.time())
    lease = payload if payload is not None else (read_restart_lease() or {})
    if str(signal_name).upper() in ("SIGTERM", "SIGKILL"):
        drain = open_drain_for(pid, lease)
        deadline = (drain or {}).get("deadline_ts")
        if isinstance(deadline, (int, float)) and moment < float(deadline):
            logger.warning(
                "restart_signal_refused signal=%s pid=%s actor=%s reason=drain-deadline-not-reached "
                "deadline=%.1f remaining=%.0fs drain_actor=%s",
                signal_name, pid, actor, float(deadline), float(deadline) - moment,
                (drain or {}).get("actor") or "?",
            )
            return SignalOutcome(
                sent=False, signal_name=str(signal_name), pid=int(pid), actor=actor,
                reason="drain-deadline-not-reached", deadline_ts=float(deadline),
            )
    logger.info("restart_signal signal=%s pid=%s actor=%s reason=%s",
                signal_name, pid, actor, "deadline-reached-or-no-drain")
    return SignalOutcome(
        sent=True, signal_name=str(signal_name), pid=int(pid), actor=actor, reason="allowed"
    )


def _send_signal(pid: int, signum: int) -> None:
    """The single OS signal seam (patchable in fixtures; never a bound default).

    A default argument would capture ``os.kill`` at import time, so a fixture could not prove
    "exactly one signal was sent" without a real gateway on the other end — and a global
    ``os.kill`` patch would also intercept every liveness probe.
    """
    os.kill(pid, signum)


def request_graceful_restart(
    pid: int,
    *,
    budget_s: float,
    actor: str,
    label: str = "",
    key: Optional[str] = None,
    kill: Optional[Callable[[int, int], None]] = None,
    wait_exit: Optional[Callable[..., bool]] = None,
    wait_for_exit: bool = False,
    signal_name: str = "SIGUSR1",
) -> SignalOutcome:
    """The single restart path: SIGUSR1, once, from the lease holder, deadline recorded first (§3.1 S3).

    ``kill`` defaults to the live ``os.kill`` *at call time* (not as a bound default), so the signal
    seam stays patchable — tests and fixtures must be able to prove "exactly one signal" without a
    real gateway on the other end.

    ``wait_exit`` is the poll used when ``wait_for_exit`` is set; the CLI's
    ``gateway._graceful_restart_via_sigusr1`` already owns that wait, so callers that route
    through it leave ``wait_for_exit`` off and let it own the poll.
    """
    send = kill if kill is not None else os.kill
    if not hasattr(signal, signal_name):
        return SignalOutcome(sent=False, signal_name=signal_name, pid=int(pid), actor=actor,
                             reason="signal-unsupported")
    if int(pid) <= 0:
        return SignalOutcome(sent=False, signal_name=signal_name, pid=int(pid), actor=actor,
                             reason="no-pid")
    send = kill if kill is not None else _send_signal
    gate = restart_signal_gate(pid, signal_name, actor=actor)
    if not gate.sent:
        return gate
    try:
        send(int(pid), getattr(signal, signal_name))
    except ProcessLookupError:
        return SignalOutcome(sent=True, signal_name=signal_name, pid=int(pid), actor=actor,
                             reason="already-exited", exited=True)
    except (PermissionError, OSError) as exc:
        logger.error("restart_signal_failed signal=%s pid=%s actor=%s error=%s",
                     signal_name, pid, actor, exc)
        return SignalOutcome(sent=False, signal_name=signal_name, pid=int(pid), actor=actor,
                             reason=f"send-failed:{exc}")
    deadline = time.time() + max(float(budget_s), 0.0)
    mark_drain_started(pid, deadline_ts=deadline, actor=actor, label=label, key=key)
    exited: Optional[bool] = None
    if wait_for_exit and wait_exit is not None:
        with suppress(Exception):
            exited = bool(wait_exit(int(pid), max(float(budget_s), 1.0)))
    return SignalOutcome(sent=True, signal_name=signal_name, pid=int(pid), actor=actor,
                         reason="sent", deadline_ts=deadline, exited=exited)


def escalate_to_sigterm(
    pid: int,
    *,
    actor: str,
    reason: str,
    payload: Optional[dict] = None,
    kill: Optional[Callable[[int, int], None]] = None,
) -> SignalOutcome:
    """Post-grace escalation. Only past the drain deadline (or a wedged-loop verdict) (§3.1 S3)."""
    send = kill if kill is not None else _send_signal
    gate = restart_signal_gate(pid, "SIGTERM", actor=actor, payload=payload)
    if not gate.sent:
        return gate
    try:
        send(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return SignalOutcome(sent=True, signal_name="SIGTERM", pid=int(pid), actor=actor,
                             reason="already-exited", exited=True)
    except (PermissionError, OSError) as exc:
        return SignalOutcome(sent=False, signal_name="SIGTERM", pid=int(pid), actor=actor,
                             reason=f"send-failed:{exc}")
    logger.warning("restart_signal_escalated signal=SIGTERM pid=%s actor=%s reason=%s",
                   pid, actor, reason)
    return SignalOutcome(sent=True, signal_name="SIGTERM", pid=int(pid), actor=actor,
                         reason=str(reason or "post-drain"))


def record_restart_action(*, key: str, sha: str, runtime_ids: Sequence[str], verdict: str) -> None:
    """Append the completed restart to the lease (idempotency evidence, G8)."""
    payload = read_restart_lease()
    if payload is None:
        return
    actions = payload.get("restarts")
    if not isinstance(actions, list):
        actions = []
    actions.append({
        "key": restart_action_key(sha, runtime_ids), "sha": sha, "verdict": verdict,
        "at": time.time(),
    })
    amend_restart_lease(restarts=actions[-10:])


# --------------------------------------------------------------------------- #
# Health predicate: the served code generation is a conjunct (G5)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HealthVerdict:
    """One predicate, one line: which conjunct failed, and what the gateway is actually serving."""

    healthy: bool
    sha: str
    served_sha: str = ""
    failures: tuple[str, ...] = ()

    def describe(self) -> str:
        detail = ",".join(self.failures) if self.failures else "all-conjuncts-ok"
        return (
            f"gateway_health healthy={str(self.healthy).lower()} expected_sha={(self.sha or '?')[:10]} "
            f"served_sha={(self.served_sha or '?')[:10]} failures={detail}"
        )


def host_gateway_home() -> Optional[Path]:
    """HERMES_HOME of the host gateway: the host record's ``home``, else this profile's."""
    with suppress(Exception):
        from gateway.host_rendezvous import read_record

        record = read_record("gateway")
        home = getattr(record, "home", "") if record is not None else ""
        if home:
            return Path(home)
    with suppress(Exception):
        from hermes_cli.update_cmd import get_hermes_home

        return Path(get_hermes_home())
    return None


def served_code_sha(home: Optional[Path] = None) -> Optional[str]:
    """Code generation the running gateway actually serves, or ``None`` when unpublished.

    Source is ``<home>/gateway_state.json`` — the same stamp the fleet matrix compares against
    (``gateway.status._get_code_identity_fields``), which is what makes this conjunct share the
    fleet's notion of "which code is running".
    """
    target = home if home is not None else host_gateway_home()
    if target is None:
        return None
    try:
        payload = json.loads((Path(target) / "gateway_state.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    sha = payload.get("code_sha")
    return str(sha) if sha else None


def gateway_heartbeat_age_s(
    home: Optional[Path] = None, *, pid: Optional[int] = None, now: Optional[float] = None
) -> Optional[float]:
    """Age of ``state/gateway.heartbeat`` for the running gateway, or ``None`` when unreadable."""
    target = home if home is not None else host_gateway_home()
    if target is None:
        return None
    try:
        payload = json.loads((Path(target) / "state" / "gateway.heartbeat").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if pid is not None and int(payload.get("pid") or 0) != int(pid):
        return None
    moment = float(now if now is not None else time.time())
    stamp = payload.get("updated_at")
    if isinstance(stamp, (int, float)):
        return max(moment - float(stamp), 0.0)
    if isinstance(stamp, str) and stamp:
        with suppress(Exception):
            from datetime import datetime, timezone

            parsed = datetime.fromisoformat(stamp)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(moment - parsed.timestamp(), 0.0)
    return None


def health_predicate(
    *,
    expected_sha: str,
    supervised_pid: Optional[int],
    child_matches: Optional[bool],
    served_sha: Optional[str] = None,
    heartbeat_age_s: Optional[float] = None,
    arbiter_supervised: Optional[bool] = None,
    max_heartbeat_age_s: float = HEARTBEAT_MAX_AGE_S,
) -> HealthVerdict:
    """READY predicate with the code generation as a conjunct (§3.1 S5).

    Any conjunct that cannot be evaluated is a FAILURE, named: an unprovable health state is not a
    healthy one, which is the boundary that let a pre-update ``sys.modules`` graph pass for healthy
    through both of today's updates.
    """
    failures: list[str] = []
    if not supervised_pid or int(supervised_pid) <= 0:
        failures.append("no-supervised-pid")
    if child_matches is None:
        failures.append("child-probe-unavailable")
    elif not child_matches:
        failures.append("child-not-gateway-run")
    if not expected_sha:
        failures.append("expected-sha-unknown")
    elif not served_sha:
        failures.append("served-sha-unpublished")
    elif str(served_sha) != str(expected_sha):
        failures.append("code-generation-stale")
    if heartbeat_age_s is None:
        failures.append("heartbeat-unreadable")
    elif float(heartbeat_age_s) > float(max_heartbeat_age_s):
        failures.append("heartbeat-stale")
    if arbiter_supervised is None:
        failures.append("arbiter-unavailable")
    elif not arbiter_supervised:
        failures.append("arbiter-not-supervised")
    return HealthVerdict(
        healthy=not failures, sha=str(expected_sha or ""), served_sha=str(served_sha or ""),
        failures=tuple(failures),
    )


# --------------------------------------------------------------------------- #
# One verdict (G3)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RestartVerdict:
    """The restart's single verdict, written once, derived from the fleet matrix's own predicate."""

    verdict: str
    incomplete: bool
    sha: str
    failing: tuple[str, ...] = ()
    rows: int = 0

    def as_receipt_fields(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "incomplete": self.incomplete,
            "expected_sha": self.sha,
            "matrix_rows": self.rows,
            "failing_states": list(self.failing),
        }


def fleet_verdict(
    rows: Optional[Sequence[dict]],
    expected_sha: str,
    *,
    rows_expected: bool,
    matrix_incomplete: Optional[bool] = None,
) -> RestartVerdict:
    """Compute THE verdict from the fleet matrix (S6).

    ``matrix_incomplete`` is ``print_fleet_version_matrix``'s own return value, passed in rather
    than re-derived so the printed rows and the receipt's ``incomplete`` flag cannot disagree —
    receipt B recorded ``incomplete=false`` beside a ``stale`` row, and every downstream reader
    that trusted the flag believed a broken restart succeeded.
    """
    fleet = list(rows or ())
    failing = tuple(sorted({str(row.get("state")) for row in fleet
                            if row.get("state") in FLEET_FAILING_STATES}))
    if not fleet:
        if rows_expected:
            return RestartVerdict(verdict="no-rows", incomplete=True, sha=str(expected_sha or ""),
                                  failing=("no-rows",), rows=0)
        return RestartVerdict(verdict="idle", incomplete=False, sha=str(expected_sha or ""), rows=0)
    if "stale" in failing:
        return RestartVerdict(verdict="stale", incomplete=True, sha=str(expected_sha or ""),
                              failing=failing, rows=len(fleet))
    if "down" in failing:
        return RestartVerdict(verdict="down", incomplete=True, sha=str(expected_sha or ""),
                              failing=failing, rows=len(fleet))
    incomplete = bool(matrix_incomplete)
    return RestartVerdict(
        verdict="incomplete" if incomplete else "current", incomplete=incomplete,
        sha=str(expected_sha or ""), failing=failing, rows=len(fleet),
    )


# --------------------------------------------------------------------------- #
# Catch-up + escalation (S8, G4)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CatchUpPlan:
    """The scheduled self-firing attempt: attempt N at ``next_at``, or exhausted → escalate."""

    attempt: int
    delay_s: float
    next_at: float
    exhausted: bool

    def describe(self) -> str:
        if self.exhausted:
            return f"catch_up attempt={self.attempt} exhausted=true"
        return f"catch_up attempt={self.attempt} next_at={self.next_at:.0f} delay={self.delay_s:.0f}s"


def schedule_catch_up(*, sha: str, now: Optional[float] = None) -> Optional[CatchUpPlan]:
    """Arm the next bounded catch-up attempt on the standing obligation (S8).

    Does nothing when no obligation is armed: a host that owes no restart has nothing to catch up.
    """
    from hermes_cli.update_host_obligation import amend_host_obligation, read_host_obligation

    record = read_host_obligation()
    if record is None:
        return None
    moment = float(now if now is not None else time.time())
    previous = record.get("catch_up")
    attempt = int(previous.get("attempt") or 0) + 1 if isinstance(previous, dict) else 1
    if attempt > len(CATCH_UP_DELAYS_S):
        plan = CatchUpPlan(attempt=attempt, delay_s=0.0, next_at=moment, exhausted=True)
        amend_host_obligation(catch_up={"attempt": attempt, "next_at": moment, "exhausted": True,
                                        "sha": sha or ""})
        logger.error("restart_catch_up exhausted after %d attempts sha=%s", attempt - 1, (sha or "?")[:10])
        return plan
    delay = float(CATCH_UP_DELAYS_S[attempt - 1])
    plan = CatchUpPlan(attempt=attempt, delay_s=delay, next_at=moment + delay, exhausted=False)
    amend_host_obligation(
        catch_up={"attempt": attempt, "next_at": plan.next_at, "exhausted": False, "sha": sha or ""}
    )
    logger.info("restart_catch_up armed attempt=%d next_at=%.0f delay=%.0fs sha=%s",
                attempt, plan.next_at, delay, (sha or "?")[:10])
    return plan


def due_catch_up(*, now: Optional[float] = None) -> Optional[CatchUpPlan]:
    """The scheduled catch-up when it is due, else ``None``."""
    from hermes_cli.update_host_obligation import read_host_obligation

    record = read_host_obligation()
    if record is None:
        return None
    plan = record.get("catch_up")
    if not isinstance(plan, dict) or plan.get("exhausted"):
        return None
    next_at = plan.get("next_at")
    if not isinstance(next_at, (int, float)):
        return None
    moment = float(now if now is not None else time.time())
    if moment < float(next_at):
        return None
    return CatchUpPlan(
        attempt=int(plan.get("attempt") or 1), delay_s=0.0, next_at=float(next_at), exhausted=False
    )


def run_due_catch_up(*, runner: Optional[Callable[[], bool]] = None, now: Optional[float] = None) -> str:
    """Run the scheduled catch-up once it is due (S8 self-firing entry point).

    Returns ``not-due`` · ``no-obligation`` · ``ran`` · ``failed`` · ``exhausted``.

    Reachable without a human: ``hermes update --run-restart-catch-up`` (safe on any tick — it
    exits 0 and never touches the checkout). *Installing* the trigger (launchd one-shot or a
    ``gateway_watchdog.sh`` hook) is a host change and stays with §5 item 2, which is unanswered:
    whether the watchdog may act on staleness or only alert.
    """
    plan = due_catch_up(now=now)
    if plan is None:
        from hermes_cli.update_host_obligation import read_host_obligation

        return "no-obligation" if read_host_obligation() is None else "not-due"
    if runner is None:
        from hermes_cli.update_cmd_fleet import _run_pending_fleet_restart

        runner = _run_pending_fleet_restart
    ok = bool(runner())
    logger.info("restart_catch_up ran attempt=%d ok=%s", plan.attempt, ok)
    if ok:
        return "ran"
    schedule_catch_up(sha=((read_restart_lease() or {}).get("sha") or ""))
    return "failed" if plan.attempt < len(CATCH_UP_DELAYS_S) else "exhausted"


def escalate_restart_failure(*, sha: str, reason: str, notify: Optional[Callable[[str], None]] = None) -> bool:
    """One escalation per pulled sha: durable record + loud operator line (S7).

    Returns ``False`` when this sha was already escalated — repeated failures of the same restart
    must not page the same human again. Delivery is a ``notify`` hook rather than a hard-wired
    transport: the updater's terminal is not the person who needs to know.
    """
    from hermes_cli.update_host_obligation import amend_host_obligation, read_host_obligation

    record = read_host_obligation()
    if record is None:
        return False
    previous = record.get("escalation")
    if isinstance(previous, dict) and str(previous.get("sha") or "") == str(sha or ""):
        logger.debug("restart escalation already sent for sha=%s", (sha or "?")[:10])
        return False
    message = (
        f"⚠ Gateway restart incomplete for {((sha or '?')[:10])}: {reason}. "
        "Run `hermes gateway restart`, then `hermes gateway status`."
    )
    amend_host_obligation(
        escalation={"sha": sha or "", "reason": str(reason or ""), "at": time.time(), "message": message}
    )
    logger.error("restart_escalation sha=%s reason=%s", (sha or "?")[:10], reason)
    print(message)
    if notify is not None:
        with suppress(Exception):
            notify(message)
    return True
