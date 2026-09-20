"""Structured update receipts + post-update fleet version verification.

The updater must *prove* its outcome instead of assuming it. Every public entry point is
exception-swallowing so a failure inside receipts can never break an update.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RECEIPT_KEEP = 20  # keep the last N receipts per profile home
COMMAND_BOUNDARY_STOP_REASON = "completed at command boundary"

# ``hermes update`` is a single-threaded CLI command; a module singleton lets the 7k-line updater
# record steps from any depth without threading a handle through every helper.
_current: Optional["UpdateReceipt"] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_identity(refresh: bool = False) -> dict[str, Any]:
    """Running-code identity, or ``{}`` when the probe fails."""
    with suppress(Exception):
        from hermes_cli.build_info import get_code_identity

        return get_code_identity(refresh=refresh) or {}
    return {}


def _str_records(entries: Any, keys: tuple[str, ...], *, pid: bool = False) -> list[dict[str, Any]]:
    """Dict entries reduced to stringified ``keys`` (plus an int ``pid`` first when requested)."""
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record: dict[str, Any] = {"pid": int(entry.get("pid", 0) or 0)} if pid else {}
        record.update({key: str(entry.get(key, "")) for key in keys})
        records.append(record)
    return records


class UpdateReceipt:
    """Collects the observable facts of one ``hermes update`` run."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "schema": 1, "started_at": _utc_now_iso(), "finished_at": None,
            "argv": list(sys.argv), "pid": os.getpid(),
            "outcome": "running",  # running | success | partial | failed
            "pre_update": _code_identity(), "post_update": {},
            "steps": [], "skips": [], "gateway_restart": {}, "fleet": [],
        }

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.data["steps"].append({"name": name, "ok": bool(ok), "detail": detail, "at": _utc_now_iso()})

    def skip(self, name: str, reason: str) -> None:
        self.data["skips"].append({"name": name, "reason": reason, "at": _utc_now_iso()})

    def gateway_restart_result(
        self, *, restarted_services: list | None = None, relaunched_profiles: list | None = None,
        externally_supervised_profiles: list | None = None, killed_pids: list | None = None,
        failed_units: list | None = None, incomplete: bool = False, phase_error: str = "",
        fresh_recovery: dict[str, Any] | None = None,
    ) -> None:
        result: dict[str, Any] = {
            "restarted_services": list(restarted_services or []),
            "relaunched_profiles": list(relaunched_profiles or []),
            "externally_supervised_profiles": list(externally_supervised_profiles or []),
            "killed_pids": [int(p) for p in (killed_pids or [])],
            "failed_units": [str(u) for u in (failed_units or [])],
            "incomplete": bool(incomplete),
            "phase_error": phase_error,
        }
        if fresh_recovery is not None:
            # Conservative outcome vocabulary: "verified" is the only bucket allowed to claim
            # supervisor coverage; "relaunch_attempted" means the relaunch exited 0 without
            # independent supervisor observation. "skipped" preserves runtimes (manual gateways,
            # serve/dashboard entries) the pass deliberately did not touch.
            persisted: dict[str, Any] = {
                key: [str(profile) for profile in fresh_recovery.get(key, [])]
                for key in ("requested", "verified", "relaunch_attempted", "failed")
            }
            persisted["skipped"] = _str_records(
                fresh_recovery.get("skipped", []), ("profile", "kind", "supervisor", "reason")
            )
            # ``hermes serve`` hosts tui_gateway and is not a gateway profile, so neither the
            # per-profile buckets above nor the fleet-version matrix can describe it. Persist its
            # unit outcomes and any process that survived on the pre-update generation, or the
            # receipt keeps claiming a clean recovery the operator's box contradicts.
            serve_units = fresh_recovery.get("serve_units") or {}
            persisted["serve_units"] = {
                key: [str(unit) for unit in (serve_units.get(key) or [])] for key in ("verified", "failed")
            }
            persisted["stale_runtimes"] = _str_records(
                fresh_recovery.get("stale_runtimes", []), ("kind", "profile", "supervisor"), pid=True
            )
            result["fresh_recovery"] = persisted
        self.data["gateway_restart"] = result

    def finalize(self, outcome: str) -> None:
        self.data["outcome"] = outcome
        self.data["finished_at"] = _utc_now_iso()
        self.data["post_update"] = _code_identity(refresh=True)


def _receipt_dir() -> Path:
    # ``hermes_constants`` (stdlib-only), never ``hermes_cli.config``: the receipt must be
    # writable from the refused/failed paths where config loading itself may be what broke
    # (#112465, #112558).
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "update_receipts"


def begin_update_receipt() -> None:
    """Start recording a new update receipt. Never raises."""
    global _current
    try:
        _current = UpdateReceipt()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not start update receipt: %s", exc)
        _current = None


def detach_update_receipt() -> Optional[dict[str, Any]]:
    """Hand the open receipt to another process: return its data and forget it here.

    The post-swap child resumes it via :func:`resume_update_receipt`; the parent's
    command-boundary finalize then no-ops, so the run still produces exactly one receipt.
    """
    global _current
    receipt, _current = _current, None
    return None if receipt is None else receipt.data


def resume_update_receipt(data: dict[str, Any]) -> None:
    """Continue a receipt detached by the pre-swap interpreter (``started_at``, ``pre_update``,
    ``argv``, steps and plan intact); records this process as the one that finished it."""
    global _current
    receipt = UpdateReceipt()
    receipt.data = data
    receipt.data["post_swap_pid"] = os.getpid()
    _current = receipt


def _record(method: str, what: str, *args: Any, **kwargs: Any) -> None:
    """Invoke ``method`` on the active receipt; no-op when none, never raises."""
    try:
        if _current is not None:
            getattr(_current, method)(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record %s: %s", what, exc)


def record_step(name: str, ok: bool, detail: str = "") -> None:
    """Record one update step outcome. No-op when no receipt is active."""
    _record("step", f"update step {name}", name, ok, detail)


def record_skip(name: str, reason: str) -> None:
    """Record a skipped step WITH the reason it was skipped."""
    _record("skip", f"update skip {name}", name, reason)


def record_gateway_restart(**kwargs: Any) -> None:
    """Record the gateway restart phase outcome (see UpdateReceipt)."""
    _record("gateway_restart_result", "gateway restart result", **kwargs)


def finalize_update_receipt(outcome: str, fleet: list | None = None, stop_reason: str = "") -> Optional[Path]:
    """Finalize + persist the receipt (``success``/``partial``/``failed``/``refused``); path or None.

    Exactly-once by construction: the module singleton is popped first, so a second call (e.g. the
    command-boundary safety net after an inner path already finalized) is a no-op returning None.
    """
    global _current
    receipt = _current
    _current = None
    if receipt is None:
        return None
    try:
        receipt.finalize(outcome)
        if stop_reason:
            receipt.data["stop_reason"] = stop_reason
        if fleet is not None:
            receipt.data["fleet"] = fleet
        from hermes_cli.update_serve_obligations import retain_receipt_manual_serves
        pending = retain_receipt_manual_serves(read_latest_receipt() or {})
        if pending:
            receipt.data["pending_manual_serves"] = pending
        directory = _receipt_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"update_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
        body = json.dumps(receipt.data, indent=2, default=str)
        path.write_text(body, encoding="utf-8")
        with suppress(OSError):  # stable pointer for the dashboard/desktop
            (directory / "latest.json").write_text(body, encoding="utf-8")
        _prune_old_receipts(directory)
        return path
    except Exception as exc:
        # Visible, not debug: a run that pulled code and left no receipt is exactly the run
        # operators need to post-mortem, and INFO-level logs discard debug (#112465, #112558).
        logger.warning("Could not write update receipt (%s): %s", outcome, exc)
        print(f"  ⚠ Update receipt not written: {exc}")
        return None


def finalize_pending_update_receipt(exit_code: Optional[int] = None, stop_reason: str = "") -> Optional[Path]:
    """Command-boundary safety net: persist a still-open receipt, if any. Never raises.

    ``hermes update`` has many early ``sys.exit`` paths (preflight refusals, venv-holder refusal,
    fetch failure) predating the inner finalize calls; finalizing here means refused/failed runs —
    where a receipt matters most — leave a record. Exit 0/None → ``success``, exit 2 → ``refused``
    (preflight convention), else → ``failed``.

    No-op when no receipt is open (the inner paths already finalized — exactly-once via the popped
    singleton) or when recording was never started. See #91283.
    """
    if _current is None:
        return None
    outcome = "success" if exit_code in (0, None) else "refused" if exit_code == 2 else "failed"
    if exit_code is not None:
        with suppress(Exception):
            _current.data["exit_code"] = int(exit_code)
    return finalize_update_receipt(outcome, stop_reason=stop_reason)


def _prune_old_receipts(directory: Path) -> None:
    with suppress(Exception):
        receipts = (p for p in directory.glob("update_*.json") if p.is_file())
        for stale in sorted(receipts, key=lambda p: p.stat().st_mtime, reverse=True)[_RECEIPT_KEEP:]:
            with suppress(OSError):
                stale.unlink()


def read_latest_receipt() -> Optional[dict[str, Any]]:
    """Read the most recent update receipt, or None. Never raises."""
    with suppress(Exception):
        path = _receipt_dir() / "latest.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
    return None


def _profile_homes() -> list[tuple[str, Path]]:
    """``(profile, home)`` for the default home plus every valid named profile dir, sorted."""
    from hermes_cli.profiles import _get_default_hermes_home, _get_profiles_root, _PROFILE_ID_RE

    homes: list[tuple[str, Path]] = []
    default_home = _get_default_hermes_home()
    if default_home.is_dir():
        homes.append(("default", default_home))
    root = _get_profiles_root()
    if root.is_dir():
        homes.extend(
            (entry.name, entry)
            for entry in sorted(root.iterdir())
            if entry.is_dir() and entry.name != "default" and _PROFILE_ID_RE.match(entry.name)
        )
    return homes


def _socket_identity(home: Path) -> Optional[tuple[int, dict]]:
    """``(pid, identity)`` declared by the gateway owning ``home``'s control socket, else None.

    A live ``identify`` answer is authoritative — no PID-reuse or stale-file heuristics. Callers
    fall back to ``gateway_state.json`` for gateways that predate the socket or whose socket
    didn't bind.
    """
    try:
        # Prefer the gateway-owned control socket (#92091): identity declared by the process itself,
        # including its own supervisor provenance — no argv/PID inference. Scan fallback below.
        # Prefer the gateway-owned control socket (#92091): a live `identify` answer is authoritative — no
        # PID-reuse or stale-file heuristics.
        from gateway.control_socket import identify_gateway

        identity = identify_gateway(home)
        return (int(identity.get("pid")), identity) if identity else None
    except Exception:  # probe failure, no gateway, or an unparseable pid
        return None


def _fleet_row(
    profile: str, pid: int, code_sha: Any, code_version: Any, expected_sha: Any,
    state: str = "unknown", served_profiles: Any = None,
) -> dict[str, Any]:
    if state == "unknown" and code_sha and expected_sha:
        state = "current" if str(code_sha) == str(expected_sha) else "stale"
    row = {
        "profile": profile, "pid": pid, "code_sha": str(code_sha) if code_sha else None,
        "code_version": code_version, "state": state,
    }
    # A live, identity-verified multiplexer represents every profile in this
    # list. Keep the field only when its shape is usable: callers use it to
    # discharge per-profile restart obligations, so corrupt status must not
    # widen coverage.
    if isinstance(served_profiles, list) and served_profiles and all(
        isinstance(name, str) and name for name in served_profiles
    ):
        row["served_profiles"] = list(dict.fromkeys(served_profiles))
    return row


# Runtime-status states that do not describe a gateway that should be running now — no down row.
_NOT_EXPECTED_STATES = {"stopped", "startup_failed"}


def collect_fleet_versions(*, pre_restart_pids: Optional[list[int]] = None) -> list[dict[str, Any]]:
    """Snapshot every profile's gateway code identity vs. the current tree.

    Rollout safety: ``down`` requires membership in ``pre_restart_pids`` — a stale state file from a
    long-dead gateway (machine reboot, manual kill weeks ago) must NOT fail every future update.
    Without a pre-restart snapshot (``None``/empty) dead PIDs are skipped (historical behavior).

    ``stale``   — gateway stamped a code_sha that differs from the updated checkout's HEAD (it is still
    serving pre-update modules). ``unknown`` — gateway predates the code-identity stamp (started before this
    feature landed), identity could not be resolved, or the state file's live PID is not the
    verified gateway for that home (``live_gateway_pid_for_home``): ``write_runtime_status`` re-stamps
    ``pid``/``code_sha`` for whatever process writes it, so a foreign writer must never read as
    ``current`` (#110420, sibling of #109680). ``down``    — the gateway was ALIVE when this update
    started (``pre_restart_pids``), its runtime status still says running, but the PID is dead and no
    successor rewrote the record: the restart phase stopped it and nothing came back. Without this row a
    killed-and-never-replaced gateway produced NO entry at all and the matrix passed silently (Phase-1
    verification gap, #88848/#74973 class).
    """
    _pre_restart = {int(p) for p in (pre_restart_pids or []) if isinstance(p, int)}
    results: list[dict[str, Any]] = []
    expected_sha = _code_identity(refresh=True).get("sha")
    try:
        from gateway.status import (
            live_gateway_pid_for_home,
            read_runtime_status,
            runtime_status_pid_is_live,
        )

        for profile, home in _profile_homes():
            sock = _socket_identity(home)
            if sock is not None:
                pid, identity = sock
                row = _fleet_row(
                    profile, pid, identity.get("code_sha"), identity.get("code_version"), expected_sha,
                    served_profiles=identity.get("served_profiles"),
                )
                results.append({**row, "source": "socket"})
                continue
            record = read_runtime_status(home / "gateway_state.json")
            if not record:
                continue
            try:
                pid = int(record.get("pid"))
            except (TypeError, ValueError):
                continue
            # A state file is only a fallback claim. Its SHA is evidence about
            # its own PID only when the profile's canonical identity resolver
            # verifies that same live gateway.
            if live_gateway_pid_for_home(home) == pid:
                results.append(
                    _fleet_row(
                        profile, pid, record.get("code_sha"), record.get("code_version"), expected_sha,
                        served_profiles=record.get("served_profiles"),
                    )
                )
                continue
            # A live non-gateway (or a gateway for another profile) can write a
            # plausible state file. Keep the fail-open visibility row, but never
            # let that file's self-reported SHA or version classify or label
            # the process — both claims have the same trust problem.
            if runtime_status_pid_is_live(record):
                results.append(_fleet_row(profile, pid, None, None, None))
                continue
            # Dead PID (or a live PID recycled by an unrelated process during the update's own
            # churn): a DOWN row only when this exact pid was alive at update start AND the record
            # still claims a running state — "the restart phase stopped it and nothing came back."
            # Everything else (clean stop, startup failure, long-dead stale record) keeps the no-row
            # behavior so the rollout can't false-positive. ``_pre_restart`` is a bare PID set, not
            # (pid, start_time) pairs, so a recycled PID from gateway A landing in B's stale record
            # could still mislabel B as down — inherent to the snapshot's data model.
            # See #93258.
            gw_state = record.get("gateway_state")
            if pid in _pre_restart and isinstance(gw_state, str) and gw_state and gw_state not in _NOT_EXPECTED_STATES:
                results.append(_fleet_row(profile, pid, None, record.get("code_version"), None, state="down"))
    except Exception as exc:
        logger.debug("Fleet version probe failed: %s", exc)
    return results


_FLEET_ROW_LINES = {
    "current": "  ✓ {profile} (pid {pid}) @ {short} — up to date",
    "stale": "  ✗ {profile} (pid {pid}) @ {short} — STALE (pre-update code)",
    "down": "  ✗ {profile} — DOWN (gateway was running before the update; pid {pid} is gone and nothing replaced it)",
}
_FLEET_ROW_UNKNOWN = "  ? {profile} (pid {pid}) — version unknown (gateway predates version stamping; restart to enable)"
# A gateway pid the pre-update snapshot did not know that had not published its code identity when
# the settle window closed (#112634): most likely the successor this update relaunched, still
# booting, so "restart to enable" would be wrong — but the poll never observed the restart itself,
# so the copy does not claim one.
_FLEET_ROW_IDENTITY_PENDING = (
    "  ? {profile} (pid {pid}) — new pid since the update, code identity not published yet"
    " — re-check with `hermes gateway status`"
)


def print_fleet_version_matrix(fleet: list[dict[str, Any]]) -> bool:
    """Print the post-update fleet version matrix.

    Returns True when at least one gateway is provably stale (still serving pre-update code) OR
    provably down (killed by the restart phase, nothing came back), so the caller can escalate.
    ``unknown`` entries are reported but do NOT fail the update: gateways started before the
    code-identity stamp existed have no sha to compare, and failing them would be a false-positive
    storm.
    """
    if not fleet:
        return False
    print()
    print("Fleet version check:")
    states = set()
    for entry in fleet:
        sha = entry.get("code_sha")
        states.add(entry.get("state"))
        fallback = _FLEET_ROW_IDENTITY_PENDING if entry.get("identity_pending") else _FLEET_ROW_UNKNOWN
        print(_FLEET_ROW_LINES.get(entry.get("state"), fallback).format(
            profile=entry.get("profile"), pid=entry.get("pid"), short=sha[:8] if isinstance(sha, str) and sha else "?",
        ))
    stale_or_down = sum(1 for entry in fleet if entry.get("state") in ("stale", "down"))
    if stale_or_down:
        print()
        if "stale" in states:
            print("  ⚠ Stale gateways keep serving pre-update code until restarted.")
        if "down" in states:
            print("  ⚠ Down gateways stopped serving messaging entirely.")
        # ``✓ Update complete!`` was already printed before the restart phase (the exit code
        # must land before a systemd restart can kill this process), so this verdict has to
        # supersede it explicitly — otherwise the output says success while the exit code is 1.
        print()
        print(
            f"✗ Update not complete: {stale_or_down} gateway(s) still running the old code (or stopped).")
        print("  Run `hermes gateway restart` (or `hermes -p <profile> gateway restart` for a named")
        print("  profile), then `hermes gateway status` to confirm.")
    return stale_or_down > 0
