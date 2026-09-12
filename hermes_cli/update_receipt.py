"""Structured update receipts + post-update fleet version verification.

Phase 1 of the fleet-update reliability plan (#91277): the updater must
*prove* its outcome instead of assuming it.

Two additive capabilities, both designed so a failure inside them can never
break an update (every public entry point is exception-swallowing):

1. **Update receipt** — a machine-readable JSON record of what one
   ``hermes update`` run discovered, did, skipped (and why), written to
   ``<HERMES_HOME>/logs/update_receipts/``. Silent-failure classes this
   makes visible: #88848 (helper died after "success" printed), #74973
   (restart silently skipped), #85753 (restart phase never ran), #81193
   (desktop shows failure for a successful update).

2. **Fleet version verification** — after the restart phase, read every
   profile's ``gateway_state.json``, compare each live gateway's stamped
   ``code_sha`` (written by ``gateway/status.py`` on every runtime-status
   write) against the freshly-updated checkout's HEAD, and print a fleet
   version matrix. Mixed-version fleets (#88654, #69754, #77553, #56717)
   become a loud, actionable report instead of a latent state.

Deployment-kind awareness (docker/image-managed installs) rides on
``hermes_cli.version_info.get_code_identity()``: a packaged build reports
its install-stamp provenance (``source="docker"``/``"nix"``/…) and the
receipt records that the install is not in-place updatable.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RECEIPT_KEEP = 20  # keep the last N receipts per profile home
COMMAND_BOUNDARY_STOP_REASON = "completed at command boundary"

# Receipt state is per-CONTEXT, not a module global: a nested
# ``hermes update`` receipt (or one in another thread) must never clobber
# the outer one, and the boundary finalize must see exactly its own
# process's receipt. Same pattern as pm.receipt's ContextVars — no
# manager object.
_current: contextvars.ContextVar[Optional["UpdateReceipt"]] = contextvars.ContextVar(
    "update_receipt_current", default=None
)


@contextmanager
def update_receipt_scope():
    """Keep the command's finalization guard away from an enclosing update."""
    token = _current.set(None)
    try:
        yield
    finally:
        _current.reset(token)


def current_correlation_id() -> Optional[str]:
    """The update correlation id in force in this context, or None.

    Derived from the OPEN update receipt itself — one source of truth, no
    duplicate id variable: a nested update's begin replaces the current
    receipt (so syncs begun under it capture the nested id), and its
    finalize RESTORES the outer receipt via the ContextVar token (so the
    outer id comes back for the rest of the outer update)."""
    current = _current.get()
    return current.correlation_id if current is not None else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_identity(refresh: bool = False) -> dict[str, Any]:
    """Running-code identity, or ``{}`` when the probe fails."""
    with suppress(Exception):
        from hermes_cli.version_info import get_code_identity

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
        # The id binds this update to the pm sync receipts begun under it
        # (pm.receipt captures it via the _correlation ContextVar).
        self.correlation_id = uuid.uuid4().hex
        self.data["update_id"] = self.correlation_id

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
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "update_receipts"


def begin_update_receipt(*, previous: dict | None = None, correlation_id: str | None = None) -> None:
    """Start recording a new update receipt.

    Nested updates are safe: the previous receipt (if any) is preserved
    behind the ContextVar token and comes back when this one finalizes —
    a nested begin/finalize never drops the outer update's receipt or
    correlation. Never raises."""
    try:
        receipt = UpdateReceipt()
        if previous:
            receipt.data.update(copy.deepcopy(previous))
        receipt.correlation_id = correlation_id or receipt.correlation_id
        receipt.data.update(update_id=receipt.correlation_id, outcome="running", finished_at=None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not start update receipt: %s", exc)
        return
    receipt.current_token = _current.set(receipt)


def _record(method: str, what: str, *args: Any, **kwargs: Any) -> None:
    """Invoke ``method`` on the active receipt; no-op when none, never raises.

    Copy-on-write, same rule as pm.receipt: a copied context (copy_context,
    asyncio.to_thread) inherits the SAME receipt object — mutate a clone
    and re-set it in THIS context only, so a child's records never leak
    into (or corrupt) the parent's receipt.
    """
    try:
        import copy

        current = _current.get()
        if current is None:
            return
        clone = copy.copy(current)
        clone.data = copy.deepcopy(current.data)
        getattr(clone, method)(*args, **kwargs)
        _current.set(clone)
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

    Exactly-once by construction: the context's receipt is popped first, so a second call (e.g. the
    command-boundary safety net after an inner path already finalized) is a no-op returning None.
    The receipt is popped via its OWN begin token, so a nested update's finalize RESTORES the outer
    update's open receipt and correlation instead of discarding them.
    """
    current = _current.get()
    if current is None:
        return None
    receipt = copy.copy(current)
    receipt.data = copy.deepcopy(current.data)
    token = getattr(receipt, "current_token", None)
    try:
        if token is not None:
            _current.reset(token)
        else:  # pragma: no cover - receipts begun before token binding
            _current.set(None)
    except ValueError:
        # Token from another context (finalize ran in a copied context) —
        # pop THIS context only, so exactly-once still holds.
        _current.set(None)
    try:
        receipt.finalize(outcome)
        if stop_reason:
            receipt.data["stop_reason"] = stop_reason
        if fleet is not None:
            receipt.data["fleet"] = fleet
        # EMBED the pm sync sections (the settled receipts contract): the
        # update's rebuild/bisect ran through pm's own sync receipt, which
        # finalizes before this one. Fold in only the completion filed
        # under THIS update's correlation id (pm.receipt.last_for_update,
        # per-context, never latest.json) — a sync from before this update
        # began, a standalone sync, or one in a concurrent thread cannot
        # be misattributed, and a nested update's sync cannot displace
        # this update's own. ONE file still carries the whole story
        # (desktop reads a single latest.json).
        try:
            from pm import receipt as pm_receipt

            sync = pm_receipt.last_for_update(receipt.correlation_id, consume=True)
            if isinstance(sync, dict):
                for key in ("venv_rebuild", "plugin_bisect", "feature_list", "steps", "exit_code"):
                    if sync.get(key) is not None:
                        receipt.data[f"pm_{key}"] = sync[key]
                if sync.get("outcome") is not None:
                    receipt.data["pm_sync_outcome"] = sync.get("outcome")
                if sync.get("warnings"):
                    receipt.data["pm_warnings"] = sync["warnings"]
                if sync.get("refusal") is not None:
                    receipt.data["pm_refusal"] = sync["refusal"]
        except Exception as exc:  # pragma: no cover — embedding is additive
            logger.debug("pm sync-section embed skipped: %s", exc)
        directory = _receipt_dir()
        directory.mkdir(parents=True, exist_ok=True)
        # Unique name: stamp+pid collides for nested/concurrent receipts in
        # the same process+second — the correlation id makes the name unique
        # per update run. Atomic write for BOTH the stamped receipt and the
        # latest.json pointer (no torn readers).
        from hermes_cli.runtime_state import _atomic_bytes

        path = directory / (
            f"update_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_"
            f"{receipt.correlation_id}.json"
        )
        payload = (json.dumps(receipt.data, indent=2, default=str) + "\n").encode("utf-8")
        _atomic_bytes(path, payload)
        with suppress(Exception):  # stable pointer for the dashboard/desktop
            _atomic_bytes(directory / "latest.json", payload)
        _prune_old_receipts(directory)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not write update receipt: %s", exc)
        return None


def finalize_pending_update_receipt(exit_code: Optional[int] = None, stop_reason: str = "") -> Optional[Path]:
    """Command-boundary safety net: persist a still-open receipt, if any. Never raises.

    ``hermes update`` has many early ``sys.exit`` paths (preflight refusals, venv-holder refusal,
    fetch failure) predating the inner finalize calls; finalizing here means refused/failed runs —
    where a receipt matters most — leave a record. Exit 0/None → ``success``, exit 2 → ``refused``
    (preflight convention), else → ``failed``.

    No-op when no receipt is open (the inner paths already finalized — exactly-once via the popped
    per-context receipt) or when recording was never started. See #91283.
    """
    current = _current.get()
    if current is None:
        return None
    outcome = "success" if exit_code in (0, None) else "refused" if exit_code == 2 else "failed"
    if exit_code is not None:
        with suppress(Exception):
            clone = copy.copy(current)
            clone.data = copy.deepcopy(current.data)
            clone.data["exit_code"] = int(exit_code)
            _current.set(clone)
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
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
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
        from gateway.control_socket import identify_gateway

        identity = identify_gateway(home)
        return (int(identity.get("pid")), identity) if identity else None
    except Exception:  # probe failure, no gateway, or an unparseable pid
        return None


def _fleet_row(
    profile: str, pid: int, code_sha: Any, code_version: Any, expected_sha: Any, state: str = "unknown"
) -> dict[str, Any]:
    if state == "unknown" and code_sha and expected_sha:
        state = "current" if str(code_sha) == str(expected_sha) else "stale"
    return {
        "profile": profile, "pid": pid, "code_sha": str(code_sha) if code_sha else None,
        "code_version": code_version, "state": state,
    }


# Runtime-status states that do not describe a gateway that should be running now — no down row.
_NOT_EXPECTED_STATES = {"stopped", "startup_failed"}


def collect_fleet_versions(*, pre_restart_pids: Optional[list[int]] = None) -> list[dict[str, Any]]:
    """Snapshot every profile's gateway code identity vs. the current tree.

    Rollout safety: ``down`` requires membership in ``pre_restart_pids`` — a stale state file from a
    long-dead gateway (machine reboot, manual kill weeks ago) must NOT fail every future update.
    Without a pre-restart snapshot (``None``/empty) dead PIDs are skipped (historical behavior).

    ``stale``   — gateway stamped a code_sha that differs from the updated checkout's HEAD (it is still
    serving pre-update modules). ``unknown`` — gateway predates the code-identity stamp (started before this
    feature landed) or identity could not be resolved. ``down``    — the gateway was ALIVE when this update
    started (``pre_restart_pids``), its runtime status still says running, but the PID is dead and no
    successor rewrote the record: the restart phase stopped it and nothing came back. Without this row a
    killed-and-never-replaced gateway produced NO entry at all and the matrix passed silently (Phase-1
    verification gap, #88848/#74973 class).
    """
    _pre_restart = {int(p) for p in (pre_restart_pids or []) if isinstance(p, int)}
    results: list[dict[str, Any]] = []
    expected_sha = _code_identity(refresh=True).get("sha")
    try:
        from gateway.status import read_runtime_status, runtime_status_pid_is_live

        for profile, home in _profile_homes():
            sock = _socket_identity(home)
            if sock is not None:
                pid, identity = sock
                row = _fleet_row(profile, pid, identity.get("code_sha"), identity.get("code_version"), expected_sha)
                results.append({**row, "source": "socket"})
                continue
            record = read_runtime_status(home / "gateway_state.json")
            if not record:
                continue
            try:
                pid = int(record.get("pid"))
            except (TypeError, ValueError):
                continue
            if runtime_status_pid_is_live(record):
                results.append(
                    _fleet_row(profile, pid, record.get("code_sha"), record.get("code_version"), expected_sha)
                )
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
        print(_FLEET_ROW_LINES.get(entry.get("state"), _FLEET_ROW_UNKNOWN).format(
            profile=entry.get("profile"), pid=entry.get("pid"), short=sha[:8] if isinstance(sha, str) and sha else "?",
        ))
    any_stale, any_down = "stale" in states, "down" in states
    if any_stale or any_down:
        print()
        if any_stale:
            print("  ⚠ Stale gateways keep serving pre-update code until restarted:")
        if any_down:
            print("  ⚠ Down gateways stopped serving messaging entirely — restart them:")
        print("      hermes gateway restart                # active profile")
        print("      hermes -p <profile> gateway restart   # named profile")
    return any_stale or any_down
