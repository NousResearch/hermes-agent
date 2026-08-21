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
``hermes_cli.build_info.get_code_identity()``: an image build reports
``source="build-file"`` and the receipt records that the install is not
in-place updatable.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)

_RECEIPT_DIR_NAME = "update_receipts"
_RECEIPT_KEEP = 20  # keep the last N receipts per profile home

# Module-level current receipt. ``hermes update`` is a single-threaded CLI
# command; a module singleton lets the 7k-line updater record steps from
# any depth without threading a handle through every helper.
_current: Optional["UpdateReceipt"] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UpdateReceipt:
    """Collects the observable facts of one ``hermes update`` run."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "schema": 1,
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "argv": list(sys.argv),
            "pid": os.getpid(),
            "outcome": "running",  # running | success | partial | failed
            "pre_update": {},
            "post_update": {},
            "steps": [],
            "skips": [],
            "gateway_restart": {},
            "fleet": [],
        }
        try:
            from hermes_cli.build_info import get_code_identity

            self.data["pre_update"] = get_code_identity()
        except Exception:
            pass

    # -- recording ---------------------------------------------------------
    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.data["steps"].append(
            {"name": name, "ok": bool(ok), "detail": detail, "at": _utc_now_iso()}
        )

    def skip(self, name: str, reason: str) -> None:
        self.data["skips"].append(
            {"name": name, "reason": reason, "at": _utc_now_iso()}
        )

    def gateway_restart_result(
        self,
        *,
        restarted_services: list | None = None,
        relaunched_profiles: list | None = None,
        externally_supervised_profiles: list | None = None,
        killed_pids: list | None = None,
        failed_units: list | None = None,
        incomplete: bool = False,
        phase_error: str = "",
    ) -> None:
        self.data["gateway_restart"] = {
            "restarted_services": list(restarted_services or []),
            "relaunched_profiles": list(relaunched_profiles or []),
            "externally_supervised_profiles": list(
                externally_supervised_profiles or []
            ),
            "killed_pids": [int(p) for p in (killed_pids or [])],
            "failed_units": [str(u) for u in (failed_units or [])],
            "incomplete": bool(incomplete),
            "phase_error": phase_error,
        }

    def finalize(self, outcome: str) -> None:
        self.data["outcome"] = outcome
        self.data["finished_at"] = _utc_now_iso()
        try:
            from hermes_cli.build_info import get_code_identity

            self.data["post_update"] = get_code_identity(refresh=True)
        except Exception:
            pass


def _receipt_dir() -> Path:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "logs" / _RECEIPT_DIR_NAME


def begin_update_receipt() -> None:
    """Start recording a new update receipt. Never raises."""
    global _current
    try:
        _current = UpdateReceipt()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not start update receipt: %s", exc)
        _current = None


def record_step(name: str, ok: bool, detail: str = "") -> None:
    """Record one update step outcome. No-op when no receipt is active."""
    try:
        if _current is not None:
            _current.step(name, ok, detail)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record update step %s: %s", name, exc)


def record_skip(name: str, reason: str) -> None:
    """Record a skipped step WITH the reason it was skipped."""
    try:
        if _current is not None:
            _current.skip(name, reason)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record update skip %s: %s", name, exc)


def record_gateway_restart(**kwargs: Any) -> None:
    """Record the gateway restart phase outcome (see UpdateReceipt)."""
    try:
        if _current is not None:
            _current.gateway_restart_result(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not record gateway restart result: %s", exc)


def finalize_update_receipt(outcome: str, fleet: list | None = None) -> Optional[Path]:
    """Finalize + persist the receipt. Returns the written path or None.

    ``outcome`` is one of ``success`` / ``partial`` / ``failed``.
    """
    global _current
    receipt = _current
    _current = None
    if receipt is None:
        return None
    try:
        receipt.finalize(outcome)
        if fleet is not None:
            receipt.data["fleet"] = fleet
        directory = _receipt_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = directory / f"update_{stamp}_{os.getpid()}.json"
        path.write_text(
            json.dumps(receipt.data, indent=2, default=str), encoding="utf-8"
        )
        # Stable pointer for the dashboard/desktop: latest receipt.
        latest = directory / "latest.json"
        try:
            latest.write_text(
                json.dumps(receipt.data, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            pass
        _prune_old_receipts(directory)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not write update receipt: %s", exc)
        return None


def _prune_old_receipts(directory: Path) -> None:
    try:
        receipts = sorted(
            (p for p in directory.glob("update_*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in receipts[_RECEIPT_KEEP:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except Exception:
        pass


def read_latest_receipt() -> Optional[dict[str, Any]]:
    """Read the most recent update receipt, or None. Never raises."""
    try:
        path = _receipt_dir() / "latest.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fleet version verification
# ---------------------------------------------------------------------------

def collect_fleet_versions() -> list[dict[str, Any]]:
    """Snapshot every profile's gateway code identity vs. the current tree.

    Returns one entry per profile home that has a ``gateway_state.json``
    with a live PID::

        {"profile": str, "pid": int, "code_sha": str|None,
         "code_version": str|None, "state": "current"|"stale"|"unknown"}

    ``stale``   — gateway stamped a code_sha that differs from the updated
                  checkout's HEAD (it is still serving pre-update modules).
    ``unknown`` — gateway predates the code-identity stamp (started before
                  this feature landed) or identity could not be resolved.
    Never raises; a probe failure yields an empty list.
    """
    results: list[dict[str, Any]] = []
    try:
        from hermes_cli.build_info import get_code_identity

        expected_sha = (get_code_identity(refresh=True) or {}).get("sha")
    except Exception:
        expected_sha = None

    try:
        from gateway.status import _pid_exists, read_runtime_status
        from hermes_cli.profiles import (
            _get_default_hermes_home,
            _get_profiles_root,
            _PROFILE_ID_RE,
        )

        homes: list[tuple[str, Path]] = []
        default_home = _get_default_hermes_home()
        if default_home.is_dir():
            homes.append(("default", default_home))
        profiles_root = _get_profiles_root()
        if profiles_root.is_dir():
            for entry in sorted(profiles_root.iterdir()):
                if entry.is_dir() and entry.name != "default" and _PROFILE_ID_RE.match(entry.name):
                    homes.append((entry.name, entry))

        for profile, home in homes:
            status_path = home / "gateway_state.json"
            record = read_runtime_status(status_path)
            if not record:
                continue
            pid = record.get("pid")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            if not _pid_exists(pid):
                continue
            code_sha = record.get("code_sha")
            if not code_sha or not expected_sha:
                state = "unknown"
            elif str(code_sha) == str(expected_sha):
                state = "current"
            else:
                state = "stale"
            results.append(
                {
                    "profile": profile,
                    "pid": pid,
                    "code_sha": str(code_sha) if code_sha else None,
                    "code_version": record.get("code_version"),
                    "state": state,
                }
            )
    except Exception as exc:
        logger.debug("Fleet version probe failed: %s", exc)
    return results


def print_fleet_version_matrix(fleet: list[dict[str, Any]]) -> bool:
    """Print the post-update fleet version matrix.

    Returns True when at least one gateway is provably stale (still
    serving pre-update code), so the caller can escalate. ``unknown``
    entries are reported but do NOT fail the update: gateways started
    before the code-identity stamp existed have no sha to compare, and
    failing on them would turn this feature's own rollout into a
    false-positive storm.
    """
    if not fleet:
        return False
    any_stale = False
    print()
    print("Fleet version check:")
    for entry in fleet:
        sha = entry.get("code_sha")
        short = sha[:8] if isinstance(sha, str) and sha else "?"
        state = entry.get("state")
        profile = entry.get("profile")
        pid = entry.get("pid")
        if state == "current":
            print(f"  ✓ {profile} (pid {pid}) @ {short} — up to date")
        elif state == "stale":
            any_stale = True
            print(f"  ✗ {profile} (pid {pid}) @ {short} — STALE (pre-update code)")
        else:
            print(
                f"  ? {profile} (pid {pid}) — version unknown "
                "(gateway predates version stamping; restart to enable)"
            )
    if any_stale:
        print()
        print("  ⚠ Stale gateways keep serving pre-update code until restarted:")
        print("      hermes gateway restart                # active profile")
        print("      hermes -p <profile> gateway restart   # named profile")
    return any_stale
