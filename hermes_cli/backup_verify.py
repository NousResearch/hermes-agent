"""Isolated-restore verification for Hermes backup archives (#117005).

A readable zip, a complete selection report, or a successful snapshot call is not the
same thing as a restore that works. ``hermes import --verify-only <archive>`` closes that
gap: the real import pipeline runs into a throwaway ``HERMES_HOME`` and the candidate is
checked against the same primitives the live restore uses.

Contracts kept by a drill:

* the active profile is only *read* — the drill runs against its own temporary home and
  never writes over ``get_hermes_home()``;
* no provider is contacted and no token is refreshed (auth state is JSON-parsed only);
* ``_external/`` members (memory-provider files that restore to the invoking user's home
  outside ``HERMES_HOME``) are policy exclusions — a drill must not publish them;
* runtime state (PIDs, locks, WAL/SHM sidecars) is never promoted as user state: the
  importer's skip list applies, and the receipt names what was skipped.

The receipt is machine-readable and a failed drill cannot report ``verified``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli import backup as _backup

_TMP_PREFIX = "hermes-backup-verify-"
_RECEIPT_SCHEMA = 1
# Parsed structurally, offline, so a torn file is caught without a provider round-trip.
_AUTH_REL = "auth.json"


def _member_targets(members: List[str], prefix: str) -> List[str]:
    """Home-relative target for every member that is not a policy exclusion."""
    rels: List[str] = []
    for member in members:
        if member.startswith(_backup._EXTERNAL_PREFIX):
            continue
        rel = member[len(prefix):] if prefix and member.startswith(prefix) else member
        if rel:
            rels.append(rel)
    return rels


def _check_stores(candidate: Path) -> tuple[Dict[str, Any], List[str]]:
    """Run the production SQLite health primitive on every restored database."""
    stores: Dict[str, Any] = {}
    errors: List[str] = []
    for db_path in sorted(p for p in candidate.rglob("*.db") if p.is_file()):
        rel = db_path.relative_to(candidate).as_posix()
        result = _backup.verify_sqlite_integrity(db_path)
        entry: Dict[str, Any] = {
            "status": "healthy" if result.get("valid") else "unhealthy",
            "detail": result.get("message"),
        }
        counts = _backup._count_session_rows(db_path)
        if counts is not None:
            entry["sessions"], entry["messages"] = counts
        stores[rel] = entry
        if not result.get("valid"):
            errors.append(f"{rel}: {result.get('message')}")
    return stores, errors


def _check_config(candidate: Path) -> tuple[Dict[str, str], List[str]]:
    """Parse ``config.yaml`` through the effective-config pipeline (fail-closed)."""
    config_path = candidate / "config.yaml"
    if not config_path.is_file():
        return {"status": "absent"}, []
    from hermes_cli.config_effective import load_user_config_effective

    try:
        # fail_closed: a torn YAML in the *candidate* must fail the drill, not silently
        # recover the last-good copy the way the live config pipeline does.
        effective = load_user_config_effective(config_path, fail_closed=True)
    except Exception as exc:  # noqa: BLE001 — any parse failure means "not restorable"
        return {"status": "invalid"}, [f"config.yaml: {exc}"]
    if not isinstance(effective, dict):
        return {"status": "invalid"}, ["config.yaml: effective config is not a mapping"]
    return {"status": "valid"}, []


def _check_auth(candidate: Path) -> tuple[Dict[str, str], List[str]]:
    """Structural, offline auth check — no refresh, no provider contact."""
    auth_path = candidate / _AUTH_REL
    if not auth_path.is_file():
        return {"status": "absent"}, []
    try:
        with open(auth_path, "r", encoding="utf-8") as fh:
            json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid"}, [f"{_AUTH_REL}: {exc}"]
    return {"status": "valid"}, []


def _check_cron(candidate: Path) -> tuple[Dict[str, Any], List[str]]:
    """Cron state parses; no job is started and no scheduler is touched."""
    jobs_path = candidate / _backup._CRON_JOBS_REL
    if not jobs_path.is_file():
        return {"status": "absent"}, []
    count = _backup._count_cron_jobs(jobs_path)
    if count is None:
        return {"status": "invalid"}, [f"{_backup._CRON_JOBS_REL}: unreadable or malformed"]
    return {"status": "valid", "jobs": count}, []


def verify_backup_archive(
    zip_path: Path,
    *,
    keep_candidate: bool = False,
    candidate_home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Restore *zip_path* into an isolated home and return the verification receipt.

    ``status`` is ``"verified"`` only when every required object landed and every
    postcondition passed; otherwise (or on any import error) it is ``"failed"`` and the
    receipt's ``errors``/``missing_objects`` name what went wrong.

    The temporary restore is removed on success unless *keep_candidate*; a failed drill
    or an explicit *keep_candidate* leaves it in place for inspection.
    """
    archive = Path(zip_path).expanduser().resolve()
    receipt: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "archive": str(archive),
        "status": "failed",
        "candidate_home": None,
        "candidate_retained": False,
        "required_objects": 0,
        "restored_objects": 0,
        "missing_objects": [],
        "policy_exclusions": 0,
        "runtime_skipped": [],
        "stores": {},
        "config": {"status": "absent"},
        "auth": {"status": "absent"},
        "cron": {"status": "absent"},
        "warnings": [],
        "errors": [],
    }
    if not archive.is_file():
        receipt["errors"].append(f"archive not found: {archive}")
        return receipt
    if not zipfile.is_zipfile(archive):
        receipt["errors"].append(f"not a valid zip file: {archive}")
        return receipt

    owned_candidate = candidate_home is None
    candidate = (
        Path(candidate_home) if candidate_home is not None
        else Path(tempfile.mkdtemp(prefix=_TMP_PREFIX))
    )
    receipt["candidate_home"] = str(candidate)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            valid, reason = _backup._validate_backup_zip(zf)
            if not valid:
                receipt["errors"].append(reason)
                return receipt
            prefix = _backup._detect_prefix(zf)
            members = [n for n in zf.namelist() if not n.endswith("/")]
            # External members restore to the invoking user's home (~), NOT the candidate:
            # a drill must never publish them, so they are counted as policy exclusions.
            external = [m for m in members if m.startswith(_backup._EXTERNAL_PREFIX)]
            staged = [m for m in members if not m.startswith(_backup._EXTERNAL_PREFIX)]
            receipt["policy_exclusions"] = len(external)
            required_rels = _member_targets(staged, prefix)
            receipt["required_objects"] = len(required_rels)

            candidate.mkdir(parents=True, exist_ok=True)
            _restored, _restored_external, import_errors, skipped_runtime, _shrunk = (
                _backup._import_members(zf, staged, prefix, candidate, len(staged))
            )
            receipt["runtime_skipped"] = sorted(skipped_runtime)
            receipt["errors"].extend(import_errors)

        skipped = set(receipt["runtime_skipped"])
        missing = [
            rel for rel in required_rels
            if rel not in skipped and not (candidate / rel).is_file()
        ]
        receipt["missing_objects"] = missing
        if missing:
            receipt["errors"].append(
                f"{len(missing)} required object(s) did not restore: "
                + ", ".join(missing[:10])
            )
        receipt["restored_objects"] = receipt["required_objects"] - len(missing)

        stores, store_errors = _check_stores(candidate)
        receipt["stores"] = stores
        receipt["errors"].extend(store_errors)
        receipt["config"], config_errors = _check_config(candidate)
        receipt["errors"].extend(config_errors)
        receipt["auth"], auth_errors = _check_auth(candidate)
        receipt["errors"].extend(auth_errors)
        receipt["cron"], cron_errors = _check_cron(candidate)
        receipt["errors"].extend(cron_errors)

        receipt["status"] = "verified" if not receipt["errors"] else "failed"
    finally:
        if owned_candidate:
            if receipt["status"] == "verified" and not keep_candidate:
                shutil.rmtree(candidate, ignore_errors=True)
                receipt["candidate_home"] = None
            else:
                receipt["candidate_retained"] = candidate.exists()
    return receipt


def run_verify_import(args) -> None:
    """``hermes import --verify-only``: prove the archive restores, then exit.

    Nonzero when the drill fails — a caller (e.g. a pre-rebuild recovery lane) treats a
    non-``verified`` exit as "do not destroy the only copy".
    """
    receipt = verify_backup_archive(
        Path(args.zipfile),
        keep_candidate=bool(getattr(args, "keep_candidate", False)),
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "verified":
        print("\n✗ Backup verification failed:", file=sys.stderr)
        for err in receipt["errors"][:10]:
            print(f"  - {err}", file=sys.stderr)
        if receipt.get("candidate_home"):
            print(f"  Candidate kept for inspection: {receipt['candidate_home']}",
                  file=sys.stderr)
        raise SystemExit(1)
    print(
        f"\n✓ Backup verified: {receipt['restored_objects']}/"
        f"{receipt['required_objects']} required objects restored into an isolated home."
    )
    if receipt.get("candidate_home"):
        print(f"  Candidate home kept: {receipt['candidate_home']}")
    else:
        print("  Active profile untouched; temporary restore removed.")
