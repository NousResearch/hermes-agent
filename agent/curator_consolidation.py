"""Validated, recoverable source-to-destination curator consolidation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from hermes_constants import get_hermes_home


def _package_manifest(root: Path) -> Dict[str, str]:
    if not root.is_dir() or not (root / "SKILL.md").is_file():
        raise ValueError(f"incomplete skill package: {root}")
    manifest: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"skill package contains unsupported symlink: {path}")
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "SKILL.md" not in manifest:
        raise ValueError(f"incomplete skill package: {root}")
    return manifest


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _report_paths(operation_id: str) -> Tuple[Path, Path]:
    base = get_hermes_home() / "logs" / "curator" / "consolidations"
    return base / f"{operation_id}.json", base / f"{operation_id}.md"


def _write_receipt(receipt: Dict[str, Any]) -> None:
    json_path, markdown_path = _report_paths(receipt["operation_id"])
    payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    markdown = "\n".join([
        "# Curator consolidation receipt",
        "",
        f"- operation: `{receipt['operation_id']}`",
        f"- source: `{receipt['source']}`",
        f"- destination: `{receipt['destination']}`",
        f"- success: `{receipt['success']}`",
        f"- archive: `{receipt.get('archive_location') or 'none'}`",
        f"- rollback handle: `{receipt.get('rollback_handle') or 'none'}`",
        f"- forwarding readback: `{receipt.get('forwarding', {}).get('readback', False)}`",
        f"- recovery: `{receipt.get('recovery', {}).get('attempted', False)}`",
        "",
    ])
    _atomic_write_bytes(json_path, payload.encode("utf-8"))
    _atomic_write_bytes(markdown_path, markdown.encode("utf-8"))
    if json.loads(json_path.read_text(encoding="utf-8")) != receipt:
        raise RuntimeError("receipt readback mismatch")


def _archive_destination(source_dir: Path) -> Path:
    from tools import skill_usage

    root = skill_usage._archive_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / source_dir.name
    if target.exists():
        target = target.with_name(
            f"{source_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
    return target


def _move(source: Path, destination: Path) -> None:
    try:
        source.rename(destination)
    except OSError:
        shutil.move(str(source), str(destination))


def _preflight(source: str, destination: str) -> Tuple[Path, Path, Path, bytes, bytes]:
    from tools import skill_ledger, skill_usage

    if not source or not destination or source == destination:
        raise ValueError("source and destination must be distinct skill names")
    source_dir = skill_usage._find_skill_dir(source)
    destination_dir = skill_usage._find_skill_dir(destination)
    if source_dir is None:
        raise ValueError(f"source skill '{source}' is not an active local skill")
    if destination_dir is None:
        raise ValueError(f"destination skill '{destination}' is not an active local skill")
    # ``prune_builtins`` permits the deterministic aging pass to archive an
    # old bundled skill, but consolidation is a source-to-destination rewrite
    # and must never re-home upstream-owned packages.
    if skill_usage.is_bundled(source):
        raise ValueError(f"source skill '{source}' is a bundled built-in and cannot be consolidated")
    if skill_usage.is_hub_installed(source):
        raise ValueError(f"source skill '{source}' is hub-installed and cannot be consolidated")
    if skill_usage.is_protected_builtin(source):
        raise ValueError(f"source skill '{source}' is a protected built-in and cannot be consolidated")
    if not skill_usage.is_curation_eligible(source, source_dir):
        raise ValueError(f"source skill '{source}' is not eligible for curator consolidation")
    if not skill_usage.is_curation_eligible(destination, destination_dir):
        raise ValueError(f"destination skill '{destination}' is not eligible for curator consolidation")
    if not skill_usage.is_curator_managed(source):
        raise ValueError(f"source skill '{source}' is not curator-managed")
    if not skill_usage.is_curator_managed(destination):
        raise ValueError(f"destination skill '{destination}' is not curator-managed")
    if skill_usage.get_record(source).get("pinned"):
        raise ValueError(f"source skill '{source}' is pinned; unpin it before consolidation")
    _package_manifest(source_dir)
    _package_manifest(destination_dir)
    if not skill_ledger.ledger_enabled():
        raise ValueError("skill ledger is disabled; consolidation requires a durable receipt")
    jobs_file = get_hermes_home() / "cron" / "jobs.json"
    try:
        cron_bytes = jobs_file.read_bytes()
        parsed = json.loads(cron_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cron store precondition failed: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("jobs"), list):
        raise ValueError("cron store precondition failed: expected a jobs list")
    usage_file = get_hermes_home() / "skills" / ".usage.json"
    try:
        usage_bytes = usage_file.read_bytes()
    except OSError as exc:
        raise ValueError(f"usage state precondition failed: {exc}") from exc
    return source_dir, destination_dir, jobs_file, cron_bytes, usage_bytes


def _restore_after_failure(
    *, source_dir: Path, archive_dir: Path | None, jobs_file: Path, cron_bytes: bytes,
    usage_file: Path, usage_bytes: bytes,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"attempted": True, "source_restored": False, "cron_restored": False, "errors": []}
    try:
        _atomic_write_bytes(jobs_file, cron_bytes)
        result["cron_restored"] = jobs_file.read_bytes() == cron_bytes
    except Exception as exc:  # recovery must report every failed component
        result["errors"].append(f"cron: {exc}")
    try:
        if not source_dir.exists() and archive_dir is not None and archive_dir.exists():
            _move(archive_dir, source_dir)
        _atomic_write_bytes(usage_file, usage_bytes)
        result["source_restored"] = bool(source_dir.is_dir() and _package_manifest(source_dir))
    except Exception as exc:
        result["errors"].append(f"source: {exc}")
    return result


def consolidate_skills(source: str, destination: str, *, actor: str = "user") -> Dict[str, Any]:
    """Archive *source*, forward cron references to *destination*, or recover exactly.

    This is the single mutation primitive used by the public CLI and background
    curator consolidation. It requires a snapshot, durable ledger, complete local
    packages, and a readable cron store before its first mutation.
    """
    operation_id = uuid.uuid4().hex[:12]
    receipt: Dict[str, Any] = {
        "operation_id": operation_id,
        "success": False,
        "source": source,
        "destination": destination,
        "archive_location": None,
        "rollback_handle": None,
        "forwarding": {"rewrites": [], "readback": False},
        "recovery": {"attempted": False},
    }
    archive_dir: Path | None = None
    source_dir: Path | None = None
    jobs_file: Path | None = None
    consolidated_after: list[dict[str, str]] = []
    ledger_id: str | None = None
    cron_bytes = usage_bytes = b""
    try:
        source_dir, _destination_dir, jobs_file, cron_bytes, usage_bytes = _preflight(source, destination)
        from agent import curator_backup
        from cron import jobs as cron_jobs
        from tools import skill_ledger, skill_usage

        snapshot = curator_backup.snapshot_skills(reason=f"pre-consolidation {source} -> {destination}")
        if snapshot is None:
            raise RuntimeError("required pre-operation snapshot could not be created")
        receipt["rollback_handle"] = snapshot.name
        usage_file = get_hermes_home() / "skills" / ".usage.json"
        usage_before = skill_ledger.snapshot_paths(usage_file)
        if not usage_before:
            raise RuntimeError("required usage ledger snapshot could not be captured")
        before = (
            skill_ledger.snapshot_paths(source_dir, complete_package=True)
            + skill_ledger.snapshot_paths(jobs_file)
            + usage_before
        )
        if not before:
            raise RuntimeError("required ledger snapshot could not be captured")

        archive_dir = _archive_destination(source_dir)
        _move(source_dir, archive_dir)
        receipt["archive_location"] = str(archive_dir)
        if _package_manifest(archive_dir) != _package_manifest_from_bytes(before, source_dir):
            raise RuntimeError("archive verification failed: source package hash mismatch")
        if not skill_usage.set_state(source, skill_usage.STATE_ARCHIVED):
            raise RuntimeError("archive lifecycle state could not be persisted")

        rewrites = cron_jobs.rewrite_skill_refs(consolidated={source: destination}, pruned=[])
        readback = cron_jobs.load_jobs()
        _validate_forwarding(readback, source, destination)
        receipt["forwarding"] = {"rewrites": rewrites.get("rewrites", []), "readback": True}

        usage_after = skill_ledger.snapshot_paths(usage_file)
        if not usage_after:
            raise RuntimeError("required usage ledger snapshot could not be captured")
        consolidated_after = (
            skill_ledger.snapshot_paths(archive_dir, complete_package=True)
            + skill_ledger.snapshot_paths(jobs_file)
            + usage_after
        )
        ledger_id = skill_ledger.append_entry(
            "consolidate", source, before=before, after=consolidated_after, actor=actor,
            evidence={"absorbed_into": destination, "archived": True, "archive_location": str(archive_dir),
                      "forwarding": receipt["forwarding"], "rollback_handle": snapshot.name,
                      "operation_id": operation_id, "transaction_created_dirs": [str(archive_dir)]},
        )
        if ledger_id is None:
            raise RuntimeError("durable consolidation ledger receipt could not be written")
        receipt["ledger_entry"] = ledger_id
        receipt["success"] = True
        _write_receipt(receipt)
        return receipt
    except Exception as exc:
        receipt["success"] = False
        receipt["error"] = str(exc)
        if source_dir is not None and jobs_file is not None:
            failed_after: list[dict[str, str]] = []
            # A post-archive failure has real on-disk state worth recording even
            # when the authoritative consolidation ledger was not reached yet.
            # Capture before recovery so this compensating ledger entry is
            # truthful and undoable like the post-receipt recovery path.
            if archive_dir is not None and archive_dir.exists():
                try:
                    from tools import skill_ledger

                    failed_after = (
                        skill_ledger.snapshot_paths(archive_dir, complete_package=True)
                        + skill_ledger.snapshot_paths(jobs_file)
                        + skill_ledger.snapshot_paths(get_hermes_home() / "skills" / ".usage.json")
                    )
                except Exception:
                    failed_after = []
            receipt["recovery"] = _restore_after_failure(
                source_dir=source_dir, archive_dir=archive_dir, jobs_file=jobs_file, cron_bytes=cron_bytes,
                usage_file=get_hermes_home() / "skills" / ".usage.json", usage_bytes=usage_bytes,
            )
            if receipt["recovery"]["source_restored"]:
                receipt["archive_location"] = None
            receipt["forwarding"]["recovered"] = receipt["recovery"]["cron_restored"]
            if ledger_id is not None or failed_after:
                try:
                    from tools import skill_ledger

                    recovery_evidence = {
                        "operation_id": operation_id,
                        "recovery_reason": str(exc),
                        "source_restored": receipt["recovery"]["source_restored"],
                        "cron_restored": receipt["recovery"]["cron_restored"],
                        "archived": False,
                    }
                    if ledger_id is not None:
                        recovery_evidence["recovered_consolidation_entry"] = ledger_id
                    recovery_ledger_id = skill_ledger.append_entry(
                        "consolidate-recovery", source,
                        before=consolidated_after if ledger_id is not None else failed_after,
                        after=(
                            skill_ledger.snapshot_paths(source_dir, complete_package=True)
                            + skill_ledger.snapshot_paths(jobs_file)
                            + skill_ledger.snapshot_paths(get_hermes_home() / "skills" / ".usage.json")
                        ),
                        actor=actor,
                        evidence=recovery_evidence,
                    )
                    if recovery_ledger_id is None:
                        receipt["recovery"]["ledger_error"] = "recovery ledger entry could not be written"
                    else:
                        receipt["recovery"]["ledger_entry"] = recovery_ledger_id
                        if ledger_id is not None:
                            receipt["recovery"]["consolidation_ledger_entry"] = ledger_id
                except Exception as ledger_exc:
                    receipt["recovery"]["ledger_error"] = str(ledger_exc)
        try:
            _write_receipt(receipt)
        except Exception as receipt_exc:
            receipt["receipt_error"] = str(receipt_exc)
        return receipt


def _package_manifest_from_bytes(before: list[dict[str, str]], source_dir: Path) -> Dict[str, str]:
    prefix = str(source_dir) + os.sep
    return {item["path"][len(prefix):].replace(os.sep, "/"): item["sha256"]
            for item in before if str(item.get("path", "")).startswith(prefix)}


def _validate_forwarding(jobs: list[dict[str, Any]], source: str, destination: str) -> None:
    from cron.jobs import _normalize_skill_list

    for job in jobs:
        refs = _normalize_skill_list(job.get("skill"), job.get("skills"))
        if source in refs:
            raise RuntimeError(f"cron readback still references source in job {job.get('id')}")
        if destination in refs and refs.count(destination) != 1:
            raise RuntimeError(f"cron readback duplicated destination in job {job.get('id')}")
