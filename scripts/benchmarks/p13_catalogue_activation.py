#!/usr/bin/env python3
"""Fail-closed P13 cron catalogue staging and activation utility.

Planning is read-only. Staging always creates jobs disabled and requires a
checksum-bound, complete `STAGE_APPROVED` disposition matrix. Enablement
requires the exact KENSEI_MIGRATION_AUTHORITY=!go value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable

GO_AUTHORITY = "!go"
CATALOGUE_STATUS = "PRE_GO_PRIVATE_DISABLED_CATALOGUE"
MIGRATION_MARKER = "P13_DISABLED_CATALOGUE"
RECEIPT_KIND_STAGE = "p13_wave_stage_v1"
RECEIPT_KIND_REGISTRATION = "p13_registration_v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_go_authority(value: str | None) -> None:
    if value != GO_AUTHORITY:
        raise PermissionError(
            "mutating P13 catalogue actions require exact "
            "KENSEI_MIGRATION_AUTHORITY=!go"
        )


def load_catalogue(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise ValueError(f"private catalogue must have mode 0600, found {mode:04o}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"catalogue checksum mismatch: expected {expected_sha256}, found {actual}"
        )
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    if catalogue.get("status") != CATALOGUE_STATUS:
        raise ValueError(f"unexpected catalogue status: {catalogue.get('status')!r}")
    entries = catalogue.get("entries")
    if not isinstance(entries, list):
        raise ValueError("catalogue entries must be a list")
    if len(entries) != catalogue.get("source_denominator"):
        raise ValueError("catalogue denominator does not match entry count")
    if catalogue.get("accounting", {}).get("registered") != 0:
        raise ValueError("catalogue claims pre-existing target registrations")
    if catalogue.get("accounting", {}).get("enabled") != 0:
        raise ValueError("catalogue claims enabled target jobs")
    if any(entry.get("target_registration") or entry.get("target_enabled") for entry in entries):
        raise ValueError("catalogue contains a registered or enabled target entry")
    return catalogue


def select_wave(catalogue: dict[str, Any], wave: str) -> list[dict[str, Any]]:
    selected = [entry for entry in catalogue["entries"] if entry.get("activation_wave") == wave]
    if not selected:
        raise ValueError(f"no catalogue entries found for wave {wave!r}")
    for entry in selected:
        if entry.get("target_status") != "ACTIVATION_CANDIDATE":
            raise ValueError(
                f"wave {wave!r} includes non-candidate {entry.get('source_instance')}: "
                f"{entry.get('target_status')}"
            )
        if entry.get("blockers"):
            raise ValueError(f"candidate has blockers: {entry.get('source_instance')}")
        if not isinstance(entry.get("target_job"), dict):
            raise ValueError(f"candidate lacks target job: {entry.get('source_instance')}")
    return selected


def select_authorised_wave(
    catalogue: dict[str, Any],
    disposition: dict[str, Any],
    wave: str,
    *,
    p13_policy: dict[str, Any] | None = None,
    provider_validation_fn: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return a catalogue wave only when every row is explicitly stage-approved
    and bound to its expected current-contract fingerprint.

    Fails closed when a disposition row is missing its ``contract_fingerprint``
    (no silent skip) and computes the expected fingerprint using the same
    explicit ``p13_policy`` that :func:`stage_wave` will use, so the row
    binding matches the contract that will actually be persisted.

    When ``provider_validation_fn`` is supplied, every explicit P13 policy
    provider resolved for a selected row is checked via that seam; a provider
    the seam reports as disabled fails closed.  No fallback to the root
    global default.
    """
    matrix_entries = [
        item
        for item in disposition.get("entries", [])
        if isinstance(item, dict) and item.get("source_instance")
    ]
    states = {
        item["source_instance"]: item.get("audit_no_stage_state")
        for item in matrix_entries
    }
    row_fingerprints: dict[str, str | None] = {
        item["source_instance"]: item.get("contract_fingerprint")
        for item in matrix_entries
    }
    catalogue_sources = {
        entry.get("source_instance") for entry in catalogue.get("entries", [])
    }
    if (
        len(matrix_entries) != len(states)
        or set(states) != catalogue_sources
        or len(matrix_entries) != len(catalogue_sources)
    ):
        raise ValueError("disposition matrix entry set mismatch")
    selected = select_wave(catalogue, wave)
    for entry in selected:
        source_instance = entry["source_instance"]
        if states.get(source_instance) != "STAGE_APPROVED":
            raise ValueError(f"entry is not stage-approved: {source_instance}")
        expected_fp = _target_fingerprint(entry, p13_policy=p13_policy)
        row_fp = row_fingerprints.get(source_instance)
        # Fail closed: a missing contract_fingerprint is a rejected binding,
        # not a silently skipped one.
        if not row_fp:
            raise ValueError(
                f"disposition row for {source_instance} is missing "
                f"contract_fingerprint; fail-closed"
            )
        if row_fp != expected_fp:
            raise ValueError(
                f"contract fingerprint mismatch for {source_instance}: "
                f"disposition row does not match expected current contract"
            )
        # Provider validation seam: explicit P13 policy providers must not be
        # explicitly disabled in the effective config.
        if provider_validation_fn is not None:
            resolved_provider, _ = _resolve_p13_policy(
                entry, p13_policy=p13_policy
            )
            if resolved_provider is not None and not provider_validation_fn(resolved_provider):
                raise ValueError(
                    f"P13 policy provider {resolved_provider!r} for "
                    f"{source_instance} is disabled in the effective config; "
                    f"refusing to stage"
                )
    return selected


def load_disposition_matrix(path: Path, catalogue_sha256: str) -> dict[str, Any]:
    """Load the independent row-authority matrix bound to this catalogue."""
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("catalogue_sha256") != catalogue_sha256:
        raise ValueError("disposition matrix catalogue checksum mismatch")
    if not isinstance(matrix.get("entries"), list):
        raise ValueError("disposition matrix entries must be a list")
    return matrix


def _schedule_text(schedule: Any, display: str | None = None) -> str:
    if isinstance(schedule, str):
        return schedule
    if not isinstance(schedule, dict):
        raise ValueError(f"unsupported schedule: {schedule!r}")
    kind = schedule.get("kind")
    if kind == "cron" and schedule.get("expr"):
        return str(schedule["expr"])
    if kind == "interval" and schedule.get("minutes") is not None:
        minutes = schedule["minutes"]
        return f"every {minutes}m"
    if kind == "once" and (schedule.get("run_at") or schedule.get("timestamp")):
        return str(schedule.get("run_at") or schedule.get("timestamp"))
    if display:
        return display
    raise ValueError(f"cannot convert schedule: {schedule!r}")


def _normalize_model_provider(
    value: Any, *, field: str, source_instance: str
) -> str | None:
    """Normalize a legacy model/provider value to a string or fail closed.

    Legacy catalogues may carry ``{model: ..., provider: ...}`` dicts.  A bare
    string passes through.  A dict with a single ``name`` key (or a flat dict
    whose first value is a string) is extracted.  Anything else fails closed
    to prevent silent loss of inference configuration.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        # Common legacy shapes: {"name": "..."} or {"model": "..."} / {"provider": "..."}
        for key in (field, "name"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        # Flat dict with a single string value as a last resort.
        str_values = [v for v in value.values() if isinstance(v, str) and v.strip()]
        if len(str_values) == 1:
            return str_values[0].strip()
        raise ValueError(
            f"cannot normalize dict {field} for {source_instance}: "
            f"no single string value in {value!r}"
        )
    raise ValueError(
        f"unsupported {field} type {type(value).__name__} for {source_instance}"
    )


def _resolve_p13_policy(
    entry: dict[str, Any],
    *,
    p13_policy: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Resolve model/provider for an unpinned agent job via explicit P13 policy.

    Returns (provider, model).  When the job is already pinned (has explicit
    model/provider strings), returns (None, None) — caller should keep the
    pinned values.  When unpinned and no_agent, returns (None, None) — no
    inference axes needed.  When unpinned and agent-bound, requires an
    explicit policy mapping (per-source or a declared default); never falls
    back to the root global default.
    """
    job = entry["target_job"]
    if bool(job.get("no_agent")):
        return None, None
    has_model = _normalize_model_provider(job.get("model"), field="model", source_instance=entry["source_instance"])
    has_provider = _normalize_model_provider(job.get("provider"), field="provider", source_instance=entry["source_instance"])
    if has_model is not None or has_provider is not None:
        return None, None  # already pinned
    # Unpinned agent job — require explicit P13 policy.
    if not p13_policy:
        raise ValueError(
            f"unpinned agent job {entry['source_instance']} requires an "
            f"explicit P13 policy mapping; refusing to fall back to root global default"
        )
    source_instance = entry["source_instance"]
    mapping = p13_policy.get(source_instance) or p13_policy.get("default")
    if not isinstance(mapping, dict):
        raise ValueError(
            f"no P13 policy mapping for {source_instance} and no declared default"
        )
    provider = mapping.get("provider")
    model = mapping.get("model")
    if not (isinstance(provider, str) and provider.strip() and isinstance(model, str) and model.strip()):
        raise ValueError(
            f"P13 policy mapping for {source_instance} must define non-empty "
            f"provider and model strings"
        )
    return provider.strip(), model.strip()


def build_create_kwargs(
    entry: dict[str, Any],
    *,
    p13_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job = entry["target_job"]
    source_instance = entry["source_instance"]
    if job.get("profile") or job.get("fallback_providers"):
        raise ValueError(
            f"unsupported legacy execution fields for {source_instance}"
        )
    if job.get("deliver") == "origin":
        raise ValueError(
            f"origin delivery cannot be reconstructed safely for {source_instance}"
        )
    repeat = job.get("repeat")
    repeat_times = repeat.get("times") if isinstance(repeat, dict) else repeat
    # Normalize legacy dict model/provider to strings or fail closed.
    model = _normalize_model_provider(job.get("model"), field="model", source_instance=source_instance)
    provider = _normalize_model_provider(job.get("provider"), field="provider", source_instance=source_instance)
    # Require explicit P13 policy for unpinned agent jobs.
    policy_provider, policy_model = _resolve_p13_policy(entry, p13_policy=p13_policy)
    if policy_provider is not None:
        provider = policy_provider
    if policy_model is not None:
        model = policy_model
    return {
        "prompt": job.get("prompt") or "",
        "schedule": _schedule_text(job.get("schedule"), job.get("schedule_display")),
        "name": job.get("name"),
        "repeat": repeat_times,
        "deliver": job.get("deliver") or "local",
        "skill": job.get("skill"),
        "skills": job.get("skills"),
        "model": model,
        "provider": provider,
        "base_url": job.get("base_url"),
        "script": job.get("script"),
        "context_from": None,
        "enabled_toolsets": job.get("enabled_toolsets"),
        "workdir": job.get("workdir"),
        "no_agent": bool(job.get("no_agent")),
        "enabled": False,
    }


def _target_fingerprint(
    entry: dict[str, Any], *, p13_policy: dict[str, Any] | None = None
) -> str:
    return hashlib.sha256(
        _canonical(build_create_kwargs(entry, p13_policy=p13_policy)).encode()
    ).hexdigest()


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _persisted_schedule_text(job: dict[str, Any]) -> str:
    """Extract the schedule as the contract string from a persisted job.

    Real cron jobs store a parsed schedule dict in ``schedule`` and the
    human-readable form in ``schedule_display``.  Mock jobs (unit tests)
    may store the raw string directly in ``schedule``.  Normalize both to
    the contract string for comparison.
    """
    display = job.get("schedule_display")
    if isinstance(display, str) and display.strip():
        return display.strip()
    schedule = job.get("schedule")
    if isinstance(schedule, str):
        return schedule
    if isinstance(schedule, dict):
        return _schedule_text(schedule, display)
    return ""


def _persisted_contract(job: dict[str, Any]) -> dict[str, Any]:
    """Extract the runtime-contract fields from a persisted cron job.

    This is the set of fields that must match the expected current contract
    for a reused P13 job.  We compare the *actual persisted values*, not the
    stored origin fingerprint, so that drift introduced via update_job (which
    permits origin and payload updates) is detected.
    """
    return {
        "prompt": job.get("prompt") or "",
        "schedule": _persisted_schedule_text(job),
        "skills": job.get("skills") or [],
        "model": job.get("model"),
        "provider": job.get("provider"),
        "base_url": job.get("base_url"),
        "script": job.get("script"),
        "no_agent": bool(job.get("no_agent")),
        "deliver": job.get("deliver") or "local",
        "workdir": job.get("workdir"),
        "enabled_toolsets": job.get("enabled_toolsets"),
        "enabled": bool(job.get("enabled")),
    }


def _receipt_target_for_source(receipt_path: Path, source_instance: str) -> str | None:
    """Return the target_id recorded for *source_instance* in a prior stage
    receipt at *receipt_path*, or None when no receipt exists.

    Fail-closed: a non-stage receipt (e.g. a registration receipt) raises
    ValueError so a tampered receipt can never silently downgrade to the
    by-source index.  The receipt\'s catalogue_sha256 is NOT validated
    here — the caller (stage_wave) validates the catalogue sha against the
    persisted origin separately.  We only need the source_instance→target_id
    mapping from the prior stage.
    """
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("migration") != MIGRATION_MARKER:
        raise ValueError(f"prior receipt at {receipt_path} is not a migration receipt")
    kind = receipt.get("receipt_kind")
    if kind != RECEIPT_KIND_STAGE:
        raise ValueError(
            f"prior receipt at {receipt_path} is not a stage receipt "
            f"(kind={kind!r}); refusing to use it for reuse authority"
        )
    for item in receipt.get("jobs", []):
        if isinstance(item, dict) and item.get("source_instance") == source_instance:
            tid = item.get("target_id")
            if tid:
                return str(tid)
    return None


def _find_job_by_id(jobs: list[dict[str, Any]], job_id: str) -> dict[str, Any] | None:
    for job in jobs:
        if job.get("id") == job_id:
            return job
    return None


def stage_wave(
    entries: list[dict[str, Any]],
    *,
    catalogue_sha256: str,
    disposition_sha256: str,
    receipt_path: Path,
    create_job_fn: Callable[..., dict[str, Any]],
    list_jobs_fn: Callable[..., list[dict[str, Any]]],
    update_job_fn: Callable[[str, dict[str, Any]], dict[str, Any] | None],
    remove_job_fn: Callable[[str], bool],
    p13_policy: dict[str, Any] | None = None,
    provider_validation_fn: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    existing_jobs = list_jobs_fn(include_disabled=True)
    by_source: dict[str, dict[str, Any]] = {}
    for job in existing_jobs:
        origin = job.get("origin")
        if isinstance(origin, dict) and origin.get("migration") == MIGRATION_MARKER:
            source_instance = origin.get("source_instance")
            if source_instance:
                by_source[str(source_instance)] = job

    created_ids: list[str] = []
    mapping: dict[str, str] = {}
    source_id_mapping: dict[str, str] = {}
    reused = 0
    job_fingerprints: dict[str, str] = {}
    # Provider validation seam: every explicit P13 policy provider must
    # be checked as not explicitly disabled in the effective config.  No
    # fallback to the root global default.
    if provider_validation_fn is not None:
        for entry in entries:
            resolved_provider, _ = _resolve_p13_policy(entry, p13_policy=p13_policy)
            if resolved_provider is not None and not provider_validation_fn(resolved_provider):
                raise ValueError(
                    f"P13 policy provider {resolved_provider!r} for "
                    f"{entry['source_instance']} is disabled in the effective "
                    f"config; refusing to stage"
                )
    try:
        for entry in entries:
            source_instance = entry["source_instance"]
            fingerprint = _target_fingerprint(entry, p13_policy=p13_policy)
            job_fingerprints[source_instance] = fingerprint
            current = by_source.get(source_instance)
            # Authoritative prior-stage mapping: when a prior stage receipt
            # exists at receipt_path, the source_instance->target_id binding
            # recorded there is the authority for reuse.  We do NOT trust the
            # by_source index alone, because update_job can rebind a job's
            # origin to a different source_instance (creating an orphan) or
            # steal another job's source_instance (identity theft).  The
            # receipt records the exact target_id that was staged for this
            # source_instance; we validate that target still exists and its
            # full persisted contract/origin match expected.
            prior_target_id = _receipt_target_for_source(receipt_path, source_instance)
            if prior_target_id is not None:
                receipt_job = _find_job_by_id(list_jobs_fn(include_disabled=True), prior_target_id)
                if receipt_job is None:
                    raise ValueError(
                        f"prior stage receipt named target {prior_target_id!r} "
                        f"for {source_instance} but that job no longer exists; "
                        f"refusing to re-stage (possible orphan/rebind)"
                    )
                # The receipt's target must be the authoritative reused job.
                # If by_source points at a *different* job, that is a rebind.
                if current is not None and current.get("id") != prior_target_id:
                    raise ValueError(
                        f"origin rebind detected for staged source {source_instance}: "
                        f"by-source index points at {current.get('id')!r} but prior "
                        f"stage receipt recorded target {prior_target_id!r}"
                    )
                current = receipt_job
            if current is not None:
                origin = current.get("origin") or {}
                # Reject origin rebinding: the persisted origin must still
                # name this source_instance (update_job permits origin updates,
                # so we re-validate the *actual* origin on the persisted job).
                if origin.get("source_instance") != source_instance:
                    raise ValueError(
                        f"origin rebind detected for staged job {source_instance}: "
                        f"persisted origin names {origin.get('source_instance')!r}"
                    )
                if origin.get("catalogue_sha256") != catalogue_sha256:
                    raise ValueError(f"catalogue checksum drift for staged job {source_instance}")
                # Strict target_fingerprint check: the persisted origin's
                # target_fingerprint must equal the recomputed expected
                # fingerprint for the current contract.  A mismatch means the
                # contract was changed (via update_job) and the stored
                # fingerprint was either not updated or tampered with.
                origin_fp = origin.get("target_fingerprint")
                if not origin_fp:
                    raise ValueError(
                        f"staged job {source_instance} origin is missing "
                        f"target_fingerprint; fail-closed"
                    )
                if origin_fp != fingerprint:
                    raise ValueError(
                        f"target fingerprint mismatch for staged job {source_instance}: "
                        f"origin target_fingerprint {origin_fp!r} != expected {fingerprint!r}"
                    )
                # Re-validate the *persisted* job against the expected current
                # contract, not just the stored origin fingerprint.  This
                # catches drift introduced via update_job even if the stored
                # fingerprint was also mutated.
                persisted_contract = _persisted_contract(current)
                expected_kwargs = build_create_kwargs(entry, p13_policy=p13_policy)
                expected_contract = {
                    "prompt": expected_kwargs["prompt"],
                    "schedule": expected_kwargs["schedule"],
                    "skills": expected_kwargs["skills"] or [],
                    "model": expected_kwargs["model"],
                    "provider": expected_kwargs["provider"],
                    "base_url": expected_kwargs["base_url"],
                    "script": expected_kwargs["script"],
                    "no_agent": expected_kwargs["no_agent"],
                    "deliver": expected_kwargs["deliver"],
                    "workdir": expected_kwargs["workdir"],
                    "enabled_toolsets": expected_kwargs["enabled_toolsets"],
                    "enabled": expected_kwargs["enabled"],
                }
                if persisted_contract != expected_contract:
                    raise ValueError(
                        f"persisted contract drift for staged job {source_instance}: "
                        f"actual {persisted_contract!r} != expected {expected_contract!r}"
                    )
                if current.get("enabled") or current.get("state") != "paused":
                    raise ValueError(f"staged job is not paused: {current.get('id')}")
                target_id = current["id"]
                reused += 1
            else:
                kwargs = build_create_kwargs(entry, p13_policy=p13_policy)
                kwargs["origin"] = {
                    "migration": MIGRATION_MARKER,
                    "source_instance": source_instance,
                    "catalogue_sha256": catalogue_sha256,
                    "target_fingerprint": fingerprint,
                }
                created = create_job_fn(**kwargs)
                if created.get("enabled") or created.get("state") != "paused":
                    raise RuntimeError(f"create_job did not persist disabled: {source_instance}")
                target_id = created["id"]
                created_ids.append(target_id)
            mapping[source_instance] = target_id
            source_id_mapping[entry["source_id"]] = target_id

        current_ids = {job.get("id") for job in list_jobs_fn(include_disabled=True)} | set(mapping.values())
        for entry in entries:
            refs = entry["target_job"].get("context_from") or []
            if isinstance(refs, str):
                refs = [refs]
            remapped: list[str] = []
            for ref in refs:
                target_ref = source_id_mapping.get(ref, ref)
                if target_ref not in current_ids:
                    raise ValueError(
                        f"unresolved context dependency {ref!r} for {entry['source_instance']}"
                    )
                remapped.append(target_ref)
            if remapped:
                updated = update_job_fn(mapping[entry["source_instance"]], {"context_from": remapped})
                if updated is None:
                    raise RuntimeError(f"failed to update dependencies for {entry['source_instance']}")
    except Exception:
        for job_id in reversed(created_ids):
            remove_job_fn(job_id)
        raise

    receipt = {
        "schema_version": 1,
        "receipt_kind": RECEIPT_KIND_STAGE,
        "migration": MIGRATION_MARKER,
        "catalogue_sha256": catalogue_sha256,
        "disposition_sha256": disposition_sha256,
        "wave": entries[0]["activation_wave"] if entries else None,
        "jobs": [
            {
                "source_instance": entry["source_instance"],
                "source_id": entry["source_id"],
                "target_id": mapping[entry["source_instance"]],
                "target_fingerprint": job_fingerprints[entry["source_instance"]],
            }
            for entry in entries
        ],
    }
    _write_private_json(receipt_path, receipt)
    return {
        "wave": receipt["wave"],
        "created": len(created_ids),
        "reused": reused,
        "receipt": str(receipt_path.resolve()),
        "target_ids": [job["target_id"] for job in receipt["jobs"]],
    }


def _read_receipt(
    path: Path,
    catalogue_sha256: str,
    *,
    expected_kind: str | None = None,
    expected_wave: str | None = None,
) -> dict[str, Any]:
    """Load and optionally strictly validate a migration receipt.

    When *expected_kind* is given the receipt is fail-closed validated: the
    receipt_kind must match, the jobs list must be non-empty, every job must
    carry a target_id, and the receipt wave must be present.  If
    *expected_wave* is also given it must match the receipt wave.  Any
    deviation — registration receipt, legacy receipt (no kind), malformed
    payload, empty jobs, or out-of-wave binding — raises ValueError so the
    caller never reaches resume_job / pause_job.
    """
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("migration") != MIGRATION_MARKER:
        raise ValueError("invalid migration receipt")
    if receipt.get("catalogue_sha256") != catalogue_sha256:
        raise ValueError("receipt catalogue checksum mismatch")
    if expected_kind is not None:
        kind = receipt.get("receipt_kind")
        if kind != expected_kind:
            raise ValueError(
                f"receipt kind {kind!r} is not a valid {expected_kind} receipt; "
                f"refusing to proceed"
            )
        jobs = receipt.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("receipt has no staged jobs; refusing to proceed")
        wave = receipt.get("wave")
        if not wave:
            raise ValueError("receipt is missing wave binding; refusing to proceed")
        for item in jobs:
            if not isinstance(item, dict):
                raise ValueError("receipt job is malformed; refusing to proceed")
            for required in ("source_instance", "source_id", "target_id"):
                if not item.get(required):
                    raise ValueError(
                        f"receipt job missing {required}; refusing to proceed"
                    )
        if expected_wave is not None and wave != expected_wave:
            raise ValueError(
                f"receipt wave {wave!r} does not match expected wave {expected_wave!r}"
            )
    return receipt


def enable_receipt(
    receipt: dict[str, Any],
    *,
    resume_job_fn: Callable[[str], dict[str, Any] | None],
    pause_job_fn: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    enabled: list[str] = []
    try:
        for item in receipt["jobs"]:
            target_id = item["target_id"]
            result = resume_job_fn(target_id)
            if result is None or not result.get("enabled"):
                raise RuntimeError(f"failed to enable {target_id}")
            enabled.append(target_id)
    except Exception:
        for target_id in reversed(enabled):
            pause_job_fn(target_id, reason="p13_wave_enable_rollback")
        raise
    return {"enabled": len(enabled), "target_ids": enabled}


def pause_receipt(
    receipt: dict[str, Any],
    *,
    pause_job_fn: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    paused: list[str] = []
    for item in receipt["jobs"]:
        target_id = item["target_id"]
        result = pause_job_fn(target_id, reason="p13_operator_rollback")
        if result is not None:
            paused.append(target_id)
    return {"paused": len(paused), "target_ids": paused}


def _build_provider_validation_fn() -> Callable[[str], bool] | None:
    """Build a provider validation seam from the effective config.

    Returns a callable ``(provider_name) -> bool`` that reports whether the
    provider is enabled in the effective config, or None when no config can
    be loaded (e.g. running outside a Hermes home).  An explicitly disabled
    provider (``enabled: false`` under ``providers.<name>``) reports False.

    The seam is injectable so tests can supply a disposable config without
    touching live user config.
    """
    try:
        from hermes_cli.config import is_provider_enabled, load_config
    except Exception:
        return None
    try:
        cfg = load_config()
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        # No providers section — nothing is explicitly disabled.
        return None

    def _is_enabled(provider_name: str) -> bool:
        # Only an explicit enabled:false under a matching provider block
        # disables the provider.  Unknown providers default to enabled.
        for _key, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            names = {str(entry.get("name") or "").strip().lower(),
                     str(_key).strip().lower()}
            if provider_name.strip().lower() in names:
                return is_provider_enabled(entry)
        return True

    return _is_enabled


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "stage", "enable", "pause"])
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--disposition", type=Path)
    parser.add_argument("--wave")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--p13-policy", type=Path,
                        help="JSON file with explicit P13 model/provider policy mappings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalogue = load_catalogue(args.catalogue, args.sha256)
    entries: list[dict[str, Any]] = []

    p13_policy = None
    if args.p13_policy:
        p13_policy = json.loads(args.p13_policy.read_text(encoding="utf-8"))
        if not isinstance(p13_policy, dict):
            raise ValueError("P13 policy file must be a JSON object")

    if args.action in {"plan", "stage"}:
        if not args.wave:
            raise SystemExit("--wave is required for plan/stage")
        if not args.disposition:
            raise SystemExit("--disposition is required for plan/stage")
        disposition = load_disposition_matrix(args.disposition, args.sha256)
        provider_validation_fn = _build_provider_validation_fn()
        entries = select_authorised_wave(
            catalogue,
            disposition,
            args.wave,
            p13_policy=p13_policy,
            provider_validation_fn=provider_validation_fn,
        )
        if args.action == "plan":
            print(json.dumps({"wave": args.wave, "jobs": len(entries), "mutation": False}, indent=2))
            return 0

    if not args.receipt:
        raise SystemExit("--receipt is required for stage/enable/pause")

    from cron.jobs import create_job, list_jobs, pause_job, remove_job, resume_job, update_job

    if args.action == "pause":
        receipt = _read_receipt(
            args.receipt, args.sha256, expected_kind=RECEIPT_KIND_STAGE
        )
        print(json.dumps(pause_receipt(receipt, pause_job_fn=pause_job), indent=2))
        return 0

    if args.action == "enable":
        if not args.wave:
            raise SystemExit("--wave is required for enable")
        require_go_authority(os.environ.get("KENSEI_MIGRATION_AUTHORITY"))
    if args.action == "stage":
        result = stage_wave(
            entries,
            catalogue_sha256=args.sha256,
            disposition_sha256=_sha256(args.disposition),
            receipt_path=args.receipt,
            create_job_fn=create_job,
            list_jobs_fn=list_jobs,
            update_job_fn=update_job,
            remove_job_fn=remove_job,
            p13_policy=p13_policy,
            provider_validation_fn=provider_validation_fn,
        )
    else:
        receipt = _read_receipt(
            args.receipt,
            args.sha256,
            expected_kind=RECEIPT_KIND_STAGE,
            expected_wave=args.wave,
        )
        result = enable_receipt(receipt, resume_job_fn=resume_job, pause_job_fn=pause_job)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
