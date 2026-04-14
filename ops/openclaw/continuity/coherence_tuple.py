#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional

POLICY_TRACKED_PATHS = [
    "ops/openclaw/architecture/swarm_role_contracts.v1.yaml",
    "ops/openclaw/architecture/web_interaction_idd.v1.yaml",
    "ops/openclaw/architecture/ui_design_edd.v1.yaml",
    "ops/openclaw/architecture/trading_terminal_design_language.v1.yaml",
    "ops/openclaw/architecture/competitive_parity_harness.v1.yaml",
    "ops/openclaw/architecture/ground_truth_connectors.v2.yaml",
    "ops/openclaw/architecture/schemas/web_capture_macro.schema.json",
    "ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json",
    "ops/openclaw/architecture/schemas/competitive_scorecard.schema.json",
    "ops/openclaw/architecture/schemas/gtc_evidence.schema.json",
    "ops/openclaw/architecture/schemas/gtc_latest.schema.json",
    "ops/openclaw/architecture/schemas/gtc_event.schema.json",
    "ops/openclaw/architecture/schemas/web_capture_scheduler_state.schema.json",
    "ops/web_capture/macros/bybit_derivatives_capture.yaml",
]

EVALUATOR_TRACKED_PATHS = [
    "ops/openclaw/architecture/validate_contracts.sh",
    "ops/openclaw/architecture/check_swarm_operability.sh",
    "ops/openclaw/continuity/db_integrity_check.sh",
    "ops/openclaw/continuity/verify_then_resume.sh",
    "ops/openclaw/continuity/continuity_now.sh",
    "ops/openclaw/continuity/continuity_current.sh",
]

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover - helper import is optional
    _helper_now_iso_utc = None
    _helper_now_ts = None


def now_dt() -> dt.datetime:
    if _helper_now_ts is not None:
        try:
            return dt.datetime.fromtimestamp(int(_helper_now_ts()), tz=dt.timezone.utc)
        except Exception:
            pass
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return now_dt().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(value: Any) -> Optional[dt.datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        d = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def atomic_write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = pathlib.Path(fh.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def to_rel(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def read_boot_id() -> str:
    p = pathlib.Path("/proc/sys/kernel/random/boot_id")
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    except Exception:
        pass
    return "boot_id_unknown"


def _hash_inputs(root: pathlib.Path, rel_paths: List[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for rel in rel_paths:
        p = (root / rel).resolve()
        sha = sha256_file(p)
        exists = sha is not None
        if not exists:
            missing.append(rel)
        rows.append({"path": rel, "exists": exists, "sha256": sha})
    digest = sha256_text(canonical_json(rows))
    return {
        "digest": digest,
        "items": rows,
        "missing_paths": missing,
    }


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _policy_epoch_candidates(doc: Dict[str, Any]) -> List[int]:
    if not isinstance(doc, dict):
        return []

    candidates: List[Optional[int]] = []

    policy = doc.get("policy") if isinstance(doc.get("policy"), dict) else {}
    tuple_obj = doc.get("tuple") if isinstance(doc.get("tuple"), dict) else {}
    tuple_policy = tuple_obj.get("policy") if isinstance(tuple_obj.get("policy"), dict) else {}

    coherence = doc.get("coherence") if isinstance(doc.get("coherence"), dict) else {}
    coherence_policy = coherence.get("policy") if isinstance(coherence.get("policy"), dict) else {}

    coherence_stamp = doc.get("coherence_stamp") if isinstance(doc.get("coherence_stamp"), dict) else {}
    coherence_stamp_policy = coherence_stamp.get("policy") if isinstance(coherence_stamp.get("policy"), dict) else {}
    coherence_stamp_tuple = coherence_stamp.get("tuple") if isinstance(coherence_stamp.get("tuple"), dict) else {}
    coherence_stamp_tuple_policy = (
        coherence_stamp_tuple.get("policy") if isinstance(coherence_stamp_tuple.get("policy"), dict) else {}
    )

    continuity_now = doc.get("continuity_now") if isinstance(doc.get("continuity_now"), dict) else {}
    continuity_now_coherence = (
        continuity_now.get("coherence") if isinstance(continuity_now.get("coherence"), dict) else {}
    )
    continuity_now_policy = (
        continuity_now_coherence.get("policy") if isinstance(continuity_now_coherence.get("policy"), dict) else {}
    )

    for raw in [
        doc.get("policy_epoch"),
        policy.get("policy_epoch"),
        tuple_policy.get("policy_epoch"),
        coherence_policy.get("policy_epoch"),
        coherence_stamp_policy.get("policy_epoch"),
        coherence_stamp_tuple_policy.get("policy_epoch"),
        continuity_now_policy.get("policy_epoch"),
    ]:
        candidates.append(_as_positive_int(raw))

    return [int(x) for x in candidates if isinstance(x, int) and x > 0]


def _published_policy_epoch_floor(root: pathlib.Path) -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []
    max_epoch = 0

    candidate_paths = [
        root / "state" / "continuity" / "current.json",
        root / "state" / "continuity" / "latest" / "coherence_stamp.json",
        root / "state" / "continuity" / "latest" / "coherence_bundle_latest.json",
    ]

    for p in candidate_paths:
        doc = load_json(p)
        epochs = _policy_epoch_candidates(doc)
        if not epochs:
            continue
        epoch = max(epochs)
        max_epoch = max(max_epoch, epoch)
        sources.append(
            {
                "path": to_rel(root, p),
                "policy_epoch": epoch,
            }
        )

    sources.sort(key=lambda row: str(row.get("path") or ""))
    return {
        "policy_epoch_floor": max_epoch,
        "sources": sources,
    }


def compute_policy_freshness(root: pathlib.Path, *, update_epoch: bool = False) -> Dict[str, Any]:
    policy = _hash_inputs(root, POLICY_TRACKED_PATHS)
    evaluator = _hash_inputs(root, EVALUATOR_TRACKED_PATHS)
    signature = sha256_text(f"{policy['digest']}|{evaluator['digest']}")

    state_path = root / "state" / "continuity" / "latest" / "policy_freshness_state.json"
    state = load_json(state_path)
    prev_sig = str(state.get("signature") or "")
    prev_epoch_raw = state.get("policy_epoch")
    try:
        prev_epoch = max(0, int(prev_epoch_raw))
    except Exception:
        prev_epoch = 0

    published_floor = _published_policy_epoch_floor(root)
    published_epoch_floor = max(0, int(published_floor.get("policy_epoch_floor") or 0))
    epoch_base = max(prev_epoch, published_epoch_floor)

    epoch_resolution = "steady"
    if update_epoch:
        if not prev_sig:
            if prev_epoch > 0:
                # Signatureless epoch history is unanchored; re-anchor with explicit movement.
                epoch = epoch_base + 1
                epoch_resolution = "signature_missing_reanchor"
            else:
                epoch = max(epoch_base, 1)
                epoch_resolution = "bootstrap" if epoch_base <= 0 else "recovered_from_published_floor"
        elif prev_sig != signature:
            epoch = epoch_base + 1
            epoch_resolution = "signature_changed_increment"
        else:
            epoch = epoch_base
            epoch_resolution = "state_clamped_to_published_floor" if published_epoch_floor > prev_epoch else "steady"

        state_out = {
            "schema_version": "continuity.policy_freshness_state.v1",
            "updated_at": now_iso(),
            "policy_epoch": epoch,
            "signature": signature,
            "policy_digest": policy["digest"],
            "evaluator_hash": evaluator["digest"],
            "missing_paths": sorted(set(policy["missing_paths"] + evaluator["missing_paths"])),
            "epoch_resolution": epoch_resolution,
            "published_policy_epoch_floor": published_epoch_floor,
            "published_policy_epoch_floor_sources": published_floor.get("sources") or [],
        }
        atomic_write_json(state_path, state_out)
    else:
        epoch = epoch_base if epoch_base > 0 else 1
        if published_epoch_floor > prev_epoch:
            epoch_resolution = "read_only_floor_recovery"

    return {
        "policy_epoch": epoch,
        "policy_digest": policy["digest"],
        "evaluator_hash": evaluator["digest"],
        "signature": signature,
        "policy_missing_paths": policy["missing_paths"],
        "evaluator_missing_paths": evaluator["missing_paths"],
        "state_path": to_rel(root, state_path),
        "computed_at": now_iso(),
        "epoch_resolution": epoch_resolution,
        "published_policy_epoch_floor": published_epoch_floor,
        "published_policy_epoch_floor_sources": published_floor.get("sources") or [],
    }


def compute_connector_freshness(root: pathlib.Path) -> Dict[str, Any]:
    gtc_current_path = root / "state" / "gtc-v2" / "latest" / "continuity_current.json"
    gateboard_path = root / "state" / "gtc-v2" / "latest" / "gateboard.json"

    gtc_current = load_json(gtc_current_path)
    gateboard = load_json(gateboard_path)

    required = {str(x) for x in (gateboard.get("required_connectors") or []) if str(x).strip()}
    if not required:
        required = {"runtime.gateway::gateway-main", "validation.gates::core"}

    now = now_dt()
    current_boot_id = read_boot_id()
    issuer_boot_default = str(gateboard.get("issuer_boot_id") or gtc_current.get("issuer_boot_id") or "").strip() or None

    rows = gtc_current.get("connectors") if isinstance(gtc_current.get("connectors"), list) else []
    items: List[Dict[str, Any]] = []
    blocking: List[str] = []
    warning: List[str] = []
    critical_valid_until: List[dt.datetime] = []

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        ctype = str(raw.get("connector_type") or "").strip()
        cid = str(raw.get("connector_id") or "").strip()
        if not ctype or not cid:
            continue
        key = f"{ctype}::{cid}"
        observed_at = str(raw.get("observed_at") or "").strip() or None
        obs_dt = parse_iso(observed_at)
        ttl_ms = int(raw.get("freshness_ttl_ms") or 0)
        if ttl_ms < 0:
            ttl_ms = 0

        valid_until_dt = (obs_dt + dt.timedelta(milliseconds=ttl_ms)) if obs_dt is not None else None
        valid_until = valid_until_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z") if valid_until_dt else None
        expired = True if valid_until_dt is None else now > valid_until_dt

        issuer_boot_id = str(raw.get("issuer_boot_id") or issuer_boot_default or "").strip() or None
        boot_mismatch = bool(issuer_boot_id and current_boot_id and issuer_boot_id != current_boot_id)

        stale_severity = str(raw.get("stale_severity") or "warning")
        critical = bool(key in required or stale_severity == "critical")
        stale_suppressed_reason = str(raw.get("stale_suppressed") or "").strip() or None
        stale_suppressed = bool(stale_suppressed_reason and expired and not critical)

        monotonic_seq = raw.get("monotonic_seq")
        try:
            connector_epoch = int(monotonic_seq)
        except Exception:
            connector_epoch = 0

        item = {
            "name": key,
            "connector_type": ctype,
            "connector_id": cid,
            "connector_epoch": connector_epoch,
            "connector_digest": sha256_text(
                canonical_json(
                    {
                        "key": key,
                        "latest_evidence_id": raw.get("latest_evidence_id"),
                        "monotonic_seq": connector_epoch,
                        "observed_at": observed_at,
                        "valid_until": valid_until,
                        "issuer_boot_id": issuer_boot_id,
                    }
                )
            ),
            "lease_id": raw.get("latest_evidence_id"),
            "issued_at": observed_at,
            "valid_until": valid_until,
            "critical": critical,
            "issuer_boot_id": issuer_boot_id,
            "expired": expired,
            "boot_mismatch": boot_mismatch,
            "stale_suppressed": stale_suppressed_reason,
        }
        items.append(item)

        if critical and valid_until_dt is not None:
            critical_valid_until.append(valid_until_dt)

        if critical and expired:
            blocking.append(f"connector_expired:{key}")
        elif (not critical) and expired and not stale_suppressed:
            warning.append(f"connector_expired:{key}")
        if critical and boot_mismatch:
            blocking.append(f"connector_boot_mismatch:{key}")
        elif (not critical) and boot_mismatch:
            warning.append(f"connector_boot_mismatch:{key}")

    if not rows:
        blocking.append("connector_surface_missing")

    critical_valid_until_min = None
    if critical_valid_until:
        critical_valid_until_min = min(critical_valid_until).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    vector_digest = sha256_text(
        canonical_json(
            [
                {
                    "name": item.get("name"),
                    "connector_epoch": item.get("connector_epoch"),
                    "lease_id": item.get("lease_id"),
                    "valid_until": item.get("valid_until"),
                    "issuer_boot_id": item.get("issuer_boot_id"),
                }
                for item in sorted(items, key=lambda x: str(x.get("name") or ""))
            ]
        )
    )

    return {
        "vector_digest": vector_digest,
        "items": sorted(items, key=lambda x: str(x.get("name") or "")),
        "critical_valid_until_min": critical_valid_until_min,
        "required_connectors": sorted(required),
        "blocking_reasons": sorted(set(blocking)),
        "warning_reasons": sorted(set(warning)),
        "current_boot_id": current_boot_id,
        "source_paths": {
            "gtc_current": to_rel(root, gtc_current_path),
            "gateboard": to_rel(root, gateboard_path),
        },
    }


def compute_scheduler_head(root: pathlib.Path) -> Dict[str, Any]:
    state_path = root / "state" / "continuity" / "latest" / "web_capture_scheduler_state.json"
    state = load_json(state_path)
    payload_sha = sha256_file(state_path)
    updated_at = str(state.get("updated_at") or "").strip() or None

    try:
        freshness_limit_sec = max(0, int(os.environ.get("OPENCLAW_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC", "21600")))
    except Exception:
        freshness_limit_sec = 21600

    fresh = None
    if updated_at:
        d = parse_iso(updated_at)
        if d is not None:
            fresh = max(0, int((now_dt() - d).total_seconds())) <= freshness_limit_sec

    contract = state.get("contract") if isinstance(state.get("contract"), dict) else {}
    return {
        "scheduler_digest": payload_sha,
        "scheduler_epoch": str(state.get("updated_at") or state.get("selection_status") or "none"),
        "state_exists": state_path.exists(),
        "updated_at": updated_at,
        "fresh": fresh,
        "freshness_limit_sec": freshness_limit_sec,
        "contract_state_valid": contract.get("state_valid"),
        "source_path": to_rel(root, state_path),
    }


def compute_anchor_head(root: pathlib.Path) -> Dict[str, Any]:
    lp_path = root / "state" / "continuity" / "latest" / "latest_pointer.json"
    gt_path = root / "state" / "ground_truth" / "latest.json"

    lp = load_json(lp_path)
    gt = load_json(gt_path)

    checkpoint_id = str(lp.get("checkpoint_id") or "").strip() or None
    checkpoint_sha = str(lp.get("json_sha256") or "").strip() or None
    snapshot_id = str(gt.get("snapshot_id") or "").strip() or None

    anchor_digest = checkpoint_sha or sha256_text(canonical_json({"checkpoint_id": checkpoint_id, "snapshot_id": snapshot_id}))

    return {
        "anchor_seq": checkpoint_id,
        "anchor_id": snapshot_id,
        "anchor_digest": anchor_digest,
        "checkpoint_id": checkpoint_id,
        "source_paths": {
            "latest_pointer": to_rel(root, lp_path),
            "ground_truth_latest": to_rel(root, gt_path),
        },
    }


def build_coherence_tuple(root: pathlib.Path, *, update_policy_epoch: bool = False) -> Dict[str, Any]:
    anchor = compute_anchor_head(root)
    policy = compute_policy_freshness(root, update_epoch=update_policy_epoch)
    connectors = compute_connector_freshness(root)
    scheduler = compute_scheduler_head(root)
    boot_id = read_boot_id()

    tuple_obj = {
        "tuple_version": 1,
        "anchor": {
            "anchor_seq": anchor.get("anchor_seq"),
            "anchor_id": anchor.get("anchor_id"),
            "anchor_digest": anchor.get("anchor_digest"),
        },
        "policy": {
            "policy_epoch": policy.get("policy_epoch"),
            "policy_digest": policy.get("policy_digest"),
            "evaluator_hash": policy.get("evaluator_hash"),
            "signature": policy.get("signature"),
        },
        "connectors": {
            "vector_digest": connectors.get("vector_digest"),
            "items": connectors.get("items"),
            "critical_valid_until_min": connectors.get("critical_valid_until_min"),
        },
        "scheduler": {
            "scheduler_epoch": scheduler.get("scheduler_epoch"),
            "scheduler_digest": scheduler.get("scheduler_digest"),
        },
        "time_basis": {
            "issuer_boot_id": boot_id,
            "clock_model": "realtime+boot-id",
        },
    }
    tuple_hash = sha256_text(canonical_json(tuple_obj))

    return {
        "schema_version": "continuity.coherence_stamp.v1",
        "generated_at": now_iso(),
        "tuple": tuple_obj,
        "tuple_hash": tuple_hash,
        "policy": policy,
        "connectors": connectors,
        "scheduler": scheduler,
        "anchor": anchor,
    }
