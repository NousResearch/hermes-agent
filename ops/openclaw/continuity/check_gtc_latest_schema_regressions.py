#!/usr/bin/env python3
"""Local regression harness for intentionally broken GTC latest surfaces.

Focus:
- Corruption/negative-path checks for latest-surface schema gate.
- Verify fail-close behavior (`--strict` non-zero) and operator-readable failure payloads.
- Use deterministic self-seeded fixtures (no dependency on live state/gtc-v2/latest).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[3]
CHECKER = ROOT / "ops" / "openclaw" / "continuity" / "gtc_latest_schema_check.sh"

SEED_FIXTURE_VERSION = "gtc.latest.schema.regressions.seed.v1"
SEED_GENERATED_AT = "2026-01-01T00:00:00Z"
SEED_VALID_UNTIL = "2026-01-01T00:30:00Z"
SEED_BUILD_GENERATION_ID = "seed-gen-20260101-0001"
SEED_BOOT_ID = "seed-boot-01"
SEED_CONNECTOR_DIGEST = "0123456789abcdef0123456789abcdef"
SEED_CONNECTOR_FILE = "runtime.gateway__seed-gateway.json"
SEED_MANIFEST_HMAC_KEY = "seed_manifest_hmac_secret_v1"
SEED_MANIFEST_HMAC_AUTH_PROFILE = "gtc.publish_manifest.hmac.v1"
SEED_MANIFEST_ED25519_AUTH_PROFILE = "gtc.publish_manifest.ed25519.v1"
SEED_MANIFEST_AUTH_FIELDS: list[str] = [
    "schema_version",
    "generated_at",
    "build_generation_id",
    "base_generation_id",
    "base_coherence_guard",
    "valid_until",
    "latest_paths",
    "latest_sha256",
]
SEED_SCHEMA_SENTINEL_VERSION = "gtc.latest.schema.seed_sentinel.v4"

SEED_REQUIRED_CONTRACT_SURFACES: list[dict[str, str]] = [
    {
        "surface": "continuity_current",
        "seed_relative_path": "latest/continuity_current.json",
        "schema": "gtc_latest.schema.json",
    },
    {
        "surface": "gateboard",
        "seed_relative_path": "latest/gateboard.json",
        "schema": "gtc_gateboard.schema.json",
    },
    {
        "surface": "event_projection",
        "seed_relative_path": "latest/event_projection.json",
        "schema": "gtc_event.schema.json",
    },
    {
        "surface": "incident_replay",
        "seed_relative_path": "latest/incident_replay.json",
        "schema": "gtc_incident_replay.schema.json",
    },
    {
        "surface": "publish_anchor",
        "seed_relative_path": "latest/publish_anchor.json",
        "schema": "gtc_publish_anchor.schema.json",
    },
    {
        "surface": "publish_manifest",
        "seed_relative_path": "latest/publish_manifest.json",
        "schema": "gtc_publish_manifest.schema.json",
    },
    {
        "surface": f"connector::{SEED_CONNECTOR_FILE}",
        "seed_relative_path": f"latest/connectors/{SEED_CONNECTOR_FILE}",
        "schema": "gtc_connector_latest.schema.json",
    },
]

DRIFT_CLASS_PAYLOAD_FIELDS: tuple[tuple[str, str], ...] = (
    ("required", "missing_required_fields"),
    ("const", "const_drift"),
    ("enum", "enum_drift"),
    ("type", "type_drift"),
    ("format", "format_drift"),
)
DRIFT_CLASS_PAYLOAD_KEY_BY_CLASS: dict[str, str] = {
    drift_class: payload_key for drift_class, payload_key in DRIFT_CLASS_PAYLOAD_FIELDS
}
SEED_REQUIRED_CONTRACT_SURFACE_PRECEDENCE: dict[str, int] = {
    str(spec.get("surface") or ""): index
    for index, spec in enumerate(SEED_REQUIRED_CONTRACT_SURFACES)
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert_true(isinstance(obj, dict), f"expected json object at {path}")
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resign_publish_manifest(path: Path, *, hmac_key: str = SEED_MANIFEST_HMAC_KEY) -> None:
    doc = load_json(path)
    payload_sha256, signature = _manifest_auth_signature(doc, hmac_key=hmac_key)
    auth = doc.get("manifest_auth") if isinstance(doc.get("manifest_auth"), dict) else {}
    auth["scheme"] = "hmac-sha256"
    auth["key_id"] = str(auth.get("key_id") or _manifest_hmac_key_id(hmac_key))
    auth["canonical_profile"] = str(auth.get("canonical_profile") or SEED_MANIFEST_HMAC_AUTH_PROFILE)
    auth["payload_sha256"] = payload_sha256
    auth["signature"] = signature
    doc["manifest_auth"] = auth
    write_json(path, doc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_connectors_dir(path: Path) -> str:
    rows: list[dict[str, str]] = []
    for p in sorted(path.glob("*.json")):
        if not p.is_file():
            continue
        rows.append({"file": p.name, "sha256": sha256_file(p)})
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _manifest_auth_payload(manifest_doc: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in SEED_MANIFEST_AUTH_FIELDS:
        if field in manifest_doc:
            payload[field] = manifest_doc.get(field)
    return payload


def _manifest_auth_fields_sha256() -> str:
    return hashlib.sha256(_canonical_json(SEED_MANIFEST_AUTH_FIELDS).encode("utf-8")).hexdigest()


def _manifest_auth_signature(manifest_doc: dict[str, Any], *, hmac_key: str) -> tuple[str, str]:
    payload_json = _canonical_json(_manifest_auth_payload(manifest_doc))
    payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    signature = hmac.new(hmac_key.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256).hexdigest()
    return payload_sha256, signature


def _manifest_hmac_key_id(hmac_key: str) -> str:
    return f"ksha256_{hashlib.sha256(hmac_key.encode('utf-8')).hexdigest()[:16]}"


def _openssl(args: list[str]) -> bytes:
    cp = subprocess.run(["openssl", *args], capture_output=True, check=False)
    if cp.returncode != 0:
        stderr = (cp.stderr or b"").decode("utf-8", errors="replace").strip()
        raise AssertionError(f"openssl failed ({' '.join(args)}): {stderr}")
    return cp.stdout or b""


def _generate_ed25519_keypair() -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="gtc_seed_ed25519_") as td:
        priv = Path(td) / "priv.pem"
        pub = Path(td) / "pub.pem"
        _openssl(["genpkey", "-algorithm", "Ed25519", "-out", str(priv)])
        _openssl(["pkey", "-in", str(priv), "-pubout", "-out", str(pub)])
        return priv.read_text(encoding="utf-8"), pub.read_text(encoding="utf-8")


def _manifest_ed25519_key_id(public_key_pem: str) -> str:
    with tempfile.TemporaryDirectory(prefix="gtc_seed_pubsha_") as td:
        pub = Path(td) / "pub.pem"
        pub.write_text(public_key_pem.strip() + "\n", encoding="utf-8")
        pub_der = _openssl(["pkey", "-pubin", "-in", str(pub), "-outform", "DER"])
    return f"ed25519_pksha256_{hashlib.sha256(pub_der).hexdigest()[:16]}"


def _manifest_ed25519_public_key_sha256(public_key_pem: str) -> str:
    with tempfile.TemporaryDirectory(prefix="gtc_seed_pubsha_") as td:
        pub = Path(td) / "pub.pem"
        pub.write_text(public_key_pem.strip() + "\n", encoding="utf-8")
        pub_der = _openssl(["pkey", "-pubin", "-in", str(pub), "-outform", "DER"])
    return hashlib.sha256(pub_der).hexdigest()


def _manifest_auth_signature_ed25519(manifest_doc: dict[str, Any], *, private_key_pem: str) -> tuple[str, str]:
    payload_json = _canonical_json(_manifest_auth_payload(manifest_doc))
    payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    with tempfile.TemporaryDirectory(prefix="gtc_seed_sign_") as td:
        priv = Path(td) / "priv.pem"
        payload = Path(td) / "payload.bin"
        sig = Path(td) / "sig.bin"
        priv.write_text(private_key_pem.strip() + "\n", encoding="utf-8")
        payload.write_bytes(payload_json.encode("utf-8"))
        _openssl(["pkeyutl", "-sign", "-rawin", "-inkey", str(priv), "-in", str(payload), "-out", str(sig)])
        return payload_sha256, sig.read_bytes().hex()


def run_checker(gtc_root: Path, *, strict: bool, extra_env: dict[str, str] | None = None) -> tuple[int, dict[str, Any], str]:
    cmd = ["bash", str(CHECKER), "--gtc-root", str(gtc_root), "--json"]
    if strict:
        cmd.append("--strict")

    env = os.environ.copy()
    env["OPENCLAW_ROOT"] = str(ROOT)
    env["OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY"] = SEED_MANIFEST_HMAC_KEY
    env["OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY_ID"] = _manifest_hmac_key_id(SEED_MANIFEST_HMAC_KEY)
    if extra_env:
        env.update(extra_env)

    cp = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    stdout = (cp.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"_parse_error": "stdout_not_json", "stdout": stdout, "stderr": (cp.stderr or "").strip()}
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload": payload}

    return cp.returncode, payload, (cp.stderr or "").strip()


def seed_latest_surfaces(latest_dir: Path) -> None:
    continuity_current = {
        "schema_version": "gtc.latest.v2",
        "generated_at": SEED_GENERATED_AT,
        "build_generation_id": SEED_BUILD_GENERATION_ID,
        "connectors": [
            {
                "connector_type": "runtime.gateway",
                "connector_id": "seed-gateway",
                "latest_evidence_id": "evt-seed-0001",
                "observed_at": SEED_GENERATED_AT,
                "freshness_ttl_ms": 60000,
                "stale": False,
                "stale_severity": "none",
                "age_ms": 0,
                "subject": {"kind": "gateway", "id": "seed-gateway"},
            }
        ],
        "readiness": {"mutate_allowed": True, "status": "green", "reasons": []},
    }

    gateboard = {
        "schema_version": "gtc.gateboard.v2",
        "generated_at": SEED_GENERATED_AT,
        "build_generation_id": SEED_BUILD_GENERATION_ID,
        "valid_until": SEED_VALID_UNTIL,
        "issuer_boot_id": SEED_BOOT_ID,
        "connector_vector_digest": SEED_CONNECTOR_DIGEST,
        "mutate_allowed": True,
        "status": "green",
        "blocking_reasons": [],
        "warning_reasons": [],
        "required_connectors": ["runtime.gateway"],
        "verify_status": "ok",
        "runtime_critical_anomaly_count": 0,
        "open_incident_count": 0,
        "queue_active_nonterminal_count": 0,
    }

    event_projection = {
        "schema_version": "gtc.event.v2",
        "generated_at": SEED_GENERATED_AT,
        "routes": [],
        "summary": {
            "route_count": 0,
            "open_incident_count": 0,
            "critical_open_count": 0,
            "warn_open_count": 0,
        },
        "build_generation_id": SEED_BUILD_GENERATION_ID,
    }

    incident_replay = {
        "schema_version": "gtc.incident_replay.v1",
        "generated_at": SEED_GENERATED_AT,
        "build_generation_id": SEED_BUILD_GENERATION_ID,
        "valid_until": SEED_VALID_UNTIL,
        "open_incidents": [],
        "recommended_commands": ["openclaw gtc refresh --dry-run"],
    }

    manifest_key_id = _manifest_hmac_key_id(SEED_MANIFEST_HMAC_KEY)

    publish_anchor = {
        "schema_version": "gtc.publish_anchor.v1",
        "generated_at": SEED_GENERATED_AT,
        "build_generation_id": SEED_BUILD_GENERATION_ID,
        "valid_until": SEED_VALID_UNTIL,
        "issuer_boot_id": SEED_BOOT_ID,
        "manifest_auth_root": {
            "scheme": "hmac-sha256",
            "key_id": manifest_key_id,
            "canonical_profile": SEED_MANIFEST_HMAC_AUTH_PROFILE,
            "payload_fields": list(SEED_MANIFEST_AUTH_FIELDS),
            "payload_fields_sha256": _manifest_auth_fields_sha256(),
            "key_source": "seed_fixture_env",
        },
    }

    connector = {
        "connector_type": "runtime.gateway",
        "connector_id": "seed-gateway",
        "display_name": "Seed Gateway",
        "latest_evidence_id": "evt-seed-0001",
        "observed_at": SEED_GENERATED_AT,
        "valid_until": SEED_VALID_UNTIL,
        "freshness_ttl_ms": 60000,
        "age_ms": 0,
        "stale": False,
        "stale_severity": "none",
        "monotonic_seq": 1,
        "issuer_boot_id": SEED_BOOT_ID,
        "severity_max": "info",
        "build_generation_id": SEED_BUILD_GENERATION_ID,
        "subject": {"kind": "gateway", "id": "seed-gateway"},
        "refs": {"source": "self-seeded-regression-fixture"},
    }

    write_json(latest_dir / "continuity_current.json", continuity_current)
    write_json(latest_dir / "gateboard.json", gateboard)
    write_json(latest_dir / "event_projection.json", event_projection)
    write_json(latest_dir / "incident_replay.json", incident_replay)
    write_json(latest_dir / "publish_anchor.json", publish_anchor)
    write_json(latest_dir / "connectors" / SEED_CONNECTOR_FILE, connector)

    publish_manifest = {
        "schema_version": "gtc.publish_manifest.v1",
        "generated_at": SEED_GENERATED_AT,
        "build_generation_id": SEED_BUILD_GENERATION_ID,
        "valid_until": SEED_VALID_UNTIL,
        "latest_paths": {
            "publish_anchor": "latest/publish_anchor.json",
            "continuity_current": "latest/continuity_current.json",
            "gateboard": "latest/gateboard.json",
            "event_projection": "latest/event_projection.json",
            "incident_replay": "latest/incident_replay.json",
            "connectors_dir": "latest/connectors",
        },
        "latest_sha256": {
            "publish_anchor": sha256_file(latest_dir / "publish_anchor.json"),
            "continuity_current": sha256_file(latest_dir / "continuity_current.json"),
            "gateboard": sha256_file(latest_dir / "gateboard.json"),
            "event_projection": sha256_file(latest_dir / "event_projection.json"),
            "incident_replay": sha256_file(latest_dir / "incident_replay.json"),
            "connectors_dir": sha256_connectors_dir(latest_dir / "connectors"),
        },
    }
    payload_sha256, signature = _manifest_auth_signature(publish_manifest, hmac_key=SEED_MANIFEST_HMAC_KEY)
    publish_manifest["manifest_auth"] = {
        "scheme": "hmac-sha256",
        "key_id": manifest_key_id,
        "canonical_profile": SEED_MANIFEST_HMAC_AUTH_PROFILE,
        "payload_sha256": payload_sha256,
        "signature": signature,
    }

    write_json(latest_dir / "publish_manifest.json", publish_manifest)


def rekey_fixture_manifest_to_ed25519(gtc_root: Path) -> dict[str, str]:
    latest_dir = gtc_root / "latest"
    anchor_path = latest_dir / "publish_anchor.json"
    manifest_path = latest_dir / "publish_manifest.json"

    private_key_pem, public_key_pem = _generate_ed25519_keypair()
    key_id = _manifest_ed25519_key_id(public_key_pem)
    public_key_sha256 = _manifest_ed25519_public_key_sha256(public_key_pem)

    anchor = load_json(anchor_path)
    root_auth = anchor.get("manifest_auth_root") if isinstance(anchor.get("manifest_auth_root"), dict) else {}
    root_auth["scheme"] = "ed25519-sha256"
    root_auth["key_id"] = key_id
    root_auth["canonical_profile"] = SEED_MANIFEST_ED25519_AUTH_PROFILE
    root_auth["payload_fields"] = list(SEED_MANIFEST_AUTH_FIELDS)
    root_auth["payload_fields_sha256"] = _manifest_auth_fields_sha256()
    root_auth["public_key_sha256"] = public_key_sha256
    root_auth["key_source"] = "seed_fixture_env"
    anchor["manifest_auth_root"] = root_auth
    write_json(anchor_path, anchor)

    manifest = load_json(manifest_path)
    latest_sha = manifest.get("latest_sha256") if isinstance(manifest.get("latest_sha256"), dict) else {}
    latest_sha["publish_anchor"] = sha256_file(anchor_path)
    manifest["latest_sha256"] = latest_sha

    payload_sha256, signature = _manifest_auth_signature_ed25519(manifest, private_key_pem=private_key_pem)
    manifest["manifest_auth"] = {
        "scheme": "ed25519-sha256",
        "key_id": key_id,
        "canonical_profile": SEED_MANIFEST_ED25519_AUTH_PROFILE,
        "payload_sha256": payload_sha256,
        "signature": signature,
    }
    write_json(manifest_path, manifest)

    return {
        "OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PUBLIC_KEY_PEM": public_key_pem,
        "OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_KEY_ID": key_id,
    }


def build_fixture() -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix="gtc_latest_schema_regressions_"))
    gtc_root = td / "gtc-v2"
    latest_dir = gtc_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    seed_latest_surfaces(latest_dir)
    return td, gtc_root


def _json_pointer(tokens: list[Any]) -> str:
    if not tokens:
        return "$"
    escaped = [str(tok).replace("~", "~0").replace("/", "~1") for tok in tokens]
    return "$/" + "/".join(escaped)


def _missing_required_field_paths(errors: list[Any]) -> list[str]:
    missing: set[str] = set()
    for err in errors:
        if getattr(err, "validator", None) != "required":
            continue
        required = getattr(err, "validator_value", None)
        instance = getattr(err, "instance", None)
        if not isinstance(required, list) or not isinstance(instance, dict):
            continue
        base_tokens = list(getattr(err, "absolute_path", []) or [])
        for field in required:
            if not isinstance(field, str):
                continue
            if field in instance:
                continue
            missing.add(_json_pointer(base_tokens + [field]))
    return sorted(missing)


def _value_preview(value: Any, *, max_len: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _const_drift_rows(errors: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for err in errors:
        if getattr(err, "validator", None) != "const":
            continue
        path = _json_pointer(list(getattr(err, "absolute_path", []) or []))
        expected = _value_preview(getattr(err, "validator_value", None))
        actual = _value_preview(getattr(err, "instance", None))
        key = (path, expected, actual)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "path": path,
                "expected_const": expected,
                "actual": actual,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _enum_drift_rows(errors: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for err in errors:
        if getattr(err, "validator", None) != "enum":
            continue
        path = _json_pointer(list(getattr(err, "absolute_path", []) or []))
        expected = _value_preview(getattr(err, "validator_value", None))
        actual = _value_preview(getattr(err, "instance", None))
        key = (path, expected, actual)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "path": path,
                "expected_enum": expected,
                "actual": actual,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _type_drift_rows(errors: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for err in errors:
        if getattr(err, "validator", None) != "type":
            continue
        path = _json_pointer(list(getattr(err, "absolute_path", []) or []))
        expected_raw = getattr(err, "validator_value", None)
        if isinstance(expected_raw, list):
            expected = ",".join(str(v) for v in expected_raw)
        else:
            expected = str(expected_raw)
        actual_type = _json_type_name(getattr(err, "instance", None))
        key = (path, expected, actual_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "path": path,
                "expected_type": expected,
                "actual_type": actual_type,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _format_drift_rows(errors: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for err in errors:
        if getattr(err, "validator", None) != "format":
            continue
        path = _json_pointer(list(getattr(err, "absolute_path", []) or []))
        expected_format = str(getattr(err, "validator_value", ""))
        actual = _value_preview(getattr(err, "instance", None))
        key = (path, expected_format, actual)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "path": path,
                "expected_format": expected_format,
                "actual": actual,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _classify_seed_schema_drifts(errors: list[Any]) -> dict[str, Any]:
    missing_required = _missing_required_field_paths(errors)
    const_drift = _const_drift_rows(errors)
    enum_drift = _enum_drift_rows(errors)
    type_drift = _type_drift_rows(errors)
    format_drift = _format_drift_rows(errors)

    out: dict[str, Any] = {}
    if missing_required:
        out["missing_required_fields"] = missing_required
    if const_drift:
        out["const_drift"] = const_drift
    if enum_drift:
        out["enum_drift"] = enum_drift
    if type_drift:
        out["type_drift"] = type_drift
    if format_drift:
        out["format_drift"] = format_drift

    out["drift_classes"] = _ordered_drift_classes(
        {
            drift_class
            for drift_class, payload_key in DRIFT_CLASS_PAYLOAD_FIELDS
            if out.get(payload_key)
        }
    )
    return out


def _ordered_drift_classes(classes: Iterable[str]) -> list[str]:
    normalized: set[str] = {
        str(cls).strip()
        for cls in classes
        if str(cls).strip()
    }

    ordered: list[str] = []
    if "required" in normalized:
        ordered.append("required")

    ordered.extend(
        name
        for name, _ in DRIFT_CLASS_PAYLOAD_FIELDS
        if name != "required" and name in normalized
    )

    known = {name for name, _ in DRIFT_CLASS_PAYLOAD_FIELDS}
    extras = sorted(normalized - known)
    return [*ordered, *extras]


def _drift_paths(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(row.get("path") or "$")
            for row in rows
            if isinstance(row, dict) and str(row.get("path") or "").strip()
        }
    )


def _surface_drift_parts(row: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    observed_classes = _ordered_drift_classes(
        drift_class
        for drift_class, payload_key in DRIFT_CLASS_PAYLOAD_FIELDS
        if row.get(payload_key)
    )

    for drift_class in observed_classes:
        payload_key = DRIFT_CLASS_PAYLOAD_KEY_BY_CLASS.get(drift_class)
        if not payload_key:
            continue
        payload = row.get(payload_key) or []
        if drift_class == "required":
            paths = [str(path) for path in payload if str(path).strip()]
        else:
            paths = _drift_paths(payload)
        if paths:
            parts.append(f"{drift_class}:{','.join(paths)}")
    return parts


def _row_observed_drift_classes(row: dict[str, Any]) -> list[str]:
    observed = {
        drift_class
        for drift_class, payload_key in DRIFT_CLASS_PAYLOAD_FIELDS
        if row.get(payload_key)
    }
    observed.update(
        str(drift_class).strip()
        for drift_class in (row.get("drift_classes") or [])
        if str(drift_class).strip()
    )
    return _ordered_drift_classes(observed)


def _aggregate_drift_classes(rows: list[dict[str, Any]]) -> list[str]:
    observed: set[str] = set()
    for row in rows:
        observed.update(_row_observed_drift_classes(row))
    return _ordered_drift_classes(observed)


def _ordered_seed_drift_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
        surface = str(row.get("surface") or "")
        precedence = SEED_REQUIRED_CONTRACT_SURFACE_PRECEDENCE.get(surface, len(SEED_REQUIRED_CONTRACT_SURFACE_PRECEDENCE))
        seed_rel = str(row.get("seed_relative_path") or "")
        return (precedence, surface, seed_rel)

    return sorted(rows, key=_sort_key)


def scenario_seed_schema_fixture_contract() -> dict[str, Any]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except Exception as exc:
        return {
            "name": "seed_schema_fixture_contract",
            "ok": False,
            "error": f"jsonschema_import_failed:{exc}",
            "hint": "Install python jsonschema dependency before running gtc latest schema regressions.",
        }

    td, gtc_root = build_fixture()
    schema_root = ROOT / "ops" / "openclaw" / "architecture" / "schemas"
    drifts: list[dict[str, Any]] = []
    try:
        for spec in SEED_REQUIRED_CONTRACT_SURFACES:
            surface = spec["surface"]
            seed_rel = spec["seed_relative_path"]
            schema_name = spec["schema"]

            seed_path = gtc_root / seed_rel
            schema_path = schema_root / schema_name

            if not seed_path.exists():
                drifts.append(
                    {
                        "surface": surface,
                        "seed_relative_path": seed_rel,
                        "schema": schema_name,
                        "error": "seed_surface_missing",
                    }
                )
                continue

            if not schema_path.exists():
                drifts.append(
                    {
                        "surface": surface,
                        "seed_relative_path": seed_rel,
                        "schema": schema_name,
                        "error": "schema_missing",
                    }
                )
                continue

            doc = load_json(seed_path)
            schema_doc = load_json(schema_path)
            validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
            schema_errors = list(validator.iter_errors(doc))
            classification = _classify_seed_schema_drifts(schema_errors)
            if classification.get("drift_classes"):
                row: dict[str, Any] = {
                    "surface": surface,
                    "seed_relative_path": seed_rel,
                    "schema": schema_name,
                    "drift_classes": classification.get("drift_classes") or [],
                }
                row.update(
                    {
                        key: value
                        for key, value in classification.items()
                        if key != "drift_classes" and value
                    }
                )
                drifts.append(row)

        if drifts:
            ordered_drifts = _ordered_seed_drift_rows(drifts)
            drift_classes_observed = _aggregate_drift_classes(ordered_drifts)
            drift_parts: list[str] = []
            for row in ordered_drifts:
                if row.get("error"):
                    drift_parts.append(f"{row['surface']}=>{row.get('error')}")
                    continue
                class_parts = _surface_drift_parts(row)
                detail = ";".join(class_parts) if class_parts else "unknown"
                drift_parts.append(f"{row['surface']}=>{detail}")

            required_only = drift_classes_observed == ["required"]
            error_code = "seed_fixture_required_field_drift" if required_only else "seed_fixture_schema_contract_drift"
            if required_only:
                hint = (
                    "Schema-required fields drifted from the self-seeded fixture. "
                    "Update seed_latest_surfaces() in "
                    "ops/openclaw/continuity/check_gtc_latest_schema_regressions.py "
                    "to satisfy new required fields before running negative cases."
                )
            else:
                hint = (
                    "Schema contract drifted from the self-seeded fixture (required/const/enum/type/format). "
                    "Update seed_latest_surfaces() in "
                    "ops/openclaw/continuity/check_gtc_latest_schema_regressions.py "
                    "to satisfy evolved schema constraints before running negative cases."
                )
            return {
                "name": "seed_schema_fixture_contract",
                "ok": False,
                "error": error_code,
                "drift_classes_observed": drift_classes_observed,
                "drifts": ordered_drifts,
                "hint": hint,
                "detail": "; ".join(drift_parts),
            }

        return {
            "name": "seed_schema_fixture_contract",
            "ok": True,
            "checked_surfaces": len(SEED_REQUIRED_CONTRACT_SURFACES),
            "sentinel_version": SEED_SCHEMA_SENTINEL_VERSION,
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def failing_surfaces(payload: dict[str, Any]) -> set[str]:
    rows = payload.get("checks") or []
    out: set[str] = set()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("ok") is True:
            continue
        s = str(row.get("surface") or "").strip()
        if s:
            out.add(s)
    return out


def run_negative_case(
    name: str,
    mutator: Callable[[Path], str | tuple[str, dict[str, str]]],
    *,
    expected_surface: str | None = None,
    expected_surface_prefix: str | None = None,
    extra_assert: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    td, gtc_root = build_fixture()
    try:
        mutator_result = mutator(gtc_root)
        checker_env: dict[str, str] = {}
        if isinstance(mutator_result, tuple):
            detail, checker_env = mutator_result
        else:
            detail = mutator_result

        rc_strict, payload_strict, stderr_strict = run_checker(gtc_root, strict=True, extra_env=checker_env)
        assert_true(rc_strict != 0, f"{name}: expected strict rc!=0, got {rc_strict}")
        assert_true(payload_strict.get("ok") is False, f"{name}: expected strict ok=false, got {payload_strict.get('ok')}")
        assert_true(int(payload_strict.get("error_count") or 0) > 0, f"{name}: expected strict error_count>0")

        surfaces_strict = failing_surfaces(payload_strict)
        if expected_surface is not None:
            assert_true(
                expected_surface in surfaces_strict,
                f"{name}: expected failing surface '{expected_surface}' not found in {sorted(surfaces_strict)}",
            )
        if expected_surface_prefix is not None:
            assert_true(
                any(s.startswith(expected_surface_prefix) for s in surfaces_strict),
                f"{name}: expected failing surface prefix '{expected_surface_prefix}' not found in {sorted(surfaces_strict)}",
            )
        if extra_assert is not None:
            extra_assert(payload_strict)

        rc_soft, payload_soft, stderr_soft = run_checker(gtc_root, strict=False, extra_env=checker_env)
        assert_true(rc_soft == 0, f"{name}: expected non-strict rc=0, got {rc_soft}")
        assert_true(payload_soft.get("ok") is False, f"{name}: expected non-strict ok=false")

        return {
            "name": name,
            "ok": True,
            "detail": detail,
            "strict_returncode": rc_strict,
            "non_strict_returncode": rc_soft,
            "strict_error_count": payload_strict.get("error_count"),
            "failing_surfaces": sorted(surfaces_strict),
            "stderr": {"strict": stderr_strict, "non_strict": stderr_soft},
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scenario_missing_gateboard(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "gateboard.json"
    p.unlink(missing_ok=True)
    return "removed latest/gateboard.json"


def scenario_invalid_continuity_shape(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "continuity_current.json"
    p.write_text("{}\n", encoding="utf-8")
    return "replaced latest/continuity_current.json with empty object"


def scenario_malformed_connector_object(gtc_root: Path) -> str:
    connectors = sorted((gtc_root / "latest" / "connectors").glob("*.json"))
    assert_true(bool(connectors), "expected at least one connector fixture")
    target = connectors[0]
    target.write_text("[]\n", encoding="utf-8")
    return f"replaced latest/connectors/{target.name} with []"


def scenario_no_connectors(gtc_root: Path) -> str:
    cdir = gtc_root / "latest" / "connectors"
    for p in cdir.glob("*.json"):
        p.unlink()
    return "removed all latest/connectors/*.json files"


def scenario_publish_manifest_paths_mismatch(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "publish_manifest.json"
    doc = load_json(p)
    latest_paths = doc.get("latest_paths") if isinstance(doc.get("latest_paths"), dict) else {}
    latest_paths["gateboard"] = "latest/not_gateboard.json"
    doc["latest_paths"] = latest_paths
    write_json(p, doc)
    resign_publish_manifest(p)
    return "set publish_manifest.latest_paths.gateboard to non-canonical target"


def assert_publish_manifest_paths_mismatch(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") or []
    assert_true(isinstance(checks, list), "checks missing")
    row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_paths"), None)
    assert_true(isinstance(row, dict), "publish_manifest_paths check missing")
    assert_true(row.get("ok") is False, "expected publish_manifest_paths.ok=false")
    mismatched = row.get("mismatched_paths") or []
    assert_true(any(str(entry.get("key") or "") == "gateboard" for entry in mismatched if isinstance(entry, dict)),
                f"expected gateboard mismatch entry, got {mismatched}")


def scenario_publish_manifest_paths_escape(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "publish_manifest.json"
    doc = load_json(p)
    latest_paths = doc.get("latest_paths") if isinstance(doc.get("latest_paths"), dict) else {}
    latest_paths["connectors_dir"] = "/etc"
    doc["latest_paths"] = latest_paths
    write_json(p, doc)
    resign_publish_manifest(p)
    return "set publish_manifest.latest_paths.connectors_dir to out-of-tree absolute path"


def assert_publish_manifest_paths_escape(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") or []
    assert_true(isinstance(checks, list), "checks missing")
    row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_paths"), None)
    assert_true(isinstance(row, dict), "publish_manifest_paths check missing")
    assert_true(row.get("ok") is False, "expected publish_manifest_paths.ok=false")
    out_of_tree = set(str(v) for v in (row.get("out_of_tree_keys") or []))
    assert_true("connectors_dir" in out_of_tree, f"expected connectors_dir out_of_tree, got {sorted(out_of_tree)}")


def scenario_publish_manifest_digest_mismatch(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "publish_manifest.json"
    doc = load_json(p)
    latest_sha = doc.get("latest_sha256") if isinstance(doc.get("latest_sha256"), dict) else {}
    latest_sha["gateboard"] = "0" * 64
    doc["latest_sha256"] = latest_sha
    write_json(p, doc)
    resign_publish_manifest(p)
    return "set publish_manifest.latest_sha256.gateboard to mismatched digest"


def assert_publish_manifest_digest_mismatch(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") or []
    assert_true(isinstance(checks, list), "checks missing")
    row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_digests"), None)
    assert_true(isinstance(row, dict), "publish_manifest_digests check missing")
    assert_true(row.get("ok") is False, "expected publish_manifest_digests.ok=false")
    mismatched = row.get("mismatched") or []
    assert_true(any(str(entry.get("key") or "") == "gateboard" for entry in mismatched if isinstance(entry, dict)),
                f"expected gateboard digest mismatch entry, got {mismatched}")


def scenario_publish_manifest_digest_tamper(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "gateboard.json"
    doc = load_json(p)
    doc["status"] = "red"
    write_json(p, doc)
    return "tampered latest/gateboard.json content after publish_manifest digest generation"


def assert_publish_manifest_digest_tamper(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") or []
    assert_true(isinstance(checks, list), "checks missing")
    row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_digests"), None)
    assert_true(isinstance(row, dict), "publish_manifest_digests check missing")
    assert_true(row.get("ok") is False, "expected publish_manifest_digests.ok=false")
    mismatched = row.get("mismatched") or []
    assert_true(any(str(entry.get("key") or "") == "gateboard" for entry in mismatched if isinstance(entry, dict)),
                f"expected gateboard digest mismatch after tamper, got {mismatched}")


def scenario_manifest_auth_signature_tamper(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "publish_manifest.json"
    doc = load_json(p)
    auth = doc.get("manifest_auth") if isinstance(doc.get("manifest_auth"), dict) else {}
    auth["signature"] = "0" * 64
    doc["manifest_auth"] = auth
    write_json(p, doc)
    return "set publish_manifest.manifest_auth.signature to mismatched hmac"


def scenario_manifest_auth_signature_tamper_ed25519(gtc_root: Path) -> tuple[str, dict[str, str]]:
    checker_env = rekey_fixture_manifest_to_ed25519(gtc_root)
    p = gtc_root / "latest" / "publish_manifest.json"
    doc = load_json(p)
    auth = doc.get("manifest_auth") if isinstance(doc.get("manifest_auth"), dict) else {}
    auth["signature"] = "0" * 128
    doc["manifest_auth"] = auth
    write_json(p, doc)
    return "set publish_manifest.manifest_auth.signature to mismatched ed25519 signature", checker_env


def assert_manifest_auth_signature_tamper(payload: dict[str, Any], *, expected_scheme: str | None = None) -> None:
    checks = payload.get("checks") or []
    assert_true(isinstance(checks, list), "checks missing")

    auth_row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_authenticity"), None)
    assert_true(isinstance(auth_row, dict), "publish_manifest_authenticity check missing")
    assert_true(auth_row.get("ok") is False, "expected publish_manifest_authenticity.ok=false")
    if expected_scheme is not None and auth_row.get("scheme") is not None:
        observed_scheme = str(auth_row.get("scheme") or "")
        assert_true(observed_scheme == expected_scheme, f"unexpected authenticity scheme: {observed_scheme}")
    assert_true(
        str(auth_row.get("error") or "") in {"manifest_signature_mismatch", "manifest_payload_sha256_mismatch"},
        f"unexpected authenticity error: {auth_row}",
    )

    paths_row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_paths"), None)
    digests_row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_digests"), None)
    assert_true(isinstance(paths_row, dict), "publish_manifest_paths check missing")
    assert_true(isinstance(digests_row, dict), "publish_manifest_digests check missing")
    assert_true(paths_row.get("error") == "manifest_untrusted_authenticity_failed", f"unexpected paths row: {paths_row}")
    assert_true(digests_row.get("error") == "manifest_untrusted_authenticity_failed", f"unexpected digests row: {digests_row}")


def scenario_generation_mismatch(gtc_root: Path) -> str:
    p = gtc_root / "latest" / "publish_manifest.json"
    doc = load_json(p)
    doc["build_generation_id"] = "broken-generation-0001"
    write_json(p, doc)
    resign_publish_manifest(p)
    return "set publish_manifest.build_generation_id to mismatched value"


def assert_generation_mismatch(payload: dict[str, Any]) -> None:
    gc = payload.get("generation_consistency") or {}
    assert_true(isinstance(gc, dict), "generation_consistency missing")
    assert_true(gc.get("ok") is False, f"expected generation_consistency.ok=false, got {gc.get('ok')}")
    expected = str(gc.get("expected_generation_id") or "")
    assert_true(expected == "broken-generation-0001", f"unexpected expected_generation_id: {expected}")
    mismatched = gc.get("mismatched_generation") or []
    assert_true("continuity_current" in mismatched, f"expected continuity_current mismatch, got {mismatched}")


def scenario_baseline_valid() -> dict[str, Any]:
    td, gtc_root = build_fixture()
    try:
        rc, payload, stderr = run_checker(gtc_root, strict=True)
        assert_true(rc == 0, f"baseline: expected rc=0, got {rc}")
        assert_true(payload.get("ok") is True, f"baseline: expected ok=true, got {payload.get('ok')}")
        assert_true(int(payload.get("error_count") or 0) == 0, "baseline: expected error_count=0")
        return {
            "name": "baseline_valid_fixture",
            "ok": True,
            "strict_returncode": rc,
            "strict_error_count": payload.get("error_count"),
            "surface_count": payload.get("surface_count"),
            "stderr": stderr,
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _manifest_auth_row(payload: dict[str, Any], *, context: str) -> dict[str, Any]:
    checks = payload.get("checks") or []
    assert_true(isinstance(checks, list), f"{context}: checks missing")
    auth_row = next((r for r in checks if isinstance(r, dict) and r.get("surface") == "publish_manifest_authenticity"), None)
    assert_true(isinstance(auth_row, dict), f"{context}: publish_manifest_authenticity check missing")
    return auth_row


def _run_publish_manifest_auth_mode_valid_case(
    *,
    name: str,
    expected_scheme: str,
    fixture_mutator: Callable[[Path], dict[str, str]] | None = None,
) -> dict[str, Any]:
    td, gtc_root = build_fixture()
    try:
        checker_env = fixture_mutator(gtc_root) if fixture_mutator is not None else {}
        rc, payload, stderr = run_checker(gtc_root, strict=True, extra_env=checker_env)
        assert_true(rc == 0, f"{name}: expected strict rc=0, got {rc}")
        assert_true(payload.get("ok") is True, f"{name}: expected strict ok=true, got {payload.get('ok')}")
        assert_true(int(payload.get("error_count") or 0) == 0, f"{name}: expected strict error_count=0")

        auth_row = _manifest_auth_row(payload, context=name)
        observed_scheme = str(auth_row.get("scheme") or "")
        assert_true(observed_scheme == expected_scheme, f"{name}: unexpected scheme {observed_scheme}")

        return {
            "name": name,
            "ok": True,
            "strict_returncode": rc,
            "strict_error_count": payload.get("error_count"),
            "auth_scheme": observed_scheme,
            "auth_canonical_profile": auth_row.get("canonical_profile"),
            "auth_key_id": auth_row.get("key_id"),
            "stderr": stderr,
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scenario_publish_manifest_auth_mode_compat_hmac_valid() -> dict[str, Any]:
    return _run_publish_manifest_auth_mode_valid_case(
        name="publish_manifest_auth_mode_compat_hmac_valid",
        expected_scheme="hmac-sha256",
    )


def scenario_publish_manifest_auth_mode_default_ed25519_valid() -> dict[str, Any]:
    return _run_publish_manifest_auth_mode_valid_case(
        name="publish_manifest_auth_mode_default_ed25519_valid",
        expected_scheme="ed25519-sha256",
        fixture_mutator=rekey_fixture_manifest_to_ed25519,
    )


def _print_result_line(row: dict[str, Any]) -> None:
    name = str(row.get("name") or "unknown")
    ok = bool(row.get("ok"))
    if ok:
        print(f"PASS {name}")
        return
    detail = str(row.get("detail") or row.get("error") or "")
    if detail:
        print(f"FAIL {name}: {detail}")
    else:
        print(f"FAIL {name}")


def main() -> int:
    if not CHECKER.exists():
        print(json.dumps({"ok": False, "error": "checker_missing", "path": str(CHECKER)}, ensure_ascii=False, indent=2))
        return 2

    scenarios: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("baseline_valid_fixture", scenario_baseline_valid),
        (
            "missing_gateboard_surface",
            lambda: run_negative_case(
                "missing_gateboard_surface",
                scenario_missing_gateboard,
                expected_surface="gateboard",
            ),
        ),
        (
            "invalid_continuity_shape",
            lambda: run_negative_case(
                "invalid_continuity_shape",
                scenario_invalid_continuity_shape,
                expected_surface="continuity_current",
            ),
        ),
        (
            "malformed_connector_object",
            lambda: run_negative_case(
                "malformed_connector_object",
                scenario_malformed_connector_object,
                expected_surface_prefix="connector::",
            ),
        ),
        (
            "no_connector_surfaces",
            lambda: run_negative_case(
                "no_connector_surfaces",
                scenario_no_connectors,
                expected_surface="connectors",
            ),
        ),
        (
            "publish_manifest_paths_mismatch",
            lambda: run_negative_case(
                "publish_manifest_paths_mismatch",
                scenario_publish_manifest_paths_mismatch,
                expected_surface="publish_manifest_paths",
                extra_assert=assert_publish_manifest_paths_mismatch,
            ),
        ),
        (
            "publish_manifest_paths_escape",
            lambda: run_negative_case(
                "publish_manifest_paths_escape",
                scenario_publish_manifest_paths_escape,
                expected_surface="publish_manifest_paths",
                extra_assert=assert_publish_manifest_paths_escape,
            ),
        ),
        (
            "publish_manifest_digest_mismatch",
            lambda: run_negative_case(
                "publish_manifest_digest_mismatch",
                scenario_publish_manifest_digest_mismatch,
                expected_surface="publish_manifest_digests",
                extra_assert=assert_publish_manifest_digest_mismatch,
            ),
        ),
        (
            "publish_manifest_digest_tamper",
            lambda: run_negative_case(
                "publish_manifest_digest_tamper",
                scenario_publish_manifest_digest_tamper,
                expected_surface="publish_manifest_digests",
                extra_assert=assert_publish_manifest_digest_tamper,
            ),
        ),
        (
            "publish_manifest_auth_mode_compat_hmac_valid",
            scenario_publish_manifest_auth_mode_compat_hmac_valid,
        ),
        (
            "publish_manifest_auth_mode_default_ed25519_valid",
            scenario_publish_manifest_auth_mode_default_ed25519_valid,
        ),
        (
            "publish_manifest_auth_signature_tamper",
            lambda: run_negative_case(
                "publish_manifest_auth_signature_tamper",
                scenario_manifest_auth_signature_tamper,
                expected_surface="publish_manifest_authenticity",
                extra_assert=lambda payload: assert_manifest_auth_signature_tamper(payload, expected_scheme="hmac-sha256"),
            ),
        ),
        (
            "publish_manifest_auth_signature_tamper_ed25519",
            lambda: run_negative_case(
                "publish_manifest_auth_signature_tamper_ed25519",
                scenario_manifest_auth_signature_tamper_ed25519,
                expected_surface="publish_manifest_authenticity",
                extra_assert=lambda payload: assert_manifest_auth_signature_tamper(payload, expected_scheme="ed25519-sha256"),
            ),
        ),
        (
            "generation_mismatch",
            lambda: run_negative_case(
                "generation_mismatch",
                scenario_generation_mismatch,
                expected_surface="generation_consistency",
                extra_assert=assert_generation_mismatch,
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    failed = 0

    sentinel = scenario_seed_schema_fixture_contract()
    results.append(sentinel)
    _print_result_line(sentinel)
    if not bool(sentinel.get("ok")):
        failed += 1
        summary = {
            "ok": False,
            "total": len(results),
            "failed": failed,
            "results": results,
            "checker": str(CHECKER),
            "fixture_mode": "self_seeded_deterministic",
            "fixture_seed_version": SEED_FIXTURE_VERSION,
            "seed_schema_sentinel_version": SEED_SCHEMA_SENTINEL_VERSION,
            "fail_fast": {
                "triggered": True,
                "reason": "seed_schema_fixture_contract",
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    for name, fn in scenarios:
        try:
            result = fn()
            results.append(result)
            _print_result_line(result)
        except Exception as exc:
            failed += 1
            row = {"name": name, "ok": False, "error": str(exc)}
            results.append(row)
            _print_result_line(row)

    summary = {
        "ok": failed == 0,
        "total": len(results),
        "failed": failed,
        "results": results,
        "checker": str(CHECKER),
        "fixture_mode": "self_seeded_deterministic",
        "fixture_seed_version": SEED_FIXTURE_VERSION,
        "seed_schema_sentinel_version": SEED_SCHEMA_SENTINEL_VERSION,
        "fail_fast": {
            "triggered": False,
            "reason": None,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
