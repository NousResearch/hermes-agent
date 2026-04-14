#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
GTC_ROOT="${OPENCLAW_GTC_ROOT:-$ROOT/state/gtc-v2}"
MANIFEST_PATH_ROOT="${OPENCLAW_GTC_PUBLISH_MANIFEST_PATH_ROOT:-}"
JSON_OUT=0
STRICT=0

usage() {
  cat <<'EOF'
Usage: gtc_latest_schema_check.sh [options]

Validate generated GTC latest surfaces against architecture schemas.

Options:
  --gtc-root <path>           GTC root (default: state/gtc-v2)
  --manifest-path-root <path> Expected root for publish_manifest.latest_paths
                              (default: --gtc-root)
  --json                      JSON output
  --strict                    Exit non-zero when any validation check fails
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gtc-root)
      GTC_ROOT="${2:-}"; shift 2 ;;
    --manifest-path-root)
      MANIFEST_PATH_ROOT="${2:-}"; shift 2 ;;
    --json)
      JSON_OUT=1; shift ;;
    --strict)
      STRICT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ROOT" "$GTC_ROOT" "$MANIFEST_PATH_ROOT" "$JSON_OUT" "$STRICT" <<'PY'
import datetime as dt
import hashlib
import hmac
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
gtc_root = pathlib.Path(sys.argv[2]).resolve()
manifest_path_root_raw = str(sys.argv[3] or "").strip()
manifest_path_root = pathlib.Path(manifest_path_root_raw).resolve() if manifest_path_root_raw else gtc_root
json_out = bool(int(sys.argv[4]))
strict = bool(int(sys.argv[5]))

latest_dir = gtc_root / "latest"
connectors_dir = latest_dir / "connectors"
manifest_path_latest_dir = manifest_path_root / "latest"
manifest_path_connectors_dir = manifest_path_latest_dir / "connectors"
schema_dir = root / "ops" / "openclaw" / "architecture" / "schemas"
MANIFEST_AUTH_CANONICAL_PROFILE_BY_SCHEME: Dict[str, str] = {
    "hmac-sha256": "gtc.publish_manifest.hmac.v1",
    "ed25519-sha256": "gtc.publish_manifest.ed25519.v1",
}
MANIFEST_AUTH_FIELDS: List[str] = [
    "schema_version",
    "generated_at",
    "build_generation_id",
    "base_generation_id",
    "base_coherence_guard",
    "valid_until",
    "latest_paths",
    "latest_sha256",
]
manifest_hmac_key_file = pathlib.Path(
    str(
        os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY_FILE")
        or (root / "state" / "continuity" / "secrets" / "gtc_publish_manifest_hmac.key")
    )
).resolve()
manifest_hmac_key_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY") or "")
manifest_hmac_key_id_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY_ID") or "").strip()
manifest_ed25519_public_key_file = pathlib.Path(
    str(
        os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PUBLIC_KEY_FILE")
        or (root / "state" / "continuity" / "secrets" / "gtc_publish_manifest_ed25519_public.pem")
    )
).resolve()
manifest_ed25519_public_key_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PUBLIC_KEY_PEM") or "")
manifest_ed25519_key_id_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_KEY_ID") or "").strip()

try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker
except Exception as exc:
    payload = {
        "ok": False,
        "schema_version": "gtc.latest.schema_check.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "gtc_root": str(gtc_root),
        "error": f"jsonschema_unavailable:{exc}",
        "checks": [],
        "error_count": 1,
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("GTC LATEST SCHEMA CHECK")
        print("- ok: False")
        print(f"- error: {payload['error']}")
    if strict:
        raise SystemExit(1)
    raise SystemExit(0)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_subpath(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def sha256_file(path: pathlib.Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_connectors_dir(path: pathlib.Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    rows: List[Dict[str, str]] = []
    for p in sorted(path.glob("*.json")):
        if not p.is_file():
            continue
        digest = sha256_file(p)
        if not digest:
            return None
        rows.append({"file": p.name, "sha256": digest})
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_manifest_hmac_key_id(secret: str, explicit_key_id: Optional[str] = None) -> str:
    candidate = str(explicit_key_id or "").strip()
    if candidate:
        return candidate
    return f"ksha256_{sha256_text(secret)[:16]}"


def run_openssl_bytes(args: List[str]) -> bytes:
    try:
        cp = subprocess.run(
            ["openssl", *args],
            capture_output=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:
        raise RuntimeError(f"openssl_exec_failed:{exc}")
    if cp.returncode != 0:
        stderr = (cp.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"openssl_failed:{' '.join(args)}:{stderr}")
    return cp.stdout or b""


def derive_manifest_ed25519_public_key_sha256(public_key_pem: str) -> str:
    with tempfile.TemporaryDirectory(prefix="gtc_schema_pubsha_") as td:
        pub_path = pathlib.Path(td) / "pub.pem"
        pub_path.write_text(public_key_pem.strip() + "\n", encoding="utf-8")
        pub_der = run_openssl_bytes(["pkey", "-pubin", "-in", str(pub_path), "-outform", "DER"])
    return hashlib.sha256(pub_der).hexdigest()


def derive_manifest_ed25519_key_id(public_key_pem: str, explicit_key_id: Optional[str] = None) -> str:
    candidate = str(explicit_key_id or "").strip()
    if candidate:
        return candidate
    return f"ed25519_pksha256_{derive_manifest_ed25519_public_key_sha256(public_key_pem)[:16]}"


def load_manifest_hmac_secret_for_verify() -> Tuple[Optional[str], Optional[str], str]:
    env_secret = manifest_hmac_key_env.strip()
    if env_secret:
        key_id = derive_manifest_hmac_key_id(env_secret, manifest_hmac_key_id_env)
        return env_secret, key_id, "env_override"

    try:
        if manifest_hmac_key_file.exists():
            file_secret = manifest_hmac_key_file.read_text(encoding="utf-8").strip()
            if file_secret:
                key_id = derive_manifest_hmac_key_id(file_secret, manifest_hmac_key_id_env)
                return file_secret, key_id, rel(manifest_hmac_key_file)
    except Exception:
        pass

    return None, None, rel(manifest_hmac_key_file)


def load_manifest_ed25519_public_key_for_verify() -> Tuple[Optional[str], Optional[str], str, Optional[str], Optional[str]]:
    env_public = manifest_ed25519_public_key_env.strip()
    if env_public:
        try:
            public_pem = env_public.strip() + "\n"
            key_id = derive_manifest_ed25519_key_id(public_pem, manifest_ed25519_key_id_env)
            digest = derive_manifest_ed25519_public_key_sha256(public_pem)
            return public_pem, key_id, "env_override", digest, None
        except Exception as exc:
            return None, None, "env_override", None, str(exc)

    try:
        if manifest_ed25519_public_key_file.exists():
            public_pem = manifest_ed25519_public_key_file.read_text(encoding="utf-8").strip()
            if public_pem:
                normalized = public_pem + "\n"
                key_id = derive_manifest_ed25519_key_id(normalized, manifest_ed25519_key_id_env)
                digest = derive_manifest_ed25519_public_key_sha256(normalized)
                return normalized, key_id, rel(manifest_ed25519_public_key_file), digest, None
    except Exception as exc:
        return None, None, rel(manifest_ed25519_public_key_file), None, str(exc)

    return None, None, rel(manifest_ed25519_public_key_file), None, None


def manifest_auth_payload(manifest_doc: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in MANIFEST_AUTH_FIELDS:
        if field in manifest_doc:
            payload[field] = manifest_doc.get(field)
    return payload


def manifest_auth_fields_sha256() -> str:
    return sha256_text(canonical_json(MANIFEST_AUTH_FIELDS))


def verify_manifest_signature_ed25519(payload_json: str, signature_hex: str, public_key_pem: str) -> Tuple[bool, Optional[str]]:
    try:
        with tempfile.TemporaryDirectory(prefix="gtc_schema_verify_ed25519_") as td:
            pub_path = pathlib.Path(td) / "pub.pem"
            payload_path = pathlib.Path(td) / "payload.bin"
            sig_path = pathlib.Path(td) / "sig.bin"

            pub_path.write_text(public_key_pem.strip() + "\n", encoding="utf-8")
            payload_path.write_bytes(payload_json.encode("utf-8"))
            sig_path.write_bytes(bytes.fromhex(signature_hex))

            cp = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-rawin",
                    "-pubin",
                    "-inkey",
                    str(pub_path),
                    "-in",
                    str(payload_path),
                    "-sigfile",
                    str(sig_path),
                ],
                capture_output=True,
                check=False,
                timeout=15,
            )
            if cp.returncode == 0:
                return True, None
            stderr = (cp.stderr or b"").decode("utf-8", errors="replace").strip()
            stdout = (cp.stdout or b"").decode("utf-8", errors="replace").strip()
            detail = stderr or stdout or f"verify_exit_{cp.returncode}"
            return False, detail[:180]
    except Exception as exc:
        return False, f"verify_exec_failed:{exc}"


def verify_manifest_authenticity(manifest_doc: Optional[Dict[str, Any]], anchor_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "surface": "publish_manifest_authenticity",
        "ok": False,
    }

    if not isinstance(anchor_doc, dict):
        row["error"] = "publish_anchor_unavailable"
        return row
    if not isinstance(manifest_doc, dict):
        row["error"] = "publish_manifest_unavailable"
        return row

    root_obj = anchor_doc.get("manifest_auth_root") if isinstance(anchor_doc.get("manifest_auth_root"), dict) else None
    auth_obj = manifest_doc.get("manifest_auth") if isinstance(manifest_doc.get("manifest_auth"), dict) else None
    if root_obj is None:
        row["error"] = "manifest_auth_root_missing"
        return row
    if auth_obj is None:
        row["error"] = "manifest_auth_missing"
        return row

    root_scheme = str(root_obj.get("scheme") or "").strip().lower()
    auth_scheme = str(auth_obj.get("scheme") or "").strip().lower()
    if root_scheme != auth_scheme:
        row["error"] = "manifest_auth_scheme_mismatch"
        row["root_scheme"] = root_scheme or None
        row["manifest_scheme"] = auth_scheme or None
        return row

    expected_profile = MANIFEST_AUTH_CANONICAL_PROFILE_BY_SCHEME.get(root_scheme)
    if not expected_profile:
        row["error"] = "unsupported_manifest_auth_scheme"
        row["root_scheme"] = root_scheme or None
        return row

    root_profile = str(root_obj.get("canonical_profile") or "").strip()
    auth_profile = str(auth_obj.get("canonical_profile") or "").strip()
    if root_profile != expected_profile or auth_profile != expected_profile:
        row["error"] = "canonical_profile_mismatch"
        row["expected_profile"] = expected_profile
        row["root_profile"] = root_profile or None
        row["manifest_profile"] = auth_profile or None
        return row

    root_fields = root_obj.get("payload_fields")
    if not isinstance(root_fields, list) or [str(v) for v in root_fields] != MANIFEST_AUTH_FIELDS:
        row["error"] = "payload_fields_mismatch"
        row["expected_payload_fields"] = list(MANIFEST_AUTH_FIELDS)
        row["anchor_payload_fields"] = [str(v) for v in root_fields] if isinstance(root_fields, list) else None
        return row

    root_fields_sha = str(root_obj.get("payload_fields_sha256") or "").strip().lower()
    expected_fields_sha = manifest_auth_fields_sha256()
    if root_fields_sha != expected_fields_sha:
        row["error"] = "payload_fields_sha256_mismatch"
        row["expected_payload_fields_sha256"] = expected_fields_sha
        row["anchor_payload_fields_sha256"] = root_fields_sha or None
        return row

    root_key_id = str(root_obj.get("key_id") or "").strip()
    auth_key_id = str(auth_obj.get("key_id") or "").strip()
    if not root_key_id or not auth_key_id or root_key_id != auth_key_id:
        row["error"] = "manifest_key_id_mismatch"
        row["anchor_key_id"] = root_key_id or None
        row["manifest_key_id"] = auth_key_id or None
        return row

    payload = manifest_auth_payload(manifest_doc)
    payload_json = canonical_json(payload)
    expected_payload_sha = sha256_text(payload_json)
    claimed_payload_sha = str(auth_obj.get("payload_sha256") or "").strip().lower()
    if claimed_payload_sha != expected_payload_sha:
        row["error"] = "manifest_payload_sha256_mismatch"
        row["expected_payload_sha256"] = expected_payload_sha
        row["claimed_payload_sha256"] = claimed_payload_sha or None
        return row

    claimed_signature = str(auth_obj.get("signature") or "").strip().lower()

    if root_scheme == "hmac-sha256":
        if not SHA256_RE.fullmatch(claimed_signature):
            row["error"] = "manifest_signature_invalid"
            return row

        secret, configured_key_id, key_source = load_manifest_hmac_secret_for_verify()
        row["key_source"] = key_source
        if not secret:
            row["error"] = "manifest_hmac_secret_unavailable"
            return row

        if configured_key_id and configured_key_id != root_key_id:
            row["error"] = "manifest_key_id_not_configured"
            row["configured_key_id"] = configured_key_id
            row["anchor_key_id"] = root_key_id
            return row

        expected_signature = hmac.new(secret.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_signature, claimed_signature):
            row["error"] = "manifest_signature_mismatch"
            row["expected_signature_prefix"] = expected_signature[:12]
            row["claimed_signature_prefix"] = claimed_signature[:12]
            return row
    elif root_scheme == "ed25519-sha256":
        if not ED25519_SIG_HEX_RE.fullmatch(claimed_signature):
            row["error"] = "manifest_signature_invalid"
            return row

        public_key_pem, configured_key_id, key_source, configured_pub_sha, key_load_error = load_manifest_ed25519_public_key_for_verify()
        row["key_source"] = key_source
        if key_load_error:
            row["error"] = "manifest_ed25519_public_key_invalid"
            row["detail"] = key_load_error
            return row
        if not public_key_pem:
            row["error"] = "manifest_ed25519_public_key_unavailable"
            return row

        if configured_key_id and configured_key_id != root_key_id:
            row["error"] = "manifest_key_id_not_configured"
            row["configured_key_id"] = configured_key_id
            row["anchor_key_id"] = root_key_id
            return row

        anchor_pub_sha = str(root_obj.get("public_key_sha256") or "").strip().lower()
        if not SHA256_RE.fullmatch(anchor_pub_sha):
            row["error"] = "manifest_public_key_sha256_invalid"
            row["anchor_public_key_sha256"] = anchor_pub_sha or None
            return row
        if configured_pub_sha and configured_pub_sha != anchor_pub_sha:
            row["error"] = "manifest_public_key_sha256_mismatch"
            row["configured_public_key_sha256"] = configured_pub_sha
            row["anchor_public_key_sha256"] = anchor_pub_sha
            return row

        verified, verify_error = verify_manifest_signature_ed25519(payload_json, claimed_signature, public_key_pem)
        if not verified:
            row["error"] = "manifest_signature_mismatch"
            row["detail"] = verify_error
            return row
        row["public_key_sha256"] = anchor_pub_sha
    else:
        row["error"] = "unsupported_manifest_auth_scheme"
        row["root_scheme"] = root_scheme or None
        return row

    row["ok"] = True
    row["scheme"] = root_scheme
    row["key_id"] = root_key_id
    row["canonical_profile"] = expected_profile
    row["payload_sha256"] = expected_payload_sha
    return row


def resolve_manifest_ref_candidates(raw: str) -> List[pathlib.Path]:
    value = str(raw or "").strip()
    if not value:
        return []

    ref = pathlib.Path(value)
    candidates: List[pathlib.Path] = []
    if ref.is_absolute():
        candidates.append(ref.resolve())
    else:
        candidates.append((root / ref).resolve())
        candidates.append((gtc_root / ref).resolve())

    seen: set[str] = set()
    deduped: List[pathlib.Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
ED25519_SIG_HEX_RE = re.compile(r"^[a-f0-9]{128}$")

checks: List[Dict[str, Any]] = []
errors: List[str] = []


def add_check(row: Dict[str, Any]) -> None:
    checks.append(row)
    if not bool(row.get("ok")):
        errors.append(str(row.get("surface") or row.get("path") or "unknown"))


def validate_surface(surface: str, data_path: pathlib.Path, schema_name: str) -> Optional[Dict[str, Any]]:
    schema_path = schema_dir / schema_name
    if not data_path.exists():
        add_check(
            {
                "surface": surface,
                "ok": False,
                "path": rel(data_path),
                "schema": f"ops/openclaw/architecture/schemas/{schema_name}",
                "error": "missing_surface",
            }
        )
        return None

    if not schema_path.exists():
        add_check(
            {
                "surface": surface,
                "ok": False,
                "path": rel(data_path),
                "schema": f"ops/openclaw/architecture/schemas/{schema_name}",
                "error": "missing_schema",
            }
        )
        return None

    doc = load_json(data_path)
    if doc is None:
        add_check(
            {
                "surface": surface,
                "ok": False,
                "path": rel(data_path),
                "schema": f"ops/openclaw/architecture/schemas/{schema_name}",
                "error": "surface_not_json_object",
            }
        )
        return None

    schema_doc = load_json(schema_path)
    if schema_doc is None:
        add_check(
            {
                "surface": surface,
                "ok": False,
                "path": rel(data_path),
                "schema": f"ops/openclaw/architecture/schemas/{schema_name}",
                "error": "schema_not_json_object",
            }
        )
        return None

    try:
        Draft202012Validator(schema_doc, format_checker=FormatChecker()).validate(doc)
        add_check(
            {
                "surface": surface,
                "ok": True,
                "path": rel(data_path),
                "schema": f"ops/openclaw/architecture/schemas/{schema_name}",
            }
        )
    except Exception as exc:
        add_check(
            {
                "surface": surface,
                "ok": False,
                "path": rel(data_path),
                "schema": f"ops/openclaw/architecture/schemas/{schema_name}",
                "error": str(exc),
            }
        )
    return doc


continuity_current = validate_surface(
    "continuity_current",
    latest_dir / "continuity_current.json",
    "gtc_latest.schema.json",
)
gateboard = validate_surface(
    "gateboard",
    latest_dir / "gateboard.json",
    "gtc_gateboard.schema.json",
)
event_projection = validate_surface(
    "event_projection",
    latest_dir / "event_projection.json",
    "gtc_event.schema.json",
)
incident_replay = validate_surface(
    "incident_replay",
    latest_dir / "incident_replay.json",
    "gtc_incident_replay.schema.json",
)
publish_manifest = validate_surface(
    "publish_manifest",
    latest_dir / "publish_manifest.json",
    "gtc_publish_manifest.schema.json",
)
publish_anchor = validate_surface(
    "publish_anchor",
    latest_dir / "publish_anchor.json",
    "gtc_publish_anchor.schema.json",
)

manifest_auth_check = verify_manifest_authenticity(publish_manifest, publish_anchor)
add_check(manifest_auth_check)
manifest_trusted = bool(manifest_auth_check.get("ok") is True)

manifest_path_expected: Dict[str, pathlib.Path] = {
    "publish_anchor": manifest_path_latest_dir / "publish_anchor.json",
    "continuity_current": manifest_path_latest_dir / "continuity_current.json",
    "gateboard": manifest_path_latest_dir / "gateboard.json",
    "event_projection": manifest_path_latest_dir / "event_projection.json",
    "incident_replay": manifest_path_latest_dir / "incident_replay.json",
    "connectors_dir": manifest_path_connectors_dir,
}
manifest_digest_expected: Dict[str, pathlib.Path] = {
    "publish_anchor": latest_dir / "publish_anchor.json",
    "continuity_current": latest_dir / "continuity_current.json",
    "gateboard": latest_dir / "gateboard.json",
    "event_projection": latest_dir / "event_projection.json",
    "incident_replay": latest_dir / "incident_replay.json",
    "connectors_dir": connectors_dir,
}

manifest_missing_keys: List[str] = []
manifest_invalid_type: List[str] = []
manifest_mismatched_paths: List[Dict[str, str]] = []
manifest_path_out_of_tree: List[str] = []
manifest_missing_targets: List[str] = []

if not manifest_trusted:
    add_check(
        {
            "surface": "publish_manifest_paths",
            "ok": False,
            "error": "manifest_untrusted_authenticity_failed",
        }
    )
elif isinstance(publish_manifest, dict):
    latest_paths = publish_manifest.get("latest_paths")
    if not isinstance(latest_paths, dict):
        add_check(
            {
                "surface": "publish_manifest_paths",
                "ok": False,
                "error": "latest_paths_not_object",
            }
        )
    else:
        for key, expected_target in manifest_path_expected.items():
            raw = latest_paths.get(key)
            if raw is None:
                manifest_missing_keys.append(key)
                continue
            if not isinstance(raw, str) or not raw.strip():
                manifest_invalid_type.append(key)
                continue

            expected_resolved = expected_target.resolve()
            candidates = resolve_manifest_ref_candidates(raw)

            if not any(candidate == expected_resolved for candidate in candidates):
                manifest_mismatched_paths.append(
                    {
                        "key": key,
                        "expected": rel(expected_target),
                        "actual": str(raw),
                    }
                )

            if not any(is_subpath(candidate, manifest_path_root) for candidate in candidates):
                manifest_path_out_of_tree.append(key)

            if not expected_target.exists():
                manifest_missing_targets.append(key)

        add_check(
            {
                "surface": "publish_manifest_paths",
                "ok": not (
                    manifest_missing_keys
                    or manifest_invalid_type
                    or manifest_mismatched_paths
                    or manifest_path_out_of_tree
                    or manifest_missing_targets
                ),
                "missing_keys": sorted(manifest_missing_keys),
                "invalid_type_keys": sorted(manifest_invalid_type),
                "mismatched_paths": manifest_mismatched_paths,
                "out_of_tree_keys": sorted(set(manifest_path_out_of_tree)),
                "missing_targets": sorted(set(manifest_missing_targets)),
            }
        )
else:
    add_check(
        {
            "surface": "publish_manifest_paths",
            "ok": False,
            "error": "publish_manifest_unavailable",
        }
    )

manifest_digest_missing_keys: List[str] = []
manifest_digest_invalid_type: List[str] = []
manifest_digest_missing_targets: List[str] = []
manifest_digest_mismatch: List[Dict[str, str]] = []

if not manifest_trusted:
    add_check(
        {
            "surface": "publish_manifest_digests",
            "ok": False,
            "error": "manifest_untrusted_authenticity_failed",
        }
    )
elif isinstance(publish_manifest, dict):
    latest_sha256 = publish_manifest.get("latest_sha256")
    if not isinstance(latest_sha256, dict):
        add_check(
            {
                "surface": "publish_manifest_digests",
                "ok": False,
                "error": "latest_sha256_not_object",
            }
        )
    else:
        for key, expected_target in manifest_digest_expected.items():
            raw = latest_sha256.get(key)
            if raw is None:
                manifest_digest_missing_keys.append(key)
                continue
            if not isinstance(raw, str):
                manifest_digest_invalid_type.append(key)
                continue
            expected_digest = raw.strip().lower()
            if not SHA256_RE.fullmatch(expected_digest):
                manifest_digest_invalid_type.append(key)
                continue

            if key == "connectors_dir":
                actual_digest = sha256_connectors_dir(expected_target)
            else:
                actual_digest = sha256_file(expected_target)

            if not actual_digest:
                manifest_digest_missing_targets.append(key)
                continue
            if actual_digest != expected_digest:
                manifest_digest_mismatch.append(
                    {
                        "key": key,
                        "expected_sha256": expected_digest,
                        "actual_sha256": actual_digest,
                    }
                )

        add_check(
            {
                "surface": "publish_manifest_digests",
                "ok": not (
                    manifest_digest_missing_keys
                    or manifest_digest_invalid_type
                    or manifest_digest_missing_targets
                    or manifest_digest_mismatch
                ),
                "algorithm": "sha256",
                "missing_keys": sorted(manifest_digest_missing_keys),
                "invalid_type_keys": sorted(manifest_digest_invalid_type),
                "missing_targets": sorted(manifest_digest_missing_targets),
                "mismatched": manifest_digest_mismatch,
            }
        )
else:
    add_check(
        {
            "surface": "publish_manifest_digests",
            "ok": False,
            "error": "publish_manifest_unavailable",
        }
    )

if not connectors_dir.exists() or not connectors_dir.is_dir():
    add_check(
        {
            "surface": "connectors_dir",
            "ok": False,
            "path": rel(connectors_dir),
            "schema": "ops/openclaw/architecture/schemas/gtc_connector_latest.schema.json",
            "error": "connectors_dir_missing",
        }
    )
    connector_files: List[pathlib.Path] = []
else:
    connector_files = sorted(connectors_dir.glob("*.json"))

if not connector_files:
    add_check(
        {
            "surface": "connectors",
            "ok": False,
            "path": rel(connectors_dir),
            "schema": "ops/openclaw/architecture/schemas/gtc_connector_latest.schema.json",
            "error": "no_connector_surfaces_found",
        }
    )

for p in connector_files:
    validate_surface(
        f"connector::{p.name}",
        p,
        "gtc_connector_latest.schema.json",
    )

# Generation consistency cross-surface invariant.
generation_sources: Dict[str, Optional[str]] = {}
for name, doc in [
    ("publish_anchor", publish_anchor),
    ("continuity_current", continuity_current),
    ("gateboard", gateboard),
    ("event_projection", event_projection),
    ("incident_replay", incident_replay),
    ("publish_manifest", publish_manifest),
]:
    val = None
    if isinstance(doc, dict):
        raw = str(doc.get("build_generation_id") or "").strip()
        if raw:
            val = raw
    generation_sources[name] = val

for p in connector_files:
    doc = load_json(p)
    val = None
    if isinstance(doc, dict):
        raw = str(doc.get("build_generation_id") or "").strip()
        if raw:
            val = raw
    generation_sources[f"connector::{p.name}"] = val

expected_generation = None
for key in ["publish_manifest", "publish_anchor", "continuity_current", "gateboard", "event_projection", "incident_replay"]:
    candidate = generation_sources.get(key)
    if candidate:
        expected_generation = candidate
        break

missing_generation = [k for k, v in generation_sources.items() if not v]
mismatched_generation = [k for k, v in generation_sources.items() if v and expected_generation and v != expected_generation]
generation_ok = bool(expected_generation) and len(missing_generation) == 0 and len(mismatched_generation) == 0

add_check(
    {
        "surface": "generation_consistency",
        "ok": generation_ok,
        "expected_generation_id": expected_generation,
        "missing_generation": missing_generation,
        "mismatched_generation": mismatched_generation,
    }
)

summary = {
    "ok": len(errors) == 0,
    "schema_version": "gtc.latest.schema_check.v1",
    "generated_at": now_iso(),
    "gtc_root": rel(gtc_root),
    "manifest_path_root": rel(manifest_path_root),
    "latest_dir": rel(latest_dir),
    "checks": checks,
    "surface_count": len(checks),
    "connector_count": len(connector_files),
    "error_count": len(errors),
    "manifest_authenticity": {
        "ok": manifest_trusted,
        "scheme": manifest_auth_check.get("scheme"),
        "key_id": manifest_auth_check.get("key_id"),
        "canonical_profile": manifest_auth_check.get("canonical_profile"),
        "error": manifest_auth_check.get("error"),
    },
    "generation_consistency": {
        "ok": generation_ok,
        "expected_generation_id": expected_generation,
        "missing_generation": missing_generation,
        "mismatched_generation": mismatched_generation,
    },
}

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("GTC LATEST SCHEMA CHECK")
    print(f"- ok: {summary['ok']}")
    print(f"- connector_count: {summary['connector_count']}")
    print(f"- checks: {summary['surface_count']}")
    print(f"- errors: {summary['error_count']}")
    ma = summary.get("manifest_authenticity") or {}
    print(
        "- manifest_authenticity: "
        f"ok={ma.get('ok')} "
        f"scheme={ma.get('scheme') or 'n/a'} "
        f"key_id={ma.get('key_id') or 'n/a'} "
        f"error={ma.get('error') or 'none'}"
    )
    gc = summary.get("generation_consistency") or {}
    print(
        "- generation_consistency: "
        f"ok={gc.get('ok')} expected={gc.get('expected_generation_id') or 'n/a'} "
        f"missing={len(gc.get('missing_generation') or [])} mismatched={len(gc.get('mismatched_generation') or [])}"
    )

if strict and not summary["ok"]:
    raise SystemExit(1)
PY
