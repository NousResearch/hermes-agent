#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
GTC_ROOT="${OPENCLAW_GTC_ROOT:-$ROOT/state/gtc-v2}"
JSON_OUT=0
STRICT=0
NO_SURFACES=0
SKIP_SCHEMA_GATE=0
MAX_ROWS="${OPENCLAW_GTC_SYNC_MAX_ROWS:-500}"
RECOVERY_MODE="${OPENCLAW_GTC_RECOVERY_MODE:-recover-then-publish}"
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: gtc_v2_sync.sh [options]

Sync Ground-Truth Connectors v2 evidence/index/latest surfaces from continuity runtime state.

Options:
  --db <path>          Continuity SQLite path (default: state/continuity/continuity_os.sqlite)
  --gtc-root <path>    GTC root directory (default: state/gtc-v2)
  --max-rows <n>       Per-stream ingest cap for queue/event append streams (default: 500)
  --recovery-mode <mode>  Publish boundary mode: recover-then-publish (default) or recover-only
  --no-surfaces        Skip markdown surface generation (JSON latest pointers still written)
  --skip-schema-gate   Skip post-publish schema validation gate (break-glass)
  --json               JSON output
  --strict             Exit non-zero when mutate_allowed=false
  --action-token <value>  Canonical mutation token for direct entrypoint use
  --truth-anchor <value>  Legacy alias of --action-token
  --allow-legacy-anchor   Allow legacy anchor-only token mode for direct token validation
  --mutation-ticket <value>  Authority ticket JSON string, @path, or path (high-risk token path)
  --attestation <name>    Satisfied authority attestation (repeatable)
  --attestation-object <value> Structured attestation JSON string, @path, or path (repeatable)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --gtc-root)
      GTC_ROOT="${2:-}"; shift 2 ;;
    --max-rows)
      MAX_ROWS="${2:-}"; shift 2 ;;
    --recovery-mode)
      RECOVERY_MODE="${2:-}"; shift 2 ;;
    --no-surfaces)
      NO_SURFACES=1; shift ;;
    --skip-schema-gate)
      SKIP_SCHEMA_GATE=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    --strict)
      STRICT=1; shift ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      MUTATION_ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      MUTATION_ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! [[ "$MAX_ROWS" =~ ^[0-9]+$ ]] || [[ "$MAX_ROWS" -lt 1 ]]; then
  echo "invalid --max-rows: $MAX_ROWS" >&2
  exit 2
fi

case "$RECOVERY_MODE" in
  recover-then-publish|recover-only)
    ;;
  *)
    echo "invalid --recovery-mode: $RECOVERY_MODE (expected recover-then-publish|recover-only)" >&2
    exit 2
    ;;
esac

guard_args=(
  --script "gtc_v2_sync.sh"
  --risk-tier "high"
  --mutation-operation "gtc_v2_sync:publish"
)
if [[ -n "$ACTION_TOKEN" ]]; then
  guard_args+=(--action-token "$ACTION_TOKEN")
fi
if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
  guard_args+=(--allow-legacy-anchor)
fi
if [[ -n "$MUTATION_TICKET" ]]; then
  guard_args+=(--mutation-ticket "$MUTATION_TICKET")
fi
for att in "${MUTATION_ATTESTATIONS[@]}"; do
  if [[ -n "${att:-}" ]]; then
    guard_args+=(--attestation "$att")
  fi
done
for att_obj in "${MUTATION_ATTESTATION_OBJECTS[@]}"; do
  if [[ -n "${att_obj:-}" ]]; then
    guard_args+=(--attestation-object "$att_obj")
  fi
done
"$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

python3 - "$ROOT" "$DB_PATH" "$GTC_ROOT" "$JSON_OUT" "$STRICT" "$NO_SURFACES" "$MAX_ROWS" "$SKIP_SCHEMA_GATE" "$RECOVERY_MODE" <<'PY'
import datetime as dt
import errno
import fcntl
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
db_path = pathlib.Path(sys.argv[2]).resolve()
gtc_root = pathlib.Path(sys.argv[3]).resolve()
json_out = bool(int(sys.argv[4]))
strict = bool(int(sys.argv[5]))
no_surfaces = bool(int(sys.argv[6]))
max_rows = max(1, int(sys.argv[7]))
skip_schema_gate = bool(int(sys.argv[8]))
recovery_mode = str(sys.argv[9] or "recover-then-publish").strip().lower()
if recovery_mode not in {"recover-then-publish", "recover-only"}:
    raise SystemExit(f"invalid_recovery_mode:{recovery_mode or 'empty'}")

continuity_dir = (root / "ops" / "openclaw" / "continuity").resolve()
if str(continuity_dir) not in sys.path:
    sys.path.insert(0, str(continuity_dir))

try:
    from continuity_policy import (
        DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC as _DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC,
        DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC as _DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC,
        is_severe_verify_gate_preflight_blocker as _policy_is_severe_verify_gate_preflight_blocker,
        read_nonnegative_int_env as _read_nonnegative_int_env,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC = 1800
    _DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC = 1800

    def _read_nonnegative_int_env(name: str, *, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(int(default)))))
        except Exception:
            return int(default)

    _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKERS = {"strict_autonomy_required_override_denied"}
    _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKER_PREFIXES = (
        "layered_health_gate:",
        "execution_supervisor_launch_readiness_severity_gate:",
        "execution_supervisor_probe_execution_gate:",
        "execution_supervisor_worker_health_canary_gate:",
    )

    def _policy_is_severe_verify_gate_preflight_blocker(reason: Any) -> bool:
        blocker = str(reason or "").strip()
        if not blocker:
            return False
        if blocker in _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKERS:
            return True
        return any(blocker.startswith(prefix) for prefix in _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKER_PREFIXES)

schema_dir = gtc_root / "schema"
evidence_root = gtc_root / "evidence"
live_latest_dir = gtc_root / "latest"
live_surfaces_dir = gtc_root / "surfaces"
live_connectors_latest_dir = live_latest_dir / "connectors"
live_cursors_path = live_latest_dir / "cursors.json"
live_publish_manifest_path = live_latest_dir / "publish_manifest.json"
live_publish_anchor_path = live_latest_dir / "publish_anchor.json"
continuity_current_path = root / "state" / "continuity" / "current.json"
continuity_now_latest_path = root / "state" / "continuity" / "latest" / "continuity_now_latest.json"
coherence_bundle_path = root / "state" / "continuity" / "latest" / "coherence_bundle_latest.json"
publish_lock_dir = gtc_root / "locks"
publish_lock_path = publish_lock_dir / "gtc_latest_publish.lock"
publish_journal_dir = gtc_root / "publish_journal"
publish_journal_latest_path = publish_journal_dir / "latest_transaction.json"
publish_journal_events_path = publish_journal_dir / "transactions.jsonl"

MANIFEST_AUTH_MODE = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_AUTH_MODE") or "ed25519").strip().lower()
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
manifest_ed25519_private_key_file = pathlib.Path(
    str(
        os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PRIVATE_KEY_FILE")
        or (root / "state" / "continuity" / "secrets" / "gtc_publish_manifest_ed25519_private.pem")
    )
).resolve()
manifest_ed25519_public_key_file = pathlib.Path(
    str(
        os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PUBLIC_KEY_FILE")
        or (root / "state" / "continuity" / "secrets" / "gtc_publish_manifest_ed25519_public.pem")
    )
).resolve()
manifest_ed25519_private_key_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PRIVATE_KEY_PEM") or "")
manifest_ed25519_public_key_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PUBLIC_KEY_PEM") or "")
manifest_ed25519_key_id_env = str(os.environ.get("OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_KEY_ID") or "").strip()

staging_root = gtc_root / ".staging" / f"gtc_sync_{uuid.uuid4().hex[:12]}"
staging_latest_dir = staging_root / "latest"
staging_surfaces_dir = staging_root / "surfaces"
connectors_latest_dir = staging_latest_dir / "connectors"
cursors_path = staging_latest_dir / "cursors.json"
latest_dir = staging_latest_dir
surfaces_dir = staging_surfaces_dir

for p in [
    schema_dir,
    evidence_root,
    gtc_root / ".staging",
    staging_latest_dir,
    connectors_latest_dir,
    publish_lock_dir,
    publish_journal_dir,
]:
    p.mkdir(parents=True, exist_ok=True)
if not no_surfaces:
    staging_surfaces_dir.mkdir(parents=True, exist_ok=True)


def now_dt() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def now_iso() -> str:
    return now_dt().isoformat().replace("+00:00", "Z")


def read_boot_id() -> str:
    p = pathlib.Path("/proc/sys/kernel/random/boot_id")
    try:
        raw = p.read_text(encoding="utf-8").strip()
        if raw:
            return raw
    except Exception:
        pass
    return "boot_id_unknown"


def parse_iso(value: str) -> Optional[dt.datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        d = dt.datetime.fromisoformat(raw)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_connectors_dir(path: pathlib.Path) -> str:
    rows: List[Dict[str, str]] = []
    if path.exists() and path.is_dir():
        for p in sorted(path.glob("*.json")):
            if not p.is_file():
                continue
            rows.append({"file": p.name, "sha256": sha256_file(p)})
    return sha256_text(canonical_json(rows))


durability_metrics: Dict[str, int] = {
    "file_fsync_count": 0,
    "dir_fsync_count": 0,
    "dir_fsync_unsupported_count": 0,
}


def fsync_file(path: pathlib.Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())
    durability_metrics["file_fsync_count"] = int(durability_metrics.get("file_fsync_count") or 0) + 1


def fsync_dir(path: pathlib.Path) -> bool:
    fd: Optional[int] = None
    try:
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        durability_metrics["dir_fsync_count"] = int(durability_metrics.get("dir_fsync_count") or 0) + 1
        return True
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.EROFS, errno.EPERM}:
            durability_metrics["dir_fsync_unsupported_count"] = int(
                durability_metrics.get("dir_fsync_unsupported_count") or 0
            ) + 1
            return False
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def durable_replace(src: pathlib.Path, dst: pathlib.Path) -> None:
    os.replace(src, dst)
    fsync_dir(dst.parent)


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        durable_replace(tmp, path)
        fsync_file(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def manifest_auth_payload(manifest_doc: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in MANIFEST_AUTH_FIELDS:
        if field in manifest_doc:
            payload[field] = manifest_doc.get(field)
    return payload


def manifest_auth_fields_sha256() -> str:
    return sha256_text(canonical_json(MANIFEST_AUTH_FIELDS))


def derive_manifest_hmac_key_id(secret: str, explicit_key_id: Optional[str] = None) -> str:
    candidate = str(explicit_key_id or "").strip()
    if candidate:
        return candidate
    return f"ksha256_{sha256_text(secret)[:16]}"


def run_openssl_bytes(args: List[str], *, stdin: Optional[bytes] = None) -> bytes:
    try:
        cp = subprocess.run(
            ["openssl", *args],
            input=stdin,
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
    with tempfile.TemporaryDirectory(prefix="gtc_ed25519_pubsha_") as td:
        pub_path = pathlib.Path(td) / "pub.pem"
        pub_path.write_text(public_key_pem.strip() + "\n", encoding="utf-8")
        pub_der = run_openssl_bytes(["pkey", "-pubin", "-in", str(pub_path), "-outform", "DER"])
    return hashlib.sha256(pub_der).hexdigest()


def derive_manifest_ed25519_key_id(public_key_pem: str, explicit_key_id: Optional[str] = None) -> str:
    candidate = str(explicit_key_id or "").strip()
    if candidate:
        return candidate
    return f"ed25519_pksha256_{derive_manifest_ed25519_public_key_sha256(public_key_pem)[:16]}"


def derive_ed25519_public_key_from_private(private_key_pem: str) -> str:
    with tempfile.TemporaryDirectory(prefix="gtc_ed25519_pubout_") as td:
        priv_path = pathlib.Path(td) / "priv.pem"
        pub_path = pathlib.Path(td) / "pub.pem"
        priv_path.write_text(private_key_pem.strip() + "\n", encoding="utf-8")
        run_openssl_bytes(["pkey", "-in", str(priv_path), "-pubout", "-out", str(pub_path)])
        return pub_path.read_text(encoding="utf-8").strip() + "\n"


def load_or_create_manifest_hmac_secret() -> Tuple[str, str, str]:
    env_secret = manifest_hmac_key_env.strip()
    if env_secret:
        key_id = derive_manifest_hmac_key_id(env_secret, manifest_hmac_key_id_env)
        return env_secret, key_id, "env_override"

    manifest_hmac_key_file.parent.mkdir(parents=True, exist_ok=True)
    file_secret = ""
    try:
        if manifest_hmac_key_file.exists():
            file_secret = manifest_hmac_key_file.read_text(encoding="utf-8").strip()
    except Exception:
        file_secret = ""

    if not file_secret:
        file_secret = secrets.token_hex(32)
        atomic_write(manifest_hmac_key_file, file_secret + "\n")

    try:
        os.chmod(manifest_hmac_key_file, 0o600)
    except Exception:
        pass

    key_id = derive_manifest_hmac_key_id(file_secret, manifest_hmac_key_id_env)
    return file_secret, key_id, rel(manifest_hmac_key_file)


def load_or_create_manifest_ed25519_keypair() -> Tuple[str, str, str, str, str, str]:
    env_private = manifest_ed25519_private_key_env.strip()
    env_public = manifest_ed25519_public_key_env.strip()

    if env_private or env_public:
        if not env_private:
            raise RuntimeError("manifest_ed25519_private_key_missing")
        private_key_pem = env_private.strip() + "\n"
        public_key_pem = env_public.strip() + "\n" if env_public else derive_ed25519_public_key_from_private(private_key_pem)
        key_id = derive_manifest_ed25519_key_id(public_key_pem, manifest_ed25519_key_id_env)
        public_key_sha256 = derive_manifest_ed25519_public_key_sha256(public_key_pem)
        public_source = "env_public_override" if env_public else "derived_from_env_private"
        return private_key_pem, public_key_pem, key_id, "env_private_override", public_source, public_key_sha256

    manifest_ed25519_private_key_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_ed25519_public_key_file.parent.mkdir(parents=True, exist_ok=True)

    private_key_pem = ""
    try:
        if manifest_ed25519_private_key_file.exists():
            private_key_pem = manifest_ed25519_private_key_file.read_text(encoding="utf-8").strip()
    except Exception:
        private_key_pem = ""

    public_key_pem = ""
    try:
        if manifest_ed25519_public_key_file.exists():
            public_key_pem = manifest_ed25519_public_key_file.read_text(encoding="utf-8").strip()
    except Exception:
        public_key_pem = ""

    if not private_key_pem:
        with tempfile.TemporaryDirectory(prefix="gtc_ed25519_gen_") as td:
            priv_path = pathlib.Path(td) / "priv.pem"
            pub_path = pathlib.Path(td) / "pub.pem"
            run_openssl_bytes(["genpkey", "-algorithm", "Ed25519", "-out", str(priv_path)])
            run_openssl_bytes(["pkey", "-in", str(priv_path), "-pubout", "-out", str(pub_path)])
            private_key_pem = priv_path.read_text(encoding="utf-8").strip()
            public_key_pem = pub_path.read_text(encoding="utf-8").strip()

        atomic_write(manifest_ed25519_private_key_file, private_key_pem + "\n")
        atomic_write(manifest_ed25519_public_key_file, public_key_pem + "\n")
    elif not public_key_pem:
        public_key_pem = derive_ed25519_public_key_from_private(private_key_pem).strip()
        atomic_write(manifest_ed25519_public_key_file, public_key_pem + "\n")

    try:
        os.chmod(manifest_ed25519_private_key_file, 0o600)
    except Exception:
        pass
    try:
        os.chmod(manifest_ed25519_public_key_file, 0o644)
    except Exception:
        pass

    private_out = private_key_pem.strip() + "\n"
    public_out = public_key_pem.strip() + "\n"
    key_id = derive_manifest_ed25519_key_id(public_out, manifest_ed25519_key_id_env)
    public_key_sha256 = derive_manifest_ed25519_public_key_sha256(public_out)
    return (
        private_out,
        public_out,
        key_id,
        rel(manifest_ed25519_private_key_file),
        rel(manifest_ed25519_public_key_file),
        public_key_sha256,
    )


def select_manifest_auth_scheme() -> str:
    if MANIFEST_AUTH_MODE in {"hmac", "hmac-sha256"}:
        return "hmac-sha256"
    if MANIFEST_AUTH_MODE in {"ed25519", "ed25519-sha256"}:
        return "ed25519-sha256"
    raise RuntimeError(f"unsupported_manifest_auth_mode:{MANIFEST_AUTH_MODE or 'empty'}")


def build_manifest_auth_hmac(manifest_doc: Dict[str, Any], *, hmac_secret: str, key_id: str) -> Dict[str, str]:
    payload_obj = manifest_auth_payload(manifest_doc)
    payload_json = canonical_json(payload_obj)
    payload_sha256 = sha256_text(payload_json)
    signature = hmac.new(hmac_secret.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "scheme": "hmac-sha256",
        "key_id": key_id,
        "canonical_profile": MANIFEST_AUTH_CANONICAL_PROFILE_BY_SCHEME["hmac-sha256"],
        "payload_sha256": payload_sha256,
        "signature": signature,
    }


def build_manifest_auth_ed25519(manifest_doc: Dict[str, Any], *, private_key_pem: str, key_id: str) -> Dict[str, str]:
    payload_obj = manifest_auth_payload(manifest_doc)
    payload_json = canonical_json(payload_obj)
    payload_sha256 = sha256_text(payload_json)
    with tempfile.TemporaryDirectory(prefix="gtc_ed25519_sign_") as td:
        priv_path = pathlib.Path(td) / "priv.pem"
        payload_path = pathlib.Path(td) / "payload.bin"
        sig_path = pathlib.Path(td) / "sig.bin"
        priv_path.write_text(private_key_pem.strip() + "\n", encoding="utf-8")
        payload_path.write_bytes(payload_json.encode("utf-8"))
        run_openssl_bytes(
            [
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(priv_path),
                "-in",
                str(payload_path),
                "-out",
                str(sig_path),
            ]
        )
        signature_hex = sig_path.read_bytes().hex()

    return {
        "scheme": "ed25519-sha256",
        "key_id": key_id,
        "canonical_profile": MANIFEST_AUTH_CANONICAL_PROFILE_BY_SCHEME["ed25519-sha256"],
        "payload_sha256": payload_sha256,
        "signature": signature_hex,
    }


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def is_severe_verify_gate_preflight_blocker(reason: Optional[str]) -> bool:
    return bool(_policy_is_severe_verify_gate_preflight_blocker(reason))


def compute_verify_gate_preflight_posture(now_payload: Dict[str, Any]) -> Dict[str, Any]:
    verify = now_payload.get("verify") if isinstance(now_payload.get("verify"), dict) else {}
    gate_preflight = verify.get("gate_preflight") if isinstance(verify.get("gate_preflight"), dict) else {}
    if not gate_preflight:
        return {
            "mode": "unknown",
            "source": "unavailable",
            "ready_to_run": None,
            "predicted_blocker_reason": None,
            "severity": "warn",
        }

    strict_mode = gate_preflight.get("strict_autonomy") if isinstance(gate_preflight.get("strict_autonomy"), dict) else {}
    predicted = gate_preflight.get("predicted_gate") if isinstance(gate_preflight.get("predicted_gate"), dict) else {}

    available = bool(gate_preflight.get("available") is True)
    enabled = bool(strict_mode.get("enabled") is True)
    required = strict_mode.get("required") if isinstance(strict_mode.get("required"), bool) else None
    source = str(strict_mode.get("source") or "disabled").strip() or "disabled"
    ready_to_run = predicted.get("ready_to_run") if isinstance(predicted.get("ready_to_run"), bool) else None
    predicted_blocker = str(predicted.get("predicted_blocker_reason") or "").strip() or None

    if not available:
        mode = "unknown"
    elif enabled and required is True:
        mode = "required"
    elif enabled:
        mode = "enabled"
    else:
        mode = "disabled"

    severity = "info"
    if not available:
        severity = "warn"
    elif predicted_blocker:
        severity = "blocker" if is_severe_verify_gate_preflight_blocker(predicted_blocker) else "warn"
    elif ready_to_run is False:
        severity = "warn"

    return {
        "mode": mode,
        "source": source,
        "ready_to_run": ready_to_run,
        "predicted_blocker_reason": predicted_blocker,
        "severity": severity,
    }


def live_publish_generation() -> Optional[str]:
    anchor = load_json(live_publish_anchor_path)
    generation = str(anchor.get("build_generation_id") or "").strip()
    if generation:
        return generation
    manifest = load_json(live_publish_manifest_path)
    generation = str(manifest.get("build_generation_id") or "").strip()
    return generation or None


def read_coherence_guard() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "coherence_tuple_hash": None,
        "coherence_build_generation_id": None,
        "policy_signature": None,
        "policy_epoch": None,
    }

    cur = load_json(continuity_current_path)
    coherence = cur.get("coherence") if isinstance(cur.get("coherence"), dict) else {}
    policy = coherence.get("policy") if isinstance(coherence.get("policy"), dict) else {}

    tuple_hash = str(coherence.get("tuple_hash") or "").strip()
    coherence_generation = str(coherence.get("build_generation_id") or "").strip()
    policy_signature = str(policy.get("signature") or "").strip()
    policy_epoch = str(policy.get("policy_epoch") or "").strip()

    if tuple_hash:
        out["coherence_tuple_hash"] = tuple_hash
    if coherence_generation:
        out["coherence_build_generation_id"] = coherence_generation
    if policy_signature:
        out["policy_signature"] = policy_signature
    if policy_epoch:
        out["policy_epoch"] = policy_epoch

    if any(v for v in out.values()):
        return out

    bundle = load_json(coherence_bundle_path)
    coherence_stamp = bundle.get("coherence_stamp") if isinstance(bundle.get("coherence_stamp"), dict) else {}
    policy_fallback = coherence_stamp.get("policy") if isinstance(coherence_stamp.get("policy"), dict) else {}

    tuple_hash = str(coherence_stamp.get("tuple_hash") or "").strip()
    coherence_generation = str(
        bundle.get("build_generation_id")
        or (((bundle.get("continuity_now") or {}).get("coherence") or {}).get("build_generation_id"))
        or ""
    ).strip()
    policy_signature = str(policy_fallback.get("signature") or "").strip()
    policy_epoch = str(policy_fallback.get("policy_epoch") or "").strip()

    if tuple_hash:
        out["coherence_tuple_hash"] = tuple_hash
    if coherence_generation:
        out["coherence_build_generation_id"] = coherence_generation
    if policy_signature:
        out["policy_signature"] = policy_signature
    if policy_epoch:
        out["policy_epoch"] = policy_epoch

    return out


def compare_publish_guard(base: Dict[str, Optional[str]], live: Dict[str, Optional[str]]) -> List[str]:
    mismatch: List[str] = []
    for key in ["coherence_tuple_hash", "coherence_build_generation_id", "policy_signature", "policy_epoch"]:
        base_val = str(base.get(key) or "").strip()
        live_val = str(live.get(key) or "").strip()
        if base_val != live_val:
            mismatch.append(key)
    return mismatch


NON_TERMINAL_TX_STATES = {"prepared", "promoting", "verifying"}
TERMINAL_TX_STATES = {"committed", "aborted", "failed"}
TEST_CRASH_STEP = str(os.environ.get("OPENCLAW_GTC_TEST_CRASH_STEP") or "").strip()


def append_jsonl(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    fsync_dir(path.parent)


def acquire_publish_lock() -> Tuple[Optional[Any], Optional[str]]:
    try:
        wait_sec = max(0.0, float(os.environ.get("OPENCLAW_GTC_PUBLISH_LOCK_WAIT_SEC", "30")))
    except Exception:
        wait_sec = 30.0
    fd = publish_lock_path.open("a+")
    deadline = time.monotonic() + wait_sec
    while True:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd, None
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fd.close()
                return None, f"publish_lock_busy:{publish_lock_path}:wait_sec={wait_sec:g}"
            time.sleep(0.1)


def release_publish_lock(fd: Optional[Any]) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        fd.close()
    except Exception:
        pass


def remove_tree(path: pathlib.Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


def resolve_stage_paths(base_stage_root: pathlib.Path) -> Dict[str, pathlib.Path]:
    stage_root_abs = base_stage_root.resolve()
    return {
        "staging_root": stage_root_abs,
        "staging_latest": stage_root_abs / "latest",
        "staging_surfaces": stage_root_abs / "surfaces",
        "backup_latest": stage_root_abs / "_backup_live_latest",
        "backup_surfaces": stage_root_abs / "_backup_live_surfaces",
    }


def stage_paths_from_tx(tx: Dict[str, Any]) -> Dict[str, pathlib.Path]:
    stage_abs = str(tx.get("staging_root_abs") or "").strip()
    stage_rel = str(tx.get("staging_root") or "").strip()
    if stage_abs:
        return resolve_stage_paths(pathlib.Path(stage_abs))
    if stage_rel:
        p = pathlib.Path(stage_rel)
        if not p.is_absolute():
            p = (root / p).resolve()
        return resolve_stage_paths(p)
    return resolve_stage_paths(staging_root)


def cleanup_staging_root(path: Optional[pathlib.Path] = None) -> None:
    remove_tree((path or staging_root).resolve())


def write_publish_latest(tx: Dict[str, Any]) -> None:
    atomic_write(publish_journal_latest_path, json.dumps(tx, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def append_publish_event(tx: Dict[str, Any], *, event: str, details: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "schema_version": "gtc.publish.transaction.event.v1",
        "recorded_at": now_iso(),
        "tx_id": tx.get("tx_id"),
        "state": tx.get("state"),
        "step": tx.get("step"),
        "event": event,
    }
    if details:
        payload["details"] = details
    append_jsonl(publish_journal_events_path, payload)


def transition_publish_tx(
    tx: Dict[str, Any],
    *,
    event: str,
    state: Optional[str] = None,
    step: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    terminal: bool = False,
) -> Dict[str, Any]:
    out = dict(tx)
    ts = now_iso()
    if state is not None:
        out["state"] = state
    if step is not None:
        out["step"] = step
    out["updated_at"] = ts
    if terminal:
        out["completed_at"] = ts
    write_publish_latest(out)
    append_publish_event(out, event=event, details=details)
    return out


def create_publish_tx(
    *,
    build_generation_id: str,
    base_generation_id: Optional[str],
    base_coherence_guard: Dict[str, Optional[str]],
    include_surfaces: bool,
    skip_schema_gate_tx: bool,
) -> Dict[str, Any]:
    tx_id = f"pubtx_{now_dt().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    stage_paths = resolve_stage_paths(staging_root)
    tx: Dict[str, Any] = {
        "schema_version": "gtc.publish.transaction.v1",
        "tx_id": tx_id,
        "build_generation_id": build_generation_id,
        "base_generation_id": base_generation_id,
        "base_coherence_guard": base_coherence_guard,
        "state": "prepared",
        "step": "staging_ready",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "staging_root": rel(stage_paths["staging_root"]),
        "staging_root_abs": str(stage_paths["staging_root"]),
        "include_surfaces": bool(include_surfaces),
        "schema_gate_skipped": bool(skip_schema_gate_tx),
        "lock_path": rel(publish_lock_path),
    }
    write_publish_latest(tx)
    append_publish_event(tx, event="prepared")
    return tx


def maybe_test_crash(step: str) -> None:
    if TEST_CRASH_STEP and TEST_CRASH_STEP == step:
        os._exit(91)


def recover_inflight_publish_transaction() -> Dict[str, Any]:
    latest = load_json(publish_journal_latest_path)
    tx_id = str(latest.get("tx_id") or "").strip()
    state = str(latest.get("state") or "").strip()
    result: Dict[str, Any] = {
        "attempted": False,
        "status": "none",
        "tx_id": tx_id or None,
        "state": state or None,
    }
    if not tx_id or not state:
        return result
    if state in TERMINAL_TX_STATES:
        result["status"] = "terminal"
        return result
    if state not in NON_TERMINAL_TX_STATES:
        result["status"] = "unknown_state"
        return result

    result["attempted"] = True
    result["status"] = "pending"

    lock_fd, lock_err = acquire_publish_lock()
    if lock_err:
        result["status"] = "lock_busy"
        result["error"] = lock_err
        return result

    tx = dict(latest)
    stage_paths = stage_paths_from_tx(tx)
    include_surfaces = bool(tx.get("include_surfaces"))
    try:
        live_generation_id = live_publish_generation()
        tx_generation_id = str(tx.get("build_generation_id") or "").strip()
        backup_latest_exists = stage_paths["backup_latest"].exists()
        backup_surfaces_exists = stage_paths["backup_surfaces"].exists()
        staging_latest_exists = stage_paths["staging_latest"].exists()
        staging_surfaces_exists = stage_paths["staging_surfaces"].exists()
        live_latest_matches_tx = bool(tx_generation_id and live_generation_id == tx_generation_id)
        live_promotion_fully_visible = bool(
            live_latest_matches_tx
            and not staging_latest_exists
            and (
                not include_surfaces
                or (not staging_surfaces_exists and live_surfaces_dir.exists())
            )
        )

        if live_latest_matches_tx and (
            (not backup_latest_exists and not backup_surfaces_exists)
            or live_promotion_fully_visible
        ):
            surfaces_promoted_on_recovery = False
            if include_surfaces and staging_surfaces_exists and not live_surfaces_dir.exists():
                os.replace(stage_paths["staging_surfaces"], live_surfaces_dir)
                fsync_dir(live_surfaces_dir.parent)
                surfaces_promoted_on_recovery = True
            cleanup_staging_root(stage_paths["staging_root"])
            tx = transition_publish_tx(
                tx,
                event="recovered_committed",
                state="committed",
                step="recovered_committed",
                details={
                    "live_generation_id": live_generation_id,
                    "backup_latest_discarded": bool(backup_latest_exists),
                    "backup_surfaces_discarded": bool(include_surfaces and backup_surfaces_exists),
                    "surfaces_promoted_on_recovery": surfaces_promoted_on_recovery,
                },
                terminal=True,
            )
            result["status"] = "recovered_committed"
            return result

        if stage_paths["backup_latest"].exists():
            if live_latest_dir.exists():
                remove_tree(live_latest_dir)
            os.replace(stage_paths["backup_latest"], live_latest_dir)
            fsync_dir(live_latest_dir.parent)

        if include_surfaces and stage_paths["backup_surfaces"].exists():
            if live_surfaces_dir.exists():
                remove_tree(live_surfaces_dir)
            os.replace(stage_paths["backup_surfaces"], live_surfaces_dir)
            fsync_dir(live_surfaces_dir.parent)

        cleanup_staging_root(stage_paths["staging_root"])
        tx = transition_publish_tx(
            tx,
            event="recovered_rolled_back",
            state="aborted",
            step="recovered_rollback",
            details={
                "live_generation_id": live_generation_id,
                "backup_latest_restored": bool(backup_latest_exists),
                "backup_surfaces_restored": bool(include_surfaces and backup_surfaces_exists),
            },
            terminal=True,
        )
        result["status"] = "recovered_rolled_back"
        return result
    except Exception as exc:
        tx = transition_publish_tx(
            tx,
            event="recovery_failed",
            state="failed",
            step="recovery_failed",
            details={"error": str(exc)},
            terminal=True,
        )
        result["status"] = "recovery_failed"
        result["error"] = str(exc)
        return result
    finally:
        release_publish_lock(lock_fd)


def promote_staged_outputs(tx: Dict[str, Any], include_surfaces: bool) -> Tuple[Dict[str, Any], Optional[str]]:
    stage_paths = stage_paths_from_tx(tx)
    tx_next = tx
    try:
        tx_next = transition_publish_tx(tx_next, event="promotion_started", state="promoting", step="promotion_started")

        if live_latest_dir.exists():
            remove_tree(stage_paths["backup_latest"])
            os.replace(live_latest_dir, stage_paths["backup_latest"])
            fsync_dir(stage_paths["backup_latest"].parent)
            tx_next = transition_publish_tx(tx_next, event="live_latest_backed_up", step="live_latest_backed_up")

        if not stage_paths["staging_latest"].exists():
            raise RuntimeError(f"staging_latest_missing:{stage_paths['staging_latest']}")

        os.replace(stage_paths["staging_latest"], live_latest_dir)
        fsync_dir(live_latest_dir.parent)
        tx_next = transition_publish_tx(tx_next, event="latest_promoted", step="latest_promoted")
        maybe_test_crash("latest_promoted")

        if include_surfaces:
            if live_surfaces_dir.exists():
                remove_tree(stage_paths["backup_surfaces"])
                os.replace(live_surfaces_dir, stage_paths["backup_surfaces"])
                fsync_dir(stage_paths["backup_surfaces"].parent)
                tx_next = transition_publish_tx(tx_next, event="live_surfaces_backed_up", step="live_surfaces_backed_up")

            if stage_paths["staging_surfaces"].exists():
                os.replace(stage_paths["staging_surfaces"], live_surfaces_dir)
                fsync_dir(live_surfaces_dir.parent)
            tx_next = transition_publish_tx(tx_next, event="surfaces_promoted", step="surfaces_promoted")
            maybe_test_crash("surfaces_promoted")

        remove_tree(stage_paths["backup_latest"])
        remove_tree(stage_paths["backup_surfaces"])
        cleanup_staging_root(stage_paths["staging_root"])
        tx_next = transition_publish_tx(tx_next, event="promotion_cleanup_complete", step="promotion_cleanup_complete")
        return tx_next, None
    except Exception as exc:
        tx_next = transition_publish_tx(
            tx_next,
            event="promotion_error",
            state="promoting",
            step="promotion_error",
            details={"error": str(exc)},
            terminal=False,
        )
        return tx_next, str(exc)


def copy_schema(rel_src: str, dst_name: str) -> None:
    src = root / rel_src
    if not src.exists():
        return
    dst = schema_dir / dst_name
    try:
        txt = src.read_text(encoding="utf-8")
        atomic_write(dst, txt)
    except Exception:
        return


copy_schema("ops/openclaw/architecture/schemas/gtc_evidence.schema.json", "gtc_evidence.schema.json")
copy_schema("ops/openclaw/architecture/schemas/gtc_latest.schema.json", "gtc_latest.schema.json")
copy_schema("ops/openclaw/architecture/schemas/gtc_event.schema.json", "gtc_event.schema.json")

issuer_boot_id = read_boot_id()

con = sqlite3.connect(db_path)
cur = con.cursor()
cur.executescript(
    """
CREATE TABLE IF NOT EXISTS gtc_connector (
  connector_id TEXT PRIMARY KEY,
  connector_type TEXT NOT NULL,
  display_name TEXT NOT NULL,
  freshness_ttl_ms INTEGER NOT NULL DEFAULT 60000,
  stale_severity TEXT NOT NULL DEFAULT 'warning',
  config_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_monotonic_seq INTEGER NOT NULL DEFAULT 0,
  UNIQUE(connector_type, connector_id)
);
CREATE INDEX IF NOT EXISTS idx_gtc_connector_type ON gtc_connector(connector_type);

CREATE TABLE IF NOT EXISTS gtc_evidence_index (
  evidence_id TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES gtc_connector(connector_id),
  connector_type TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  monotonic_seq INTEGER NOT NULL,
  subject_kind TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  severity_max TEXT,
  jsonl_path TEXT NOT NULL,
  jsonl_line_no INTEGER NOT NULL,
  payload_sha256 TEXT NOT NULL,
  facts_json TEXT NOT NULL,
  refs_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gtc_evidence_subject ON gtc_evidence_index(subject_kind, subject_id, monotonic_seq);
CREATE INDEX IF NOT EXISTS idx_gtc_evidence_time ON gtc_evidence_index(observed_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gtc_evidence_connector_seq ON gtc_evidence_index(connector_id, monotonic_seq);

CREATE TABLE IF NOT EXISTS gtc_artifact (
  sha256 TEXT PRIMARY KEY,
  media_type TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gtc_evidence_artifact (
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  sha256 TEXT NOT NULL REFERENCES gtc_artifact(sha256),
  role TEXT NOT NULL,
  PRIMARY KEY (evidence_id, sha256, role)
);

CREATE TABLE IF NOT EXISTS gtc_task_evidence (
  task_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  PRIMARY KEY (task_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS gtc_checkpoint_evidence (
  checkpoint_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  PRIMARY KEY (checkpoint_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS gtc_latest_pointer (
  pointer_key TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES gtc_connector(connector_id),
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  observed_at TEXT NOT NULL,
  freshness_ttl_ms INTEGER NOT NULL,
  updated_at TEXT NOT NULL
);
"""
)

connector_specs = [
    {
        "connector_type": "runtime.gateway",
        "connector_id": "gateway-main",
        "display_name": "Gateway Runtime",
        "freshness_ttl_ms": 300000,
        "stale_severity": "critical",
        "config": {"source": "state/ground_truth/latest.json", "sync": "snapshot_ground_truth.sh"},
    },
    {
        "connector_type": "queue.task",
        "connector_id": "default-queue",
        "display_name": "Queue Task Transitions",
        "freshness_ttl_ms": 600000,
        "stale_severity": "warning",
        "config": {"source_table": "task_transitions", "max_rows": max_rows},
    },
    {
        "connector_type": "validation.gates",
        "connector_id": "core",
        "display_name": "Verify-Then-Resume Gate",
        "freshness_ttl_ms": 1800000,
        "stale_severity": "warning",
        "config": {"source": "state/continuity/latest/verify_last.json"},
    },
    {
        "connector_type": "operator.actions",
        "connector_id": "local",
        "display_name": "Operator / Router Events",
        "freshness_ttl_ms": 1800000,
        "stale_severity": "warning",
        "config": {"source_table": "continuity_events", "predicate": "emitted=1"},
    },
]

created_at = now_iso()
for spec in connector_specs:
    cur.execute(
        """
INSERT INTO gtc_connector (
  connector_id, connector_type, display_name, freshness_ttl_ms, stale_severity,
  config_json, created_at, updated_at, last_monotonic_seq
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
ON CONFLICT(connector_id) DO UPDATE SET
  connector_type=excluded.connector_type,
  display_name=excluded.display_name,
  freshness_ttl_ms=excluded.freshness_ttl_ms,
  stale_severity=excluded.stale_severity,
  config_json=excluded.config_json,
  updated_at=excluded.updated_at
""",
        (
            spec["connector_id"],
            spec["connector_type"],
            spec["display_name"],
            int(spec["freshness_ttl_ms"]),
            str(spec["stale_severity"]),
            canonical_json(spec.get("config") or {}),
            created_at,
            created_at,
        ),
    )

con.commit()

cursors = load_json(live_cursors_path)
if cursors.get("schema_version") != "gtc.cursors.v1":
    cursors = {"schema_version": "gtc.cursors.v1", "updated_at": "", "sources": {}}

source_cursors = cursors.setdefault("sources", {})

line_counts: Dict[str, int] = {}


def get_line_count(path: pathlib.Path) -> int:
    key = str(path)
    if key in line_counts:
        return line_counts[key]
    if not path.exists():
        line_counts[key] = 0
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            count = sum(1 for _ in fh)
    except Exception:
        count = 0
    line_counts[key] = count
    return count


def append_jsonl(path: pathlib.Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = get_line_count(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(payload) + "\n")
    current += 1
    line_counts[str(path)] = current
    return current


def evidence_exists(evidence_id: str) -> bool:
    row = cur.execute("SELECT 1 FROM gtc_evidence_index WHERE evidence_id = ?", (evidence_id,)).fetchone()
    return row is not None


def next_monotonic_seq(connector_id: str) -> int:
    row = cur.execute("SELECT last_monotonic_seq FROM gtc_connector WHERE connector_id = ?", (connector_id,)).fetchone()
    last = int(row[0] or 0) if row else 0
    return last + 1


def update_monotonic_seq(connector_id: str, seq: int) -> None:
    cur.execute(
        "UPDATE gtc_connector SET last_monotonic_seq = ?, updated_at = ? WHERE connector_id = ?",
        (int(seq), now_iso(), connector_id),
    )


def add_artifact_links(evidence_id: str, artifacts: List[Dict[str, Any]]) -> None:
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        sha = str(item.get("sha256") or "").strip().lower()
        if len(sha) < 16:
            continue
        media_type = str(item.get("media_type") or "application/octet-stream")
        role = str(item.get("role") or "artifact")
        path_raw = str(item.get("path") or "")
        path_abs = pathlib.Path(path_raw) if path_raw else None
        path_rel = path_raw
        size = 0
        if path_abs is not None and path_abs.exists() and path_abs.is_file():
            try:
                size = int(path_abs.stat().st_size)
                path_rel = rel(path_abs)
            except Exception:
                size = 0
        cur.execute(
            """
INSERT OR IGNORE INTO gtc_artifact (sha256, media_type, bytes, path, created_at)
VALUES (?, ?, ?, ?, ?)
""",
            (sha, media_type, size, path_rel or f"artifact://{sha}", now_iso()),
        )
        cur.execute(
            """
INSERT OR IGNORE INTO gtc_evidence_artifact (evidence_id, sha256, role)
VALUES (?, ?, ?)
""",
            (evidence_id, sha, role),
        )


def parse_metadata_json(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_sha256_hex(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw):
        return raw
    return None


def media_type_for_path(path_obj: pathlib.Path) -> str:
    suffix = path_obj.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".md", ".txt", ".log", ".csv", ".tsv", ".yaml", ".yml"}:
        return "text/plain"
    if suffix == ".html":
        return "text/html"
    if suffix in {".sqlite", ".db"}:
        return "application/x-sqlite3"
    if suffix == ".zip":
        return "application/zip"
    return "application/octet-stream"


def build_queue_task_artifact_envelopes(task_id: str, evidence_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    artifacts: List[Dict[str, Any]] = []
    task_artifact_manifest: List[Dict[str, Any]] = []
    dedupe: set[Tuple[str, str, str]] = set()

    def add_artifact(
        *,
        path_value: str,
        role: str,
        artifact_type: Optional[str] = None,
        sha_hint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_path = str(path_value or "").strip()
        if not raw_path:
            return
        path_obj = pathlib.Path(raw_path)
        if not path_obj.is_absolute():
            path_obj = (root / raw_path).resolve()

        sha = normalize_sha256_hex(sha_hint)
        if sha is None and path_obj.exists() and path_obj.is_file():
            try:
                sha = hashlib.sha256(path_obj.read_bytes()).hexdigest()
            except Exception:
                sha = None

        path_ref = str(path_obj)
        dedupe_key = (str(role or "artifact"), path_ref, str(sha or ""))
        if dedupe_key in dedupe:
            return
        dedupe.add(dedupe_key)

        media_type = media_type_for_path(path_obj)
        row: Dict[str, Any] = {
            "media_type": media_type,
            "role": str(role or "artifact"),
            "path": path_ref,
        }
        if sha:
            row["sha256"] = sha
        if metadata:
            row["metadata"] = metadata
        artifacts.append(row)

        if artifact_type:
            manifest_row: Dict[str, Any] = {
                "artifact_type": str(artifact_type),
                "artifact_path": path_ref,
                "role": str(role or "artifact"),
                "media_type": media_type,
            }
            if sha:
                manifest_row["sha256"] = sha
            if metadata:
                manifest_row["metadata"] = metadata
            task_artifact_manifest.append(manifest_row)

    for path_raw in evidence_paths:
        add_artifact(path_value=path_raw, role="transition_evidence")

    task_rows = cur.execute(
        """
SELECT artifact_type, artifact_path, sha256, metadata_json
FROM task_artifacts
WHERE task_id = ?
ORDER BY artifact_type ASC, artifact_path ASC
""",
        (task_id,),
    ).fetchall()

    for row in task_rows:
        artifact_type = str(row[0] or "artifact").strip() or "artifact"
        artifact_path = str(row[1] or "").strip()
        if not artifact_path:
            continue
        metadata = parse_metadata_json(row[3])
        add_artifact(
            path_value=artifact_path,
            role=f"task_artifact:{artifact_type}",
            artifact_type=artifact_type,
            sha_hint=str(row[2] or ""),
            metadata=metadata if metadata else None,
        )

    artifacts.sort(key=lambda x: (str(x.get("role") or ""), str(x.get("path") or "")))
    task_artifact_manifest.sort(
        key=lambda x: (str(x.get("artifact_type") or ""), str(x.get("artifact_path") or ""))
    )
    return artifacts, task_artifact_manifest


def parse_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def compact_gate_summary(raw: Any) -> Dict[str, Any]:
    summary = parse_json_object(raw)
    if not summary:
        return {}

    keep_keys = [
        "schema_version",
        "summary_signature",
        "ingress_classification",
        "queue_reason",
        "decision_path",
        "decision_sha256",
        "completion_packet_path",
        "completion_packet_sha256",
        "completion_packet_source",
        "retry_profile",
        "provider_failure",
        "gate_outcome",
        "required_deliverable_path",
    ]
    out: Dict[str, Any] = {}
    for key in keep_keys:
        value = summary.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool, dict, list)):
            out[key] = value
    return out


DELEGATED_GATE_SUMMARY_SCHEMA_VERSION = "autopilot.delegated_gate_summary.v1"


def normalize_artifact_path_for_compare(path_value: Any) -> Optional[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    path_obj = pathlib.Path(raw)
    if not path_obj.is_absolute():
        path_obj = root / path_obj
    try:
        return str(path_obj.resolve())
    except Exception:
        return str(path_obj)


def project_delegated_gate_summary_binding_status(
    gate_summary: Dict[str, Any],
    task_artifact_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    projection: Dict[str, Any] = {
        "schema_version": "gtc.queue_handoff_gate_binding_projection.v1",
        "status": "not_applicable",
        "checked_count": 0,
        "matched_count": 0,
        "missing_count": 0,
        "mismatch_count": 0,
        "issues": [],
    }

    if str(gate_summary.get("schema_version") or "") != DELEGATED_GATE_SUMMARY_SCHEMA_VERSION:
        projection["reason"] = "non_delegated_gate_summary"
        return projection

    required_binding_specs = [
        ("completion_packet_path", "completion_packet_sha256", "completion_packet"),
        ("decision_path", "decision_sha256", "gate_decision"),
    ]

    expected_bindings: List[Dict[str, str]] = []
    missing_binding_fields: List[Dict[str, Any]] = []
    for path_field, sha_field, binding_kind in required_binding_specs:
        binding_path = normalize_artifact_path_for_compare(gate_summary.get(path_field))
        binding_sha = normalize_sha256_hex(gate_summary.get(sha_field))
        has_path = bool(binding_path)
        has_sha = bool(binding_sha)

        if has_path and has_sha:
            expected_bindings.append({"binding": binding_kind, "path": binding_path, "sha256": binding_sha})
            continue

        issue_code = "missing_summary_binding_path_and_sha"
        if has_path and not has_sha:
            issue_code = "missing_summary_binding_sha"
        elif has_sha and not has_path:
            issue_code = "missing_summary_binding_path"

        missing_binding_fields.append(
            {
                "binding": binding_kind,
                "code": issue_code,
                "path_field": path_field,
                "sha_field": sha_field,
                "expected_path": binding_path,
                "expected_sha256": binding_sha,
            }
        )

    projection["checked_count"] = len(expected_bindings)
    if missing_binding_fields:
        projection["status"] = "degraded"
        projection["missing_count"] = int(projection.get("missing_count") or 0) + len(missing_binding_fields)
        projection["issues"].extend(missing_binding_fields)

    if not expected_bindings:
        projection["reason"] = "delegated_gate_bindings_unverifiable"
        return projection

    if str(projection.get("status") or "") != "degraded":
        projection["status"] = "verified"

    manifest_rows: List[Dict[str, str]] = []
    for row in task_artifact_manifest:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if str(metadata.get("source") or "") != "delegated_gate_summary":
            continue
        manifest_rows.append(
            {
                "binding": str(metadata.get("binding") or "").strip(),
                "path": str(normalize_artifact_path_for_compare(row.get("artifact_path")) or ""),
                "sha256": str(normalize_sha256_hex(row.get("sha256")) or ""),
            }
        )

    for expected in expected_bindings:
        expected_binding = str(expected.get("binding") or "")
        expected_path = str(expected.get("path") or "")
        expected_sha = str(expected.get("sha256") or "")

        candidates = [row for row in manifest_rows if str(row.get("binding") or "") == expected_binding]
        if not candidates:
            projection["status"] = "degraded"
            projection["missing_count"] = int(projection.get("missing_count") or 0) + 1
            projection["issues"].append(
                {
                    "binding": expected_binding,
                    "code": "missing_task_artifact_binding",
                    "expected_path": expected_path,
                    "expected_sha256": expected_sha,
                }
            )
            continue

        matched = any(
            str(row.get("path") or "") == expected_path and str(row.get("sha256") or "") == expected_sha
            for row in candidates
        )
        if matched:
            projection["matched_count"] = int(projection.get("matched_count") or 0) + 1
            continue

        projection["status"] = "degraded"
        projection["mismatch_count"] = int(projection.get("mismatch_count") or 0) + 1

        sample = candidates[0] if candidates else {}
        observed_path = str(sample.get("path") or "")
        observed_sha = str(sample.get("sha256") or "")
        mismatch_code = "binding_path_sha_mismatch"
        if observed_path == expected_path:
            mismatch_code = "binding_sha_mismatch"
        elif observed_sha == expected_sha:
            mismatch_code = "binding_path_mismatch"

        projection["issues"].append(
            {
                "binding": expected_binding,
                "code": mismatch_code,
                "expected_path": expected_path,
                "expected_sha256": expected_sha,
                "observed_path": observed_path or None,
                "observed_sha256": observed_sha or None,
            }
        )

    projection["issues"] = [row for row in projection.get("issues") or [] if isinstance(row, dict)][:4]
    return projection


def load_handoff_projection_by_transition(transition_event_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    ids = sorted({str(x or "").strip() for x in transition_event_ids if str(x or "").strip()})
    if not ids:
        return {}

    placeholders = ",".join("?" for _ in ids)
    rows = cur.execute(
        f"""
SELECT transition_event_id, packet_id, from_role, to_role, next_gate, retry_count, failure_signature, gate_metadata_json, created_at
FROM task_handoff_packets
WHERE transition_event_id IN ({placeholders})
ORDER BY created_at DESC, packet_id DESC
""",
        tuple(ids),
    ).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        transition_event_id = str(row[0] or "").strip()
        if not transition_event_id or transition_event_id in out:
            continue

        gate_meta = parse_json_object(row[7])
        gate_summary = compact_gate_summary(gate_meta.get("gate_summary"))

        proj: Dict[str, Any] = {
            "packet_id": str(row[1] or "").strip() or None,
            "transition_event_id": transition_event_id,
            "from_role": str(row[2] or "").strip() or None,
            "to_role": str(row[3] or "").strip() or None,
            "next_gate": str(row[4] or "").strip() or None,
            "retry_count": int(row[5] or 0),
            "failure_signature": str(row[6] or "").strip() or None,
            "created_at": str(row[8] or "").strip() or None,
        }
        if gate_summary:
            proj["gate_summary"] = gate_summary
        out[transition_event_id] = proj
    return out


def persist_evidence(
    connector_type: str,
    connector_id: str,
    subject_kind: str,
    subject_id: str,
    observed_at: str,
    facts: Dict[str, Any],
    refs: Dict[str, Any],
    severity_max: Optional[str],
    source_identity: str,
) -> Tuple[bool, str]:
    observed_at = observed_at or now_iso()
    core_seed = {
        "connector_type": connector_type,
        "connector_id": connector_id,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "observed_at": observed_at,
        "facts": facts,
        "refs": refs,
        "source_identity": source_identity,
    }
    evidence_id = "ev_" + sha256_text(canonical_json(core_seed))[:24]
    if evidence_exists(evidence_id):
        return False, evidence_id

    seq = next_monotonic_seq(connector_id)
    payload_base = {
        "schema_version": "gtc.evidence.v2",
        "evidence_id": evidence_id,
        "connector_type": connector_type,
        "connector_id": connector_id,
        "observed_at": observed_at,
        "monotonic_seq": seq,
        "subject": {
            "kind": subject_kind,
            "id": subject_id,
            "workspace": "clawd-architect",
        },
        "facts": facts,
        "refs": refs,
    }
    payload_sha = sha256_text(canonical_json(payload_base))
    payload = dict(payload_base)
    payload["integrity"] = {
        "payload_sha256": payload_sha,
        "producer": {"name": "gtc_v2_sync", "version": "1.0.0"},
    }

    date_key = parse_iso(observed_at) or now_dt()
    jsonl_path = evidence_root / connector_type / connector_id / f"{date_key.strftime('%Y-%m-%d')}.jsonl"
    line_no = append_jsonl(jsonl_path, payload)

    cur.execute(
        """
INSERT INTO gtc_evidence_index (
  evidence_id, connector_id, connector_type, observed_at, monotonic_seq,
  subject_kind, subject_id, severity_max, jsonl_path, jsonl_line_no,
  payload_sha256, facts_json, refs_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            evidence_id,
            connector_id,
            connector_type,
            observed_at,
            seq,
            subject_kind,
            subject_id,
            severity_max,
            rel(jsonl_path),
            int(line_no),
            payload_sha,
            canonical_json(facts),
            canonical_json(refs),
            now_iso(),
        ),
    )

    pointer_key = f"{connector_type}::{connector_id}"
    ttl_row = cur.execute(
        "SELECT freshness_ttl_ms FROM gtc_connector WHERE connector_id = ?",
        (connector_id,),
    ).fetchone()
    ttl_ms = int(ttl_row[0] or 60000) if ttl_row else 60000
    cur.execute(
        """
INSERT INTO gtc_latest_pointer (pointer_key, connector_id, evidence_id, observed_at, freshness_ttl_ms, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(pointer_key) DO UPDATE SET
  evidence_id=excluded.evidence_id,
  observed_at=excluded.observed_at,
  freshness_ttl_ms=excluded.freshness_ttl_ms,
  updated_at=excluded.updated_at
""",
        (pointer_key, connector_id, evidence_id, observed_at, ttl_ms, now_iso()),
    )

    checkpoint_id = str(refs.get("checkpoint_id") or "").strip()
    if checkpoint_id:
        cur.execute(
            "INSERT OR IGNORE INTO gtc_checkpoint_evidence (checkpoint_id, evidence_id) VALUES (?, ?)",
            (checkpoint_id, evidence_id),
        )

    task_id = str(refs.get("task_id") or "").strip()
    if task_id:
        cur.execute(
            "INSERT OR IGNORE INTO gtc_task_evidence (task_id, evidence_id) VALUES (?, ?)",
            (task_id, evidence_id),
        )

    artifacts = refs.get("artifacts") if isinstance(refs.get("artifacts"), list) else []
    add_artifact_links(evidence_id, artifacts)
    update_monotonic_seq(connector_id, seq)
    return True, evidence_id


inserted = 0
skipped = 0
connector_insert_counts: Dict[str, int] = {}
stream_updates: Dict[str, Any] = {}


def bump(connector_type: str, ok: bool) -> None:
    global inserted, skipped
    if ok:
        inserted += 1
        connector_insert_counts[connector_type] = int(connector_insert_counts.get(connector_type) or 0) + 1
    else:
        skipped += 1


# runtime.gateway from ground_truth snapshot pointer
latest_pointer = load_json(root / "state/ground_truth/latest.json")
gt_snapshot = {}
gt_snapshot_path = pathlib.Path()
if latest_pointer:
    p = root / str(latest_pointer.get("snapshot_path") or "")
    if p.exists():
        gt_snapshot = load_json(p)
        gt_snapshot_path = p

if latest_pointer and gt_snapshot:
    anomalies = gt_snapshot.get("anomalies") if isinstance(gt_snapshot.get("anomalies"), list) else []
    critical_keys = [
        str(a.get("key"))
        for a in anomalies
        if isinstance(a, dict) and str(a.get("severity") or "") == "critical"
    ]
    warn_keys = [
        str(a.get("key"))
        for a in anomalies
        if isinstance(a, dict) and str(a.get("severity") or "") == "warn"
    ]
    gateway_status = (((gt_snapshot.get("gateway") or {}).get("status") or {}).get("service") or {}).get("runtime") or {}
    rpc_ok = bool(((((gt_snapshot.get("gateway") or {}).get("status") or {}).get("rpc") or {}).get("ok")))

    facts = {
        "runtime_status": str(gateway_status.get("status") or "unknown"),
        "rpc_ok": rpc_ok,
        "critical_anomaly_count": len(critical_keys),
        "warn_anomaly_count": len(warn_keys),
        "critical_anomaly_keys": critical_keys,
        "warn_anomaly_keys": warn_keys,
    }
    refs = {
        "ground_truth_snapshot_id": latest_pointer.get("snapshot_id"),
        "ground_truth_snapshot_path": rel(gt_snapshot_path) if gt_snapshot_path else None,
        "ground_truth_snapshot_sha256": latest_pointer.get("snapshot_sha256"),
        "checkpoint_id": str((load_json(root / "state/continuity/latest/latest_pointer.json").get("checkpoint_id") or "")),
    }
    sev = "critical" if critical_keys else ("warning" if warn_keys else "info")
    ok, _eid = persist_evidence(
        "runtime.gateway",
        "gateway-main",
        "gateway",
        "main",
        str(gt_snapshot.get("snapshot_ts_utc") or latest_pointer.get("snapshot_ts_utc") or now_iso()),
        facts,
        refs,
        sev,
        f"ground_truth:{latest_pointer.get('snapshot_id')}",
    )
    bump("runtime.gateway", ok)

# validation.gates from verify_last.json
verify_last = load_json(root / "state/continuity/latest/verify_last.json")
continuity_latest = load_json(root / "state/continuity/latest/latest_pointer.json")
if verify_last:
    gate_status = str(verify_last.get("status") or "unknown")
    facts = {
        "status": gate_status,
        "reason": str(verify_last.get("reason") or ""),
        "ready": gate_status == "READY",
    }
    refs = {
        "verify_report_path": "state/continuity/latest/verify_last.json",
        "checkpoint_id": continuity_latest.get("checkpoint_id"),
    }
    sev = "info" if gate_status == "READY" else ("critical" if gate_status == "BLOCKER" else "warning")
    ok, _eid = persist_evidence(
        "validation.gates",
        "core",
        "gate",
        "verify_then_resume",
        str(verify_last.get("timestamp") or now_iso()),
        facts,
        refs,
        sev,
        f"verify_last:{verify_last.get('timestamp')}:{gate_status}",
    )
    bump("validation.gates", ok)

# queue.task stream from task_transitions
queue_cursor = int((source_cursors.get("task_transitions_rowid") or 0))
rows = cur.execute(
    """
SELECT rowid, event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
FROM task_transitions
WHERE rowid > ?
ORDER BY rowid ASC
LIMIT ?
""",
    (queue_cursor, max_rows),
).fetchall()

handoff_projection_by_transition = load_handoff_projection_by_transition(
    [str(r[1] or "").strip() for r in rows]
)

max_queue_rowid = queue_cursor
for row in rows:
    rowid = int(row[0])
    max_queue_rowid = max(max_queue_rowid, rowid)
    event_id = str(row[1] or "")
    task_id = str(row[2] or "")
    from_status = str(row[3] or "")
    to_status = str(row[4] or "")
    actor_role = str(row[5] or "")
    reason = str(row[6] or "")
    evidence_ref = str(row[7] or "")
    created_at_row = str(row[8] or now_iso())

    evidence_paths = [p.strip() for p in evidence_ref.split("|") if p.strip()]
    artifacts, task_artifact_manifest = build_queue_task_artifact_envelopes(task_id, evidence_paths)
    handoff_projection = handoff_projection_by_transition.get(event_id)
    handoff_gate_summary = (
        handoff_projection.get("gate_summary")
        if isinstance(handoff_projection, dict) and isinstance(handoff_projection.get("gate_summary"), dict)
        else {}
    )
    handoff_gate_binding_projection = project_delegated_gate_summary_binding_status(handoff_gate_summary, task_artifact_manifest)

    refs = {
        "task_id": task_id,
        "transition_event_id": event_id,
        "checkpoint_id": continuity_latest.get("checkpoint_id"),
        "evidence_paths": evidence_paths,
        "artifacts": artifacts,
        "task_artifact_manifest": task_artifact_manifest,
        "handoff_packet": handoff_projection if isinstance(handoff_projection, dict) and handoff_projection else None,
    }
    if str(handoff_gate_binding_projection.get("status") or "") != "not_applicable":
        refs["handoff_gate_binding_projection"] = handoff_gate_binding_projection

    facts = {
        "from_status": from_status,
        "to_status": to_status,
        "actor_role": actor_role,
        "reason": reason,
        "has_evidence_ref": bool(evidence_paths),
        "artifact_count": len(artifacts),
        "task_artifact_count": len(task_artifact_manifest),
        "has_task_artifact_manifest": bool(task_artifact_manifest),
        "has_handoff_packet": bool(handoff_projection),
        "handoff_queue_reason": str(handoff_gate_summary.get("queue_reason") or "") or None,
        "handoff_summary_signature": str(handoff_gate_summary.get("summary_signature") or "") or None,
        "handoff_ingress_classification": str(handoff_gate_summary.get("ingress_classification") or "") or None,
        "handoff_gate_binding_status": str(handoff_gate_binding_projection.get("status") or "not_applicable"),
        "handoff_gate_binding_checked_count": int(handoff_gate_binding_projection.get("checked_count") or 0),
        "handoff_gate_binding_matched_count": int(handoff_gate_binding_projection.get("matched_count") or 0),
        "handoff_gate_binding_missing_count": int(handoff_gate_binding_projection.get("missing_count") or 0),
        "handoff_gate_binding_mismatch_count": int(handoff_gate_binding_projection.get("mismatch_count") or 0),
    }
    severity = "info"
    if to_status in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
        severity = "warning"
    if str(handoff_gate_binding_projection.get("status") or "") == "degraded" and severity == "info":
        severity = "warning"
    ok, _eid = persist_evidence(
        "queue.task",
        "default-queue",
        "task",
        task_id or f"task_row_{rowid}",
        created_at_row,
        facts,
        refs,
        severity,
        f"task_transitions:rowid:{rowid}",
    )
    bump("queue.task", ok)

source_cursors["task_transitions_rowid"] = max_queue_rowid
stream_updates["task_transitions_rowid"] = {"from": queue_cursor, "to": max_queue_rowid, "rows": len(rows)}

# operator.actions stream from continuity_events (emitted rows)
event_cursor = int((source_cursors.get("continuity_events_rowid") or 0))
ev_rows = cur.execute(
    """
SELECT rowid, event_id, created_at, source, event_key, severity, changed, cooldown_elapsed, suppress_reason, summary, evidence_ref
FROM continuity_events
WHERE rowid > ? AND emitted = 1
ORDER BY rowid ASC
LIMIT ?
""",
    (event_cursor, max_rows),
).fetchall()

max_event_rowid = event_cursor
for row in ev_rows:
    rowid = int(row[0])
    max_event_rowid = max(max_event_rowid, rowid)
    event_id = str(row[1] or "")
    created_at_row = str(row[2] or now_iso())
    source = str(row[3] or "")
    event_key = str(row[4] or "")
    severity = str(row[5] or "info")
    changed = bool(row[6])
    cooldown_elapsed = bool(row[7])
    suppress_reason = str(row[8] or "")
    summary = str(row[9] or "")
    evidence_ref = str(row[10] or "")

    refs = {
        "continuity_event_id": event_id,
        "evidence_ref": evidence_ref,
        "checkpoint_id": continuity_latest.get("checkpoint_id"),
    }
    facts = {
        "source": source,
        "event_key": event_key,
        "severity": severity,
        "changed": changed,
        "cooldown_elapsed": cooldown_elapsed,
        "suppress_reason": suppress_reason,
        "summary": summary,
    }

    ok, _eid = persist_evidence(
        "operator.actions",
        "local",
        "event",
        event_id or f"continuity_event_row_{rowid}",
        created_at_row,
        facts,
        refs,
        severity,
        f"continuity_events:rowid:{rowid}",
    )
    bump("operator.actions", ok)

source_cursors["continuity_events_rowid"] = max_event_rowid
stream_updates["continuity_events_rowid"] = {"from": event_cursor, "to": max_event_rowid, "rows": len(ev_rows)}

cursors["updated_at"] = now_iso()
atomic_write(cursors_path, json.dumps(cursors, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

con.commit()

# Event-layer projection (`gtc.event.v2`) + GTC-driven incident replay hints.
event_rows = cur.execute(
    """
SELECT rowid, event_id, created_at, source, event_key, severity, emitted, changed, cooldown_elapsed, suppress_reason, summary, evidence_ref, route_key
FROM continuity_events
ORDER BY rowid ASC
"""
).fetchall()

event_evidence_map: Dict[str, str] = {}
for erow in cur.execute(
    """
SELECT evidence_id, refs_json
FROM gtc_evidence_index
WHERE connector_type = 'operator.actions'
ORDER BY monotonic_seq DESC
LIMIT 5000
"""
).fetchall():
    evid = str(erow[0] or "").strip()
    refs_obj = {}
    try:
        refs_obj = json.loads(str(erow[1] or "{}"))
        if not isinstance(refs_obj, dict):
            refs_obj = {}
    except Exception:
        refs_obj = {}
    ev_id = str(refs_obj.get("continuity_event_id") or "").strip()
    if ev_id and ev_id not in event_evidence_map:
        event_evidence_map[ev_id] = evid

route_stats: Dict[str, Dict[str, Any]] = {}
for row in event_rows:
    created_at = str(row[2] or "")
    route_key = str(row[12] or "")
    source = str(row[3] or "")
    event_key = str(row[4] or "")
    severity = str(row[5] or "info")
    emitted = bool(row[6])
    changed = bool(row[7])
    suppress_reason = str(row[9] or "")
    summary = str(row[10] or "")
    evidence_ref = str(row[11] or "")
    event_id = str(row[1] or "")

    bucket = route_stats.setdefault(
        route_key,
        {
            "route_key": route_key,
            "source": source,
            "event_key": event_key,
            "first_seen_at": created_at,
            "last_seen_at": created_at,
            "last_emitted_at": None,
            "last_changed_at": None,
            "last_closed_at": None,
            "open_since": None,
            "status": "closed",
            "severity": "info",
            "latest_event_id": event_id,
            "latest_summary": summary,
            "latest_evidence_ref": evidence_ref,
            "latest_gtc_evidence_id": event_evidence_map.get(event_id),
            "emitted_count": 0,
            "suppressed_count": 0,
            "changed_count": 0,
            "cooldown_suppressed_count": 0,
            "recent_event_ids": [],
            "related_task_ids": set(),
        },
    )

    bucket["source"] = source or bucket.get("source")
    bucket["event_key"] = event_key or bucket.get("event_key")
    bucket["last_seen_at"] = created_at or bucket.get("last_seen_at")
    bucket["latest_event_id"] = event_id or bucket.get("latest_event_id")
    bucket["latest_summary"] = summary
    bucket["latest_evidence_ref"] = evidence_ref
    bucket["latest_gtc_evidence_id"] = event_evidence_map.get(event_id)
    bucket["severity"] = severity

    if emitted:
        bucket["emitted_count"] = int(bucket.get("emitted_count") or 0) + 1
        bucket["last_emitted_at"] = created_at
    else:
        bucket["suppressed_count"] = int(bucket.get("suppressed_count") or 0) + 1
        if suppress_reason == "unchanged_within_cooldown":
            bucket["cooldown_suppressed_count"] = int(bucket.get("cooldown_suppressed_count") or 0) + 1

    if changed:
        bucket["changed_count"] = int(bucket.get("changed_count") or 0) + 1
        bucket["last_changed_at"] = created_at

    if severity in {"warn", "critical", "warning"}:
        if not bucket.get("open_since"):
            bucket["open_since"] = created_at
        bucket["status"] = "open"
    elif severity == "info" and emitted:
        bucket["status"] = "closed"
        bucket["last_closed_at"] = created_at
        bucket["open_since"] = None

    recent_ids = bucket.get("recent_event_ids") if isinstance(bucket.get("recent_event_ids"), list) else []
    if event_id:
        recent_ids.append(event_id)
        if len(recent_ids) > 6:
            recent_ids[:] = recent_ids[-6:]
    bucket["recent_event_ids"] = recent_ids

    for part in [p.strip() for p in evidence_ref.split("|") if p.strip()]:
        if part.startswith("autopilot:") or part.startswith("continuity:") or part.startswith("parity:"):
            bucket["related_task_ids"].add(part)

projection_rows: List[Dict[str, Any]] = []
open_incidents: List[Dict[str, Any]] = []
now_for_projection = now_dt()
for rk, bucket in route_stats.items():
    open_since = parse_iso(str(bucket.get("open_since") or ""))
    age_sec = int((now_for_projection - open_since).total_seconds()) if open_since else None
    related_tasks = sorted({str(t) for t in (bucket.get("related_task_ids") or set()) if str(t).strip()})
    row = {
        "route_key": rk,
        "source": bucket.get("source"),
        "event_key": bucket.get("event_key"),
        "status": bucket.get("status"),
        "severity": bucket.get("severity"),
        "first_seen_at": bucket.get("first_seen_at"),
        "last_seen_at": bucket.get("last_seen_at"),
        "open_since": bucket.get("open_since"),
        "open_age_sec": age_sec,
        "last_emitted_at": bucket.get("last_emitted_at"),
        "last_changed_at": bucket.get("last_changed_at"),
        "last_closed_at": bucket.get("last_closed_at"),
        "latest_event_id": bucket.get("latest_event_id"),
        "latest_gtc_evidence_id": bucket.get("latest_gtc_evidence_id"),
        "latest_summary": bucket.get("latest_summary"),
        "latest_evidence_ref": bucket.get("latest_evidence_ref"),
        "recent_event_ids": bucket.get("recent_event_ids") or [],
        "related_task_ids": related_tasks,
        "counters": {
            "emitted": int(bucket.get("emitted_count") or 0),
            "suppressed": int(bucket.get("suppressed_count") or 0),
            "changed": int(bucket.get("changed_count") or 0),
            "cooldown_suppressed": int(bucket.get("cooldown_suppressed_count") or 0),
        },
    }
    projection_rows.append(row)
    if row.get("status") == "open":
        open_incidents.append(row)

projection_rows.sort(
    key=lambda x: (
        0 if x.get("status") == "open" else 1,
        0 if str(x.get("severity") or "") == "critical" else (1 if str(x.get("severity") or "") in {"warn", "warning"} else 2),
        str(x.get("last_seen_at") or ""),
    ),
    reverse=False,
)

open_incidents_sorted = sorted(
    open_incidents,
    key=lambda x: (
        0 if str(x.get("severity") or "") == "critical" else 1,
        -int(x.get("open_age_sec") or 0),
    ),
)

verify_gate_preflight_posture = compute_verify_gate_preflight_posture(load_json(continuity_now_latest_path))
if str(verify_gate_preflight_posture.get("predicted_blocker_reason") or "").strip():
    recommended_gate_cmd = "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/continuity.sh verify-gate-status --json"
else:
    recommended_gate_cmd = ""

recommended_commands: List[str] = []
if recommended_gate_cmd:
    recommended_commands.append(recommended_gate_cmd)
for row in open_incidents_sorted[:5]:
    route_key = str(row.get("route_key") or "").strip()
    if route_key:
        recommended_commands.append(
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/gtc_incident_replay.sh "
            f"--route-key {shlex.quote(route_key)} --json"
        )

    src = str(row.get("source") or "")
    if src:
        recommended_commands.append(
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh "
            f"--source {shlex.quote(src)} --include-suppressed --hours 24 --json"
        )
    for task_id in (row.get("related_task_ids") or [])[:2]:
        task_txt = str(task_id or "").strip()
        if task_txt:
            recommended_commands.append(
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh "
                f"trace --task-id {shlex.quote(task_txt)} --json"
            )
    ev_ref = str(row.get("latest_evidence_ref") or "").strip()
    if ev_ref:
        ev_path = pathlib.Path(ev_ref)
        if not ev_path.is_absolute():
            ev_path = (root / ev_ref).resolve()
        if ev_path.exists() and ev_path.is_file():
            recommended_commands.append(f"sed -n '1,160p' {shlex.quote(rel(ev_path))}")

# stable de-duplication while preserving order
seen_cmds = set()
uniq_commands: List[str] = []
for cmd in recommended_commands:
    if cmd and cmd not in seen_cmds:
        uniq_commands.append(cmd)
        seen_cmds.add(cmd)

recovery = recover_inflight_publish_transaction()
base_publish_generation_id = live_publish_generation()
base_coherence_guard = read_coherence_guard()

try:
    gtc_publish_ttl_sec = max(0, int(os.environ.get("OPENCLAW_GTC_PUBLISH_TTL_SEC", "300")))
except Exception:
    gtc_publish_ttl_sec = 300
publish_generated_dt = now_dt()
publish_generated_at = publish_generated_dt.isoformat().replace("+00:00", "Z")
publish_valid_until = (publish_generated_dt + dt.timedelta(seconds=gtc_publish_ttl_sec)).isoformat().replace("+00:00", "Z")
publish_generation_id = f"gtcgen_{uuid.uuid4().hex[:16]}"

event_projection = {
    "schema_version": "gtc.event.v2",
    "generated_at": publish_generated_at,
    "build_generation_id": publish_generation_id,
    "valid_until": publish_valid_until,
    "routes": projection_rows,
    "summary": {
        "route_count": len(projection_rows),
        "open_incident_count": len(open_incidents),
        "critical_open_count": sum(1 for x in open_incidents if str(x.get("severity") or "") == "critical"),
        "warn_open_count": sum(1 for x in open_incidents if str(x.get("severity") or "") in {"warn", "warning"}),
    },
}

incident_replay = {
    "schema_version": "gtc.incident_replay.v1",
    "generated_at": publish_generated_at,
    "build_generation_id": publish_generation_id,
    "valid_until": publish_valid_until,
    "open_incidents": open_incidents_sorted,
    "recommended_commands": uniq_commands[:12],
    "verify_gate_preflight": verify_gate_preflight_posture,
}

atomic_write(latest_dir / "event_projection.json", json.dumps(event_projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
atomic_write(latest_dir / "incident_replay.json", json.dumps(incident_replay, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

# Build latest connector state + gateboard
rows = cur.execute(
    """
SELECT
  c.connector_type,
  c.connector_id,
  c.display_name,
  c.freshness_ttl_ms,
  c.stale_severity,
  p.evidence_id,
  p.observed_at,
  e.subject_kind,
  e.subject_id,
  e.facts_json,
  e.refs_json,
  e.severity_max,
  e.monotonic_seq
FROM gtc_connector c
LEFT JOIN gtc_latest_pointer p ON p.pointer_key = (c.connector_type || '::' || c.connector_id)
LEFT JOIN gtc_evidence_index e ON e.evidence_id = p.evidence_id
ORDER BY c.connector_type, c.connector_id
"""
).fetchall()

now = now_dt()
connector_views: List[Dict[str, Any]] = []
warning_reasons: List[str] = []
blocking_reasons: List[str] = []

verify_status = str(verify_last.get("status") or "unknown")
runtime_critical_count = 0

queue_active_nonterminal = int(
    (
        cur.execute(
            """
SELECT COUNT(*)
FROM work_queue
WHERE status IN ('QUEUED','RUNNING','REVIEW','BLOCKED')
"""
        ).fetchone()
        or [0]
    )[0]
)

queue_ready_count = int(
    (
        cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        or [0]
    )[0]
)

queue_ready_oldest_updated_at = (
    (
        cur.execute(
            """
SELECT MIN(w.updated_at)
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
        ).fetchone()
        or [None]
    )[0]
)
queue_ready_oldest_age_sec: Optional[int] = None
queue_ready_oldest_dt = parse_iso(str(queue_ready_oldest_updated_at or ""))
if queue_ready_oldest_dt is not None:
    queue_ready_oldest_age_sec = max(0, int((now - queue_ready_oldest_dt).total_seconds()))

queue_running_count = int(
    (
        cur.execute(
            """
SELECT COUNT(*)
FROM work_queue
WHERE status = 'RUNNING'
"""
        ).fetchone()
        or [0]
    )[0]
)
queue_active_file_lock_count = int(
    (
        cur.execute(
            """
SELECT COUNT(*)
FROM file_locks
WHERE lock_state = 'ACTIVE'
"""
        ).fetchone()
        or [0]
    )[0]
)

queue_now_iso = now.isoformat().replace("+00:00", "Z")
queue_stale_active_file_lock_count = int(
    (
        cur.execute(
            """
SELECT COUNT(*)
FROM file_locks
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
""",
            (queue_now_iso,),
        ).fetchone()
        or [0]
    )[0]
)

orphaned_running_min_sec = _read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_ORPHANED_RUNNING_MIN_SEC",
    default=_DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC,
)
queue_ready_idle_threshold_sec = _read_nonnegative_int_env(
    "OPENCLAW_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC",
    default=_DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC,
)

queue_orphaned_running_cutoff_iso = (
    now - dt.timedelta(seconds=orphaned_running_min_sec)
).replace(microsecond=0).isoformat().replace("+00:00", "Z")
queue_orphaned_running_without_locks_count = int(
    (
        cur.execute(
            """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND w.updated_at <= ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_locks fl
    WHERE fl.locked_by_task_id = w.task_id
      AND fl.lock_state = 'ACTIVE'
  )
""",
            (queue_orphaned_running_cutoff_iso,),
        ).fetchone()
        or [0]
    )[0]
)

queue_effective_active_file_lock_count = max(0, queue_active_file_lock_count - queue_stale_active_file_lock_count)
queue_effective_running_count = max(0, queue_running_count - queue_orphaned_running_without_locks_count)
queue_in_flight_effective = bool(queue_effective_running_count > 0 or queue_effective_active_file_lock_count > 0)
queue_stale_wave_active = bool(
    queue_ready_count > 0
    and not queue_in_flight_effective
    and (
        queue_ready_oldest_age_sec is None
        or queue_ready_oldest_age_sec >= queue_ready_idle_threshold_sec
    )
)

open_incident_count = int((event_projection.get("summary") or {}).get("open_incident_count") or 0)

for row in rows:
    connector_type = str(row[0] or "")
    connector_id = str(row[1] or "")
    display_name = str(row[2] or "")
    ttl_ms = int(row[3] or 60000)
    stale_severity = str(row[4] or "warning")
    evidence_id = str(row[5] or "")
    observed_at = str(row[6] or "")
    monotonic_seq = int(row[12] or 0)
    facts_obj = {}
    refs_obj = {}
    try:
        facts_obj = json.loads(str(row[9] or "{}")) if row[9] else {}
        if not isinstance(facts_obj, dict):
            facts_obj = {}
    except Exception:
        facts_obj = {}
    try:
        refs_obj = json.loads(str(row[10] or "{}")) if row[10] else {}
        if not isinstance(refs_obj, dict):
            refs_obj = {}
    except Exception:
        refs_obj = {}

    obs_dt = parse_iso(observed_at)
    age_ms = None
    stale = True
    valid_until = None
    if obs_dt is not None:
        age_ms = max(0, int((now - obs_dt).total_seconds() * 1000))
        stale = age_ms > ttl_ms
        valid_until = (obs_dt + dt.timedelta(milliseconds=max(0, ttl_ms))).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    row_view = {
        "connector_type": connector_type,
        "connector_id": connector_id,
        "display_name": display_name,
        "latest_evidence_id": evidence_id or None,
        "observed_at": observed_at or None,
        "valid_until": valid_until,
        "freshness_ttl_ms": ttl_ms,
        "age_ms": age_ms,
        "stale": stale,
        "stale_severity": stale_severity,
        "monotonic_seq": monotonic_seq,
        "issuer_boot_id": issuer_boot_id,
        "subject": {
            "kind": str(row[7] or ""),
            "id": str(row[8] or ""),
        },
        "severity_max": str(row[11] or ""),
        "refs": refs_obj,
    }
    connector_views.append(row_view)

    if connector_type == "runtime.gateway":
        runtime_critical_count = int(facts_obj.get("critical_anomaly_count") or 0)

    if connector_type == "queue.task":
        handoff_binding_status = str(facts_obj.get("handoff_gate_binding_status") or "").strip().lower()
        if handoff_binding_status == "degraded":
            if queue_active_nonterminal > 0:
                warning_reasons.append("queue_task_handoff_gate_binding_degraded")
            else:
                row_view["handoff_binding_warning_suppressed"] = "queue_idle"

    is_required = connector_type in {"runtime.gateway", "validation.gates"}
    if stale:
        reason = f"connector_stale:{connector_type}::{connector_id}"
        if connector_type == "queue.task" and not queue_stale_wave_active:
            if queue_ready_count <= 0 and not queue_in_flight_effective:
                row_view["stale_suppressed"] = "queue_quiescent"
            elif queue_in_flight_effective:
                row_view["stale_suppressed"] = "queue_in_flight_effective"
            else:
                row_view["stale_suppressed"] = "queue_ready_backlog_within_idle_grace"
        elif connector_type == "operator.actions" and open_incident_count == 0:
            row_view["stale_suppressed"] = "no_open_incidents"
        elif is_required and stale_severity == "critical":
            blocking_reasons.append(reason)
        else:
            warning_reasons.append(reason)

# mutation readiness
if verify_status != "READY":
    blocking_reasons.append(f"verify_status_not_ready:{verify_status}")
if runtime_critical_count > 0:
    blocking_reasons.append("runtime_gateway_critical_anomalies")

mutate_allowed = len(blocking_reasons) == 0
if mutate_allowed and warning_reasons:
    readiness_status = "yellow"
elif mutate_allowed:
    readiness_status = "green"
else:
    readiness_status = "red"

connector_vector_digest = sha256_text(
    canonical_json(
        [
            {
                "key": f"{row.get('connector_type')}::{row.get('connector_id')}",
                "evidence": row.get("latest_evidence_id"),
                "monotonic_seq": row.get("monotonic_seq"),
                "valid_until": row.get("valid_until"),
                "issuer_boot_id": row.get("issuer_boot_id"),
            }
            for row in connector_views
        ]
    )
)

gateboard = {
    "schema_version": "gtc.gateboard.v2",
    "generated_at": publish_generated_at,
    "build_generation_id": publish_generation_id,
    "valid_until": publish_valid_until,
    "issuer_boot_id": issuer_boot_id,
    "connector_vector_digest": connector_vector_digest,
    "mutate_allowed": mutate_allowed,
    "status": readiness_status,
    "blocking_reasons": sorted(set(blocking_reasons)),
    "warning_reasons": sorted(set(warning_reasons)),
    "required_connectors": [
        "runtime.gateway::gateway-main",
        "validation.gates::core",
    ],
    "verify_status": verify_status,
    "runtime_critical_anomaly_count": runtime_critical_count,
    "open_incident_count": open_incident_count,
    "queue_active_nonterminal_count": queue_active_nonterminal,
}

continuity_current = {
    "schema_version": "gtc.latest.v2",
    "generated_at": publish_generated_at,
    "build_generation_id": publish_generation_id,
    "valid_until": publish_valid_until,
    "issuer_boot_id": issuer_boot_id,
    "connector_vector_digest": connector_vector_digest,
    "db_path": rel(db_path),
    "gtc_root": rel(gtc_root),
    "connectors": connector_views,
    "event_projection": {
        "path": rel(live_latest_dir / "event_projection.json"),
        "open_incident_count": open_incident_count,
    },
    "incident_replay": {
        "path": rel(live_latest_dir / "incident_replay.json"),
        "recommended_commands": (incident_replay.get("recommended_commands") or [])[:5],
        "verify_gate_preflight": incident_replay.get("verify_gate_preflight"),
    },
    "readiness": {
        "mutate_allowed": mutate_allowed,
        "status": readiness_status,
        "reasons": sorted(set(blocking_reasons + warning_reasons)),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "warning_reasons": sorted(set(warning_reasons)),
        "open_incident_count": open_incident_count,
        "queue_active_nonterminal_count": queue_active_nonterminal,
    },
    "ingest": {
        "inserted_evidence": inserted,
        "skipped_existing": skipped,
        "connector_insert_counts": connector_insert_counts,
        "stream_updates": stream_updates,
    },
}

atomic_write(latest_dir / "continuity_current.json", json.dumps(continuity_current, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
atomic_write(latest_dir / "gateboard.json", json.dumps(gateboard, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

for row in connector_views:
    fname = f"{row['connector_type']}__{row['connector_id']}.json".replace("/", "_")
    row_surface = dict(row)
    row_surface["build_generation_id"] = publish_generation_id
    row_surface["valid_until"] = row_surface.get("valid_until") or publish_valid_until
    atomic_write(connectors_latest_dir / fname, json.dumps(row_surface, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

if not no_surfaces:
    lines = [
        "# GTC v2 Continuity Current",
        "",
        f"- generated_at: {continuity_current.get('generated_at')}",
        f"- mutate_allowed: {mutate_allowed}",
        f"- readiness_status: {readiness_status}",
    ]
    if blocking_reasons:
        lines.append(f"- blocking_reasons: {', '.join(sorted(set(blocking_reasons)))}")
    if warning_reasons:
        lines.append(f"- warning_reasons: {', '.join(sorted(set(warning_reasons)))}")
    lines.append("")
    lines.append("## Connector freshness")
    for row in connector_views:
        age_ms = row.get("age_ms")
        age_text = "n/a" if age_ms is None else f"{int(age_ms)}ms"
        suppressed = row.get("stale_suppressed")
        suppress_txt = f" | stale_suppressed={suppressed}" if suppressed else ""
        lines.append(
            f"- {row.get('connector_type')}::{row.get('connector_id')} | stale={row.get('stale')} | age={age_text} | ttl={row.get('freshness_ttl_ms')}ms | evidence={row.get('latest_evidence_id') or 'none'}{suppress_txt}"
        )
    atomic_write(surfaces_dir / "continuity_current.md", "\n".join(lines) + "\n")

    event_lines = [
        "# GTC Event Projection (v2)",
        "",
        f"- generated_at: {event_projection.get('generated_at')}",
        f"- open_incident_count: {open_incident_count}",
        "",
        "## Open incidents",
    ]
    for row in open_incidents_sorted[:20]:
        event_lines.append(
            f"- {row.get('route_key')} | severity={row.get('severity')} | open_age_sec={row.get('open_age_sec')} | summary={row.get('latest_summary') or ''}"
        )
    if len(open_incidents_sorted) == 0:
        event_lines.append("- (none)")
    atomic_write(surfaces_dir / "event_projection.md", "\n".join(event_lines) + "\n")

    replay_lines = [
        "# GTC Incident Replay",
        "",
        f"- generated_at: {incident_replay.get('generated_at')}",
        f"- open_incident_count: {len(open_incidents_sorted)}",
        (
            "- verify_gate_preflight: "
            f"mode={verify_gate_preflight_posture.get('mode')}; "
            f"source={verify_gate_preflight_posture.get('source')}; "
            f"ready_to_run={verify_gate_preflight_posture.get('ready_to_run') if verify_gate_preflight_posture.get('ready_to_run') is not None else 'n/a'}; "
            f"predicted_blocker={verify_gate_preflight_posture.get('predicted_blocker_reason') or 'none'}; "
            f"severity={verify_gate_preflight_posture.get('severity')}"
        ),
        "",
        "## Recommended commands",
    ]
    cmds = incident_replay.get("recommended_commands") if isinstance(incident_replay.get("recommended_commands"), list) else []
    if cmds:
        for cmd in cmds:
            replay_lines.append(f"- `{cmd}`")
    else:
        replay_lines.append("- (none)")
    atomic_write(surfaces_dir / "incident_replay.md", "\n".join(replay_lines) + "\n")

manifest_auth_scheme = select_manifest_auth_scheme()
manifest_fields_digest = manifest_auth_fields_sha256()
manifest_auth_profile = MANIFEST_AUTH_CANONICAL_PROFILE_BY_SCHEME[manifest_auth_scheme]
schema_gate_env_overrides: Dict[str, str] = {}

manifest_auth_root: Dict[str, Any] = {
    "scheme": manifest_auth_scheme,
    "key_id": "",
    "canonical_profile": manifest_auth_profile,
    "payload_fields": list(MANIFEST_AUTH_FIELDS),
    "payload_fields_sha256": manifest_fields_digest,
}
manifest_auth_summary: Dict[str, Any] = {
    "scheme": manifest_auth_scheme,
    "canonical_profile": manifest_auth_profile,
    "payload_fields_sha256": manifest_fields_digest,
}
manifest_auth_sign_material: Dict[str, str] = {}

if manifest_auth_scheme == "hmac-sha256":
    manifest_hmac_secret, manifest_hmac_key_id, manifest_hmac_source = load_or_create_manifest_hmac_secret()
    manifest_auth_root["key_id"] = manifest_hmac_key_id
    manifest_auth_root["key_source"] = manifest_hmac_source
    manifest_auth_summary["key_id"] = manifest_hmac_key_id
    manifest_auth_summary["key_source"] = manifest_hmac_source
    manifest_auth_sign_material = {
        "hmac_secret": manifest_hmac_secret,
        "key_id": manifest_hmac_key_id,
    }
elif manifest_auth_scheme == "ed25519-sha256":
    (
        manifest_ed25519_private_key_pem,
        manifest_ed25519_public_key_pem,
        manifest_ed25519_key_id,
        manifest_ed25519_private_source,
        manifest_ed25519_public_source,
        manifest_ed25519_public_sha256,
    ) = load_or_create_manifest_ed25519_keypair()
    manifest_auth_root["key_id"] = manifest_ed25519_key_id
    manifest_auth_root["key_source"] = manifest_ed25519_public_source
    manifest_auth_root["public_key_sha256"] = manifest_ed25519_public_sha256
    manifest_auth_summary["key_id"] = manifest_ed25519_key_id
    manifest_auth_summary["key_source"] = manifest_ed25519_public_source
    manifest_auth_summary["private_key_source"] = manifest_ed25519_private_source
    manifest_auth_summary["public_key_sha256"] = manifest_ed25519_public_sha256
    manifest_auth_sign_material = {
        "private_key_pem": manifest_ed25519_private_key_pem,
        "key_id": manifest_ed25519_key_id,
    }
    schema_gate_env_overrides["OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_PUBLIC_KEY_PEM"] = manifest_ed25519_public_key_pem
    schema_gate_env_overrides["OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_KEY_ID"] = manifest_ed25519_key_id
else:
    raise RuntimeError(f"unsupported_manifest_auth_scheme:{manifest_auth_scheme}")

publish_anchor = {
    "schema_version": "gtc.publish_anchor.v1",
    "generated_at": publish_generated_at,
    "build_generation_id": publish_generation_id,
    "valid_until": publish_valid_until,
    "issuer_boot_id": issuer_boot_id,
    "manifest_auth_root": manifest_auth_root,
}
atomic_write(latest_dir / "publish_anchor.json", json.dumps(publish_anchor, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

publish_manifest = {
    "schema_version": "gtc.publish_manifest.v1",
    "generated_at": publish_generated_at,
    "build_generation_id": publish_generation_id,
    "base_generation_id": base_publish_generation_id,
    "base_coherence_guard": base_coherence_guard,
    "valid_until": publish_valid_until,
    "latest_paths": {
        "publish_anchor": rel(live_latest_dir / "publish_anchor.json"),
        "continuity_current": rel(live_latest_dir / "continuity_current.json"),
        "gateboard": rel(live_latest_dir / "gateboard.json"),
        "event_projection": rel(live_latest_dir / "event_projection.json"),
        "incident_replay": rel(live_latest_dir / "incident_replay.json"),
        "connectors_dir": rel(live_connectors_latest_dir),
    },
    "latest_sha256": {
        "publish_anchor": sha256_file(latest_dir / "publish_anchor.json"),
        "continuity_current": sha256_file(latest_dir / "continuity_current.json"),
        "gateboard": sha256_file(latest_dir / "gateboard.json"),
        "event_projection": sha256_file(latest_dir / "event_projection.json"),
        "incident_replay": sha256_file(latest_dir / "incident_replay.json"),
        "connectors_dir": sha256_connectors_dir(connectors_latest_dir),
    },
}
if manifest_auth_scheme == "hmac-sha256":
    publish_manifest["manifest_auth"] = build_manifest_auth_hmac(
        publish_manifest,
        hmac_secret=manifest_auth_sign_material["hmac_secret"],
        key_id=manifest_auth_sign_material["key_id"],
    )
else:
    publish_manifest["manifest_auth"] = build_manifest_auth_ed25519(
        publish_manifest,
        private_key_pem=manifest_auth_sign_material["private_key_pem"],
        key_id=manifest_auth_sign_material["key_id"],
    )
atomic_write(latest_dir / "publish_manifest.json", json.dumps(publish_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

schema_gate: Dict[str, Any] = {"enforced": not skip_schema_gate, "ok": True, "skipped": bool(skip_schema_gate)}
schema_gate_ok = True
if recovery_mode == "recover-only":
    schema_gate = {
        "enforced": False,
        "ok": True,
        "skipped": True,
        "skipped_reason": "recovery_mode_recover_only",
    }
elif not skip_schema_gate:
    schema_cmd = [
        str(root / "ops" / "openclaw" / "continuity" / "gtc_latest_schema_check.sh"),
        "--gtc-root",
        str(staging_root),
        "--manifest-path-root",
        str(gtc_root),
        "--json",
    ]
    try:
        schema_env = os.environ.copy()
        if schema_gate_env_overrides:
            schema_env.update(schema_gate_env_overrides)
        cp = subprocess.run(schema_cmd, text=True, capture_output=True, timeout=90, env=schema_env)
        payload: Dict[str, Any] = {}
        parse_error = None
        if (cp.stdout or "").strip():
            try:
                parsed = json.loads(cp.stdout)
                if isinstance(parsed, dict):
                    payload = parsed
                else:
                    parse_error = "schema_gate_payload_not_object"
            except Exception as exc:
                parse_error = f"schema_gate_payload_unparseable:{exc}"

        payload_ok = bool(payload.get("ok") is True)
        schema_gate = {
            "enforced": True,
            "ok": bool(cp.returncode == 0 and payload_ok),
            "command_ok": bool(cp.returncode == 0),
            "returncode": int(cp.returncode),
            "surface_count": payload.get("surface_count"),
            "connector_count": payload.get("connector_count"),
            "error_count": payload.get("error_count"),
            "generation_consistency": payload.get("generation_consistency"),
            "check_schema_version": payload.get("schema_version"),
            "checked_at": payload.get("generated_at") or now_iso(),
            "failed_checks": [row for row in (payload.get("checks") or []) if isinstance(row, dict) and row.get("ok") is False],
        }
        if parse_error:
            schema_gate["error"] = parse_error
        if not payload and not parse_error:
            schema_gate["error"] = "schema_gate_payload_missing"
        stderr_txt = (cp.stderr or "").strip()
        if stderr_txt:
            schema_gate["stderr"] = stderr_txt[:400]
        schema_gate_ok = bool(schema_gate.get("ok") is True)
    except Exception as exc:
        schema_gate = {
            "enforced": True,
            "ok": False,
            "command_ok": False,
            "returncode": 127,
            "error": f"schema_gate_exec_failed:{exc}",
            "checked_at": now_iso(),
        }
        schema_gate_ok = False

recovery_status = str(recovery.get("status") or "")
recovery_lock_busy = recovery_status == "lock_busy"
recovery_incomplete = recovery_status in {"recovery_failed", "unknown_state", "pending"}
publish_enabled = recovery_mode == "recover-then-publish"

publish_tx: Optional[Dict[str, Any]] = None
if publish_enabled and not recovery_lock_busy:
    publish_tx = create_publish_tx(
        build_generation_id=publish_generation_id,
        base_generation_id=base_publish_generation_id,
        base_coherence_guard=base_coherence_guard,
        include_surfaces=bool(not no_surfaces),
        skip_schema_gate_tx=bool(skip_schema_gate),
    )
    if schema_gate_ok:
        publish_tx = transition_publish_tx(
            publish_tx,
            event="schema_gate_passed" if not skip_schema_gate else "schema_gate_skipped",
            state="prepared",
            step="schema_gate_passed" if not skip_schema_gate else "schema_gate_skipped",
            details={"schema_gate": schema_gate},
        )
    else:
        publish_tx = transition_publish_tx(
            publish_tx,
            event="schema_gate_failed",
            state="failed",
            step="schema_gate_failed",
            details={"schema_gate": schema_gate},
            terminal=True,
        )

publish_promotion: Dict[str, Any] = {
    "staging_root": rel(staging_root),
    "promoted": False,
    "surfaces_included": bool(not no_surfaces),
    "lock_path": rel(publish_lock_path),
    "base_generation_id": base_publish_generation_id,
    "base_coherence_guard": base_coherence_guard,
    "recovery": recovery,
    "recovery_mode": recovery_mode,
    "publish_decision": "pending",
}
promotion_ok = True
if recovery_lock_busy:
    promotion_ok = False
    publish_promotion["publish_decision"] = "blocked_recovery_lock_busy"
    publish_promotion["error"] = str(recovery.get("error") or "inflight_recovery_lock_busy")
    publish_promotion["error_class"] = "lock_busy"
    cleanup_staging_root(staging_root)
elif not publish_enabled:
    publish_promotion["publish_decision"] = "skipped_recover_only_mode"
    publish_promotion["skipped"] = True
    cleanup_staging_root(staging_root)
    if recovery_incomplete:
        promotion_ok = False
        publish_promotion["error"] = str(recovery.get("error") or f"recovery_status:{recovery_status}")
        publish_promotion["error_class"] = "recovery_incomplete"
elif schema_gate_ok:
    publish_promotion["publish_decision"] = "attempted"
    lock_fd, lock_err = acquire_publish_lock()
    if lock_err:
        promotion_ok = False
        publish_promotion["error"] = lock_err
        publish_promotion["error_class"] = "lock_busy"
        if publish_tx is not None:
            publish_tx = transition_publish_tx(
                publish_tx,
                event="lock_busy",
                state="failed",
                step="lock_busy",
                details={"error": lock_err},
                terminal=True,
            )
        cleanup_staging_root(staging_root)
    else:
        publish_promotion["lock_acquired"] = True
        if publish_tx is not None:
            publish_tx = transition_publish_tx(publish_tx, event="lock_acquired", state="promoting", step="lock_acquired")
        try:
            live_generation_id = live_publish_generation()
            publish_promotion["live_generation_id"] = live_generation_id
            base_generation_cmp = str(base_publish_generation_id or "").strip()
            live_generation_cmp = str(live_generation_id or "").strip()
            if live_generation_cmp != base_generation_cmp:
                promotion_ok = False
                publish_promotion["error"] = f"stale_publish_generation:live={live_generation_id or 'none'}"
                publish_promotion["error_class"] = "stale_generation"
                if publish_tx is not None:
                    publish_tx = transition_publish_tx(
                        publish_tx,
                        event="stale_generation",
                        state="failed",
                        step="stale_generation",
                        details={"live_generation_id": live_generation_id},
                        terminal=True,
                    )
                cleanup_staging_root(staging_root)
            else:
                live_coherence_guard = read_coherence_guard()
                publish_promotion["live_coherence_guard"] = live_coherence_guard
                coherence_mismatch_keys = compare_publish_guard(base_coherence_guard, live_coherence_guard)
                if coherence_mismatch_keys:
                    promotion_ok = False
                    publish_promotion["error"] = "stale_coherence_guard"
                    publish_promotion["error_class"] = "stale_coherence"
                    publish_promotion["coherence_mismatch_keys"] = coherence_mismatch_keys
                    if publish_tx is not None:
                        publish_tx = transition_publish_tx(
                            publish_tx,
                            event="stale_coherence",
                            state="failed",
                            step="stale_coherence",
                            details={"coherence_mismatch_keys": coherence_mismatch_keys},
                            terminal=True,
                        )
                    cleanup_staging_root(staging_root)
                else:
                    if publish_tx is None:
                        promotion_ok = False
                        publish_promotion["error"] = "publish_tx_missing"
                        publish_promotion["error_class"] = "promotion_failed"
                    else:
                        publish_tx, promotion_error = promote_staged_outputs(publish_tx, include_surfaces=not no_surfaces)
                        if promotion_error:
                            promotion_ok = False
                            publish_promotion["error"] = f"promotion_failed:{promotion_error}"
                            publish_promotion["error_class"] = "promotion_incomplete"
                        else:
                            publish_promotion["promoted"] = True
                            publish_tx = transition_publish_tx(
                                publish_tx,
                                event="committed",
                                state="committed",
                                step="verified",
                                details={"live_generation_id": live_publish_generation()},
                                terminal=True,
                            )
        finally:
            release_publish_lock(lock_fd)
else:
    publish_promotion["publish_decision"] = "schema_gate_failed"
    publish_promotion["skipped"] = True
    cleanup_staging_root(staging_root)

summary = {
    "ok": True,
    "schema_version": "gtc.sync.result.v1",
    "generated_at": publish_generated_at,
    "db_path": rel(db_path),
    "gtc_root": rel(gtc_root),
    "build_generation_id": publish_generation_id,
    "base_generation_id": base_publish_generation_id,
    "base_coherence_guard": base_coherence_guard,
    "valid_until": publish_valid_until,
    "recovery_mode": {
        "selected": recovery_mode,
        "default": "recover-then-publish",
        "publish_enabled": publish_enabled,
        "recovery_status": recovery_status,
    },
    "inserted_evidence": inserted,
    "skipped_existing": skipped,
    "connector_insert_counts": connector_insert_counts,
    "stream_updates": stream_updates,
    "latest_paths": {
        "publish_anchor": rel(live_latest_dir / "publish_anchor.json"),
        "continuity_current": rel(live_latest_dir / "continuity_current.json"),
        "gateboard": rel(live_latest_dir / "gateboard.json"),
        "event_projection": rel(live_latest_dir / "event_projection.json"),
        "incident_replay": rel(live_latest_dir / "incident_replay.json"),
        "publish_manifest": rel(live_latest_dir / "publish_manifest.json"),
        "cursors": rel(live_cursors_path),
    },
    "schema_gate": schema_gate,
    "publish_promotion": publish_promotion,
    "manifest_auth": manifest_auth_summary,
    "publish_transaction": {
        "tx_id": (publish_tx or {}).get("tx_id") if isinstance(publish_tx, dict) else None,
        "state": (publish_tx or {}).get("state") if isinstance(publish_tx, dict) else None,
        "step": (publish_tx or {}).get("step") if isinstance(publish_tx, dict) else None,
        "journal_latest_path": rel(publish_journal_latest_path),
        "journal_events_path": rel(publish_journal_events_path),
    },
    "gateboard": {
        "mutate_allowed": mutate_allowed,
        "status": readiness_status,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "warning_reasons": sorted(set(warning_reasons)),
        "open_incident_count": open_incident_count,
    },
}
summary["ok"] = bool(summary["ok"] and schema_gate_ok and promotion_ok)

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("GTC V2 SYNC")
    print(f"- ok: {summary['ok']}")
    print(f"- inserted_evidence: {inserted}")
    print(f"- skipped_existing: {skipped}")
    print(f"- mutate_allowed: {mutate_allowed}")
    print(f"- open_incident_count: {open_incident_count}")
    recovery_mode_summary = summary.get("recovery_mode") if isinstance(summary.get("recovery_mode"), dict) else {}
    print(
        "- recovery_mode: "
        f"selected={recovery_mode_summary.get('selected')} "
        f"publish_enabled={recovery_mode_summary.get('publish_enabled')} "
        f"recovery_status={recovery_mode_summary.get('recovery_status')}"
    )
    sg = summary.get("schema_gate") if isinstance(summary.get("schema_gate"), dict) else {}
    print(
        "- schema_gate: "
        f"enforced={sg.get('enforced')} "
        f"ok={sg.get('ok')} "
        f"errors={sg.get('error_count') if sg.get('error_count') is not None else 'n/a'}"
    )
    promo = summary.get("publish_promotion") if isinstance(summary.get("publish_promotion"), dict) else {}
    print(
        "- publish_promotion: "
        f"promoted={promo.get('promoted')} "
        f"skipped={promo.get('skipped', False)} "
        f"decision={promo.get('publish_decision')}"
    )
    if blocking_reasons:
        print(f"- blocking_reasons: {', '.join(sorted(set(blocking_reasons)))}")
    if warning_reasons:
        print(f"- warning_reasons: {', '.join(sorted(set(warning_reasons)))}")

con.close()

if not schema_gate_ok:
    raise SystemExit(3)
if not promotion_ok:
    err_class = str(publish_promotion.get("error_class") or "")
    if err_class == "lock_busy":
        raise SystemExit(5)
    if err_class == "stale_generation":
        raise SystemExit(6)
    if err_class == "stale_coherence":
        raise SystemExit(7)
    raise SystemExit(4)
if strict and not mutate_allowed:
    raise SystemExit(1)
PY
