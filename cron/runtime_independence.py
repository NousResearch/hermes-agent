"""Profile-scoped runtime-independence attestation for cron execution.

The guard is deliberately default-off. A separately authorized cutover writes
``cron/runtime-independence.json`` with ``enforce: true`` and one approved
execution digest per runnable job. Once enforcement is on, malformed or stale
attestations fail closed across every caller that uses the shared jobs API.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ATTESTATION_FILE = "runtime-independence.json"
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(token|secret|password|credential|api[_-]?key|auth)(?:$|[_-])",
    re.IGNORECASE,
)
_CONTRACT_FIELDS = (
    "attach_to_session",
    "base_url",
    "context_from",
    "deliver",
    "enabled_toolsets",
    "model",
    "model_snapshot",
    "monitor_script",
    "monitor_url",
    "no_agent",
    "noAgent",
    "prompt",
    "provider",
    "provider_snapshot",
    "reasoning_effort",
    "schedule",
    "script",
    "skill",
    "skills",
    "workdir",
)
_CONFIG_SECTIONS = (
    "cron",
    "filesystem",
    "mcp_servers",
    "plugins",
    "terminal",
    "tools",
)


def _stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _redact_config_value(value: object, key: str = "") -> object:
    if _SENSITIVE_KEY_RE.search(key):
        return "SECRET_REDACTED"
    if isinstance(value, dict):
        return {
            str(item_key): _redact_config_value(item, str(item_key))
            for item_key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, list):
        return [_redact_config_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value).__name__)


def _value_free_config_shape(profile_home: Path) -> object:
    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        return {"status": "missing"}
    try:
        import yaml

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"status": "malformed"}
    if not isinstance(parsed, dict):
        return {"status": "not-mapping"}
    return {
        section: _redact_config_value(parsed.get(section), section)
        for section in _CONFIG_SECTIONS
        if section in parsed
    }


def _scheduler_provider(profile_home: Path) -> str:
    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        return "builtin"
    try:
        import yaml

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "invalid"
    cron = parsed.get("cron") if isinstance(parsed, dict) else None
    provider = cron.get("provider") if isinstance(cron, dict) else None
    return str(provider or "builtin").strip().lower() or "builtin"


def _script_shape(job: dict[str, Any], profile_home: Path) -> object:
    raw = job.get("script")
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = profile_home / "scripts" / candidate
    try:
        resolved = candidate.resolve(strict=True)
        scripts_root = (profile_home / "scripts").resolve(strict=True)
        resolved.relative_to(scripts_root)
        if not resolved.is_file():
            raise OSError("not a file")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    except (OSError, RuntimeError, ValueError):
        return {"status": "missing-or-unsafe", "path": str(candidate)}
    return {
        "status": "ok",
        "path": str(resolved.relative_to(profile_home)),
        "sha256": digest,
    }


def compute_execution_contract_digest(
    job: dict[str, Any],
    profile_home: Path,
) -> str:
    """Hash the runnable contract without persisting credential values."""
    job_contract = {field: job.get(field) for field in _CONTRACT_FIELDS if field in job}
    repeat = job.get("repeat")
    if isinstance(repeat, dict) and "times" in repeat:
        job_contract["repeat"] = {"times": repeat.get("times")}
    contract = {
        "config": _value_free_config_shape(profile_home),
        "job": job_contract,
        "job_id": str(job.get("id") or ""),
        "release_digest": job.get("runtime_release_digest"),
        "script": _script_shape(job, profile_home),
    }
    return _stable_digest(contract)


def _load_attestation(profile_home: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = profile_home / "cron" / ATTESTATION_FILE
    if not path.exists():
        return None, None
    try:
        if path.is_symlink():
            return None, "attestation file may not be a symlink"
        if path.stat().st_mode & 0o077:
            return None, "attestation file permissions must be private"
    except OSError:
        return None, "attestation file is unreadable"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "attestation file is malformed or unreadable"
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None, "attestation schema is invalid"
    if not isinstance(payload.get("enforce"), bool):
        return None, "attestation enforce flag must be boolean"
    return payload, None


def verify_job_runtime_independence(
    job: dict[str, Any],
    profile_home: Path,
    *,
    action: str,
) -> tuple[bool, str]:
    """Return whether ``job`` may execute or become runnable for ``action``."""
    payload, error = _load_attestation(profile_home)
    if error:
        return False, f"runtime independence blocked {action}: {error}"
    if payload is None or payload.get("enforce") is not True:
        return True, "runtime independence enforcement is off"
    provider = _scheduler_provider(profile_home)
    if provider != "builtin":
        return False, (
            f"runtime independence blocked {action}: scheduler provider "
            f"{provider!r} is not approved for this migration"
        )
    epoch = payload.get("epoch")
    entries = payload.get("jobs")
    if not isinstance(epoch, str) or not epoch or not isinstance(entries, dict):
        return False, f"runtime independence blocked {action}: active epoch or jobs map missing"
    fence = payload.get("fence")
    if fence is not None:
        if not isinstance(fence, dict):
            return False, f"runtime independence blocked {action}: fence record is malformed"
        state = fence.get("state")
        fence_epoch = fence.get("epoch")
        if fence_epoch != epoch:
            return False, f"runtime independence blocked {action}: fence epoch is stale"
        if state == "active":
            expires_at = fence.get("expires_at")
            if not isinstance(expires_at, str):
                return False, f"runtime independence blocked {action}: active fence has no expiry"
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
            except ValueError:
                return False, f"runtime independence blocked {action}: active fence expiry is invalid"
            if expiry <= datetime.now(timezone.utc):
                return False, (
                    f"runtime independence blocked {action}: fence expired and requires "
                    "explicit recover or abort"
                )
            return False, f"runtime independence blocked {action}: cohort fence is active"
        if state == "released":
            if not isinstance(fence.get("released_at"), str):
                return False, f"runtime independence blocked {action}: released fence lacks receipt time"
        else:
            return False, f"runtime independence blocked {action}: fence state is invalid"
    job_id = str(job.get("id") or "")
    entry = entries.get(job_id)
    if not isinstance(entry, dict):
        return False, f"runtime independence blocked {action}: job has no approved attestation"
    if entry.get("status") != "approved" or entry.get("dependency_status") != "independent":
        return False, f"runtime independence blocked {action}: job is not approved independent"
    if entry.get("epoch") != epoch:
        return False, f"runtime independence blocked {action}: job epoch is stale"
    expected = entry.get("execution_digest")
    current = compute_execution_contract_digest(job, profile_home)
    if not isinstance(expected, str) or expected != current:
        return False, f"runtime independence blocked {action}: execution contract digest changed"
    return True, "runtime independence attestation verified"
