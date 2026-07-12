"""Redacted settings.snapshot (contract §4.6).

Strips secret-shaped keys before anything leaves the host. BFF re-asserts
redaction; host is the first line of defense.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hermes_constants import display_hermes_home, get_hermes_home

logger = logging.getLogger(__name__)

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|password|passwd|secret|authorization|auth|credential|"
    r"private[_-]?key|access[_-]?key|refresh[_-]?key|bearer|headers|env)$",
    re.IGNORECASE,
)

_DEFAULT_INCLUDE = ("skills", "crons", "mcp_servers", "models", "integrations")


def _is_secret_key(key: str) -> bool:
    k = str(key)
    if _SECRET_KEY_RE.search(k):
        return True
    lowered = k.lower()
    for frag in (
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
        "passwd",
        "password",
        "authorization",
    ):
        if frag in lowered:
            return True
    return False


def redact_value(value: Any, *, key: str = "") -> Any:
    """Recursively strip secret-shaped keys from nested structures."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(str(k)):
                continue
            # Drop nested env/headers blocks entirely
            if str(k).lower() in {"env", "headers", "secrets", "credentials"}:
                continue
            out[str(k)] = redact_value(v, key=str(k))
        return out
    if isinstance(value, list):
        return [redact_value(v, key=key) for v in value]
    return value


def build_settings_snapshot(
    params: Optional[Dict[str, Any]] = None,
    *,
    hermes_home: Optional[Path] = None,
) -> Dict[str, Any]:
    params = params or {}
    include_raw = params.get("include")
    if isinstance(include_raw, list) and include_raw:
        include: Set[str] = {str(x) for x in include_raw}
    else:
        include = set(_DEFAULT_INCLUDE)

    home = hermes_home or get_hermes_home()
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    snapshot: Dict[str, Any] = {
        "exported_at": exported_at,
        "hermes_home": display_hermes_home(),
        "source": "hermes",
    }

    if "skills" in include:
        snapshot["skills"] = _collect_skills(home)
    if "crons" in include:
        snapshot["crons"] = _collect_crons(home)
    if "mcp_servers" in include:
        snapshot["mcp_servers"] = _collect_mcp_servers(home)
    if "models" in include:
        snapshot["models"] = _collect_models(home)
    if "integrations" in include:
        snapshot["integrations"] = _collect_integrations(home)

    # Final redaction pass over the whole payload.
    return redact_value(snapshot)


def _load_yaml_config(home: Path) -> Dict[str, Any]:
    cfg_path = home / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("brain_rpc settings: config load failed: %s", exc)
        return {}


def _collect_skills(home: Path) -> List[Dict[str, Any]]:
    skills: List[Dict[str, Any]] = []
    roots = [home / "skills"]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for skill_md in root.rglob("SKILL.md"):
                name = skill_md.parent.name
                rel = str(skill_md.parent.relative_to(home)) if home in skill_md.parents else str(skill_md.parent)
                skills.append(
                    {
                        "id": name,
                        "name": name,
                        "enabled": True,
                        "path": rel,
                        "source": "hermes",
                    }
                )
        except OSError as exc:
            logger.debug("brain_rpc settings: skills scan failed: %s", exc)
    # Stable order
    skills.sort(key=lambda s: s.get("id") or "")
    return skills


def _collect_crons(home: Path) -> List[Dict[str, Any]]:
    jobs_path = home / "cron" / "jobs.json"
    if not jobs_path.is_file():
        return []
    try:
        import json

        raw = json.loads(jobs_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("brain_rpc settings: cron load failed: %s", exc)
        return []

    jobs = raw if isinstance(raw, list) else (raw.get("jobs") if isinstance(raw, dict) else None)
    if not isinstance(jobs, list):
        return []

    out: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        jid = str(job.get("id") or job.get("job_id") or "")
        name = str(job.get("name") or job.get("title") or jid or "job")
        schedule = str(job.get("schedule") or job.get("cron") or "")
        enabled = job.get("enabled")
        if enabled is None:
            enabled = not bool(job.get("paused") or job.get("disabled"))
        out.append(
            {
                "id": jid or name,
                "name": name,
                "schedule": schedule,
                "enabled": bool(enabled),
            }
        )
    return out


def _collect_mcp_servers(home: Path) -> List[Dict[str, Any]]:
    cfg = _load_yaml_config(home)
    servers = cfg.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        return []
    out: List[Dict[str, Any]] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            entry = {}
        # Never include command args that may embed secrets; transport only.
        transport = str(entry.get("transport") or entry.get("type") or "stdio")
        enabled = entry.get("enabled")
        if enabled is None:
            enabled = True
        out.append(
            {
                "name": str(name),
                "transport": transport,
                "enabled": bool(enabled),
            }
        )
    return out


def _collect_models(home: Path) -> List[Dict[str, Any]]:
    cfg = _load_yaml_config(home)
    model = cfg.get("model")
    out: List[Dict[str, Any]] = []
    if isinstance(model, str) and model.strip():
        out.append(
            {
                "provider": "default",
                "model": model.strip(),
                "is_active": True,
                "connected": True,
            }
        )
    elif isinstance(model, dict):
        provider = str(model.get("provider") or model.get("backend") or "default")
        model_name = str(model.get("model") or model.get("name") or model.get("default") or "")
        if model_name:
            out.append(
                {
                    "provider": provider,
                    "model": model_name,
                    "is_active": True,
                    "connected": True,
                }
            )
    return out


def _collect_integrations(home: Path) -> List[Dict[str, Any]]:
    # MVP: surface known CLI integration markers without credentials.
    cfg = _load_yaml_config(home)
    out: List[Dict[str, Any]] = []
    # Gateway platforms present in config are integrations of type "messaging".
    gw = cfg.get("gateway") or {}
    if isinstance(gw, dict):
        platforms = gw.get("platforms") or gw.get("enabled_platforms") or []
        if isinstance(platforms, list):
            for p in platforms:
                out.append({"id": str(p), "name": str(p), "type": "messaging"})
        elif isinstance(platforms, dict):
            for p, meta in platforms.items():
                enabled = True
                if isinstance(meta, dict) and meta.get("enabled") is False:
                    enabled = False
                if enabled:
                    out.append({"id": str(p), "name": str(p), "type": "messaging"})
    return out
