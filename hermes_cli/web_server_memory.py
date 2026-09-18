"""Memory-provider dashboard helpers: manifest/schema loading, setup-env and dependency probes, configured-status discovery.
"""

import logging
import json
import os
import re
import shlex
import subprocess
import yaml
from fastapi import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")

_MEMORY_PROVIDER_IMPORT_NAMES = {
    "honcho-ai": "honcho",
    "mem0ai": "mem0",
    "hindsight-client": "hindsight_client",
    "hindsight-all": "hindsight",
}


def _normalize_memory_provider_name(name: Any) -> str:
    provider = str(name or "").strip()
    return "" if provider.lower() in {"built-in", "builtin", "none"} else provider


def _load_memory_provider(name: str):
    try:
        from plugins.memory import load_memory_provider

        return load_memory_provider(name)
    except Exception:
        _log.debug("Failed to load memory provider %s", name, exc_info=True)
        return None


def _memory_provider_manifest(name: str) -> Dict[str, Any]:
    try:
        from plugins.memory import find_provider_dir

        provider_dir = find_provider_dir(name)
        if provider_dir is None:
            return {}
        manifest_path = provider_dir / "plugin.yaml"
        if not manifest_path.exists():
            return {}
        with manifest_path.open(encoding="utf-8-sig") as handle:
            manifest = yaml.safe_load(handle) or {}
        return manifest if isinstance(manifest, dict) else {}
    except Exception:
        _log.debug("Failed to read memory provider manifest for %s", name, exc_info=True)
        return {}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _memory_provider_setup_manifest(name: str) -> Dict[str, Any]:
    manifest = _memory_provider_manifest(name)
    external_dependencies: List[Dict[str, str]] = []
    for raw in manifest.get("external_dependencies") or []:
        if not isinstance(raw, dict):
            continue
        dep = {k: str(raw.get(k) or "").strip() for k in ("name", "install", "check")}
        if any(dep.values()):
            external_dependencies.append(dep)
    return {
        "pip_dependencies": _string_list(manifest.get("pip_dependencies")),
        "external_dependencies": external_dependencies,
        "required_env": _string_list(manifest.get("requires_env")),
    }


def _memory_provider_setup_info(name: str) -> Dict[str, Any]:
    setup = _memory_provider_setup_manifest(name)
    setup["dependencies_installed"] = _memory_provider_dependencies_installed(setup)
    return setup


def _memory_provider_dependency_package(dep: str) -> str:
    return re.split(r"[\[<>=!~;]", dep, maxsplit=1)[0].strip()


def _memory_provider_import_name(dep: str) -> str:
    package = _memory_provider_dependency_package(dep)
    return _MEMORY_PROVIDER_IMPORT_NAMES.get(package, package.replace("-", "_"))


def _dependency_importable(dep: str) -> bool:
    import_name = _memory_provider_import_name(dep)
    try:
        return bool(import_name) and __import__(import_name) is not None
    except ImportError:
        return False


def _memory_provider_setup_env() -> Dict[str, str]:
    # External package-manager child (npm/uv/pip): exact env preservation —
    # scrubbing or HOME rewriting could break user tool auth/config.
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    home = Path.home()
    extra_bins = [home / ".brv-cli" / "bin", home / ".local" / "bin", home / ".npm-global" / "bin", Path("/usr/local/bin")]
    prefix = os.pathsep.join(str(path) for path in extra_bins if path.exists())
    if prefix:
        env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
    return env


def _run_setup_command(
    command: Any, *, display: str, shell: bool = False, timeout: int = 180
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=shell,
        executable="/bin/bash" if shell else None,
        env=_memory_provider_setup_env(),
        capture_output=True,
        text=True,
        # Lossy UTF-8 decode — setup tools emit UTF-8; a locale-mismatched byte must never raise.
        # Force UTF-8 with lossy decoding so child output containing bytes that are invalid in the system
        # locale (e.g. GBK on Chinese Windows) can't raise UnicodeDecodeError inside the drain threads and
        # crash the gateway. See #53137.
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _memory_provider_dependencies_installed(setup: Dict[str, Any]) -> bool:
    pip_ok = all(_dependency_importable(dep) for dep in _string_list(setup.get("pip_dependencies")))
    external_ok = True
    for dep in setup.get("external_dependencies") or []:
        if not isinstance(dep, dict):
            continue
        check_cmd = str(dep.get("check") or "").strip()
        if not check_cmd:
            if str(dep.get("install") or "").strip():
                external_ok = False
            continue
        try:
            completed = _run_setup_command(shlex.split(check_cmd), display=check_cmd, timeout=20)
        except Exception:
            external_ok = False
            continue
        if completed.returncode != 0:
            external_ok = False
    return pip_ok and external_ok


def _schema_field_kind(raw: Dict[str, Any], choices: list) -> str:
    """Field kind from explicit ``kind``/``type`` hints, else inferred from ``default``."""
    explicit_kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
    default = raw.get("default")
    if raw.get("secret"):
        return "secret"
    if choices:
        return "select"
    if explicit_kind in {"bool", "boolean"} or isinstance(default, bool):
        return "boolean"
    if explicit_kind in {"int", "integer"} or (isinstance(default, int) and not isinstance(default, bool)):
        return "integer"
    if explicit_kind in {"float", "number"} or isinstance(default, float):
        return "number"
    return "text"


def _normalize_memory_provider_schema(name: str, provider: Any) -> List[Dict[str, Any]]:
    raw_schema: List[Dict[str, Any]] = []
    if provider is not None and hasattr(provider, "get_config_schema"):
        try:
            raw = provider.get_config_schema()
            if isinstance(raw, list):
                raw_schema = [field for field in raw if isinstance(field, dict)]
        except Exception:
            _log.warning("Failed to read memory provider schema for %s", name, exc_info=True)

    fields: List[Dict[str, Any]] = []
    for raw in raw_schema:
        key = str(raw.get("key") or "").strip()
        if not key:
            continue
        choices = raw.get("choices") or raw.get("options") or []
        if not isinstance(choices, list):
            choices = []
        fields.append({
            "key": key,
            "label": str(raw.get("label") or key.replace("_", " ").title()),
            "kind": _schema_field_kind(raw, choices),
            "description": str(raw.get("description") or ""),
            "placeholder": str(raw.get("placeholder") or ""),
            "required": bool(raw.get("required", False)),
            "default": raw.get("default", ""),
            "options": [{"value": str(c), "label": str(c), "description": ""} for c in choices],
            "url": str(raw.get("url") or ""),
            "when": raw.get("when") if isinstance(raw.get("when"), dict) else None,
            "minimum": raw.get("minimum"),
            "maximum": raw.get("maximum"),
            "step": raw.get("step"),
            "_env_key": str(raw.get("env_var") or "") or None,
        })
    return fields


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        _log.debug("Failed to read JSON config from %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _read_memory_provider_existing_values(name: str) -> Dict[str, Any]:
    """Best-effort read of existing provider config across legacy/native stores."""
    from hermes_cli.config import get_hermes_home, load_config

    hermes_home = get_hermes_home()
    values: Dict[str, Any] = {}
    for path in (hermes_home / f"{name}.json", hermes_home / name / "config.json"):
        values.update(_read_json_file(path))

    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    memory_cfg = cfg.get("memory")
    if isinstance(memory_cfg, dict):
        provider_cfg = memory_cfg.get(name)
        if isinstance(provider_cfg, dict):
            values.update(provider_cfg)
        legacy_cfg = memory_cfg.get("provider_config")
        if isinstance(legacy_cfg, dict):
            values = {**legacy_cfg, **values}

    # Holographic stores under plugins.hermes-memory-store.
    plugins_cfg = cfg.get("plugins")
    if name == "holographic" and isinstance(plugins_cfg, dict):
        holographic_cfg = plugins_cfg.get("hermes-memory-store")
        if isinstance(holographic_cfg, dict):
            values.update(holographic_cfg)
    return values


def _env_lookup(env_key: Optional[str]) -> str:
    from hermes_cli.config import load_env
    if not env_key:
        return ""
    return str(load_env().get(env_key) or os.environ.get(env_key) or "")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _field_default(field: Dict[str, Any]) -> Any:
    default = field.get("default", "")
    if field["kind"] == "boolean":
        return _coerce_bool(default, default=False)
    return default


def _field_value(field: Dict[str, Any], data: Dict[str, Any]) -> Any:
    if field["kind"] == "secret":
        return ""
    value = data.get(field["key"])
    if value in (None, ""):
        value = _env_lookup(field.get("_env_key"))
    if value in (None, ""):
        value = _field_default(field)

    if field["kind"] == "select":
        allowed = {opt["value"] for opt in field.get("options", [])}
        value = str(value)
        return value if value in allowed else str(_field_default(field))
    if field["kind"] == "boolean":
        return _coerce_bool(value, default=_coerce_bool(_field_default(field), default=False))
    return str(value)


def _field_is_set(field: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if field["kind"] == "secret":
        return bool(_env_lookup(field.get("_env_key")) or data.get(field["key"]))
    return _field_value(field, data) not in (None, "")


def _field_visible(
    field: Dict[str, Any], data: Dict[str, Any], fields_by_key: Optional[Dict[str, Dict[str, Any]]] = None
) -> bool:
    when = field.get("when")
    if not isinstance(when, dict) or not when:
        return True
    for dep_key, expected in when.items():
        dep_field = (fields_by_key or {}).get(str(dep_key)) or {
            "key": str(dep_key), "kind": "text", "default": "", "_env_key": None
        }
        if str(_field_value(dep_field, data)) != str(expected):
            return False
    return True


def _memory_provider_is_configured(name: str, provider: Any) -> bool:
    data = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    return all(
        _field_is_set(field, data)
        for field in fields
        if field.get("required") and _field_visible(field, data, fields_by_key)
    )


def _memory_provider_status(row: Dict[str, Any], setup: Dict[str, Any], configured: bool, schema_fields: list) -> str:
    if row["missing"]:
        return "missing"
    if not row["available"] and not setup.get("dependencies_installed", True):
        return "unavailable"
    if not configured or (not row["available"] and schema_fields):
        return "needs_config"
    return "ready" if row["available"] else "unavailable"


def _discover_memory_provider_statuses() -> List[Dict[str, Any]]:
    from hermes_cli.config import load_config
    discovered: Dict[str, Dict[str, Any]] = {}
    try:
        from plugins.memory import discover_memory_providers

        for name, description, available in discover_memory_providers():
            discovered[str(name)] = {
                "name": str(name),
                "description": str(description or ""),
                "available": bool(available),
                "missing": False,
            }
    except Exception:
        _log.exception("discover_memory_providers failed")

    mem = load_config().get("memory")
    active = _normalize_memory_provider_name(mem.get("provider")) if isinstance(mem, dict) else ""
    if active and active not in discovered:
        discovered[active] = {
            "name": active,
            "description": "Configured provider was not found.",
            "available": False,
            "missing": True,
        }

    providers: List[Dict[str, Any]] = []
    for name in sorted(discovered):
        row = discovered[name]
        missing = row["missing"]
        provider = None if missing else _load_memory_provider(name)
        setup = _memory_provider_setup_info(name)
        configured = False if missing else _memory_provider_is_configured(name, provider)
        schema_fields = [] if missing else _normalize_memory_provider_schema(name, provider)
        providers.append({
            "name": name,
            "description": row["description"],
            "available": row["available"],
            "configured": configured,
            "status": _memory_provider_status(row, setup, configured, schema_fields),
            "setup": setup,
        })
    return providers


def _require_memory_provider_ready(name: str) -> None:
    if not name:
        return
    row = next((r for r in _discover_memory_provider_statuses() if r["name"] == name), None)
    if row is None:
        raise HTTPException(status_code=400, detail=f"Unknown memory provider '{name}'.")
    if row["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Memory provider '{name}' is not ready "
                f"({row['status'].replace('_', ' ')}). Configure it in the dashboard first."
            ),
        )
