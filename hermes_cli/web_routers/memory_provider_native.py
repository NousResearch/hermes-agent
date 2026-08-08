"""Memory-provider native-config helpers (extracted verbatim from web_server.py).

Cluster c17 (memory_provider_native) of the s2 shard plan: schema
normalization, field coercion, existing-value reads, native config saves,
and provider status discovery.  Bodies are byte-identical to their previous
in-web_server form.

Cross-module helpers and config functions that tests monkeypatch on
``web_server`` are reached through the late-binding seam in
:mod:`hermes_cli.web_deps`, so ``monkeypatch.setattr(web_server, ...)``
keeps working (see web_deps).
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from hermes_cli.web_deps import late

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent) - same seam as hermes_cli/web_routers/cron.py.
get_hermes_home = late("get_hermes_home")
load_config = late("load_config")
load_env = late("load_env")
save_config = late("save_config")
save_env_value = late("save_env_value")
_load_memory_provider = late("_load_memory_provider")
_memory_provider_label = late("_memory_provider_label")
_memory_provider_setup_info = late("_memory_provider_setup_info")
_normalize_memory_provider_name = late("_normalize_memory_provider_name")

_log = logging.getLogger("hermes_cli.web_server")


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

        explicit_kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
        if raw.get("secret"):
            kind = "secret"
        elif choices:
            kind = "select"
        elif explicit_kind in {"bool", "boolean"} or isinstance(raw.get("default"), bool):
            kind = "boolean"
        elif explicit_kind in {"int", "integer"} or (
            isinstance(raw.get("default"), int) and not isinstance(raw.get("default"), bool)
        ):
            kind = "integer"
        elif explicit_kind in {"float", "number"} or isinstance(raw.get("default"), float):
            kind = "number"
        else:
            kind = "text"

        options = []
        for choice in choices:
            value = str(choice)
            options.append({"value": value, "label": value, "description": ""})

        description = str(raw.get("description") or "")
        fields.append({
            "key": key,
            "label": str(raw.get("label") or key.replace("_", " ").title()),
            "kind": kind,
            "description": description,
            "placeholder": str(raw.get("placeholder") or ""),
            "required": bool(raw.get("required", False)),
            "default": raw.get("default", ""),
            "options": options,
            "url": str(raw.get("url") or ""),
            "when": raw.get("when") if isinstance(raw.get("when"), dict) else None,
            "minimum": raw.get("minimum"),
            "maximum": raw.get("maximum"),
            "step": raw.get("step"),
            "_env_key": str(raw.get("env_var") or "") or None,
        })

    return fields


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.debug("Failed to read JSON config from %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _read_memory_provider_existing_values(name: str) -> Dict[str, Any]:
    """Best-effort read of existing provider config across legacy/native stores."""

    hermes_home = get_hermes_home()
    values: Dict[str, Any] = {}

    # Common native provider stores.
    for path in (
        hermes_home / f"{name}.json",
        hermes_home / name / "config.json",
    ):
        values.update(_read_json_file(path))

    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    memory_cfg = cfg.get("memory") if isinstance(cfg, dict) else {}
    if isinstance(memory_cfg, dict):
        provider_cfg = memory_cfg.get(name)
        if isinstance(provider_cfg, dict):
            values.update(provider_cfg)
        legacy_cfg = memory_cfg.get("provider_config")
        if isinstance(legacy_cfg, dict):
            values = {**legacy_cfg, **values}

    # Holographic stores under plugins.hermes-memory-store.
    plugins_cfg = cfg.get("plugins") if isinstance(cfg, dict) else {}
    if name == "holographic" and isinstance(plugins_cfg, dict):
        holographic_cfg = plugins_cfg.get("hermes-memory-store")
        if isinstance(holographic_cfg, dict):
            values.update(holographic_cfg)

    return values


def _env_lookup(env_key: Optional[str]) -> str:
    if not env_key:
        return ""
    env_on_disk = load_env()
    return str(env_on_disk.get(env_key) or os.environ.get(env_key) or "")


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
    value = _field_value(field, data)
    return value not in (None, "")


def _field_visible(
    field: Dict[str, Any],
    data: Dict[str, Any],
    fields_by_key: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    when = field.get("when")
    if not isinstance(when, dict) or not when:
        return True
    for dep_key, expected in when.items():
        dep_field = (fields_by_key or {}).get(str(dep_key)) or {
            "key": str(dep_key),
            "kind": "text",
            "default": "",
            "_env_key": None,
        }
        actual = _field_value(dep_field, data)
        if str(actual) != str(expected):
            return False
    return True


def _public_memory_provider_field(field: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "key": field["key"],
        "label": field["label"],
        "kind": field["kind"],
        "description": field["description"],
        "placeholder": field["placeholder"],
        "required": field["required"],
        "value": "" if field["kind"] == "secret" else _field_value(field, data),
        "is_set": _field_is_set(field, data),
        "options": field.get("options", []),
        "url": field.get("url", ""),
        "when": field.get("when"),
        "minimum": field.get("minimum"),
        "maximum": field.get("maximum"),
        "step": field.get("step"),
    }
    return entry


def _memory_provider_payload(name: str, provider: Any) -> Dict[str, Any]:
    data = _read_memory_provider_existing_values(name)
    fields = [
        _public_memory_provider_field(field, data)
        for field in _normalize_memory_provider_schema(name, provider)
    ]
    return {
        "name": name,
        "label": _memory_provider_label(name),
        "fields": fields,
        "setup": _memory_provider_setup_info(name),
    }


def _coerce_schema_field(field: Dict[str, Any], raw: Any) -> Any:
    if field["kind"] == "boolean":
        return _coerce_bool(raw, default=_coerce_bool(_field_default(field), default=False))

    if field["kind"] in {"integer", "number"}:
        value = raw if raw is not None and raw != "" else _field_default(field)
        try:
            if isinstance(value, bool):
                raise ValueError
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError
            if field["kind"] == "integer":
                if not parsed.is_integer():
                    raise ValueError
                result: int | float = int(parsed)
            else:
                result = parsed
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid numeric value for '{field['key']}'") from exc

        minimum = field.get("minimum")
        maximum = field.get("maximum")
        if minimum is not None and result < minimum:
            raise ValueError(f"'{field['key']}' must be at least {minimum}")
        if maximum is not None and result > maximum:
            raise ValueError(f"'{field['key']}' must be at most {maximum}")
        return result

    value = str(raw if raw is not None else "").strip()
    if field["kind"] == "select":
        if not value:
            value = str(_field_default(field))
        allowed = {opt["value"] for opt in field.get("options", [])}
        if value not in allowed:
            raise ValueError(f"Invalid value for '{field['key']}'")
        return value

    return value or _field_default(field)


def _save_memory_provider_native_config(name: str, provider: Any, values: Dict[str, Any]) -> None:
    if provider is not None and hasattr(provider, "save_config"):
        try:
            from agent.memory_provider import MemoryProvider as _BaseMemoryProvider
        except Exception:
            provider.save_config(values, str(get_hermes_home()))
            return
        if type(provider).save_config is not _BaseMemoryProvider.save_config:
            provider.save_config(values, str(get_hermes_home()))
            return

    cfg = load_config()
    memory_cfg = cfg.get("memory")
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
        cfg["memory"] = memory_cfg
    current = memory_cfg.get(name)
    if not isinstance(current, dict):
        current = {}
    current.update(values)
    memory_cfg[name] = current
    save_config(cfg)


def _memory_provider_is_configured(name: str, provider: Any) -> bool:
    data = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    visible_fields = [
        field for field in fields if _field_visible(field, data, fields_by_key)
    ]
    required_fields = [field for field in visible_fields if field.get("required")]
    if not required_fields:
        return True
    return all(_field_is_set(field, data) for field in required_fields)


def _discover_memory_provider_statuses() -> List[Dict[str, Any]]:
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

    cfg = load_config()
    active = ""
    mem = cfg.get("memory")
    if isinstance(mem, dict):
        active = _normalize_memory_provider_name(mem.get("provider"))
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
        provider = None if row["missing"] else _load_memory_provider(name)
        setup = _memory_provider_setup_info(name)
        configured = False if row["missing"] else _memory_provider_is_configured(name, provider)
        schema_fields = [] if row["missing"] else _normalize_memory_provider_schema(name, provider)
        if row["missing"]:
            status = "missing"
        elif not row["available"] and not setup.get("dependencies_installed", True):
            status = "unavailable"
        elif not configured:
            status = "needs_config"
        elif not row["available"] and schema_fields:
            status = "needs_config"
        elif not row["available"]:
            status = "unavailable"
        else:
            status = "ready"
        providers.append({
            "name": name,
            "description": row["description"],
            "available": row["available"],
            "configured": configured,
            "status": status,
            "setup": setup,
        })
    return providers


def _require_memory_provider_ready(name: str) -> None:
    if not name:
        return
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    row = statuses.get(name)
    if row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown memory provider '{name}'.",
        )
    if row["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Memory provider '{name}' is not ready "
                f"({row['status'].replace('_', ' ')}). Configure it in the dashboard first."
            ),
        )


def _write_memory_provider_config_values(
    name: str,
    provider: Any,
    values: Dict[str, Any],
) -> None:
    existing = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    config_values: Dict[str, Any] = {}
    secrets: Dict[str, str] = {}

    for field in fields:
        if not _field_visible(field, {**existing, **config_values}, fields_by_key):
            continue

        if field["kind"] == "secret":
            submitted = str(values.get(field["key"]) or "").strip()
            if submitted and field.get("_env_key"):
                secrets[str(field["_env_key"])] = submitted
            continue

        raw = (
            values[field["key"]]
            if field["key"] in values
            else existing.get(field["key"], _field_default(field))
        )
        config_values[field["key"]] = _coerce_schema_field(field, raw)

    _save_memory_provider_native_config(name, provider, config_values)

    for env_key, secret in secrets.items():
        save_env_value(env_key, secret)


_MEMORY_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _require_valid_memory_provider_name(name: str) -> None:
    """Reject provider names that could traverse outside the plugin dirs.

    ``name`` is interpolated into filesystem paths by ``find_provider_dir()``
    and gates which plugin manifest's setup commands run. A strict charset
    allowlist (no path separators, no dots) makes traversal impossible
    regardless of how the downstream lookup evolves.
    """
    if not _MEMORY_PROVIDER_NAME_RE.fullmatch(name or ""):
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")


# --- Monkeypatch-transparent rebinding ------------------------------------
# ``tests/hermes_cli/test_plugins_hub_perf_guard.py`` patches
# ``web_server._discover_memory_provider_statuses`` and expects in-module
# callers (e.g. ``_require_memory_provider_ready``) to see the patch, exactly
# as they did in the pre-extraction single-module layout.  The original def
# is kept for web_server's legacy re-export; the public name is rebound to a
# late proxy so every in-module call re-reads the live attribute on web_server.
_discover_memory_provider_statuses_impl = _discover_memory_provider_statuses
_discover_memory_provider_statuses = late("_discover_memory_provider_statuses")
