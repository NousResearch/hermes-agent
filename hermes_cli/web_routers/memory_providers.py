"""Memory-provider setup dashboard routes (schema, existing values, external dependency install).

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import contextlib
import json
import logging
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import late
from hermes_cli.web_server_dashboard import _invalidate_plugins_hub_cache
from hermes_cli.web_server_memory import (
    _coerce_bool, _field_default, _field_is_set, _field_value, _field_visible, _load_memory_provider, _memory_provider_manifest, _memory_provider_setup_info, _memory_provider_setup_manifest, _normalize_memory_provider_schema, _read_memory_provider_existing_values, _require_memory_provider_ready, _run_setup_command,
)
from hermes_cli.web_models import MemoryProviderConfigUpdate, MemoryProviderSetupRequest
from hermes_cli.web_routers._common import scoped_to_thread
from plugins.memory.config_schema import (
    STORAGE_HONCHO_HOST_BLOCK, ProviderConfigSchema, ProviderField, get_provider_config_schema,
)

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_discover_memory_provider_statuses = late("_discover_memory_provider_statuses", "hermes_cli.web_server_memory")
get_hermes_home = late("get_hermes_home", "hermes_cli.config")
load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")
save_env_value = late("save_env_value", "hermes_cli.config")
_dependency_importable = late("_dependency_importable", "hermes_cli.web_server_memory")
load_env = late("load_env", "hermes_cli.config")
# Sentinel: remove this key so it falls back to the host or built-in default.
_UNSET: Any = object()

_MEMORY_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _unknown_provider(name: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")


@contextlib.contextmanager
def _value_errors_as_http(log_msg: str, name: str, *, passthrough_http: bool = True):
    """``ValueError`` -> 400 with its text; any other error -> logged 500 (an
    ``HTTPException`` passes through unless ``passthrough_http`` is False)."""
    try:
        yield
    except HTTPException:
        if passthrough_http:
            raise
        _log.exception(log_msg, name)
        raise HTTPException(status_code=500, detail="Internal server error")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception(log_msg, name)
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Declared-schema surface (plugins.memory.config_schema) ────────────────────

def _coerce_field_value(field: ProviderField, raw: str) -> Any:
    """Coerce a submitted non-secret string to its native JSON type.

    A bool is stored as JSON ``false`` rather than ``"false"`` (truthy). Blank
    number/json/text clears the key (``_UNSET``); raises ``ValueError`` on malformed input.
    """
    value = (raw or "").strip()
    kind = field.kind
    if kind == "select":
        value = value or field.default
        if value not in field.allowed_values():
            raise ValueError(f"Invalid value for '{field.key}'")
        return value
    if kind == "bool":
        from utils import is_truthy_value
        return is_truthy_value(value)
    if not value:
        return _UNSET
    if kind == "number":
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid number for '{field.key}'") from exc
        return int(number) if number.is_integer() else number
    if kind == "json":
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid JSON for '{field.key}'") from exc
        if isinstance(parsed, (dict, list)):
            return parsed
        raise ValueError(f"'{field.key}' must be a JSON object or array")
    return value


def _read_json_dict(path: Path, what: str) -> Dict[str, Any]:
    """Read a JSON object from ``path``; missing/unreadable/non-dict -> ``{}``."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.warning("Failed to read %s from %s", what, path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _flat_json_path(provider: ProviderConfigSchema) -> Path:
    return get_hermes_home() / provider.name / "config.json"


def _read_flat_json(provider: ProviderConfigSchema) -> Dict[str, Any]:
    return _read_json_dict(_flat_json_path(provider), "memory provider config")


def _honcho_resolvers():
    """Lazily import the Honcho plugin's resolvers (optional plugin)."""
    from plugins.memory.honcho.client import _host_block, resolve_active_host, resolve_config_path
    return resolve_active_host, resolve_config_path, _host_block


def _save_submitted_secrets(provider: ProviderConfigSchema, values: Dict[str, str]) -> list:
    """Persist each non-blank secret submission to the env store (when the field has an
    ``env_key``); return the ``(field, submitted)`` pairs for backend-specific handling."""
    saved = []
    for field in provider.fields:
        submitted = (values.get(field.key) or "").strip() if field.is_secret else ""
        if not submitted:
            continue
        if field.env_key:
            save_env_value(field.env_key, submitted)
        saved.append((field, submitted))
    return saved


def _apply_field_values(provider: ProviderConfigSchema, values: Dict[str, str], target_for) -> None:
    """Apply submitted non-secret fields to their backend dict, in place.

    Only keys present in ``values`` are touched, so a partial save never
    clobbers fields owned by another surface. ``_UNSET`` clears the key (and
    its aliases) so it falls back to the host/default mapping.
    """
    for field in provider.fields:
        if field.is_secret or field.key not in values:
            continue
        target = target_for(field)
        coerced = _coerce_field_value(field, values[field.key])
        if coerced is _UNSET:
            for key in (field.key, *field.aliases):
                target.pop(key, None)
        else:
            target[field.key] = coerced


def _write_json_0600(path: Path, data: Dict[str, Any]) -> None:
    from utils import atomic_json_write
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data, mode=0o600)


def _write_provider_flat(provider: ProviderConfigSchema, values: Dict[str, str]) -> None:
    existing = _read_flat_json(provider)
    _save_submitted_secrets(provider, values)
    _apply_field_values(provider, values, lambda field: existing)
    _write_json_0600(_flat_json_path(provider), existing)


def _write_provider_honcho(provider: ProviderConfigSchema, values: Dict[str, str]) -> None:
    """Persist submitted fields to Honcho's real config for the active host (partial
    saves touch only submitted keys; blank text clears a key — see ``_apply_field_values``)."""
    from plugins.memory.honcho.oauth import ACCESS_TOKEN_PREFIX, _config_refresh_lock

    resolve_active_host, resolve_config_path, host_block_of = _honcho_resolvers()
    host = resolve_active_host()
    # Write the file reads resolve, or a save shadows it with a sparse copy.
    path = resolve_config_path()

    # OAuth rotation is single-use; an unlocked RMW here can revoke the grant.
    with _config_refresh_lock(path):
        cfg = _read_json_dict(path, "Honcho config")
        hosts = cfg.get("hosts")
        cfg["hosts"] = hosts = hosts if isinstance(hosts, dict) else {}
        # Update the block reads resolve (legacy dot-form included), never shadow it.
        existing = host_block_of(cfg, host)
        host_key = next((k for k, v in hosts.items() if v is existing), host) if existing else host
        host_block = hosts.setdefault(host_key, existing)

        for field, submitted in _save_submitted_secrets(provider, values):
            # Persist where the client reads first; an OAuth token owns that slot.
            stored = host_block.get(field.key)
            if not (isinstance(stored, str) and stored.startswith(ACCESS_TOKEN_PREFIX)):
                host_block[field.key] = submitted

        _apply_field_values(provider, values, lambda field: host_block if field.scope == "host" else cfg)
        _write_json_0600(path, cfg)


def _serialize_field_value(field: ProviderField, value: Any) -> str:
    """Render a stored native value as the string the generic UI edits (``None`` = key
    absent -> declared default; bools -> "true"/"false"; JSON containers re-encoded)."""
    if value is None:
        return field.default
    if field.kind == "bool":
        from utils import is_truthy_value
        return "true" if is_truthy_value(value) else "false"
    if field.kind == "json" and isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _read_field(field: ProviderField, sources: tuple, env: Dict[str, str]) -> Any:
    """Stored native value from the first source holding it, else ``None``.

    Presence (``key in source``) decides, not truthiness, so a stored ``False``
    or ``0`` survives instead of being mistaken for "unset".
    """
    for source in sources:
        for source_key in (field.key, *field.aliases):
            if source_key in source and source[source_key] is not None:
                return source[source_key]
    for env_key in field.env_fallbacks:
        if env.get(env_key):
            return env[env_key]
    return None


def _declared_field_is_set(field: ProviderField, sources: tuple, env: Dict[str, str]) -> bool:
    if any(env_key and env.get(env_key) for env_key in (field.env_key, *field.env_fallbacks)):
        return True
    return any(source.get(k) for source in sources for k in (field.key, *field.aliases))


def _declared_provider_payload(provider: ProviderConfigSchema) -> Dict[str, Any]:
    env = load_env()
    is_honcho = provider.storage == STORAGE_HONCHO_HOST_BLOCK
    if is_honcho:
        resolve_active_host, resolve_config_path, host_block_of = _honcho_resolvers()
        host = resolve_active_host()
        raw = _read_json_dict(resolve_config_path(), "Honcho config")
        host_block = host_block_of(raw, host)

        def sources_for(field: ProviderField) -> tuple:
            return (host_block, raw) if field.scope == "host" else (raw,)
    else:
        host, data = "", _read_flat_json(provider)

        def sources_for(field: ProviderField) -> tuple:
            return (data,)

    fields: List[Dict[str, Any]] = []
    for field in provider.fields:
        entry = {k: getattr(field, k) for k in ("key", "label", "kind", "description", "info", "placeholder", "inline", "group")}
        entry["options"] = [{"value": o.value, "label": o.label, "description": o.description} for o in field.options]
        sources = sources_for(field)
        if field.is_secret:
            entry["value"] = ""  # secrets are write-only over the API
            entry["is_set"] = _declared_field_is_set(field, sources, env)
            fields.append(entry)
            continue
        native = _read_field(field, sources, env)
        if is_honcho and not field.placeholder and field.key in {"workspace", "aiPeer"}:
            # Blank fields surface the resolved host Honcho will actually use.
            entry["placeholder"] = host
        value = _serialize_field_value(field, native)
        if field.kind == "select" and value not in field.allowed_values():
            value = field.default
        entry["value"] = value
        # Presence, not truthiness — a stored False/0 is still "set".
        entry["is_set"] = native is not None if is_honcho else bool(value)
        fields.append(entry)
    return {"name": provider.name, "label": provider.label, "docs_url": provider.docs_url, "fields": fields}


def _stringify_submitted(value: Any) -> str:
    """The declared-schema path edits strings; the dashboard may send natives."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _memory_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``config["memory"]`` as a dict, creating/replacing a non-dict value."""
    memory_config = config.get("memory")
    if not isinstance(memory_config, dict):
        memory_config = config["memory"] = {}
    return memory_config


def _update_memory_provider_config(provider: ProviderConfigSchema, values: Dict[str, str]) -> None:
    writer = _write_provider_honcho if provider.storage == STORAGE_HONCHO_HOST_BLOCK else _write_provider_flat
    writer(provider, values)
    config = load_config()
    memory_config = _memory_section(config)
    if memory_config.get("provider") != provider.name:
        memory_config["provider"] = provider.name
        save_config(config)


# ── Setup: dependency installation ────────────────────────────────────────────

def _trim_setup_output(value: Optional[str], limit: int = 4000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}\n... truncated ..."


def _command_result(
    *, kind: str, name: str, status: str, command: str = "",
    completed: Optional[subprocess.CompletedProcess] = None, error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind, "name": name, "status": status, "command": command,
        "returncode": None if completed is None else completed.returncode,
        "stdout": "" if completed is None else _trim_setup_output(completed.stdout),
        "stderr": _trim_setup_output(error or ("" if completed is None else completed.stderr)),
    }


def _install_memory_provider_pip_dependencies(dependencies: List[str]) -> List[Dict[str, Any]]:
    if not dependencies:
        return []
    missing = [dep for dep in dependencies if not _dependency_importable(dep)]
    if not missing:
        return [_command_result(kind="pip", name=", ".join(dependencies), status="already_installed")]
    # Route through the lazy-install pipeline rather than pip against
    # sys.executable: on hosted/immutable images the agent venv is sealed
    # read-only and installs must go to HERMES_LAZY_INSTALL_TARGET, which
    # install_specs also activates on sys.path so the recheck sees the packages.
    name = ", ".join(missing)
    try:
        from tools.lazy_deps import install_specs
        outcome = install_specs(missing, timeout=240)
    except Exception as exc:
        return [_command_result(kind="pip", name=name, status="failed", error=str(exc))]
    if outcome.blocked:
        return [_command_result(kind="pip", name=name, status="failed", command=outcome.command, error=outcome.reason)]
    return [_command_result(
        kind="pip", name=name, status="installed" if outcome.ok else "failed", command=outcome.command,
        completed=subprocess.CompletedProcess(
            args=outcome.command, returncode=0 if outcome.ok else 1, stdout=outcome.stdout, stderr=outcome.stderr,
        ),
    )]


def _run_setup_step(results: list, kind: str, name: str, command: str, status_of, **kwargs) -> Optional[int]:
    """Run a setup command, append its result row; returncode or None on spawn failure."""
    try:
        completed = _run_setup_command(command if kwargs.get("shell") else shlex.split(command), display=command, **kwargs)
    except Exception as exc:
        results.append(_command_result(kind=kind, name=name, status=status_of(None), command=command, error=str(exc)))
        return None
    results.append(_command_result(kind=kind, name=name, status=status_of(completed.returncode == 0), command=command, completed=completed))
    return completed.returncode


def _install_memory_provider_external_dependencies(dependencies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for dep in dependencies:
        name = dep.get("name") or "dependency"
        check_cmd = dep.get("check") or ""
        install_cmd = dep.get("install") or ""
        # Check first: "already_installed" short-circuits; a failed check is
        # "missing" when an install step can fix it, "failed" otherwise.
        if check_cmd and _run_setup_step(
            results, "external_check", name, check_cmd,
            lambda ok: "already_installed" if ok else ("missing" if install_cmd else "failed"), timeout=20,
        ) == 0:
            continue
        if not install_cmd:
            continue
        rc = _run_setup_step(
            results, "external_install", name, install_cmd, lambda ok: "installed" if ok else "failed",
            shell=True, timeout=300,
        )
        if check_cmd and rc == 0:
            _run_setup_step(results, "external_check", name, check_cmd, lambda ok: "verified" if ok else "failed", timeout=20)
    return results


def _install_memory_provider_setup(name: str) -> Dict[str, Any]:
    provider = _load_memory_provider(name)
    manifest = _memory_provider_manifest(name)
    if provider is None and not manifest:
        raise _unknown_provider(name)
    setup = _memory_provider_setup_manifest(name)
    results = _install_memory_provider_pip_dependencies(setup["pip_dependencies"])
    results.extend(_install_memory_provider_external_dependencies(setup["external_dependencies"]))
    if not results:
        results.append(_command_result(kind="setup", name=name, status="no_declared_steps"))
    ok = all(result["status"] != "failed" for result in results)
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    return {"ok": ok, "provider": name, "results": results, "status": statuses.get(name)}


# ── Legacy provider surface (provider.config_schema()) ────────────────────────

def _memory_provider_payload(name: str, provider: Any) -> Dict[str, Any]:
    data = _read_memory_provider_existing_values(name)
    fields = [
        {
            **{k: field[k] for k in ("key", "label", "kind", "description", "placeholder", "required")},
            "value": "" if field["kind"] == "secret" else _field_value(field, data),
            "is_set": _field_is_set(field, data), "options": field.get("options", []), "url": field.get("url", ""),
            **{k: field.get(k) for k in ("when", "minimum", "maximum", "step")},
        }
        for field in _normalize_memory_provider_schema(name, provider)
    ]
    return {
        "name": name, "label": name.replace("_", " ").replace("-", " ").title(), "fields": fields,
        "setup": _memory_provider_setup_info(name),
    }


def _coerce_schema_number(field: Dict[str, Any], raw: Any) -> "int | float":
    value = raw if raw is not None and raw != "" else _field_default(field)
    try:
        if isinstance(value, bool) or not math.isfinite(result := float(value)):
            raise ValueError
        if field["kind"] == "integer":
            if not result.is_integer():
                raise ValueError
            result = int(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric value for '{field['key']}'") from exc
    minimum, maximum = field.get("minimum"), field.get("maximum")
    if minimum is not None and result < minimum:
        raise ValueError(f"'{field['key']}' must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"'{field['key']}' must be at most {maximum}")
    return result


def _coerce_schema_field(field: Dict[str, Any], raw: Any) -> Any:
    kind = field["kind"]
    if kind == "boolean":
        return _coerce_bool(raw, default=_coerce_bool(_field_default(field), default=False))
    if kind in {"integer", "number"}:
        return _coerce_schema_number(field, raw)
    value = str(raw if raw is not None else "").strip()
    if kind == "select":
        value = value or str(_field_default(field))
        if value not in {opt["value"] for opt in field.get("options", [])}:
            raise ValueError(f"Invalid value for '{field['key']}'")
        return value
    return value or _field_default(field)


def _save_memory_provider_native_config(name: str, provider: Any, values: Dict[str, Any]) -> None:
    if provider is not None and hasattr(provider, "save_config"):
        try:
            from agent.memory_provider import MemoryProvider as _BaseMemoryProvider
        except Exception:
            _BaseMemoryProvider = None
        if _BaseMemoryProvider is None or type(provider).save_config is not _BaseMemoryProvider.save_config:
            provider.save_config(values, str(get_hermes_home()))
            return
    cfg = load_config()
    memory_cfg = _memory_section(cfg)
    current = memory_cfg.get(name)
    memory_cfg[name] = {**(current if isinstance(current, dict) else {}), **values}
    save_config(cfg)


def _write_memory_provider_config_values(name: str, provider: Any, values: Dict[str, Any]) -> None:
    existing = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    config_values: Dict[str, Any] = {}
    secrets: Dict[str, str] = {}
    for field in fields:
        if not _field_visible(field, {**existing, **config_values}, fields_by_key):
            continue
        key = field["key"]
        if field["kind"] == "secret":
            submitted = str(values.get(key) or "").strip()
            if submitted and field.get("_env_key"):
                secrets[str(field["_env_key"])] = submitted
            continue
        raw = values[key] if key in values else existing.get(key, _field_default(field))
        config_values[key] = _coerce_schema_field(field, raw)
    _save_memory_provider_native_config(name, provider, config_values)
    for env_key, secret in secrets.items():
        save_env_value(env_key, secret)


# ── Routes ────────────────────────────────────────────────────────────────────

def _require_valid_memory_provider_name(name: str) -> None:
    """Reject provider names that could traverse outside the plugin dirs.

    ``name`` is interpolated into filesystem paths by ``find_provider_dir()``
    and gates which plugin manifest's setup commands run; a strict charset
    allowlist (no path separators, no dots) makes traversal impossible.
    """
    if not _MEMORY_PROVIDER_NAME_RE.fullmatch(name or ""):
        raise _unknown_provider(name)


@router.get("/api/memory/providers/{name}/config")
async def get_memory_provider_config(name: str, surface: Optional[str] = None, profile: Optional[str] = None):
    _require_valid_memory_provider_name(name)

    def _run():
        # Undeclared providers (e.g. builtin) have no config surface; an
        # empty schema makes the generic panel render nothing.
        if surface == "declared":
            declared = get_provider_config_schema(name)
            if declared is None:
                return {"name": name, "label": name, "docs_url": "", "fields": []}
            return _declared_provider_payload(declared)
        provider = _load_memory_provider(name)
        if provider is None:
            return {"name": name, "label": name, "fields": [], "setup": _memory_provider_setup_info(name)}
        return _memory_provider_payload(name, provider)

    return await scoped_to_thread(profile, _run)


@router.post("/api/memory/providers/{name}/setup")
async def setup_memory_provider(name: str, body: MemoryProviderSetupRequest):
    _require_valid_memory_provider_name(name)
    provider = _load_memory_provider(name)
    if provider is None and not _memory_provider_manifest(name):
        # No discoverable plugin directory -> no manifest that could declare
        # setup commands; refuse before the command-running path. (provider
        # may be None with a manifest present when its pip deps aren't
        # installed yet — that's the setup use case.)
        raise _unknown_provider(name)
    if provider is not None and body.values:
        with _value_errors_as_http("Failed to persist memory provider setup values for %s", name, passthrough_http=False):
            _write_memory_provider_config_values(name, provider, body.values)
    _invalidate_plugins_hub_cache()
    return _install_memory_provider_setup(name)


@router.put("/api/memory/providers/{name}/config")
async def update_memory_provider_config(
    name: str, body: MemoryProviderConfigUpdate, surface: Optional[str] = None, profile: Optional[str] = None
):
    _require_valid_memory_provider_name(name)
    values = body.values or {}

    def _run():
        if surface == "declared":
            declared = get_provider_config_schema(name)
            if declared is None:
                raise _unknown_provider(name)
            _update_memory_provider_config(declared, {k: _stringify_submitted(v) for k, v in values.items()})
            _invalidate_plugins_hub_cache()
            return {"ok": True}
        provider = _load_memory_provider(name)
        if provider is None:
            raise _unknown_provider(name)
        _write_memory_provider_config_values(name, provider, values)
        _require_memory_provider_ready(name)
        config = load_config()
        _memory_section(config)["provider"] = name
        save_config(config)
        _invalidate_plugins_hub_cache()
        return {"ok": True, "active": name}

    with _value_errors_as_http("PUT /api/memory/providers/%s/config failed", name):
        return await scoped_to_thread(profile, _run)
