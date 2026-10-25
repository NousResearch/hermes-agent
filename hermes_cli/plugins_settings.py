"""Plugin-declared settings fields and real config-source binding (#46600, #87934, #144485).

A manifest ``config_schema`` describes a plugin's configurable keys; an optional ``config_source``
binds that schema to where the plugin actually stores its settings:
  - omitted → ``plugins.entries.<id>.settings`` in ``config.yaml`` (same namespace as ``ctx.get_config``)
  - ``{kind: config_section, path: <section>}`` → ``<section>`` in ``config.yaml``
  - ``{kind: json_file, path: $HERMES_HOME/<file>.json}`` → profile-scoped JSON sidecar under Hermes home

Hard rule (#144485): never expose an editable form when ``config_source`` cannot be resolved safely.
Secrets (``type: secret``) never touch ``config.yaml`` or JSON sidecars — they carry only ``env`` and
``has_value``, and clients store them via ``PUT /api/env``.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from hermes_constants import get_hermes_home
from hermes_cli.plugins_state import (
    _CONFIG_SECTION_RESERVED_ROOTS,
    _PLUGIN_SETTING_SEGMENT_RE,
    _config_section_segments,
    _field_segments,
    _nested_plugin_value,
    _plugin_relative_segments,
    _plugin_settings_entry,
    save_plugin_json_settings,
    save_plugin_section_setting,
    save_plugin_setting,
)

logger = logging.getLogger(__name__)

# manifest ``type`` → wire field type the renderer keys its component table on.
_FIELD_TYPES: dict[str, str] = {
    "str": "string", "string": "string",
    "int": "number", "integer": "number", "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "json", "array": "json", "dict": "json", "object": "json",
    "secret": "secret",
}
# wire field type → Python types a saved value must have (bool is excluded from number on purpose).
_VALUE_TYPES: dict[str, tuple] = {
    "string": (str,), "enum": (str,), "number": (int, float), "boolean": (bool,), "json": (list, dict),
}
_ENV_NAME_CLEAN_RE = re.compile(r"[^A-Z0-9]+")
_VALID_SOURCE_KINDS = frozenset({"config_section", "json_file"})
_FORBIDDEN_JSON_ROOTS = frozenset({"plugins", "profiles", ".git"})
_FORBIDDEN_JSON_FILES = frozenset({"auth.json"})
_CORE_CONFIG_SECTIONS = frozenset({
    "agent", "approvals", "auxiliary", "browser", "checkpoints", "code_execution", "command_allowlist",
    "compression", "context", "cron", "custom_providers", "dashboard", "delegation", "desktop", "display",
    "fallback_providers", "file_read_max_chars", "gateway", "logging", "mcp_servers", "memory", "moa",
    "model", "model_context_length", "platform_toolsets", "platforms", "plugins", "provider_routing",
    "security", "session_reset", "skills", "stt", "terminal", "timezone", "tool_output", "toolsets",
    "tts", "updates", "voice", "web",
})


@dataclass(frozen=True)
class ResolvedConfigSource:
    declared: bool
    valid: bool
    kind: Optional[str]
    path: str
    section_segments: tuple[str, ...] = ()
    json_path: Optional[Path] = None
    error: str = ""


def _read_manifest_bundle(plugin_dir: Optional[Path]) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], bool, Any]:
    """Return ``(manifest_data, config_schema, has_declared_source, raw_config_source)``."""
    if plugin_dir is None:
        return {}, {}, False, None
    root = Path(plugin_dir)
    yaml_file = next((root / n for n in ("plugin.yaml", "plugin.yml") if (root / n).is_file()), None)
    try:
        if yaml_file is not None:
            from utils import fast_safe_load
            data = fast_safe_load(yaml_file.read_text(encoding="utf-8-sig")) or {}
            if not isinstance(data, Mapping):
                return {}, {}, False, None
            raw_schema = data.get("config_schema")
            has_source = "config_source" in data and data.get("config_source") is not None
            raw_source = data.get("config_source") if has_source else None
            manifest_map: Mapping[str, Any] = data
        else:
            portable = root / "plugin.json"
            if not portable.is_file():
                return {}, {}, False, None
            from hermes_cli.agent_plugins import manifest_config_schema, manifest_config_source
            pdata = json.loads(portable.read_text(encoding="utf-8-sig"))
            if not isinstance(pdata, Mapping):
                return {}, {}, False, None
            raw_schema = manifest_config_schema(pdata)
            raw_source = manifest_config_source(pdata)
            has_source = raw_source is not None
            manifest_map = pdata
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        logger.debug("plugin settings: cannot read the manifest in %s: %s", plugin_dir, exc)
        return {}, {}, False, None
    schema = {str(k): v for k, v in raw_schema.items() if isinstance(v, Mapping)} if isinstance(raw_schema, Mapping) else {}
    return manifest_map, schema, has_source, raw_source


def _manifest_config_schema(plugin_dir: Optional[Path]) -> Mapping[str, Mapping[str, Any]]:
    """``config_schema`` from ``plugin.yaml`` or ``plugin.json`` vendor extension; ``{}`` when absent."""
    return _read_manifest_bundle(plugin_dir)[1]


def _normalize_json_source_path(raw_path: str) -> tuple[str, Path]:
    """Validate a ``json_file`` path under the active profile's ``get_hermes_home()``.

    Accepts ``$HERMES_HOME/<rel>.json``, ``${HERMES_HOME}/<rel>.json``, or ``<rel>.json``.
    Expands against ``get_hermes_home()`` (never ``os.environ``) so ContextVar profile scopes stay isolated.
    """
    cleaned = raw_path.strip()
    for prefix in ("${HERMES_HOME}/", "$HERMES_HOME/"):
        if cleaned.startswith(prefix):
            rel = cleaned[len(prefix):]
            break
    else:
        if cleaned.startswith(("/", "\\", "~", "$")) or "%" in cleaned or ":" in cleaned:
            raise ValueError(f"json_file path {raw_path!r} must live under $HERMES_HOME")
        rel = cleaned
    if "\\" in rel or not rel or not rel.lower().endswith(".json"):
        raise ValueError(f"json_file path {raw_path!r} must be a relative .json path under $HERMES_HOME")
    parts = [p for p in rel.split("/") if p]
    if not parts or len(parts) != len(rel.split("/")) or any(p in {".", ".."} for p in parts):
        raise ValueError(f"json_file path {raw_path!r} contains invalid path segments")
    if parts[0] in _FORBIDDEN_JSON_ROOTS or (len(parts) == 1 and parts[0].lower() in _FORBIDDEN_JSON_FILES):
        raise ValueError(f"json_file path {raw_path!r} targets a protected Hermes file")
    home = get_hermes_home()
    home_real = home.resolve(strict=False)
    target = home.joinpath(*parts)
    # Reject symlinks anywhere along the relative chain and any resolved path outside HERMES_HOME.
    probe = home
    for part in parts:
        probe = probe / part
        if probe.is_symlink():
            raise ValueError(f"json_file path {raw_path!r} must not traverse a symlink")
    if not target.resolve(strict=False).is_relative_to(home_real):
        raise ValueError(f"json_file path {raw_path!r} escapes HERMES_HOME")
    return f"$HERMES_HOME/{'/'.join(parts)}", target


def validate_config_source_spec(raw: Any) -> ResolvedConfigSource:
    """Validate a manifest ``config_source`` mapping without reading on-disk values."""
    if not isinstance(raw, Mapping):
        return ResolvedConfigSource(
            declared=True, valid=False, kind=None, path=str(raw) if raw is not None else "",
            error="config_source must be a mapping with 'kind' and 'path'",
        )
    raw_kind = str(raw.get("kind") or "").strip()
    raw_path = str(raw.get("path") or "").strip()
    known_kind = raw_kind if raw_kind in _VALID_SOURCE_KINDS else None
    if not raw_kind or known_kind is None:
        return ResolvedConfigSource(
            declared=True, valid=False, kind=None, path=raw_path,
            error=f"config_source.kind must be one of {', '.join(sorted(_VALID_SOURCE_KINDS))}",
        )
    if not raw_path:
        return ResolvedConfigSource(
            declared=True, valid=False, kind=known_kind, path="",
            error="config_source.path must be a non-empty string",
        )
    if known_kind == "config_section":
        try:
            segments = _config_section_segments(raw_path)
        except ValueError as exc:
            return ResolvedConfigSource(declared=True, valid=False, kind=known_kind, path=raw_path, error=str(exc))
        return ResolvedConfigSource(
            declared=True, valid=True, kind="config_section", path=raw_path, section_segments=segments,
        )
    try:
        display_path, target = _normalize_json_source_path(raw_path)
    except ValueError as exc:
        return ResolvedConfigSource(declared=True, valid=False, kind=known_kind, path=raw_path, error=str(exc))
    return ResolvedConfigSource(
        declared=True, valid=True, kind="json_file", path=display_path, json_path=target,
    )


def read_plugin_config_values(
    plugin_id: str, plugin_dir: Optional[Path]
) -> tuple[Mapping[str, Any], ResolvedConfigSource]:
    """Read current non-secret values and resolve the active ``config_source`` for *plugin_id*."""
    from hermes_cli.config import load_config_readonly

    _manifest, _schema, has_source, raw_source = _read_manifest_bundle(plugin_dir)
    if not has_source:
        cfg = load_config_readonly() or {}
        entry = _plugin_settings_entry(cfg, plugin_id) or {}
        raw_current = entry.get("settings")
        if not isinstance(raw_current, Mapping):
            raw_current = entry.get("config")
        current: Mapping[str, Any] = raw_current if isinstance(raw_current, Mapping) else {}
        return current, ResolvedConfigSource(
            declared=False, valid=True, kind="plugin_settings", path=f"plugins.entries.{plugin_id}.settings",
        )

    resolved = validate_config_source_spec(raw_source)
    if not resolved.valid:
        return {}, resolved

    if resolved.kind == "config_section":
        cfg = load_config_readonly() or {}
        section_val = _nested_plugin_value(cfg, resolved.section_segments, None)
        if section_val is None:
            return {}, resolved
        if not isinstance(section_val, Mapping):
            return {}, ResolvedConfigSource(
                declared=True, valid=False, kind="config_section", path=resolved.path,
                error=f"config.yaml section {resolved.path!r} is not a mapping",
            )
        return section_val, resolved

    assert resolved.json_path is not None
    if not resolved.json_path.exists():
        return {}, resolved
    try:
        parsed = json.loads(resolved.json_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {}, ResolvedConfigSource(
            declared=True, valid=False, kind="json_file", path=resolved.path,
            error=f"Cannot parse {resolved.path}: {exc}",
        )
    if not isinstance(parsed, dict):
        return {}, ResolvedConfigSource(
            declared=True, valid=False, kind="json_file", path=resolved.path,
            error=f"{resolved.path} root value must be a JSON object",
        )
    return parsed, resolved


def secret_env_name(plugin_id: str, key: str, spec: Mapping[str, Any]) -> str:
    """``.env`` variable a ``type: secret`` field is stored under: the manifest's ``env:`` or
    ``<PLUGIN_ID>_<KEY>`` upper-snaked (``image_gen/fal`` + ``api_key`` → ``IMAGE_GEN_FAL_API_KEY``)."""
    declared = str(spec.get("env") or "").strip()
    if declared:
        return declared
    return _ENV_NAME_CLEAN_RE.sub("_", f"{plugin_id}_{key}".upper()).strip("_")


def _field_type(spec: Mapping[str, Any]) -> str:
    if spec.get("secret") is True:
        return "secret"
    kind = _FIELD_TYPES.get(str(spec.get("type") or "str").lower(), "string")
    choices = spec.get("choices", spec.get("enum"))
    if kind == "string" and isinstance(choices, list) and choices:
        return "enum"
    return kind


def _key_segments_for_source(key: str, source: ResolvedConfigSource) -> tuple[str, ...]:
    if source.declared:
        return _field_segments(key)
    return _plugin_relative_segments(key)


def _lookup_current_value(current: Mapping[str, Any], key: str, segments: tuple[str, ...], default: Any) -> Any:
    if key in current:
        return current[key]
    return _nested_plugin_value(current, segments, default)


def _parse_py_trees(plugin_dir: Path) -> list[tuple[str, ast.AST]]:
    trees: list[tuple[str, ast.AST]] = []
    try:
        files = sorted(plugin_dir.glob("*.py"))
    except OSError:
        return trees
    for py_file in files:
        try:
            if not py_file.is_file() or py_file.stat().st_size > 131_072:
                continue
            src = py_file.read_text(encoding="utf-8-sig", errors="replace")
            trees.append((py_file.name, ast.parse(src)))
        except (OSError, SyntaxError, ValueError):
            continue
    return trees


def _defines_context_engine_ast(trees: list[tuple[str, ast.AST]]) -> bool:
    for _fname, tree in trees:
        bases = {"ContextEngine"} | {
            alias.asname for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names if alias.name == "ContextEngine" and alias.asname
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and "register_context_engine" in (
                getattr(node.func, "attr", None), getattr(node.func, "id", None),
            ):
                return True
            if isinstance(node, ast.ClassDef) and any(
                (getattr(b, "attr", None) or getattr(b, "id", None)) in bases for b in node.bases
            ):
                return True
    return False


def _ast_defines_method(trees: list[tuple[str, ast.AST]], method_name: str) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == method_name
        for _fname, tree in trees for node in ast.walk(tree)
    )


def _ast_uses_honcho_host_block(schema_file: Path) -> bool:
    try:
        tree = ast.parse(schema_file.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return False
    return any(
        (isinstance(n, ast.Name) and n.id == "STORAGE_HONCHO_HOST_BLOCK")
        or (isinstance(n, ast.Constant) and n.value == "honcho_host_block")
        for n in ast.walk(tree)
    )


def _binop_home_json_path(node: ast.BinOp) -> Optional[str]:
    if not (isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
        return None
    rhs = node.right.value.strip("/")
    if not rhs.endswith(".json") or "/" in rhs or rhs in _FORBIDDEN_JSON_FILES:
        return None
    lhs_func = getattr(node.left, "func", None)
    lhs_src = getattr(lhs_func, "id", None) or getattr(lhs_func, "attr", None)
    return f"$HERMES_HOME/{rhs}" if lhs_src == "get_hermes_home" else None


def _scan_trees_for_config_hints(
    trees: list[tuple[str, ast.AST]],
) -> tuple[bool, bool, list[str], list[str]]:
    uses_ctx_config = False
    imports_hermes_config = False
    config_get_keys: list[str] = []
    home_json_files: list[str] = []

    for fname, tree in trees:
        if fname == "config_schema.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hermes_cli.config"):
                imports_hermes_config = True
            elif isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", None)
                owner = getattr(node.func, "value", None)
                if attr in {"get_config", "set_config"} and isinstance(owner, ast.Name) and owner.id in {"ctx", "context"}:
                    uses_ctx_config = True
                elif attr == "get" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    config_get_keys.append(node.args[0].value)
            elif isinstance(node, ast.BinOp):
                json_hint = _binop_home_json_path(node)
                if json_hint:
                    home_json_files.append(json_hint)

    return uses_ctx_config, imports_hermes_config, config_get_keys, home_json_files


def _is_custom_config_section_name(candidate: str) -> bool:
    return bool(
        _PLUGIN_SETTING_SEGMENT_RE.fullmatch(candidate)
        and candidate.lower() not in _CONFIG_SECTION_RESERVED_ROOTS
        and candidate not in _CORE_CONFIG_SECTIONS
    )


def _ast_inferred_source(
    plugin_id: str, plugin_dir: Path, trees: list[tuple[str, ast.AST]], plugin_name: str
) -> tuple[Optional[str], str]:
    """Conservative AST + disk inspection for schemaless plugins (read-only, never editable)."""
    uses_ctx_config, imports_hermes_config, config_get_keys, home_json_files = _scan_trees_for_config_hints(trees)

    if uses_ctx_config:
        return "plugin_settings", f"plugins.entries.{plugin_id}.settings"
    if home_json_files:
        return "json_file", home_json_files[0]

    home = get_hermes_home()
    slugs = list(dict.fromkeys(
        s for s in (
            plugin_dir.name,
            plugin_name,
            plugin_dir.name.replace("-", "_"),
            plugin_name.replace("-", "_"),
        ) if s
    ))
    for slug in slugs:
        if (home / slug / "config.json").is_file():
            return "json_file", f"$HERMES_HOME/{slug}/config.json"
        if f"{slug}.json" not in _FORBIDDEN_JSON_FILES and (home / f"{slug}.json").is_file():
            return "json_file", f"$HERMES_HOME/{slug}.json"

    if imports_hermes_config:
        for candidate in config_get_keys:
            if _is_custom_config_section_name(candidate):
                return "config_section", candidate

    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly() or {}
    for slug in slugs:
        if _is_custom_config_section_name(slug) and isinstance(cfg.get(slug), Mapping):
            return "config_section", slug

    return None, ""


def _manifest_env_summary(manifest_data: Mapping[str, Any]) -> str:
    names: list[str] = []
    for field_name in ("requires_env", "optional_env"):
        raw = manifest_data.get(field_name)
        if not isinstance(raw, list):
            continue
        for item in raw:
            env_name = item.strip() if isinstance(item, str) else str(item.get("name") or "").strip() if isinstance(item, Mapping) else ""
            if env_name and env_name not in names:
                names.append(env_name)
    return f".env ({', '.join(names)})" if names else ""


def _static_provider_and_source_info(
    plugin_id: str, plugin_dir: Optional[Path], manifest_data: Mapping[str, Any], *, name: str = ""
) -> tuple[Optional[str], Optional[str], str]:
    """Return ``(settings_link, read_only_kind, read_only_path)`` without importing plugin code."""
    manifest_kind = str(manifest_data.get("kind") or "").strip().lower()
    is_model_provider = manifest_kind == "model-provider" or plugin_id.startswith("model-providers/")

    if plugin_dir is None or not Path(plugin_dir).is_dir():
        env_path = _manifest_env_summary(manifest_data)
        return ("providers" if is_model_provider else None), ("env" if env_path else None), env_path

    root = Path(plugin_dir)
    schema_py = root / "config_schema.py"
    has_memory_schema_py = schema_py.is_file()
    is_memory = has_memory_schema_py
    if not is_memory:
        try:
            from plugins.memory import _defines_memory_provider
            is_memory = _defines_memory_provider(root)
        except (ImportError, OSError, SyntaxError, ValueError):
            is_memory = False

    trees = _parse_py_trees(root)
    is_context_engine = not is_memory and _defines_context_engine_ast(trees)

    settings_link: Optional[str] = (
        "memory" if is_memory else "context" if is_context_engine else "providers" if is_model_provider else None
    )

    if has_memory_schema_py:
        if _ast_uses_honcho_host_block(schema_py):
            return settings_link, None, ""
        return settings_link, "json_file", f"$HERMES_HOME/{root.name}/config.json"

    if is_memory:
        home = get_hermes_home()
        if (home / root.name / "config.json").is_file():
            return settings_link, "json_file", f"$HERMES_HOME/{root.name}/config.json"
        if (home / f"{root.name}.json").is_file():
            return settings_link, "json_file", f"$HERMES_HOME/{root.name}.json"
        from hermes_cli.config import load_config_readonly
        mem_cfg = (load_config_readonly() or {}).get("memory")
        if (isinstance(mem_cfg, Mapping) and isinstance(mem_cfg.get(root.name), Mapping)) or (
            _ast_defines_method(trees, "get_config_schema") and not _ast_defines_method(trees, "save_config")
        ):
            return settings_link, "config_section", f"memory.{root.name}"

    inferred_kind, inferred_path = _ast_inferred_source(plugin_id, root, trees, name or root.name)
    if inferred_kind:
        return settings_link, inferred_kind, inferred_path

    env_path = _manifest_env_summary(manifest_data)
    if env_path:
        return settings_link, "env", env_path

    return settings_link, None, ""


def _build_source_payload(
    *, kind: Optional[str], path: str, editable: bool, settings_link: Optional[str]
) -> Optional[dict[str, Any]]:
    if not editable and kind is None and not path and settings_link is None:
        return None
    out: dict[str, Any] = {"path": path, "editable": editable}
    if kind is not None:
        out["kind"] = kind
    if settings_link is not None:
        out["settings_link"] = settings_link
    return out


def plugin_config_surface(
    plugin_id: str, plugin_dir: Optional[Path], *, name: str = ""
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """Return ``(settings_schema_fields, config_source_metadata)`` for ``plugins.manage`` listing."""
    manifest_data, schema, _has_source, _raw_source = _read_manifest_bundle(plugin_dir)
    settings_link, inferred_kind, inferred_path = _static_provider_and_source_info(
        plugin_id, plugin_dir, manifest_data, name=name
    )
    current, resolved = read_plugin_config_values(plugin_id, plugin_dir)

    # Hard rule (#144485): an unresolvable manifest config_source never exposes editable fields.
    if not resolved.valid:
        return [], _build_source_payload(
            kind=resolved.kind, path=resolved.path, editable=False, settings_link=settings_link
        )

    if not schema:
        if resolved.declared:
            return [], _build_source_payload(
                kind=resolved.kind, path=resolved.path, editable=False, settings_link=settings_link
            )
        return [], _build_source_payload(
            kind=inferred_kind, path=inferred_path, editable=False, settings_link=settings_link
        )

    from hermes_cli.config import get_env_value

    fields: list[dict[str, Any]] = []
    for key, spec in schema.items():
        try:
            segments = _key_segments_for_source(key, resolved)
        except ValueError:
            continue
        kind = _field_type(spec)
        field: dict[str, Any] = {
            "key": key, "type": kind,
            "label": str(spec.get("label") or spec.get("title") or key),
            "description": str(spec.get("description") or ""),
            "required": bool(spec.get("required")),
        }
        if kind == "secret":
            env = secret_env_name(plugin_id, key, spec)
            field.update({"env": env, "has_value": get_env_value(env) is not None})
        else:
            choices = spec.get("choices", spec.get("enum"))
            if kind == "enum":
                field["choices"] = [str(c) for c in choices]
            if "default" in spec:
                field["default"] = spec["default"]
            field["value"] = _lookup_current_value(current, key, segments, spec.get("default"))
        fields.append(field)

    if not fields:
        kind_out = resolved.kind if resolved.declared else inferred_kind
        path_out = resolved.path if resolved.declared else inferred_path
        return [], _build_source_payload(
            kind=kind_out, path=path_out, editable=False, settings_link=settings_link
        )

    return fields, _build_source_payload(
        kind=resolved.kind, path=resolved.path, editable=True, settings_link=settings_link
    )


def plugin_settings_fields(plugin_id: str, plugin_dir: Optional[Path]) -> list[dict[str, Any]]:
    """Renderable settings fields for one plugin: schema + the current value of each key."""
    return plugin_config_surface(plugin_id, plugin_dir)[0]


def save_plugin_settings(plugin_id: str, plugin_dir: Optional[Path], values: Mapping[str, Any]) -> list[str]:
    """Write ``values`` (``{key: value}``) for the plugin's schema keys; returns the keys written.

    Raises ``ValueError`` on an unresolvable ``config_source``, an unknown key, a type mismatch,
    an enum value outside ``choices``, or a secret; ``PermissionError`` propagates from managed scopes.
    """
    _manifest, schema, _has_source, _raw_source = _read_manifest_bundle(plugin_dir)
    _current, resolved = read_plugin_config_values(plugin_id, plugin_dir)
    if not resolved.valid:
        raise ValueError(resolved.error or f"Plugin {plugin_id!r} config_source cannot be resolved")

    plan: list[tuple[str, tuple[str, ...], Any]] = []
    for key, value in values.items():
        spec = schema.get(str(key))
        if spec is None:
            raise ValueError(f"{key!r} is not declared in the plugin's config_schema")
        kind = _field_type(spec)
        if kind == "secret":
            raise ValueError(f"{key!r} is a secret; it is stored in .env, not config.yaml")
        expected = _VALUE_TYPES[kind]
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            raise ValueError(f"{key!r} should be {kind} (got {type(value).__name__})")
        if kind == "enum" and value not in [str(c) for c in spec.get("choices", spec.get("enum"))]:
            raise ValueError(f"{key!r} must be one of the declared choices")
        plan.append((str(key), _key_segments_for_source(str(key), resolved), value))

    if resolved.kind == "config_section":
        for _key, segments, value in plan:
            save_plugin_section_setting(resolved.section_segments, segments, value)
    elif resolved.kind == "json_file":
        assert resolved.json_path is not None
        if plan:
            save_plugin_json_settings(resolved.json_path, [(segments, value) for _key, segments, value in plan])
    else:
        for _key, segments, value in plan:
            save_plugin_setting(plugin_id, segments, value)
    return [key for key, _segments, _value in plan]
