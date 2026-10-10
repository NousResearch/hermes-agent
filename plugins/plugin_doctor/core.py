from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# ``kind`` is optional in the loader: ``PluginManager._parse_manifest`` defaults a
# missing or unknown kind to ``standalone`` (``hermes_cli/plugins.py``), and the
# documented starter manifest omits it. Only require what the loader actually
# treats as mandatory.
REQUIRED_PLUGIN_KEYS = {"name", "version", "description"}
DEFAULT_PLUGINS_DIR = "plugins"
DEFAULT_PLUGIN_KIND = "standalone"
VALID_PLUGIN_KINDS = {"standalone", "backend", "exclusive", "platform", "model-provider"}

# Category traversal mirrors the loader: a directory without its own manifest is
# a category namespace, scanned one level deeper. Depth is capped at two
# segments, so ``<root>/a/b/c/`` is ignored.
MAX_CATEGORY_DEPTH = 1
IGNORED_PLUGIN_DIRS = {"__pycache__", "hermes_test"}

SCAN_SCHEMA = {
    "description": "Validate Hermes plugin manifests and import/register entry points.",
    "type": "object",
    "properties": {
        "plugins_dir": {
            "type": "string",
            "description": "Directory containing plugin subdirectories.",
            "default": DEFAULT_PLUGINS_DIR,
        },
        "include_import_check": {
            "type": "boolean",
            "description": (
                "Statically verify each plugin __init__.py declares a callable "
                "register(ctx) without executing it."
            ),
            "default": True,
        },
        "unsafe_execute_import_check": {
            "type": "boolean",
            "description": (
                "UNSAFE: actually execute each plugin __init__.py via exec_module "
                "to confirm it imports. Default false keeps the scan read-only."
            ),
            "default": False,
        },
    },
}


class _MetadataError(ValueError):
    pass


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise _MetadataError(f"PyYAML is required to parse {path.name}: {exc}") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _MetadataError(f"failed to parse {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise _MetadataError(f"{path.name} must contain a YAML object")
    return data


def _plugin_dirs(root: Path, depth: int = 0) -> list[Path]:
    """Return plugin directories under *root*, including category layouts.

    Mirrors ``PluginManager._scan_directory_level``: a child directory that owns
    a ``plugin.yaml`` / ``plugin.yml`` is a plugin, while a child without one is a
    category namespace scanned one level deeper. Traversal is capped at
    ``MAX_CATEGORY_DEPTH`` extra segments so ``<root>/a/b/c/`` is ignored.
    """
    if not root.is_dir():
        return []

    found: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.name in IGNORED_PLUGIN_DIRS:
            continue
        if (path / "plugin.yaml").is_file() or (path / "plugin.yml").is_file():
            found.append(path)
            continue
        # A directory with no manifest that also carries no plugin surface is
        # still reported as an error candidate: the loader ignores it, but the
        # doctor must still surface a stray top-level directory so a misplaced
        # plugin is not silently invisible.
        if depth == 0 and not any(path.iterdir()):
            found.append(path)
            continue
        if depth < MAX_CATEGORY_DEPTH:
            found.extend(_plugin_dirs(path, depth + 1))
    return found


def _validate_manifest(plugin_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = plugin_dir / "plugin.yaml"
    if not manifest_path.is_file():
        return {}, ["missing plugin.yaml"], warnings
    try:
        manifest = _load_yaml(manifest_path)
    except _MetadataError as exc:
        return {}, [str(exc)], warnings

    missing = sorted(REQUIRED_PLUGIN_KEYS - set(manifest))
    if missing:
        errors.append(f"missing required manifest key(s): {', '.join(missing)}")
    if manifest.get("name") and str(manifest["name"]).replace("-", "_") != plugin_dir.name:
        warnings.append(
            "manifest name does not match directory name after dash/underscore normalization"
        )
    raw_kind = manifest.get("kind")
    if raw_kind is None:
        # Optional in the loader: absent kind means standalone.
        manifest["kind"] = DEFAULT_PLUGIN_KIND
    elif not isinstance(raw_kind, str) or raw_kind.strip().lower() not in VALID_PLUGIN_KINDS:
        # The loader warns and falls back to standalone rather than failing.
        warnings.append(
            f"unknown kind {raw_kind!r} (valid: {', '.join(sorted(VALID_PLUGIN_KINDS))}); "
            f"treating as '{DEFAULT_PLUGIN_KIND}'"
        )
        manifest["kind"] = DEFAULT_PLUGIN_KIND
    else:
        manifest["kind"] = raw_kind.strip().lower()
    for list_key in ("provides_tools", "provides_cli"):
        value = manifest.get(list_key, []) 
        if value is not None and not isinstance(value, list):
            errors.append(f"{list_key} must be a list")
    return manifest, errors, warnings


def _check_register_static(plugin_dir: Path) -> tuple[bool, str]:
    """Verify ``register(ctx)`` exists without executing plugin code.

    The doctor scan is advertised as read-only, so the default path parses the
    module's AST instead of importing it. That means no module-level plugin code
    runs during a scan.
    """
    init_path = plugin_dir / "__init__.py"
    if not init_path.is_file():
        return False, "missing __init__.py"
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    except Exception as exc:  # pragma: no cover - unreadable file path
        return False, f"unreadable __init__.py: {type(exc).__name__}: {exc}"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "register":
            return True, "ok"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "register":
                    return True, "ok"
    return False, "register(ctx) is missing or not callable"


def _check_import_execute(plugin_dir: Path) -> tuple[bool, str]:
    """Actually import the plugin. Runs its module-level code — opt-in only."""
    init_path = plugin_dir / "__init__.py"
    if not init_path.is_file():
        return False, "missing __init__.py"
    module_name = f"_hermes_plugin_doctor_{plugin_dir.name}"
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            return False, "could not create import spec"
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}: {exc}"
    if not callable(getattr(module, "register", None)):
        return False, "register(ctx) is missing or not callable"
    return True, "ok"


def scan_plugins(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    root = Path(str(values.get("plugins_dir") or DEFAULT_PLUGINS_DIR)).expanduser()
    include_import_check = bool(values.get("include_import_check", True))
    unsafe_execute = bool(values.get("unsafe_execute_import_check", False))
    plugins: list[dict[str, Any]] = []
    duplicate_tools: dict[str, list[str]] = {}
    duplicate_cli: dict[str, list[str]] = {}

    for plugin_dir in _plugin_dirs(root):
        manifest, errors, warnings = _validate_manifest(plugin_dir)
        # Capture manifest-validation errors before any import-check detail is
        # appended, so ``manifest_ok`` reflects the manifest alone.
        manifest_errors = list(errors)
        tools = [str(item) for item in manifest.get("provides_tools", []) or []]
        cli = [str(item) for item in manifest.get("provides_cli", []) or []]
        for tool in tools:
            duplicate_tools.setdefault(tool, []).append(plugin_dir.name)
        for command in cli:
            duplicate_cli.setdefault(command, []).append(plugin_dir.name)

        import_result: dict[str, Any] | None = None
        if include_import_check:
            ok, detail = (
                _check_import_execute(plugin_dir)
                if unsafe_execute
                else _check_register_static(plugin_dir)
            )
            import_result = {
                "ok": ok,
                "detail": detail,
                "mode": "execute" if unsafe_execute else "static",
            }
            if not ok:
                errors.append(detail)

        plugins.append(
            {
                "name": str(manifest.get("name") or plugin_dir.name),
                "path": str(plugin_dir),
                "kind": str(manifest.get("kind") or DEFAULT_PLUGIN_KIND),
                "manifest_ok": not manifest_errors,
                "errors": errors,
                "warnings": warnings,
                "provides_tools": tools,
                "provides_cli": cli,
                "import": import_result,
            }
        )

    tool_conflicts = {key: value for key, value in duplicate_tools.items() if len(value) > 1}
    cli_conflicts = {key: value for key, value in duplicate_cli.items() if len(value) > 1}
    for plugin in plugins:
        for tool in plugin["provides_tools"]:
            if tool in tool_conflicts:
                plugin["errors"].append(f"duplicate tool name: {tool}")
        for command in plugin["provides_cli"]:
            if command in cli_conflicts:
                plugin["errors"].append(f"duplicate CLI command: {command}")

    error_count = sum(len(plugin["errors"]) for plugin in plugins)
    warning_count = sum(len(plugin["warnings"]) for plugin in plugins)
    return {
        "ok": error_count == 0,
        "plugins_dir": str(root),
        "plugin_count": len(plugins),
        "error_count": error_count,
        "warning_count": warning_count,
        "tool_conflicts": tool_conflicts,
        "cli_conflicts": cli_conflicts,
        "plugins": plugins,
    }


def handle_slash(raw_args: str) -> str:
    parts = (raw_args or "").split()
    plugins_dir = parts[0] if parts else DEFAULT_PLUGINS_DIR
    return to_json(scan_plugins({"plugins_dir": plugins_dir}))
