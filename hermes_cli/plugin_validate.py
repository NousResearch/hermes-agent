"""``hermes plugins validate`` — admission checks for a plugin directory.

Catalog admission performs static manifest and Python registration checks. Candidate
modules are parsed, never imported or executed: a subprocess alone is not a sandbox.
Static results do not certify runtime behavior or plugin safety.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.plugin_validate_desktop import check_desktop_surface

_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CONFIG_TYPES = {
    "str", "string", "int", "integer", "float", "number",
    "bool", "boolean", "list", "array", "dict", "mapping", "map",
}


@dataclass
class ValidationReport:
    """Result of validating one plugin directory."""

    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[str]:
        return [detail or name for name, ok, detail in self.checks if not ok]

    @property
    def ok(self) -> bool:
        return all(ok for _name, ok, _detail in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": name, "ok": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "warnings": list(self.warnings),
        }


# ─── Static checks ───────────────────────────────────────────────────────────


def _requires_hermes_spec_valid(spec: str) -> bool:
    """Strictly validate a ``requires_hermes`` spec.

    Unlike :func:`hermes_cli.plugins_manifest.version_satisfies` (permissive at load
    time), validation REJECTS clauses whose version segment doesn't parse —
    a typo'd spec should fail admission, not silently gate nothing.
    """
    from hermes_cli.plugins_manifest import _VERSION_COMPARATOR_RE, _version_tuple

    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = _VERSION_COMPARATOR_RE.match(clause)
        target = m.group(2) if m else clause
        if _version_tuple(target) is None:
            return False
    return True


def _check_manifest_fields(report: ValidationReport, manifest: dict) -> None:
    missing = [
        f for f in ("name", "version", "description") if not manifest.get(f)
    ]
    if missing:
        report.add(
            "manifest fields",
            False,
            f"plugin.yaml missing required field(s): {', '.join(missing)}",
        )
    else:
        report.add("manifest fields", True, "name, version, description present")


def _check_requires_hermes(report: ValidationReport, manifest: dict) -> None:
    spec = str(manifest.get("requires_hermes") or "").strip()
    if not spec:
        report.add("requires_hermes", True, "not declared")
        return
    if _requires_hermes_spec_valid(spec):
        report.add("requires_hermes", True, f"spec {spec!r} parses")
    else:
        report.add(
            "requires_hermes",
            False,
            f"requires_hermes spec {spec!r} does not parse "
            "(expected e.g. \">=0.19\" or \">=0.19, <1.0\")",
        )


def _check_config_spec(report: ValidationReport, manifest: dict) -> None:
    """Validate the manifest ``config_schema`` mapping (#64165 format)."""
    raw = manifest.get("config_schema")
    if raw in (None, [], {}):
        report.add("config schema", True, "not declared")
        return
    problems: List[str] = []
    if not isinstance(raw, dict):
        problems.append("config_schema: must be a mapping of key -> spec")
    else:
        for skey, spec in raw.items():
            if not isinstance(spec, dict):
                problems.append(
                    f"config_schema.{skey}: must be a mapping (e.g. {{type: str}})"
                )
                continue
            typ = spec.get("type")
            if typ is not None and str(typ).lower() not in _CONFIG_TYPES:
                problems.append(
                    f"config_schema.{skey}: type must be one of "
                    f"{'/'.join(sorted(_CONFIG_TYPES))}"
                )
            required = spec.get("required")
            if required is not None and not isinstance(required, bool):
                problems.append(
                    f"config_schema.{skey}: required must be a boolean"
                )
    if problems:
        report.add("config schema", False, "; ".join(problems))
    else:
        report.add("config schema", True, "shape valid")


def _check_requires_env(report: ValidationReport, manifest: dict) -> None:
    raw = manifest.get("requires_env") or []
    problems: List[str] = []
    if not isinstance(raw, list):
        problems.append("requires_env: must be a list")
        raw = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = str(entry.get("name") or "")
        else:
            problems.append(f"requires_env[{i}]: must be a string or mapping")
            continue
        if not _UPPER_SNAKE_RE.match(name):
            problems.append(
                f"requires_env[{i}]: {name!r} is not UPPER_SNAKE_CASE"
            )
    if problems:
        report.add("requires_env", False, "; ".join(problems))
    else:
        report.add("requires_env", True, "all entries UPPER_SNAKE")


# ─── Static capability declarations ──────────────────────────────────────────

_REGISTRATION_KINDS = {
    "register_tool": ("tools", "name"),
    "register_hook": ("hooks", "hook_name"),
    "register_middleware": ("middleware", "kind"),
    "register_command": ("commands", "name"),
    "register_cli_command": ("commands", "name"),
}


# Directories excluded from capability scanning: test fixtures and maintenance
# scripts register tools/hooks that the plugin itself never uses at runtime.
_EXCLUDED_SCAN_DIRS = frozenset({".git", ".venv", "venv", "__pycache__", "test", "tests", "_test", "_tests"})


def _module_assignments(tree: ast.Module) -> Dict[str, ast.expr]:
    """Return unambiguous simple module assignments for bounded static resolution."""
    values: Dict[str, List[ast.expr]] = {}
    assignment_targets: set[ast.Name] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values.setdefault(target.id, []).append(value)
                assignment_targets.add(target)

    mutable: set[str] = set()
    mutating_methods = {"add", "append", "clear", "discard", "extend", "insert", "pop", "remove", "reverse", "setdefault", "sort", "update"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node not in assignment_targets:
            mutable.add(node.id)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name):
            mutable.add(node.value.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in mutating_methods
            and isinstance(node.func.value, ast.Name)
        ):
            mutable.add(node.func.value.id)
    return {
        name: assignments[0]
        for name, assignments in values.items()
        if len(assignments) == 1 and name not in mutable
    }


def _target_path(target: ast.expr, name: str) -> Optional[Tuple[int, ...]]:
    if isinstance(target, ast.Name):
        return () if target.id == name else None
    if isinstance(target, (ast.Tuple, ast.List)):
        for index, child in enumerate(target.elts):
            path = _target_path(child, name)
            if path is not None:
                return (index, *path)
    return None


def _literal_sequence(
    node: ast.expr, assignments: Dict[str, ast.expr], seen: Optional[set[str]] = None
) -> Optional[ast.expr]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return node
    if not isinstance(node, ast.Name):
        return None
    seen = seen or set()
    if node.id in seen or node.id not in assignments:
        return None
    return _literal_sequence(assignments[node.id], assignments, seen | {node.id})


def _node_at_path(node: ast.expr, path: Tuple[int, ...]) -> Optional[ast.expr]:
    for index in path:
        if not isinstance(node, (ast.Tuple, ast.List)) or index >= len(node.elts):
            return None
        node = node.elts[index]
    return node


def _literal_loop_values(
    iterable: ast.expr, path: Tuple[int, ...], assignments: Dict[str, ast.expr]
) -> Optional[List[str]]:
    sequence = _literal_sequence(iterable, assignments)
    if sequence is None:
        return None
    values: List[str] = []
    for item in sequence.elts:
        value = _node_at_path(item, path)
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return None
        values.append(value.value)
    return values


def _loop_reassigns_name(loop: ast.For, name: str) -> bool:
    target_nodes = set(ast.walk(loop.target))
    return any(
        isinstance(node, ast.Name)
        and node not in target_nodes
        and node.id == name
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(loop)
    )


def _registration_loop_values(
    name: str, call: ast.Call, parents: Dict[ast.AST, ast.AST], assignments: Dict[str, ast.expr]
) -> Optional[List[str]]:
    node: ast.AST = call
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.For):
            path = _target_path(node.target, name)
            if path is not None:
                if _loop_reassigns_name(node, name):
                    return None
                return _literal_loop_values(node.iter, path, assignments)
    return None


def _top_level_statements(statements: List[ast.stmt]):
    """Walk module try blocks to find conditional import re-exports without execution."""
    for node in statements:
        yield node
        if isinstance(node, (ast.Try, ast.TryStar)):
            for field_name in ("body", "orelse", "finalbody"):
                yield from _top_level_statements(getattr(node, field_name, []))
            for handler in getattr(node, "handlers", []):
                yield from _top_level_statements(handler.body)


def _provider_profile_aliases(tree: ast.Module) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "providers.base":
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "ProviderProfile"
            )
    return aliases


def _provider_names_for_registration(
    call: ast.Call, tree: ast.Module, assignments: Dict[str, ast.expr], parents: Dict[ast.AST, ast.AST]
) -> Optional[List[str]]:
    aliases = _provider_profile_aliases(tree)
    if not aliases:
        return None

    subclasses = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id in aliases for base in node.bases)
    }
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name not in subclasses
                and any(isinstance(base, ast.Name) and base.id in subclasses for base in node.bases)
            ):
                subclasses.add(node.name)
                changed = True
    constructors = aliases | subclasses

    def profile_name(expression: ast.expr) -> Optional[str]:
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in constructors
        ):
            name = next((kw.value for kw in expression.keywords if kw.arg == "name"), None)
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                return name.value
            if isinstance(name, ast.Name):
                assigned = assignments.get(name.id)
                if isinstance(assigned, ast.Constant) and isinstance(assigned.value, str):
                    return assigned.value
        return None

    def helper_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[List[str]]:
        returns: List[str] = []
        for returned in ast.walk(function):
            if not isinstance(returned, ast.Return):
                continue
            owner: ast.AST = returned
            while owner in parents and owner is not function:
                owner = parents[owner]
                if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    break
            if owner is not function:
                continue
            name = profile_name(returned.value) if returned.value is not None else None
            if name is None:
                return None
            returns.append(name)
        return returns or None

    def names_for_argument(argument: ast.expr) -> Optional[List[str]]:
        direct = profile_name(argument)
        if direct is not None:
            return [direct]
        if isinstance(argument, ast.Name):
            assigned = profile_name(assignments.get(argument.id, ast.Constant(None)))
            return [assigned] if assigned is not None else None
        if isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name):
            helpers = [
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == argument.func.id
            ]
            if len(helpers) == 1:
                return helper_names(helpers[0])
        return None

    arguments = [*call.args, *(kw.value for kw in call.keywords)]
    resolved: List[str] = []
    for argument in arguments:
        names = names_for_argument(argument)
        if names is not None:
            resolved.extend(names)
        elif isinstance(argument, (ast.Name, ast.Call)):
            return None
    return resolved or None


def _scan_capabilities(plugin_dir: Path, *, model_provider: bool = False) -> Tuple[Optional[dict], str]:
    """Inspect literal registration calls without running candidate code.

    Dynamic names cannot establish admission declarations and fail closed. Calls in
    helpers and conditional branches are included conservatively, not claimed to run.
    """
    recorded = {"tools": [], "hooks": [], "middleware": [], "commands": [], "model_providers": []}
    entry = plugin_dir / "__init__.py"
    has_register = False
    has_provider_profile_registration = False
    for path in sorted(plugin_dir.rglob("*.py")):
        if any(part in _EXCLUDED_SCAN_DIRS for part in path.relative_to(plugin_dir).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            return None, f"cannot parse plugin Python: {exc}"
        if path == entry:
            for node in _top_level_statements(tree.body):
                # Accept register() defined directly OR re-exported via
                # `from .impl import register` (an ImportFrom whose only
                # name is "register" aliases it into this module's namespace).
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "register":
                    has_register = True
                    if any(isinstance(statement, ast.Raise) for statement in node.body):
                        return None, "register() contains an unconditional raise"
                if isinstance(node, ast.ImportFrom) and any(alias.name == "register" for alias in node.names):
                    has_register = True
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        assignments = _module_assignments(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, (ast.Name, ast.Attribute))
                    and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == "register_provider"):
                provider_names = _provider_names_for_registration(node, tree, assignments, parents)
                if provider_names:
                    has_provider_profile_registration = True
                    recorded["model_providers"].extend(provider_names)
            if isinstance(node, ast.Attribute) and node.attr in _REGISTRATION_KINDS:
                parent = parents.get(node)
                if not isinstance(parent, ast.Call) or parent.func is not node:
                    return None, "aliased registration requires manual capability review"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and node.args[1].value in _REGISTRATION_KINDS:
                    return None, "dynamic registration lookup requires manual capability review"
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            registration = _REGISTRATION_KINDS.get(node.func.attr)
            if registration is None:
                continue
            kind, keyword = registration
            name = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == keyword), None)
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                recorded[kind].append(name.value)
                continue
            if isinstance(name, ast.Name):
                loop_values = _registration_loop_values(name.id, node, parents, assignments)
                if loop_values is not None:
                    recorded[kind].extend(loop_values)
                    continue
            return None, f"dynamic {node.func.attr} name requires manual capability review ({path.name}:{node.lineno})"
    if model_provider and not has_provider_profile_registration:
        return None, "model-provider plugin registered no ProviderProfile"
    if not has_register and not model_provider:
        return None, "no statically defined register() function"
    return recorded, ""


def _declared_list(manifest: dict, key: str) -> List[str]:
    raw = manifest.get(key) or []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def _check_capabilities(
    report: ValidationReport, manifest: dict, plugin_dir: Path
) -> Optional[dict]:
    """Compare statically visible registration calls against declared capabilities.

    Returns the recorded dict (for the built-in collision check) or None
    when the probe failed / was skipped.
    """
    if not (plugin_dir / "__init__.py").is_file():
        report.warn(
            "no __init__.py — capability probe skipped (manifest-only plugin)"
        )
        report.add("capability probe", True, "skipped (no __init__.py)")
        return None

    recorded, error = _scan_capabilities(plugin_dir, model_provider=manifest.get("kind") == "model-provider")
    if recorded is None:
        report.add("capability probe", False, error)
        return None
    if manifest.get("kind") == "model-provider":
        provider_names = ", ".join(recorded.get("model_providers") or [])
        report.add("capability probe", True, f"statically registered ProviderProfile(s): {provider_names}")
    else:
        report.add("capability probe", True, "literal registration calls inspected without execution")
    report.warn("Static inspection only: runtime behavior, imported registrations, capability completeness, and plugin safety are not verified.")

    for kind, manifest_key in (
        ("tools", "provides_tools"),
        ("hooks", "provides_hooks"),
        ("middleware", "provides_middleware"),
    ):
        declared = set(_declared_list(manifest, manifest_key))
        actual = set(recorded.get(kind) or [])
        undeclared = sorted(actual - declared)
        unregistered = sorted(declared - actual)
        if undeclared:
            report.add(
                f"declared {kind}",
                False,
                f"undeclared {kind} registered (not in {manifest_key}): "
                f"{', '.join(undeclared)}",
            )
        else:
            report.add(f"declared {kind}", True, "matches statically visible calls")
        if unregistered:
            report.warn(
                f"{manifest_key} declares {', '.join(unregistered)} "
                f"but no literal registration call was found"
            )
    return recorded


def _builtin_tool_names() -> Optional[List[str]]:
    """Return the built-in tool registry names (discovery-timing safe).

    ``tools.registry`` starts empty — built-in tool modules self-register on
    import, so we must run ``discover_builtin_tools()`` first (idempotent;
    see the AGENTS.md discover_plugins timing pitfall).
    """
    try:
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        return list(registry.get_all_tool_names())
    except Exception:
        return None


def _check_builtin_collisions(
    report: ValidationReport, manifest: dict, recorded: Optional[dict]
) -> None:
    candidate_tools = set(_declared_list(manifest, "provides_tools"))
    if recorded:
        candidate_tools.update(recorded.get("tools") or [])
    if not candidate_tools:
        report.add("built-in tool collisions", True, "no tools to check")
        return
    names = _builtin_tool_names()
    if names is None:
        report.add("built-in tool collisions", False, "built-in discovery unavailable")
        return
    builtin = set(names)
    collisions = sorted(candidate_tools & builtin)
    if collisions:
        report.add(
            "built-in tool collisions",
            False,
            "tool name(s) collide with built-in tools: "
            f"{', '.join(collisions)}",
        )
    else:
        report.add("built-in tool collisions", True, "no collisions")


# ─── Entry point ─────────────────────────────────────────────────────────────


def validate_plugin_dir(plugin_dir: Path) -> ValidationReport:
    """Run every admission check against *plugin_dir* and return the report."""
    report = ValidationReport()
    plugin_dir = Path(plugin_dir)

    if not plugin_dir.is_dir():
        report.add(
            "plugin directory", False, f"{plugin_dir} is not a directory"
        )
        return report

    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.is_file():
        manifest_file = plugin_dir / "plugin.yml"
    if not manifest_file.is_file():
        # Portable Agent Plugins v1 (#81196): a plugin.json-only package is
        # admissible — validated through the portable manifest reader. When a
        # package carries both manifests the native plugin.yaml always wins
        # (this branch is only reached when no native manifest exists).
        portable_file = plugin_dir / "plugin.json"
        if portable_file.is_file():
            return _validate_portable_plugin(report, plugin_dir)
        report.add(
            "manifest", False,
            "no plugin.yaml (or portable plugin.json) in the plugin directory",
        )
        return report

    import yaml

    try:
        manifest = yaml.safe_load(
            manifest_file.read_text(encoding="utf-8")
        )
    except Exception as exc:
        report.add("manifest", False, f"plugin.yaml failed to parse: {exc}")
        return report
    if not isinstance(manifest, dict):
        report.add("manifest", False, "plugin.yaml must be a mapping")
        return report
    report.add("manifest", True, "plugin.yaml parses")

    _check_manifest_fields(report, manifest)
    _check_requires_hermes(report, manifest)
    _check_config_spec(report, manifest)
    _check_requires_env(report, manifest)
    _check_loadable(report, plugin_dir)
    _check_python_dependencies(report, plugin_dir)
    recorded = _check_capabilities(report, manifest, plugin_dir)
    _check_builtin_collisions(report, manifest, recorded)
    _check_security_scan(report, plugin_dir)
    check_desktop_surface(report, plugin_dir)
    return report


_LOADABLE_ENTRYPOINTS = ("__init__.py", "desktop/plugin.js", "plugin.json")


def _check_loadable(report: ValidationReport, plugin_dir: Path) -> None:
    """A plugin.yaml with nothing beside it that Hermes can load (no ``register()`` module, no
    desktop bundle, no portable manifest) installs "successfully" and does nothing — a pip-layout
    repo whose code lives under ``src/`` behind an entry point is the usual shape."""
    present = [rel for rel in _LOADABLE_ENTRYPOINTS if (plugin_dir / rel).is_file()]
    report.add(
        "loadable", bool(present),
        f"entry: {', '.join(present)}" if present else
        "nothing to load: no __init__.py, desktop/plugin.js or plugin.json beside plugin.yaml "
        "(pip-layout packages need a directory-plugin wrapper with a pyproject.toml declaring the deps)",
    )


def _check_python_dependencies(report: ValidationReport, plugin_dir: Path) -> None:
    """Declared deps (pyproject ``[project].dependencies`` or manifest ``python_dependencies``) must be
    well-formed PEP 508 specs the installer will accept; a plugin opting out with
    ``python_runtime: external`` declares none."""
    from hermes_cli.plugin_python_deps import read_declaration

    try:
        decl = read_declaration(plugin_dir)
    except Exception as exc:
        report.add("python dependencies", False, f"declaration invalid: {exc}")
        return
    if decl.external:
        report.add("python dependencies", True, "external runtime (plugin manages its own)")
        return
    from hermes_cli.plugin_python_deps import applicable_specs, unsupported_specs

    urls = unsupported_specs(decl.specs)
    if urls:
        report.warn("python dependencies: direct URL requirement(s) are never auto-installed, users must "
                    f"install them by hand: {', '.join(urls)}")
    installable = applicable_specs(decl.specs)
    rejected = [s for s in installable if not _spec_is_safe(s)]
    detail = f"{len(installable)} installable from {decl.source}" if decl.source else "none declared"
    report.add("python dependencies", not rejected,
               f"unsafe spec(s): {', '.join(rejected)}" if rejected else detail)


def _spec_is_safe(spec: str) -> bool:
    from tools.lazy_deps import _spec_is_safe as safe
    return safe(spec)


def _check_security_scan(report: ValidationReport, plugin_dir: Path) -> None:
    """Run the install-time scanner at admission, so a pin a reviewer approves is one the
    installer will accept: ``dangerous`` fails the entry; ``caution`` findings surface as
    warnings for the reviewer (the installer trusts them once the pin is merged)."""
    from tools.plugin_guard import scan_plugin

    result = scan_plugin(plugin_dir)
    flagged = [f for f in result.findings if f.severity in ("critical", "high")]
    summary = ", ".join(sorted({f"{f.pattern_id} ({Path(f.file).name}:{f.line})" for f in flagged})) or "no findings"
    if result.verdict == "dangerous":
        report.add("security scan", False, f"dangerous: {summary}")
        return
    report.add("security scan", True, result.verdict)
    if result.verdict == "caution":
        report.warn(f"security scan caution: {summary}")


def _validate_portable_plugin(report: ValidationReport, plugin_dir: Path) -> ValidationReport:
    """Admission checks for a portable Agent Plugins v1 (plugin.json) package.

    Portable packages have no register() entry point, so the capability
    probe does not apply; validation is the manifest reader's own
    diagnostics (schema shape, name, supported subset).
    """
    try:
        from hermes_cli.agent_plugins import read_agent_plugin_manifest

        manifest, diagnostics = read_agent_plugin_manifest(plugin_dir)
    except Exception as exc:
        report.add("portable manifest", False, f"plugin.json failed validation: {exc}")
        return report

    # The portable reader raises on hard failures; surviving diagnostics are
    # advisory (unsupported-subset notes etc.) — surface them as warnings.
    for diag in diagnostics:
        scope = getattr(diag, "scope", "")
        message = getattr(diag, "message", str(diag))
        report.warnings.append(f"{scope}: {message}" if scope else message)

    report.add("portable manifest", True, "plugin.json parses (Agent Plugins v1)")
    name = str(manifest.get("name") or "").strip()
    report.add(
        "manifest fields",
        bool(name),
        "name present" if name else "plugin.json missing required 'name'",
    )
    _check_security_scan(report, plugin_dir)
    check_desktop_surface(report, plugin_dir)
    return report
