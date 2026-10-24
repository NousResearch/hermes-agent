"""Static answer to "will this plugin run in the plugin host?" (``plugins.isolation: host``).

Reads a plugin directory's manifest and Python source (never imports it) and reports whether the
plugin reaches Hermes only through ``ctx`` surfaces that cross the host boundary. The rules come
from :mod:`hermes_cli.plugin_isolation`, the same tables the host enforces at runtime.
This is an advisory syntax audit, not an interpreter or a guarantee that a plugin will load.
It follows local name aliases, literal unpacking and uniform literal iteration. Container
lookups, call results, object attributes, arbitrary iterables and cross-function/control-flow
data flow are not inferred; the host still refuses unsupported access at runtime.

What it cannot see in a process boundary it flags:

* ctx methods that hand over live objects (platform adapters, SDK clients, argparse);
* writing attributes of Hermes modules (monkeypatching): the patch lands in the host's copy;
* mutating a global registry directly instead of through ``ctx``: the registration lands in the
  host's registry, which Hermes never reads;
* gateway platform adapters (``kind: platform``), model-provider profiles that build their own SDK
  client (``create_client``), and dashboard APIs that stream (the host bridge buffers responses).

``hermes plugins validate`` prints the verdict; ``evals/plugin_isolation/audit_catalog.py`` runs it
over every catalog entry.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

VERDICT_HOST = "host"
VERDICT_IN_PROCESS = "in_process"
VERDICT_PORTABLE = "portable"

# Top-level packages that are Hermes's own process state when imported by a plugin.
HERMES_PACKAGES: frozenset = frozenset({
    "agent", "tools", "hermes_cli", "gateway", "cron", "providers", "plugins", "tui_gateway",
    "acp_adapter", "run_agent", "cli", "model_tools", "toolsets", "hermes_state", "hermes_constants",
    "hermes_logging", "hermes_time", "utils", "registration_lifecycle", "hermes_platform",
})
# ``<module>.<callable>`` that mutate a Hermes-process registry when called from plugin code.
_DIRECT_REGISTRY_CALLS: frozenset = frozenset({
    ("tools.registry", "register"), ("tools.registry", "deregister"),
    ("gateway.platform_registry", "register"), ("providers", "register_provider"),
})
_MANIFEST_KIND_REASONS: dict[str, str] = {
    "platform": "kind 'platform': gateway platform adapters run in the Hermes process",
}
_STREAMING_MARKERS = ("StreamingResponse", "EventSourceResponse", ".websocket(", "WebSocket")
_SKIP_DIRS = frozenset({"tests", "test", ".git", "__pycache__", "node_modules", ".venv", "venv", "docs"})
_CtxValue = bool | tuple["_CtxValue", ...]


def _parameter_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
    names.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)
    return names


@dataclass
class IsolationReport:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ctx_methods: set[str] = field(default_factory=set)

    @property
    def host_ready(self) -> bool:
        return self.verdict in {VERDICT_HOST, VERDICT_PORTABLE}

    def summary(self) -> str:
        if self.verdict == VERDICT_PORTABLE:
            return "portable plugin (no Python): runs anywhere"
        if self.verdict == VERDICT_HOST:
            return "runs in the plugin host (plugins.isolation: host)"
        return "in-process only: " + "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "host_ready": self.host_ready, "reasons": list(self.reasons),
                "notes": list(self.notes), "ctx_methods": sorted(self.ctx_methods)}


def _python_files(plugin_dir: Path) -> list[Path]:
    files = []
    for path in sorted(plugin_dir.rglob("*.py")):
        rel = path.relative_to(plugin_dir).parts
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel[:-1]):
            continue
        if rel[-1].startswith("test_") or rel[-1] == "conftest.py":
            continue
        files.append(path)
    return files


class _SourceVisitor(ast.NodeVisitor):
    """Collect ctx calls and aliases; dynamic cross-function data flow is not inferred."""

    def __init__(self, rel: str, *, model_provider: bool = False):
        from hermes_cli.plugin_isolation import (
            HOST_DEGRADED_HOOKS, HOST_OBJECT_BASES, HOST_SKIPPED_CTX_METHODS,
            HOST_UNSUPPORTED_CTX_FACADES, HOST_UNSUPPORTED_CTX_METHODS,
        )
        self._unsupported, self._skipped = HOST_UNSUPPORTED_CTX_METHODS, HOST_SKIPPED_CTX_METHODS
        self._ctx_facades = HOST_UNSUPPORTED_CTX_FACADES
        self._ctx_names: set[str] = set()
        self._scope_depth = 0
        self._degraded, self._object_methods = HOST_DEGRADED_HOOKS, HOST_OBJECT_BASES
        self.rel = rel
        # A model-provider plugin's register_provider() call is its contract; the host captures it.
        self.model_provider = model_provider
        self.aliases: dict[str, str] = {}  # local name -> hermes module path (or module.attr)
        self.reasons: list[str] = []
        self.notes: list[str] = []
        self.ctx_methods: set[str] = set()

    def _hermes(self, module: str) -> bool:
        return module.split(".")[0] in HERMES_PACKAGES

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self._hermes(alias.name):
                self.aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module and self._hermes(node.module):
            for alias in node.names:
                self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def _resolve(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _flag_write(self, target: ast.AST, line: int) -> None:
        if isinstance(target, ast.Attribute):
            owner = self._resolve(target.value)
            if owner:
                self.reasons.append(f"{self.rel}:{line} patches {owner}.{target.attr} (a Hermes module "
                                    f"attribute; the host's copy is not the one Hermes runs)")

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self._ctx_value(node.value)
        self.generic_visit(node)
        for target in node.targets:
            self._flag_write(target, node.lineno)
            self._bind_ctx(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = self._ctx_value(node.value)
        self.generic_visit(node)
        self._flag_write(node.target, node.lineno)
        if node.value is not None:
            self._bind_ctx(node.target, value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._flag_write(node.target, node.lineno)
        self._bind_ctx(node.target, False)

    def _is_ctx(self, node: Optional[ast.AST]) -> bool:
        return isinstance(node, ast.Name) and node.id in self._ctx_names

    def _ctx_value(self, node: Optional[ast.AST]) -> _CtxValue:
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._ctx_value(item) for item in node.elts)
        return self._is_ctx(node)

    def _iteration_value(self, node: ast.AST) -> _CtxValue:
        items = self._ctx_value(node)
        if isinstance(items, tuple) and items and len(set(items)) == 1:
            return items[0]
        return False

    def _bind_ctx(self, node: ast.AST, value: _CtxValue) -> None:
        if isinstance(node, ast.Name):
            if value is True:
                self._ctx_names.add(node.id)
            else:
                self._ctx_names.discard(node.id)
        elif isinstance(node, ast.Starred):
            self._bind_ctx(node.value, False)  # A starred target receives a list, not a context.
        elif isinstance(node, (ast.Tuple, ast.List)):
            values = value if isinstance(value, tuple) else ()
            star = next((i for i, item in enumerate(node.elts) if isinstance(item, ast.Starred)), None)
            if star is not None and len(values) >= len(node.elts) - 1:
                tail = len(node.elts) - star - 1
                values = values[:star] + (False,) + (values[-tail:] if tail else ())
            if len(values) != len(node.elts):
                values = (False,) * len(node.elts)
            for target, item in zip(node.elts, values):
                self._bind_ctx(target, item)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_ctx(node.target, self._iteration_value(node.iter))
        self.visit(node.target)
        for statement in node.body + node.orelse:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_ctx(item.optional_vars, False)
                self.visit(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._ctx_names.discard(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_ctx(node.target, self._ctx_value(node.value))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        positional = node.args.posonlyargs + node.args.args
        params = _parameter_names(node.args)
        outer = self._ctx_names
        # Helpers conventionally name their context parameter ctx. Only a
        # module-level register function receives a context regardless of name.
        self._ctx_names = (outer - params) | (params & {"ctx"})
        if node.name == "register" and positional and self._scope_depth == 0:
            self._ctx_names.add(positional[0].arg)
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1
            self._ctx_names = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)  # Defaults evaluate in the enclosing scope.
        outer = self._ctx_names
        self._ctx_names = outer - _parameter_names(node.args)
        try:
            self.visit(node.body)
        finally:
            self._ctx_names = outer

    def _visit_comprehension(self, node: ast.AST) -> None:
        outer = self._ctx_names
        self._ctx_names = outer.copy()
        try:
            for generator in node.generators:
                self.visit(generator.iter)
                self._bind_ctx(generator.target, self._iteration_value(generator.iter))
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self._ctx_names = outer

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def _flag_ctx_member(self, name: str, line: int) -> None:
        if name in self._unsupported:
            self.reasons.append(f"{self.rel}:{line} ctx.{name}: {self._unsupported[name]}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # A facade can be used through a nested call or saved for later use.
        if isinstance(node.ctx, ast.Load) and self._is_ctx(node.value):
            self._flag_ctx_member(node.attr, node.lineno)
        self.generic_visit(node)

    def _flag_getattr(self, node: ast.Call) -> None:
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) > 1 and self._is_ctx(node.args[0])):
            name = node.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                self._flag_ctx_member(name.value, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        self._flag_getattr(node)
        if isinstance(func, ast.Name) and func.id == "setattr" and node.args:
            owner = self._resolve(node.args[0])
            if owner:
                self.reasons.append(f"{self.rel}:{node.lineno} setattr() on {owner} (patches a Hermes module)")
        if isinstance(func, ast.Attribute):
            name = func.attr
            # Preserve the existing method-call heuristic; facade accesses above
            # require a known context receiver, even when the facade is called.
            if name in self._unsupported and name not in self._ctx_facades and not self._is_ctx(func.value):
                self._flag_ctx_member(name, node.lineno)
            if name in self._skipped:
                self.notes.append(f"ctx.{name}() is skipped in the host ({self._skipped[name]})")
            if name.startswith("register_") or name in {"subscribe", "dispatch_tool", "inject_message", "emit"}:
                self.ctx_methods.add(name)
            if name == "register_hook" and node.args and isinstance(node.args[0], ast.Constant) \
                    and node.args[0].value in self._degraded:
                hook = node.args[0].value
                self.notes.append(f"hook {hook} {self._degraded[hook]}; those fields are placeholders in the host")
            self._flag_registry_call(self._resolve(func), node.lineno)
        elif isinstance(func, ast.Name):
            self._flag_registry_call(self.aliases.get(func.id), node.lineno)
        self.generic_visit(node)

    def _flag_registry_call(self, target: Optional[str], line: int) -> None:
        if not target:
            return
        module, _, attr = target.rpartition(".")
        if self.model_provider and (module, attr) == ("providers", "register_provider"):
            return
        if (module, attr) in _DIRECT_REGISTRY_CALLS or (
                module.endswith("_registry") and attr in {"register", "register_provider"}):
            self.reasons.append(f"{self.rel}:{line} calls {target}() directly instead of "
                                f"through ctx (registers in the host, not in Hermes)")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.model_provider and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                       and item.name == "create_client" for item in node.body):
            self.reasons.append(f"{self.rel}:{node.lineno} {node.name}.create_client() returns a live SDK "
                                f"client (model-provider profiles cross the host as data + methods)")
        outer = self._ctx_names
        self._ctx_names = outer.copy()
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1
            self._ctx_names = outer


def audit_plugin_dir(plugin_dir: Path, manifest: Optional[Mapping[str, Any]] = None) -> IsolationReport:
    """Classify one plugin directory without importing it."""
    plugin_dir = Path(plugin_dir)
    if manifest is None:
        manifest = _read_manifest(plugin_dir)
    if not (plugin_dir / "__init__.py").is_file():
        return IsolationReport(VERDICT_PORTABLE)
    reasons: list[str] = []
    notes: list[str] = []
    ctx_methods: set[str] = set()
    kind = str((manifest or {}).get("kind") or "").strip().lower()
    if kind in _MANIFEST_KIND_REASONS:
        reasons.append(_MANIFEST_KIND_REASONS[kind])
    dashboard = plugin_dir / "dashboard"
    if (dashboard / "manifest.json").is_file() and '"api"' in (dashboard / "manifest.json").read_text(
            encoding="utf-8", errors="replace"):
        api_source = "".join(p.read_text(encoding="utf-8-sig", errors="replace") for p in dashboard.rglob("*.py"))
        if any(marker in api_source for marker in _STREAMING_MARKERS):
            reasons.append("dashboard backend API streams (SSE/websocket); the host bridge buffers responses")
        else:
            notes.append("dashboard backend API is served through the plugin host (buffered responses)")
    for path in _python_files(plugin_dir):
        rel = str(path.relative_to(plugin_dir))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
        except SyntaxError as exc:
            notes.append(f"{rel}: not parseable ({exc.msg}); not audited")
            continue
        visitor = _SourceVisitor(rel, model_provider=kind == "model-provider")
        visitor.visit(tree)
        reasons.extend(visitor.reasons)
        notes.extend(visitor.notes)
        ctx_methods |= visitor.ctx_methods
    verdict = VERDICT_IN_PROCESS if reasons else VERDICT_HOST
    return IsolationReport(verdict, _dedupe(reasons), _dedupe(notes), ctx_methods)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _read_manifest(plugin_dir: Path) -> dict[str, Any]:
    for name in ("plugin.yaml", "plugin.yml"):
        path = plugin_dir / name
        if path.is_file():
            try:
                import hermes_yaml as yaml
                data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}
