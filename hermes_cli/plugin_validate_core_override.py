"""Admission lint: a catalog plugin must not rebind Hermes core at runtime.

Plugins extend Hermes through public surfaces (hooks, middleware, provider profiles, Desktop SDK
slots). A plugin that replaces a core function, method or module attribute in place — assigning
``AIAgent._replace_primary_openai_client``, ``tui_gateway.server.handle_request``, a
``gateway.run_turn`` helper, a private catalog dict — collides with every other plugin that patches
the same seam and breaks on any core release. This static pass finds those rebinds in the plugin's
Python (the import closure of ``__init__.py`` and ``dashboard/``; tests, benchmarks and scripts
excluded). Like the Desktop lint it is a review tripwire, not a sandbox.

A value is "core" when it is a Hermes module or an attribute chain off one: a name bound by an
absolute import of a Hermes top-level package, ``importlib.import_module(...)`` / ``sys.modules``
lookups, ``getattr(core, ...)``, iteration over core values, and calls to plugin functions that
return a core value. Rebinds are attribute assignment / deletion, ``setattr`` / ``delattr``,
``sys.modules[...] = ...``, ``mock.patch``, and passing a core value to a plugin helper that
``setattr``s its parameter.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_SKIP_DIRS = frozenset({"tests", "test", "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages"})


@lru_cache(maxsize=1)
def _hermes_top_level() -> frozenset:
    """Top-level importable names shipped by this Hermes checkout (packages and root modules)."""
    root = Path(__file__).resolve().parents[1]
    names = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    names |= {p.stem for p in root.glob("*.py")}
    # ``hermes_plugins.*`` is where the loader imports plugins (bundled platform adapters included):
    # rebinding another plugin's module is the same collision as rebinding core.
    return frozenset((names | {"hermes_plugins"}) - {"setup", "conftest"})


def _is_test_file(rel: Path) -> bool:
    name = rel.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _python_files(plugin_dir: Path) -> Iterable[tuple[Path, Path]]:
    for path in sorted(plugin_dir.rglob("*.py")):
        rel = path.relative_to(plugin_dir)
        if _SKIP_DIRS.intersection(rel.parts[:-1]) or _is_test_file(rel):
            continue
        yield path, rel


def _runtime_files(plugin_dir: Path, files: dict[Path, ast.AST], local: set[str]) -> set[Path]:
    """The import closure of what Hermes loads: ``__init__.py`` (``register``) and ``dashboard/*.py``
    (dashboard plugin API). Benchmarks, CI and smoke scripts shipped beside them never run inside
    Hermes, so a stub they install into ``sys.modules`` is not a runtime override."""
    by_module: dict[str, set[Path]] = {}
    for rel in files:
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for start in range(len(parts)):
            by_module.setdefault(".".join(parts[start:]), set()).add(rel)

    def resolve(rel: Path, node: ast.AST) -> set[Path]:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = rel.parent.parts[: len(rel.parent.parts) - (node.level - 1)] if node.level > 1 else rel.parent.parts
                prefix = ".".join(base + tuple((node.module or "").split("."))).strip(".")
                names = [prefix] + [f"{prefix}.{a.name}".strip(".") for a in node.names]
            elif node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        found: set[Path] = set()
        for name in names:
            if name.split(".")[0] in local or isinstance(node, ast.ImportFrom) and node.level:
                found |= by_module.get(name, set())
        return found

    roots = {rel for rel in files if rel == Path("__init__.py") or (len(rel.parts) == 2 and rel.parts[0] == "dashboard")}
    seen: set[Path] = set()
    stack = list(roots)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for node in ast.walk(files[rel]):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                stack.extend(resolve(rel, node) - seen)
    return seen


def _local_names(plugin_dir: Path) -> set[str]:
    """Module/package names the plugin ships itself: an import of these is local, not core."""
    names = set()
    for path in plugin_dir.rglob("*"):
        if _SKIP_DIRS.intersection(path.relative_to(plugin_dir).parts):
            continue
        if path.suffix == ".py":
            names.add(path.stem)
        elif path.is_dir():
            names.add(path.name)
    return names


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(node: ast.Call) -> str:
    return _dotted(node.func)


class _Analysis:
    def __init__(self, core_roots: frozenset, local: set[str]):
        self.core_roots = core_roots - local
        self.core_returners: set[str] = set()
        self.param_patchers: dict[str, set[int]] = {}
        self.aliases: dict[str, str] = {}  # ``from .compat import server as gateway_server``
        self.constants: dict[str, str] = {}  # module-level ``NAME = "module.path"``

    def _func_key(self, call: ast.Call) -> str:
        name = _call_name(call).split(".")[-1]
        return self.aliases.get(name, name)

    # -- what counts as a core value -------------------------------------------------------------
    def _core_module_name(self, name: object) -> bool:
        return isinstance(name, str) and name.split(".")[0] in self.core_roots

    def is_core(self, node: ast.AST, tainted: set[str]) -> bool:
        if isinstance(node, ast.Name):
            return node.id in tainted
        if isinstance(node, ast.Attribute):
            return self.is_core(node.value, tainted)
        if isinstance(node, ast.Subscript):
            if _dotted(node.value) == "sys.modules":
                return self._lookup_is_core(node.slice)
            return self.is_core(node.value, tainted)
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ("importlib.import_module", "import_module", "__import__") and node.args:
                return self._lookup_is_core(node.args[0])
            if name == "sys.modules.get" and node.args:
                return self._lookup_is_core(node.args[0])
            if name == "getattr" and node.args:
                return self.is_core(node.args[0], tainted)
            return self._func_key(node) in self.core_returners
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(self.is_core(e, tainted) for e in node.elts)
        return False

    def _lookup_is_core(self, arg: ast.AST) -> bool:
        """Whether a module-name expression names Hermes: a literal, a module-level string
        constant, or an f-string whose literal head names it (``f"hermes_plugins.{p}.adapter"``).
        A name computed some other way (the plugin locating its own package) is not judged."""
        if isinstance(arg, ast.Name):
            arg = ast.Constant(self.constants.get(arg.id))
        if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            head = str(arg.values[0].value)
            return "." in head and self._core_module_name(head)
        return isinstance(arg, ast.Constant) and self._core_module_name(arg.value)

    # -- per-scope taint --------------------------------------------------------------------------
    def scope_taint(self, body: list[ast.stmt], inherited: set[str]) -> set[str]:
        tainted = set(inherited)
        nodes = [n for stmt in body for n in _walk_scope(stmt)]
        for _ in range(4):
            before = len(tainted)
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._core_module_name(alias.name):
                            tainted.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and self._core_module_name(node.module):
                    tainted.update(a.asname or a.name for a in node.names)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    if self.is_core(node.value, tainted):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        tainted.update(_bound_names(targets))
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    if self.is_core(node.iter, tainted):
                        tainted.update(_bound_names([node.target]))
            if len(tainted) == before:
                break
        return tainted

    # -- sinks ------------------------------------------------------------------------------------
    def rebinds(self, node: ast.AST, tainted: set[str]) -> list[str]:
        hits: list[str] = []
        targets: Sequence[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            for t in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                if isinstance(t, ast.Attribute) and self.is_core(t.value, tainted):
                    hits.append(_dotted(t) or t.attr)
                elif isinstance(t, ast.Subscript) and _dotted(t.value) == "sys.modules":
                    if isinstance(t.slice, ast.Constant) and self._core_module_name(t.slice.value):
                        hits.append(f"sys.modules[{t.slice.value!r}]")
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
                        and self.is_core(t.value.value, tainted):
                    hits.append(f"{_dotted(t.value) or t.value.attr}[...]")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ("setattr", "delattr") and node.args and self.is_core(node.args[0], tainted):
                hits.append(f"{name}({_dotted(node.args[0]) or '<core>'}, ...)")
            elif name.endswith("patch.object") and node.args and self.is_core(node.args[0], tainted):
                hits.append(f"patch.object({_dotted(node.args[0])}, ...)")
            elif name.split(".")[-1] == "patch" and node.args and isinstance(node.args[0], ast.Constant) \
                    and self._core_module_name(node.args[0].value):
                hits.append(f"patch({node.args[0].value!r})")
            else:
                for index in self.param_patchers.get(self._func_key(node), ()):
                    if index < len(node.args) and self.is_core(node.args[index], tainted):
                        hits.append(f"{name}({_dotted(node.args[index]) or '<core>'}, ...)")
        return hits


def _walk_scope(stmt: ast.AST):
    """Nodes of one scope: does not descend into nested function or class bodies."""
    stack = [stmt]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(child)


def _bound_names(targets: Iterable[ast.AST]) -> set[str]:
    names = set()
    for t in targets:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            names |= _bound_names(t.elts)
    return names


def _functions(tree: ast.AST):
    """``(func, is_method)`` for every function in the module, nested ones included."""
    def visit(node, in_class):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                static = any(_dotted(d) == "staticmethod" for d in child.decorator_list)
                yield child, in_class and not static
                yield from visit(child, False)
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, True)
            else:
                yield from visit(child, in_class)
    yield from visit(tree, False)


def core_override_findings(plugin_dir: Path) -> list[str]:
    """``["<target> (<rel>:<line>)", ...]`` for every runtime rebind of Hermes core in the plugin."""
    plugin_dir = Path(plugin_dir)
    parsed: dict[Path, ast.AST] = {}
    for path, rel in _python_files(plugin_dir):
        try:
            parsed[rel] = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except (SyntaxError, ValueError):
            continue
    local = _local_names(plugin_dir)
    runtime = _runtime_files(plugin_dir, parsed, local)
    trees = [(rel, tree) for rel, tree in parsed.items() if rel in runtime]
    analysis = _Analysis(_hermes_top_level(), local)
    for _rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] in local):
                analysis.aliases.update({a.asname: a.name for a in node.names if a.asname})
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                analysis.constants.update({n: node.value.value for n in _bound_names(node.targets)})

    # Fixpoint over the whole plugin: which helpers return core values / rebind a parameter.
    for _ in range(4):
        before = (len(analysis.core_returners), sum(len(v) for v in analysis.param_patchers.values()))
        for _rel, tree in trees:
            module_taint = analysis.scope_taint(tree.body, set())
            for func, is_method in _functions(tree):
                taint = analysis.scope_taint(func.body, module_taint)
                if any(isinstance(n, ast.Return) and n.value is not None and analysis.is_core(n.value, taint)
                       for stmt in func.body for n in _walk_scope(stmt)):
                    analysis.core_returners.add(func.name)
                params = [a.arg for a in func.args.posonlyargs + func.args.args]
                # A method is called as ``obj.bind(module, ...)``: its call-site args start after ``self``.
                offset = 1 if is_method else 0
                for index, param in enumerate(params[offset:]):
                    if any(analysis.rebinds(n, {param}) for stmt in func.body for n in _walk_scope(stmt)):
                        analysis.param_patchers.setdefault(func.name, set()).add(index)
        after = (len(analysis.core_returners), sum(len(v) for v in analysis.param_patchers.values()))
        if after == before:
            break

    findings: list[str] = []
    for rel, tree in trees:
        module_taint = analysis.scope_taint(tree.body, set())
        scopes = [(tree.body, module_taint)]
        scopes += [(f.body, analysis.scope_taint(f.body, module_taint)) for f, _ in _functions(tree)]
        for body, taint in scopes:
            for stmt in body:
                for node in _walk_scope(stmt):
                    for target in analysis.rebinds(node, taint):
                        findings.append((rel.as_posix(), getattr(node, "lineno", 0), target))
    return [f"{target} ({rel}:{line})" for rel, line, target in sorted(set(findings))]


def check_core_override(report, plugin_dir: Path) -> None:
    """Fail the report when the plugin's Python rebinds Hermes core at runtime."""
    hits = core_override_findings(plugin_dir)
    report.add(
        "no core override", not hits,
        "rebinds Hermes core at runtime (use a public hook, middleware or provider profile): "
        + "; ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        if hits else "no runtime rebinds of Hermes core",
    )
