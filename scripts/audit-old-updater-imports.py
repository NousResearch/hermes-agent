#!/usr/bin/env python3
"""What an OLD `hermes update` can still import from a NEW tree.

`hermes update` swaps the checkout under its own feet. The process keeps
running the code it started with, but the files underneath it are the
ones we just pulled. Anything it loads from disk after that point is a
contract with every released updater in the wild: delete one of those
names and the users on that release get a traceback halfway through an
update, on a tree that is already half-new.

`managed_uv._reload_hermes_constants` is the scar tissue proving this is
real: an updater hit ``cannot import name 'venv_python_path' from
'hermes_constants'`` while the file on disk plainly contained the name.

WHY THIS OVER-APPROXIMATES, ON PURPOSE
--------------------------------------
An earlier version of this script tried to find the exact swap statement
(the ``git merge --ff-only``) and count only what runs after it. That was
wrong twice over. It was fragile — ``ast.unparse`` normalises quotes, so
matching source text for ``"merge", "--ff-only"`` silently matched
nothing and the whole git path reported no swap at all. And it was wrong
in the DANGEROUS direction: every miss SHRINKS the frozen set, and a
symbol wrongly dropped from the set is a bricked update for whoever
reaches that branch.

So the rule is deliberately blunt: everything reachable from the update
entrypoints counts. A false positive costs one kept symbol. A false
negative costs somebody's install, mid-update, on a half-new tree.

A dynamic trace (driving real updates and watching imports) has the
opposite bias and is the wrong tool here for the same reason: one run
takes one path. It never enters the diverged-history reset, the Windows
rollback, or the ZIP fallback, so it reports a SMALLER
surface than reality.

THE DYNAMIC PATTERNS THAT MATTER
--------------------------------------------------------------
Plain import analysis misses three things this flow does. Literal targets
are recorded. Unresolved expressions remain visible for manual review:

* ``importlib.reload(m)`` — RE-EXECUTES the new file in the old process.
  This is the most dangerous load in the whole flow and it looks like
  nothing to an import walker. ``_UPDATE_RUNTIME_RELOAD_MODULES`` and
  ``_reload_config_modules`` reload ``hermes_constants``,
  ``hermes_cli.config`` and friends by name. Treated as a whole-module
  requirement.
* ``getattr(module, "name")`` — a symbol requirement with no import
  statement. ``managed_uv._windows_runtime_holders`` looks up
  ``_detect_venv_python_processes`` on ``hermes_cli.main`` this way, and
  silently refuses the update when it is absent.
* ``importlib.import_module(x)`` with a non-literal argument — cannot be
  resolved statically. Reported as UNRESOLVED rather than ignored.

HISTORY COVERAGE AND STATIC LIMITS
---------------------------------
History discovery inventories every reachable origin/main commit, including
merge parents, and scans every distinct Python blob for entrypoint ASTs.
All historical versions of discovered paths, updater siblings, renamed seed
helpers, and statically referenced imported functions are audited. Witness
commits identify versions; unchanged descendants are not re-parsed.

This is NOT a complete Python call-graph proof. Reflection, computed module
names, arbitrary alias reassignment, class/instance dispatch, and module
objects passed through opaque callbacks require manual review. Known helper
modules are conservatively entered wholesale; other modules are followed by
selected function names, not every unrelated command. FIRST_PARTY_ROOTS
bounds imports of interest. The report and freeze retain unresolved edges.
Syntax/decoding failures abort rather than shrink coverage. The special case
of committed merge markers audits every arm combination and records recovery;
this is conservative archaeology, not a claim that broken source could run.

Usage:
    python scripts/audit-old-updater-imports.py            # report
    python scripts/audit-old-updater-imports.py --json
    python scripts/audit-old-updater-imports.py --check    # CI gate
    python scripts/audit-old-updater-imports.py --explain hermes_constants
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tokenize
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Known entrypoint seeds, not a history path filter. Discovery also scans
# every reachable Python blob for entrypoint definitions at other addresses.
UPDATE_MODULE_CANDIDATES = (
    "hermes_cli/main.py",
    "hermes_cli/subcommands/update.py",
    "hermes_cli/update_cmd.py",
)

# Known post-swap helpers: every function counts. Historical filenames are
# augmented by rename edges, updater siblings from the HISTORY inventory,
# and statically referenced imported functions (including extractions).
# This list is a conservative seed, not the universe of audited paths.
POST_SWAP_HELPER_MODULES = (
    "hermes_cli/post_update.py",
    "hermes_cli/update_lock.py",
    # Read every historical home of these helpers.
    "hermes_cli/backup.py",
    "hermes_cli/backup_restore.py",
    "hermes_cli/managed_uv.py",
    "hermes_cli/psutil_android.py",
    # pm era: `hermes update` drives the store through the pm package. A pm
    # module imported BEFORE the swap keeps running as old code afterwards,
    # so its lazy loads resolve against the NEW tree -- same failure shape
    # as managed_uv. Files absent at a given revision are simply skipped.
    "pm/__init__.py",
    "pm/cli.py",
    "pm/ensure.py",
    "pm/extras.py",
    "pm/lock.py",
    "pm/package.py",
    "pm/packages.py",
    "pm/paths.py",
    "pm/registry.py",
    "pm/store.py",
    "pm/operations.py",
    "pm/build_operations.py",
)

# Historical module-object calls that need explicit review, not a guessed
# receiver type. Keep the witness so regeneration cannot discard the contract.
REVIEWED_DYNAMIC_LOADS = (
    ("hermes_cli._subprocess_compat", "run", "2ecca1e7d3e7",
     "hermes_cli/managed_uv.py:_install_uv_windows"),
)

# Only OUR packages matter: a third-party import is pinned by the
# dependency resolver, not by this repo's file layout.
FIRST_PARTY_ROOTS = frozenset(
    {
        "agent",
        "gateway",
        "hermes_cli",
        "hermes_constants",
        "hermes_state",
        "installation",
        "plugins",
        "pm",
        "tools",
        "utils",
    }
)

# Where an update begins. Everything reachable from here can run while
# the tree is being replaced.
UPDATE_ENTRYPOINTS = (
    "cmd_update",
    "_cmd_update_impl",
    "_update_via_zip",
    "_run_update_phase_inline",
)

_AnyFunc = ast.FunctionDef | ast.AsyncFunctionDef


def _first_party(module: str) -> bool:
    return module.split(".")[0] in FIRST_PARTY_ROOTS


@dataclass(frozen=True)
class Requirement:
    """One name the updater needs to find in the NEW tree."""

    module: str
    symbol: str | None
    kind: str  # import | reload | getattr
    function: str
    source_file: str
    guarded: bool = False
    """True when the load sits in a ``try`` that catches its failure.

    A guarded requirement cannot brick an update — the old code has a
    fallback arm — so it is reported as informational, not frozen.
    """

    def key(self) -> tuple[str, str]:
        return (self.module, self.symbol or "")


@dataclass
class Analysis:
    path: str
    requirements: list[Requirement] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    reachable: set[str] = field(default_factory=set)
    dependencies: dict[str, set[str]] = field(default_factory=dict)
    parse_recoveries: set[str] = field(default_factory=set)


def _function_definitions(tree: ast.AST) -> list[_AnyFunc]:
    return [node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _called_names(node: ast.AST) -> set[str]:
    """Function names a piece of code can call.

    Covers the three shapes this codebase uses: ``foo()``,
    ``module.foo()``, and ``_m().foo()`` — update_cmd's lazy
    ``hermes_cli.main`` handle, which re-exports these same helpers.
    Attribute calls that are not ours simply find no match in the
    module's own function table.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _string_constants(tree: ast.AST) -> dict[str, list[str]]:
    """Module-level ``NAME = (...)`` / ``NAME = [...]`` string collections.

    ``_UPDATE_RUNTIME_RELOAD_MODULES`` is exactly this shape, and its
    contents are module names that get reloaded — i.e. re-executed from
    the new tree.
    """
    out: dict[str, list[str]] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List, ast.Set)):
            continue
        values = [
            el.value
            for el in node.value.elts
            if isinstance(el, ast.Constant) and isinstance(el.value, str)
        ]
        if values:
            out[target.id] = values
    return out


def _guarded_spans(func: _AnyFunc, failure: str = "ImportError") -> list[tuple[int, int]]:
    """Try bodies whose first matching handler swallows this load failure.

    AttributeError cannot guard an import, and ModuleNotFoundError cannot
    guard a missing from-import symbol. A handler that raises is not a
    fallback, even if it catches the right exception.
    """
    caught_by = {failure, "Exception", "BaseException"}
    if failure == "ModuleNotFoundError":
        caught_by.add("ImportError")
    spans: list[tuple[int, int]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            matches = handler.type is None or any(
                isinstance(t, ast.Name) and t.id in caught_by for t in types
            )
            if not matches:
                continue
            if not any(isinstance(n, ast.Raise) for n in ast.walk(handler)) and node.body:
                spans.append((node.body[0].lineno, max(stmt.end_lineno or stmt.lineno for stmt in node.body)))
            break
    return spans


def _requirements_in(
    func: _AnyFunc,
    source_file: str,
    constants: dict[str, list[str]],
) -> tuple[list[Requirement], list[str]]:
    """Every name *func* needs from the new tree, plus what we could not read."""
    reqs: list[Requirement] = []
    unresolved: list[str] = []
    guarded_spans = {
        failure: _guarded_spans(func, failure)
        for failure in ("ImportError", "ModuleNotFoundError", "AttributeError")
    }

    def _is_guarded(node: ast.AST, kind: str, symbol: str | None) -> bool:
        line = getattr(node, "lineno", None)
        if line is None:
            return False
        failure = "AttributeError" if kind == "getattr" else "ImportError" if symbol else "ModuleNotFoundError"
        return any(first <= line <= last for first, last in guarded_spans[failure])

    def add(
        module: str, symbol: str | None, kind: str, node: ast.AST
    ) -> None:
        if _first_party(module):
            reqs.append(
                Requirement(
                    module,
                    symbol,
                    kind,
                    func.name,
                    source_file,
                    guarded=_is_guarded(node, kind, symbol),
                )
            )

    for child in ast.walk(func):
        # ── plain lazy imports ─────────────────────────────────────────
        if isinstance(child, ast.ImportFrom):
            module = _import_module(child, source_file)
            if module:
                for alias in child.names:
                    add(module, alias.name, "import", child)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                add(alias.name, None, "import", child)

        # ── dynamic loads ──────────────────────────────────────────────
        elif isinstance(child, ast.Call):
            fname = (
                child.func.attr
                if isinstance(child.func, ast.Attribute)
                else child.func.id
                if isinstance(child.func, ast.Name)
                else ""
            )

            if fname in ("reload", "import_module") and child.args:
                arg = child.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    # importlib.reload("x") is not legal, but
                    # import_module("x") is — same requirement either way.
                    add(arg.value, None, "reload", child)
                elif isinstance(arg, ast.Name) and arg.id in constants:
                    for module in constants[arg.id]:
                        add(module, None, "reload", child)
                else:
                    # A reload of a loop variable: find the collection the
                    # loop walks. `for m in (...)` / `for m in CONST`.
                    resolved = False
                    for loop in ast.walk(func):
                        if not isinstance(loop, ast.For):
                            continue
                        if not (
                            isinstance(loop.target, ast.Name)
                            and isinstance(arg, ast.Name)
                        ):
                            continue
                        names: list[str] = []
                        if isinstance(loop.iter, (ast.Tuple, ast.List)):
                            names = [
                                el.value
                                for el in loop.iter.elts
                                if isinstance(el, ast.Constant)
                                and isinstance(el.value, str)
                            ]
                        elif isinstance(loop.iter, ast.Name):
                            names = constants.get(loop.iter.id, [])
                        for module in names:
                            add(module, None, "reload", child)
                            resolved = True
                    if not resolved:
                        try:
                            text = ast.unparse(child)
                        except Exception:  # noqa: BLE001
                            text = f"{fname}(...)"
                        unresolved.append(f"{source_file}:{func.name}: {text[:90]}")

            elif fname == "getattr" and len(child.args) >= 2:
                holder, attr = child.args[0], child.args[1]
                if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                    module = _module_of(holder)
                    if module:
                        add(module, attr.value, "getattr", child)

    return reqs, unresolved


def _module_of(node: ast.AST) -> str | None:
    """Best-effort: which module a getattr target refers to.

    Handles the one real shape — ``sys.modules.get("hermes_cli.main")``
    stashed in a local and then getattr'd (managed_uv does exactly this).
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str):
                    return value
    return None


def _resolve_sys_modules_locals(func: _AnyFunc) -> dict[str, str]:
    """Locals bound to ``sys.modules.get("<module>")`` inside *func*."""
    bound: dict[str, str] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                module = _module_of(node.value)
                if module:
                    bound[target.id] = module
    return bound


def _getattr_on_bound_locals(
    func: _AnyFunc, source_file: str
) -> list[Requirement]:
    """``m = sys.modules.get("x")`` … ``getattr(m, "y")`` → x.y required."""
    bound = _resolve_sys_modules_locals(func)
    if not bound:
        return []
    out: list[Requirement] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        fname = node.func.id if isinstance(node.func, ast.Name) else ""
        if fname != "getattr" or len(node.args) < 2:
            continue
        holder, attr = node.args[0], node.args[1]
        if not (isinstance(holder, ast.Name) and holder.id in bound):
            continue
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            module = bound[holder.id]
            if _first_party(module):
                out.append(
                    Requirement(
                        module, attr.value, "getattr", func.name, source_file,
                        guarded=any(first <= node.lineno <= last for first, last in _guarded_spans(func, "AttributeError")),
                    )
                )
    return out


class AuditError(RuntimeError):
    """An incomplete audit must never be mistaken for a smaller contract."""


def _parse_variants(source: str, context: str) -> tuple[list[ast.Module], bool]:
    """A committed conflict is audited as ALL arm combinations, never skipped.

    Actual history contains merge markers in config.py. Both resolutions
    can be analyzed without inventing syntax or choosing a winner. Other
    syntax errors, malformed markers, and excessive combinations fail closed.
    """
    try:
        return [ast.parse(source, filename=context)], False
    except (SyntaxError, ValueError) as original:
        if not re.search(r"^<<<<<<< ", source, re.MULTILINE):
            raise AuditError(f"Cannot parse {context}: {original}") from original
        variants = [""]
        arms: list[str] | None = None
        for line in source.splitlines(keepends=True):
            if line.startswith("<<<<<<< "):
                if arms is not None:
                    raise AuditError(f"Nested conflict in {context}") from original
                arms = [""]
            elif line.startswith("||||||| ") or line.rstrip("\r\n") == "=======":
                if arms is None:
                    raise AuditError(f"Malformed conflict in {context}") from original
                arms.append("")
            elif line.startswith(">>>>>>> "):
                if arms is None or len(arms) < 2 or len(variants) * len(arms) > 64:
                    raise AuditError(f"Malformed or excessive conflict variants in {context}") from original
                variants = [prefix + arm for prefix in variants for arm in arms]
                arms = None
            elif arms is not None:
                arms[-1] += line
            else:
                variants = [prefix + line for prefix in variants]
        if arms is not None:
            raise AuditError(f"Unterminated conflict in {context}") from original
        try:
            return [ast.parse(v, filename=context) for v in variants], True
        except (SyntaxError, ValueError) as exc:
            raise AuditError(f"Cannot parse all conflict arms of {context}: {exc}") from exc


def _import_module(node: ast.ImportFrom, source_file: str) -> str:
    if not node.level:
        return node.module or ""
    package = source_file.removesuffix(".py").split("/")[:-1]
    if node.level > len(package):
        raise AuditError(f"{source_file}:{node.lineno}: invalid relative import")
    return ".".join(package[:len(package) - node.level + 1] +
                    ([node.module] if node.module else []))


@dataclass
class FunctionFacts:
    calls: set[str]
    imports: list[ast.Import | ast.ImportFrom]
    references: list[tuple[str, tuple[str, ...], bool]]
    returned_names: set[str]
    factory_attributes: set[tuple[str, str]]
    requirements: list[Requirement]
    unresolved: list[str]
    error: str | None = None


@dataclass
class ModuleFacts:
    functions: dict[str, FunctionFacts]
    imports: list[ast.Import | ast.ImportFrom]
    reexports: list[ast.ImportFrom]


@dataclass
class VersionFacts:
    variants: list[ModuleFacts]
    recovered: bool


def _prepare_tree(tree: ast.Module, source_file: str) -> ModuleFacts:
    """Discard the big AST after retaining compact, seed-independent facts."""
    imports: list[ast.Import | ast.ImportFrom] = []

    def collect(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        for child in ast.iter_child_nodes(node):
            collect(child)

    collect(tree)
    functions = {}
    constants = _string_constants(tree)
    for func in _function_definitions(tree):
        name = func.name
        nodes = list(ast.walk(func))
        parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
        references = []
        returned_names = set()
        factory_attributes = set()
        for node in nodes:
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                returned_names.add(node.value.id)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    factory_attributes.add((node.value.func.id, node.attr))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                attrs = []
                parent = parents.get(node)
                returned = isinstance(parent, ast.Return)
                while isinstance(parent, ast.Attribute):
                    attrs.append(parent.attr)
                    parent = parents.get(parent)
                references.append((node.id, tuple(attrs), returned))
        error = None
        try:
            reqs, unresolved = _requirements_in(func, source_file, constants)
            reqs.extend(_getattr_on_bound_locals(func, source_file))
        except AuditError as exc:
            # Invalid imports in an unreachable function were never part of
            # the selected surface. Defer that failure until it is reached.
            reqs, unresolved, error = [], [], str(exc)
        facts = FunctionFacts(
            _called_names(func),
            [n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))],
            references, returned_names, factory_attributes, reqs, unresolved, error,
        )
        if name in functions:
            # Without receiver types, every same-named definition is possible.
            previous = functions[name]
            previous.calls.update(facts.calls)
            previous.imports.extend(facts.imports)
            previous.references.extend(facts.references)
            previous.returned_names.update(facts.returned_names)
            previous.factory_attributes.update(facts.factory_attributes)
            previous.requirements.extend(facts.requirements)
            previous.unresolved.extend(facts.unresolved)
            previous.error = previous.error or facts.error
        else:
            functions[name] = facts
    return ModuleFacts(functions, imports, [n for n in tree.body if isinstance(n, ast.ImportFrom)])


def _prepare_version(source: str, source_file: str) -> VersionFacts:
    trees, recovered = _parse_variants(source, source_file)
    return VersionFacts([_prepare_tree(tree, source_file) for tree in trees], recovered)


def _imported_dependencies(
    facts: ModuleFacts, reachable: set[str], source_file: str,
) -> dict[str, set[str]]:
    """Resolve bindings over the selected functions, not unrelated commands.

    Keeping bindings seed-dependent preserves conditional aliases and lazy
    module factories while their expensive AST walks are cached once.
    """
    funcs = {name: facts.functions[name] for name in sorted(reachable)}
    imports = [*facts.imports, *(node for func in funcs.values() for node in func.imports)]
    bindings: dict[str, set[tuple[str, str]]] = {}
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            module = _import_module(node, source_file)
            for alias in node.names:
                bindings.setdefault(alias.asname or alias.name, set()).add((module, alias.name))
        else:
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                bindings.setdefault(name, set()).add((alias.name if alias.asname else name, "*"))

    factories: dict[str, set[str]] = {}
    for name, func in funcs.items():
        for returned in func.returned_names:
            for module, symbol in bindings.get(returned, ()):
                if _first_party(module):
                    factories.setdefault(name, set()).add(
                        module if symbol == "*" else f"{module}.{symbol}"
                    )
    dependencies: dict[str, set[str]] = {}
    for name, func in funcs.items():
        for called, attr in func.factory_attributes:
            for module in factories.get(called, ()):
                dependencies.setdefault(module, set()).add(attr)
        for reference, attrs, returned in func.references:
            if returned and name in factories:
                continue
            for module, symbol in bindings.get(reference, ()):
                if not _first_party(module):
                    continue
                if symbol == "*" and attrs:
                    module = ".".join([module, *attrs[:-1]])
                    symbol = attrs[-1]
                elif attrs:
                    module = ".".join([module, symbol, *attrs[:-1]])
                    symbol = attrs[-1]
                dependencies.setdefault(module, set()).add(symbol)
    return dependencies


def _analyse_facts(facts: ModuleFacts, source_file: str, seeds: set[str] | None) -> Analysis:
    result = Analysis(path=source_file)
    functions = facts.functions
    if seeds is None:
        reachable = set(functions)
    else:
        reachable: set[str] = set()
        stack = [name for name in seeds if name in functions]
        while stack:
            name = stack.pop()
            if name in reachable:
                continue
            reachable.add(name)
            stack.extend(functions[name].calls & functions.keys() - reachable)
    result.reachable = reachable
    result.dependencies = _imported_dependencies(facts, reachable, source_file)
    for node in facts.reexports:
        module = _import_module(node, source_file)
        if _first_party(module):
            for alias in node.names:
                if seeds is None or (alias.asname or alias.name) in seeds:
                    result.dependencies.setdefault(module, set()).add(alias.name)
    for name in sorted(reachable):
        func = functions[name]
        if func.error:
            raise AuditError(func.error)
        result.requirements.extend(func.requirements)
        result.unresolved.extend(func.unresolved)
    return result


def _analyse_version(facts: VersionFacts, source_file: str, seeds: set[str] | None) -> Analysis:
    result = Analysis(path=source_file)
    for variant in facts.variants:
        part = _analyse_facts(variant, source_file, seeds)
        result.requirements.extend(part.requirements)
        result.unresolved.extend(part.unresolved)
        result.reachable.update(part.reachable)
        for module, symbols in part.dependencies.items():
            result.dependencies.setdefault(module, set()).update(symbols)
    if facts.recovered:
        result.parse_recoveries.add(
            f"{source_file}: audited all {len(facts.variants)} merge-conflict arm combinations"
        )
    return result


def analyse(
    source: str, source_file: str, *, entrypoints: bool,
    seeds: set[str] | None = None,
) -> Analysis:
    """Analyze selected imports without executing the source. Fail closed."""
    if entrypoints and seeds is None:
        seeds = set(UPDATE_ENTRYPOINTS)
    return _analyse_version(_prepare_version(source, source_file), source_file, seeds)


# ─── history walking ────────────────────────────────────────────────────


def _git(*args: str, input: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", *args], cwd=REPO_ROOT,
            input=input, capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise AuditError(exc.stderr.decode("utf-8", "replace").strip()) from exc


def _full_history_ref() -> str:
    if _git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise AuditError(
            "Cannot audit shallow history. Run git fetch --unshallow origin "
            "and fetch origin/main before regenerating; shallow CI must use "
            "the checked-in frozen JSON."
        )
    return _git("rev-parse", "--verify", "origin/main^{commit}").decode().strip()


def shipped_commits() -> list[str]:
    """ALL ancestors of origin/main, not tags, first-parent, or a path log."""
    return _git("rev-list", _full_history_ref()).decode().splitlines()


_ENTRYPOINT_PATTERN = re.compile(
    rb"\bdef\s+(" + "|".join(UPDATE_ENTRYPOINTS).encode() + rb")\s*\("
)


def _has_entrypoint(source: bytes, context: str) -> bool:
    # Cheap prefilter, then AST check: fixtures/docstrings mentioning
    # cmd_update are not entrypoints. A malformed candidate is still fatal.
    if not any(name.encode() in source for name in UPDATE_ENTRYPOINTS):
        return False
    # Python permits explicit line continuations between def and its name.
    normalized = source.replace(b"\\\r\n", b"").replace(b"\\\n", b"")
    if not _ENTRYPOINT_PATTERN.search(normalized):
        return False
    trees, _ = _parse_variants(_decode_source(source, context), context)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in UPDATE_ENTRYPOINTS
        for tree in trees for node in tree.body
    )


def _entrypoint_paths(index: HistoryIndex) -> set[str]:
    """Search every distinct reachable Python blob, not current filenames.

    Scanning blobs once avoids pickaxe repeatedly diffing the same large
    modules at thousands of merges. Bounded batches keep the multi-GB
    history out of memory; only likely definitions need AST parsing.
    """
    locations: dict[str, set[str]] = {}
    for path, versions in index.versions.items():
        for blob in versions:
            locations.setdefault(blob, set()).add(path)
    blobs = sorted(locations)
    paths: set[str] = set()
    candidates = 0
    for start in range(0, len(blobs), 64):
        if start % 4096 == 0:
            print(f"[audit] entrypoint discovery: {start}/{len(blobs)} distinct blobs", file=sys.stderr, flush=True)
        for blob, source in _batch_blobs(blobs[start:start + 64]).items():
            context = f"{', '.join(sorted(locations[blob]))} [blob {blob}]"
            if any(name.encode() in source for name in UPDATE_ENTRYPOINTS):
                candidates += 1
            if _has_entrypoint(source, context):
                paths.update(locations[blob])
    index.discovery_stats = {
        "python_paths_in_history": len(index.versions),
        "distinct_blobs_scanned": len(blobs),
        "blobs_mentioning_entrypoints": candidates,
    }
    return paths


@dataclass
class HistoryIndex:
    # One entry per (path, blob), with commits witnessing that version. A
    # version unchanged in later commits need not be parsed again.
    versions: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    renames: dict[str, set[str]] = field(default_factory=dict)
    discovery_stats: dict[str, int] = field(default_factory=dict)


def _history_index(ref: str) -> HistoryIndex:
    """Inventory every Python version, including deleted files and merge sides.

    A full DAG walk with root diffs covers every blob at its introduction.
    Rename edges supplement import/definition discovery; correctness does not
    depend on Git detecting an entrypoint move by its similarity threshold.
    """
    output = _git(
        "log", "--full-history", "-m", "--root", "--find-renames",
        "--format=%H", "--raw", "--no-abbrev", "-z", ref, "--", "*.py",
    )
    index = HistoryIndex()
    records = iter(output.split(b"\0"))
    commit = ""
    for record in records:
        record = record.lstrip(b"\n")
        if not record:
            continue
        if not record.startswith(b":"):
            commit = record.decode()
            continue
        _old_mode, new_mode, _old_blob, blob, status = record.decode().split()
        path = next(records).decode()
        if status.startswith(("R", "C")):
            old_path, path = path, next(records).decode()
            index.renames.setdefault(old_path, set()).add(path)
            index.renames.setdefault(path, set()).add(old_path)
        if new_mode != "000000" and path.endswith(".py"):
            index.versions.setdefault(path, {}).setdefault(blob, set()).add(commit)
    return index


def _decode_source(payload: bytes, context: str) -> str:
    try:
        encoding, _ = tokenize.detect_encoding(BytesIO(payload).readline)
        return payload.decode(encoding)
    except (SyntaxError, UnicodeError, LookupError) as exc:
        raise AuditError(f"Cannot decode {context}: {exc}") from exc


def _batch_blobs(refs: list[str]) -> dict[str, bytes]:
    """Read known blob IDs in one process; missing/corrupt objects are fatal."""
    if not refs:
        return {}
    out = _git("cat-file", "--batch", input=("\n".join(refs) + "\n").encode())
    contents: dict[str, bytes] = {}
    pos = 0
    for ref in refs:
        newline = out.find(b"\n", pos)
        header = out[pos:newline].split()
        if newline < 0 or len(header) != 3 or header[1] != b"blob":
            raise AuditError(f"Cannot read historical blob {ref}: {header!r}")
        size = int(header[2])
        pos = newline + 1
        if len(out) <= pos + size or out[pos + size:pos + size + 1] != b"\n":
            raise AuditError(f"Truncated historical blob {ref}")
        contents[ref] = out[pos:pos + size]
        pos += size + 1
    return contents


def _batch_read_blobs(refs: list[str]) -> dict[str, str]:
    return {ref: _decode_source(source, ref) for ref, source in _batch_blobs(refs).items()}


@dataclass
class Surface:
    required: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    kinds: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    sites: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    guarded_only: set[tuple[str, str]] = field(default_factory=set)
    """Pairs whose every load site sits under a swallowing ``try``.

    These cannot brick an update (the old code has a fallback arm), so
    they are informational: reported, but absent from the frozen surface
    and never fatal in ``--check``.
    """
    unresolved: set[str] = field(default_factory=set)
    parse_recoveries: set[str] = field(default_factory=set)
    stats: dict = field(default_factory=dict)


def _module_paths(module: str, paths: dict) -> list[str]:
    stem = module.replace(".", "/")
    return [p for p in (f"{stem}.py", f"{stem}/__init__.py") if p in paths]


def _audit_versions(index: HistoryIndex, entrypaths: set[str], read_sources, *, progress: bool = False) -> Surface:
    """Fixed point over historical static imports and rename edges.

    Demand is monotonic: an additional caller can only add functions. The
    memo includes the path, not just the blob: identical source at another
    address has different relative imports, provenance, and seed roles.
    """
    demand: dict[str, set[str] | None] = {}
    pending: deque[str] = deque()
    queued: set[str] = set()

    def enqueue(path: str, seeds: set[str] | None) -> None:
        if path not in index.versions:
            return
        if path in demand:
            previous = demand[path]
            if previous is None or (seeds is not None and seeds <= previous):
                return
            if seeds is not None:
                seeds = seeds | previous
        demand[path] = None if seeds is None else set(seeds)
        if path not in queued:
            pending.append(path)
            queued.add(path)

    for path in sorted(entrypaths | set(UPDATE_MODULE_CANDIDATES)):
        enqueue(path, set(UPDATE_ENTRYPOINTS))
    for path in sorted(set(POST_SWAP_HELPER_MODULES) | {
        p for p in index.versions
        if p.startswith("hermes_cli/update_cmd_") and p.endswith(".py")
    }):
        enqueue(path, None)

    memo: dict[tuple[str, str], Analysis] = {}
    prepared: dict[tuple[str, str], VersionFacts] = {}
    analysis_passes = 0
    while pending:
        path = pending.popleft()
        queued.remove(path)
        seeds = demand[path]
        for renamed in index.renames.get(path, ()):
            enqueue(renamed, seeds)
        blobs = sorted(index.versions[path])
        if progress:
            print(
                f"[audit] {path}: {len(blobs)} versions; "
                f"{len(seeds) if seeds is not None else 'all'} function seeds; "
                f"{len(pending)} paths queued; {analysis_passes} analyses so far",
                file=sys.stderr, flush=True,
            )
        for start in range(0, len(blobs), 64):
            batch = blobs[start:start + 64]
            sources = read_sources(path, [blob for blob in batch if (path, blob) not in prepared])
            for blob in batch:
                try:
                    if (path, blob) not in prepared:
                        prepared[path, blob] = _prepare_version(sources[blob], path)
                    analysis = _analyse_version(prepared[path, blob], path, seeds)
                except AuditError as exc:
                    witnesses = ", ".join(sorted(index.versions[path][blob])[:3])
                    raise AuditError(f"{exc} [blob {blob}; commits {witnesses}]") from exc
                memo[path, blob] = analysis
                analysis_passes += 1
                for module, symbols in analysis.dependencies.items():
                    for symbol in symbols:
                        submodules = _module_paths(f"{module}.{symbol}", index.versions)
                        if submodules or symbol == "*":
                            # A bare module object can escape through a callback
                            # or getattr. Do not turn that into all CLI commands.
                            # Expose the unresolved edge for manual review.
                            analysis.unresolved.append(
                                f"{path}: module object {module}.{symbol} requires manual call-graph review"
                            )
                        else:
                            for target in _module_paths(module, index.versions):
                                enqueue(target, {symbol})

    surface = Surface()
    bare_pairs: set[tuple[str, str]] = set()
    for (path, blob), analysis in memo.items():
        for req in analysis.requirements:
            surface.required.setdefault(req.key(), set()).update(
                c[:12] for c in index.versions[path][blob]
            )
            surface.kinds.setdefault(req.key(), set()).add(req.kind)
            surface.sites.setdefault(req.key(), set()).add(f"{path}:{req.function}")
            if not req.guarded:
                bare_pairs.add(req.key())
        surface.unresolved.update(analysis.unresolved)
        surface.parse_recoveries.update(
            f"{recovery} [blob {blob}; commits {', '.join(sorted(index.versions[path][blob]))}]"
            for recovery in analysis.parse_recoveries
        )
    surface.guarded_only = set(surface.required) - bare_pairs
    surface.stats = {
        "files_analyzed": sorted(demand),
        "entrypoint_paths": sorted(entrypaths & demand.keys()),
        "files_with_reachable_functions": sorted({p for (p, _), a in memo.items() if a.reachable}),
        "commits_with_audited_changes": len({c for p, b in memo for c in index.versions[p][b]}),
        "revisions_read": sum(len(index.versions[p][b]) for p, b in memo),
        "distinct_file_versions": len(memo),
        "analysis_passes": analysis_passes,
        "versions_prepared": len(prepared),
    }
    return surface


def audit_history() -> Surface:
    ref = _full_history_ref()  # pin once; a concurrent fetch cannot mix DAGs
    index = _history_index(ref)
    print(f"[audit] indexed {len(index.versions)} historical Python paths", file=sys.stderr, flush=True)
    surface = _audit_versions(
        index, _entrypoint_paths(index), lambda path, blobs: _batch_read_blobs(blobs),
        progress=True,
    )
    if not surface.stats["entrypoint_paths"]:
        raise AuditError("No updater entrypoint found in origin/main history")
    surface.stats.update({
        "mode": "history",
        "discovery": index.discovery_stats,
        "coverage": "All reachable commits inventoried; distinct selected path/blob versions analyzed. Commit evidence lists version witnesses, not every unchanged descendant.",
        "history_ref": ref,
        "complete_history": True,
        "commits": int(_git("rev-list", "--count", ref)),
        "roots": sorted(_git("rev-list", "--max-parents=0", ref).decode().split()),
    })
    return surface


def audit_tree() -> Surface:
    """Audit the current tree without reading any Git history (shallow CI safe)."""
    names = _git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py")
    paths = {p.decode() for p in names.split(b"\0") if p}
    index = HistoryIndex()
    entrypaths = set()
    for path in sorted(paths):
        file = REPO_ROOT / path
        if not file.is_file():  # tracked deletion in the working tree
            continue
        index.versions[path] = {"worktree": {"worktree"}}
        source = file.read_bytes()
        if _has_entrypoint(source, path):
            entrypaths.add(path)

    def read_sources(path: str, blobs: list[str]) -> dict[str, str]:
        return {"worktree": _decode_source((REPO_ROOT / path).read_bytes(), path)}

    surface = _audit_versions(index, entrypaths, read_sources)
    surface.stats["mode"] = "tree"
    return surface


def audit_union() -> Surface:
    """The frozen contract: every name any SHIPPED updater can lazy-load
    after a checkout swap (full history walk over origin/main, following
    the updater through every path it has lived at), UNION everything the
    CURRENT updater loads (so a future rename turns the test red before it
    bricks a live update).

    A pair is guarded_only only when EVERY load site — historical or
    current — sits under a swallowing ``try``.
    """
    history = audit_history()
    tree = audit_tree()

    surface = Surface()
    for part in (history, tree):
        for key, commits in part.required.items():
            surface.required.setdefault(key, set()).update(commits)
        for key, kinds in part.kinds.items():
            surface.kinds.setdefault(key, set()).update(kinds)
        for key, sites in part.sites.items():
            surface.sites.setdefault(key, set()).update(sites)
        surface.unresolved.update(part.unresolved)
        surface.parse_recoveries.update(part.parse_recoveries)

    bare: set[tuple[str, str]] = set()
    for part in (history, tree):
        bare |= set(part.required) - part.guarded_only
    surface.guarded_only = set(surface.required) - bare
    witnesses = {commit for commits in history.required.values() for commit in commits}
    for module, symbol, witness, site in REVIEWED_DYNAMIC_LOADS:
        if witness not in witnesses:
            continue
        key = (module, symbol)
        surface.required.setdefault(key, set()).add(witness)
        surface.kinds.setdefault(key, set()).add("reviewed-module-call")
        surface.sites.setdefault(key, set()).add(site)
        surface.guarded_only.discard(key)
    surface.stats = {"mode": "union", "history": history.stats, "tree": tree.stats}
    return surface


# ─── resolution against the working tree ────────────────────────────────


def resolve_in_tree(module: str, symbol: str | None, root: Path) -> tuple[bool, str]:
    """Does *module* (and *symbol*) exist in the tree at *root*?

    Static resolution against the FILES, deliberately: importing would
    RUN the module, and the question is what an updater finds on disk,
    not what this interpreter can execute.
    """
    rel = Path(module.replace(".", "/"))
    for candidate in (root / f"{rel}.py", root / rel / "__init__.py"):
        if candidate.is_file():
            path = candidate
            break
    else:
        return False, f"module {module} not found"

    if symbol is None:
        return True, ""

    # `from hermes_cli import gateway_windows` names a SUBMODULE, not an
    # attribute of the package body.
    submodule = root / rel / symbol
    if submodule.with_suffix(".py").is_file() or (submodule / "__init__.py").is_file():
        return True, ""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError) as exc:
        return False, f"{module}: unreadable ({exc})"

    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True, ""
            # Function locals and class members are not module exports.
            continue
        pending.extend(ast.iter_child_nodes(node))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return True, ""
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == symbol:
                return True, ""
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # A re-export counts.
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == symbol:
                    return True, ""

    return False, f"{module}.{symbol} not found"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero when this tree is missing something an old "
        "updater needs.",
    )
    parser.add_argument(
        "--explain", metavar="MODULE", help="Show every requirement on MODULE."
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Audit only the complete origin/main history. By default the "
        "audit unions that history with the current working tree. Both "
        "modes require a full clone; CI reads the checked-in frozen JSON.",
    )
    parser.add_argument(
        "--freeze",
        metavar="PATH",
        help="Write the surface as JSON (the file the enforcing test reads). "
        "The default audits the union of the current tree AND the full "
        "shipped-history walk over origin/main (needs a full clone); "
        "--history freezes the history half alone.",
    )
    ns = parser.parse_args(argv)

    try:
        surface = audit_history() if ns.history else audit_union()
    except AuditError as exc:
        parser.error(str(exc))

    if ns.freeze:
        payload = {
            "_comment": (
                "Generated by scripts/audit-old-updater-imports.py --freeze. "
                "Names an already-running `hermes update` loads from the NEW "
                "tree after the checkout swap. Deleting a bare name bricks "
                "every release that loads it, mid-update, on a half-new "
                "tree. Regenerate after changing the update flow; never "
                "hand-trim. History enumeration is complete; static call-graph "
                "limits and unresolved_dynamic still require manual review."
            ),
            "stats": surface.stats,
            "unresolved_dynamic": sorted(surface.unresolved),
            "parse_recoveries": sorted(surface.parse_recoveries),
            "bare": sorted(
                f"{m}::{s}"
                for (m, s) in surface.required
                if (m, s) not in surface.guarded_only
            ),
            "guarded_only": sorted(
                f"{m}::{s}" for (m, s) in surface.guarded_only
            ),
        }
        Path(ns.freeze).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if surface.parse_recoveries:
            print("Historical parse recoveries (all arms audited; recorded in JSON):")
            for recovery in sorted(surface.parse_recoveries):
                print(f"  {recovery}")
        print(
            f"froze {len(payload['bare'])} bare + "
            f"{len(payload['guarded_only'])} guarded pairs -> {ns.freeze}"
        )
        return 0

    missing = []
    soft_missing = []
    for (module, symbol), commits in sorted(surface.required.items()):
        ok, why = resolve_in_tree(module, symbol or None, REPO_ROOT)
        if not ok:
            if (module, symbol) in surface.guarded_only:
                soft_missing.append((module, symbol, sorted(commits), why))
            else:
                missing.append((module, symbol, sorted(commits), why))

    if ns.explain:
        print(f"Requirements on {ns.explain!r}:")
        for (module, symbol), commits in sorted(surface.required.items()):
            if module != ns.explain:
                continue
            kinds = "/".join(sorted(surface.kinds[(module, symbol)]))
            where = ", ".join(sorted(surface.sites[(module, symbol)])[:3])
            print(
                f"  {module}.{symbol or '<module>'}  [{kinds}]"
                f"  {len(commits)} commits  {where}"
            )
        return 0

    if ns.json:
        print(
            json.dumps(
                {
                    "stats": surface.stats,
                    "required": [
                        {
                            "module": m,
                            "symbol": s or None,
                            "kinds": sorted(surface.kinds[(m, s)]),
                            "commits": sorted(c),
                            "sites": sorted(surface.sites[(m, s)]),
                            "guarded_only": (m, s) in surface.guarded_only,
                        }
                        for (m, s), c in sorted(surface.required.items())
                    ],
                    "unresolved_dynamic": sorted(surface.unresolved),
                    "parse_recoveries": sorted(surface.parse_recoveries),
                    "missing": [
                        {"module": m, "symbol": s or None, "commits": c, "why": w}
                        for m, s, c, w in missing
                    ],
                    "soft_missing": [
                        {"module": m, "symbol": s or None, "commits": c, "why": w}
                        for m, s, c, w in soft_missing
                    ],
                },
                indent=2,
            )
        )
        return 1 if (missing and ns.check) else 0

    st = surface.stats
    if st.get("mode") == "union":
        hist, tree = st["history"], st["tree"]
        print(
            f"Union of {hist['commits']} reachable shipped commits in the update "
            f"flow ({hist['revisions_read']} file revisions, "
            f"{hist['distinct_file_versions']} distinct versions) and the "
            f"current tree ({len(tree['files_analyzed'])} files)."
        )
    elif st.get("mode") == "tree":
        print(
            f"Audited the update flow in this working tree: "
            f"{len(st['files_analyzed'])} files "
            f"({', '.join(st['files_analyzed'])})."
        )
    else:
        print(
            f"Walked every reachable shipped commit, auditing distinct update versions: "
            f"{st['commits']} commits, {st['revisions_read']} file revisions, "
            f"{st['distinct_file_versions']} distinct versions."
        )
    print()
    by_kind: dict[str, int] = {}
    for kinds in surface.kinds.values():
        for kind in kinds:
            by_kind[kind] = by_kind.get(kind, 0) + 1
    kinds_summary = ", ".join(f"{v} {k}" for k, v in sorted(by_kind.items()))
    hard = {k for k in surface.required if k not in surface.guarded_only}
    print(
        f"FROZEN COMPAT SURFACE — {len(hard)} module/symbol pairs an old "
        f"updater can load BARE from the NEW tree ({kinds_summary} overall; "
        f"{len(surface.guarded_only)} more guarded-only, listed after):"
    )
    by_module: dict[str, list[str]] = {}
    for module, symbol in hard:
        by_module.setdefault(module, []).append(symbol or "<module>")
    for module in sorted(by_module):
        print(f"  {module}: {', '.join(sorted(by_module[module]))}")

    if surface.guarded_only:
        print()
        print(
            f"GUARDED-ONLY — {len(surface.guarded_only)} pairs loaded solely "
            f"under a swallowing try (deleting one degrades a fallback arm, "
            f"not the update):"
        )
        by_module = {}
        for module, symbol in surface.guarded_only:
            by_module.setdefault(module, []).append(symbol or "<module>")
        for module in sorted(by_module):
            print(f"  {module}: {', '.join(sorted(by_module[module]))}")

    if surface.parse_recoveries:
        print("Historical parse recoveries (all arms audited):")
        for recovery in sorted(surface.parse_recoveries):
            print(f"    {recovery}")

    if surface.unresolved:
        print()
        print(
            f"! {len(surface.unresolved)} dynamic load(s) this script cannot "
            f"resolve — audit by hand before deleting anything they may reach:"
        )
        for item in sorted(surface.unresolved):
            print(f"    {item}")

    print()
    if missing:
        print(f"X {len(missing)} name(s) an old updater needs are GONE:")
        for module, symbol, commits, why in missing:
            kinds = "/".join(sorted(surface.kinds[(module, symbol or "")]))
            where = ", ".join(sorted(surface.sites[(module, symbol or "")])[:2])
            shown = ", ".join(commits[:3])
            more = f" +{len(commits) - 3}" if len(commits) > 3 else ""
            print(f"    [{kinds}] {why}\n        from {where}  [{shown}{more}]")
    else:
        print("OK: every name an old updater needs still exists in this tree")

    if soft_missing:
        print()
        print(
            f"o {len(soft_missing)} guarded-only name(s) gone — fallback arms "
            f"now taken, worth knowing but not fatal:"
        )
        for module, symbol, commits, why in soft_missing:
            where = ", ".join(sorted(surface.sites[(module, symbol or "")])[:2])
            print(f"    {why}  (from {where})")

    return 1 if (missing and ns.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
