"""Architectural guardrails.

These tests fail when the architecture erodes, which is the failure mode that matters
most: every individual violation looks reasonable in isolation and is expensive to undo
once it has spread.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOVA = ROOT / "nova"

#: Runtime vocabulary that must not leak into NOVA's own interfaces.
RUNTIME_WORDS = ("hermes", "profile", "soul", "kanban", "cordis", "dsh")


def nova_python_files() -> list[Path]:
    return [p for p in NOVA.rglob("*.py") if "__pycache__" not in p.parts]


def _import_roots(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        return {node.module.split(".")[0]}
    return set()


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        names |= _import_roots(node)
    return names


def module_level_imports(path: Path) -> set[str]:
    """Imports that run at import time.

    An import nested inside a function body runs only when that function is called, so it
    does not decide whether the module loads. The distinction is the whole difference
    between "NOVA requires the runtime" and "this one function requires the runtime".
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    stack: list[tuple[ast.AST, bool]] = [(tree, True)]
    while stack:
        node, at_module_level = stack.pop()
        if at_module_level:
            names |= _import_roots(node)
        deferred = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for child in ast.iter_child_nodes(node):
            stack.append((child, at_module_level and not deferred))
    return names


def adapter_package(path: Path) -> str | None:
    """``nova/runtime/<adapter>/`` for a file inside one, else ``None``."""
    parts = path.relative_to(NOVA).parts
    if len(parts) >= 3 and parts[0] == "runtime":
        return parts[1]
    return None


#: Top-level modules that only exist when the runtime is installed.
RUNTIME_PACKAGES = {
    "hermes_cli", "hermes_constants", "hermes_state", "hermes_logging",
    "agent", "gateway", "tools", "toolsets", "cli", "run_agent", "model_tools",
}


def test_platform_layer_never_imports_the_runtime_at_module_level():
    """NOVA must load and test with only the stdlib and PyYAML present.

    A module-level runtime import breaks that for the whole package, so it is forbidden
    everywhere — including inside an adapter, which must stay importable so the registry
    can list it on a machine where its runtime is absent.
    """
    offenders: list[str] = []
    for path in nova_python_files():
        leaked = module_level_imports(path) & RUNTIME_PACKAGES
        if leaked:
            offenders.append(f"{path.relative_to(ROOT)}: imports {', '.join(sorted(leaked))}")
    assert not offenders, "platform code imported the runtime at module level:\n" + "\n".join(
        offenders
    )


def test_only_an_adapter_may_import_the_runtime_lazily():
    """A deferred runtime import is a capability the adapter borrows, not a dependency.

    Some things only the runtime can do — pulling text out of a PDF, for one. Reimplementing
    them in NOVA would be worse than calling them, but the call must be confined: inside
    ``nova/runtime/<adapter>/``, in a function body, where the runtime is by definition
    installed because the adapter was selected.
    """
    offenders: list[str] = []
    for path in nova_python_files():
        lazy = (imported_modules(path) - module_level_imports(path)) & RUNTIME_PACKAGES
        if lazy and adapter_package(path) is None:
            offenders.append(
                f"{path.relative_to(ROOT)}: lazily imports {', '.join(sorted(lazy))} "
                f"outside an adapter package"
            )
    assert not offenders, "\n".join(offenders)


def test_the_adapter_packages_still_import_without_their_runtime():
    """The proof that the lazy import stays lazy. Costs one subprocess, catches a real bug."""
    probe = (
        "import importlib, sys; "
        "sys.modules.update({name: None for name in "
        f"{sorted(RUNTIME_PACKAGES)!r}"
        "}); "
        "importlib.import_module('nova.runtime.hermes.adapter')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr


def test_platform_layer_depends_only_on_stdlib_and_yaml():
    """Keeping the install-time dependency surface at zero is what makes the layer portable.

    Scoped to module-level imports for the same reason as the test above: a lazy import
    inside an adapter is not something ``pip install nova`` has to satisfy. The two tests
    that follow it police where those lazy imports may live.
    """
    allowed_third_party = {"yaml"}
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []
    for path in nova_python_files():
        for name in module_level_imports(path):
            if name in stdlib or name in allowed_third_party or name == "nova":
                continue
            offenders.append(f"{path.relative_to(ROOT)}: imports {name}")
    assert not offenders, "unexpected dependency:\n" + "\n".join(offenders)


def test_lazy_imports_outside_the_stdlib_are_confined_to_adapters():
    """Otherwise the rule above would be trivially escapable by indenting an import."""
    allowed_third_party = {"yaml"}
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []
    for path in nova_python_files():
        if adapter_package(path) is not None:
            continue
        for name in imported_modules(path) - module_level_imports(path):
            if name in stdlib or name in allowed_third_party or name == "nova":
                continue
            offenders.append(f"{path.relative_to(ROOT)}: lazily imports {name}")
    assert not offenders, "unexpected deferred dependency outside an adapter:\n" + "\n".join(
        offenders
    )


def test_runtime_contract_is_free_of_runtime_vocabulary():
    """The adapter interface must be expressed in NOVA's words, not one runtime's."""
    source = (NOVA / "runtime" / "base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            lowered = node.name.lower()
            for word in RUNTIME_WORDS:
                if word in lowered:
                    offenders.append(f"{node.name} (line {node.lineno}) contains {word!r}")
        elif isinstance(node, ast.arg):
            lowered = node.arg.lower()
            for word in RUNTIME_WORDS:
                if word in lowered:
                    offenders.append(f"argument {node.arg} (line {node.lineno}) contains {word!r}")
    assert not offenders, "runtime vocabulary leaked into the contract:\n" + "\n".join(offenders)


def code_without_prose(path: Path) -> str:
    """Source with docstrings and comments removed.

    Prose may name the runtime — explaining why a boundary exists requires it. Code may
    not: an identifier or a string literal naming the runtime outside the adapter is
    knowledge that has escaped its package.
    """
    import tokenize

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    out: list[str] = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type == tokenize.COMMENT:
                continue
            if token.type == tokenize.STRING:
                try:
                    value = ast.literal_eval(token.string)
                except (ValueError, SyntaxError):
                    value = None
                if isinstance(value, str) and value in docstrings:
                    continue
            out.append(token.string)
    return "\n".join(out)


def test_only_the_adapter_package_names_the_runtime_in_code():
    """Runtime knowledge is confined to one package, so a second adapter is additive."""
    offenders: list[str] = []
    for path in nova_python_files():
        rel = path.relative_to(ROOT).as_posix()
        if "runtime/hermes" in rel or rel.endswith("nova/runtime/registry.py"):
            continue
        if "hermes" in code_without_prose(path).lower():
            offenders.append(rel)
    assert not offenders, (
        "'hermes' appears in code outside the adapter package:\n" + "\n".join(offenders)
    )


def test_registry_names_the_runtime_only_inside_the_registration_hook():
    """The registry is the one seam allowed to know a bundled adapter's name.

    Structural rather than a count: every mention must sit inside the lazy-registration
    function, so the name cannot leak into the registry's general logic.
    """
    path = NOVA / "runtime" / "registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_builtins"
    )
    hook_lines = set(range(hook.lineno, (hook.end_lineno or hook.lineno) + 1))

    # Docstrings are prose and may name the runtime; exclude their line ranges.
    prose_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                literal = body[0].value
                if isinstance(literal.value, str):
                    prose_lines.update(
                        range(literal.lineno, (literal.end_lineno or literal.lineno) + 1)
                    )

    outside: list[int] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#", 1)[0]
        if "hermes" in code.lower() and number not in hook_lines and number not in prose_lines:
            outside.append(number)

    assert not outside, (
        f"the runtime name appears in registry code outside _ensure_builtins at line(s) "
        f"{outside}"
    )


def test_no_toplevel_platform_package_shadowing_stdlib():
    """A root 'platform' package breaks platform.system() in 40 upstream modules."""
    assert not (ROOT / "platform" / "__init__.py").exists()


def test_upstream_code_never_imports_nova():
    """The inverted dependency would make every upstream merge a conflict."""
    offenders: list[str] = []
    skip = {".git", "node_modules", "__pycache__", "apps", "website", ".venv"}
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in skip for part in rel.parts):
            continue
        if rel.as_posix().startswith(("nova/", "tests/platform/", "scripts/check_protected")):
            continue
        try:
            names = imported_modules(path)
        except SyntaxError:
            continue
        if "nova" in names:
            offenders.append(rel.as_posix())
    assert not offenders, "upstream code imports nova:\n" + "\n".join(offenders)


def test_protected_identifier_script_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_protected_identifiers.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_protected_identifier_script_catches_a_stdlib_shadow(tmp_path):
    """The guardrail must actually fire, not just pass vacuously."""
    shadow = ROOT / "platform" / "__init__.py"
    shadow.parent.mkdir(exist_ok=True)
    shadow.write_text("# temporary\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_protected_identifiers.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 1
        assert "shadows the stdlib" in result.stderr
    finally:
        shadow.unlink()
        if not any(shadow.parent.iterdir()):
            shadow.parent.rmdir()


@pytest.mark.parametrize("module", ["nova", "nova.spec", "nova.runtime", "nova.audit", "nova.apply"])
def test_public_modules_import_cleanly(module):
    __import__(module)
