"""Structural invariants for the gateway/slash_commands/ package.

These assertions pin relationships rather than a command or method count, so adding
a command does not require snapshot churn.
"""

from __future__ import annotations

import ast
import builtins
import dis
import importlib
import pkgutil
import subprocess
import sys
import types
from pathlib import Path

from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.slash_commands.registry import GATEWAY_SLASH_HANDLERS

ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = ROOT / "gateway" / "slash_commands"
LEAF_MIXINS = list(GatewaySlashCommandsMixin.__bases__)


def test_every_registry_binding_names_a_real_command_and_method() -> None:
    """The binding table may not drift from command metadata or the mixin."""
    from hermes_cli.commands import resolve_command

    for command, method in GATEWAY_SLASH_HANDLERS.items():
        assert resolve_command(command) is not None, (
            f"registry binds unknown command {command!r}"
        )
        assert hasattr(GatewaySlashCommandsMixin, method), (
            f"registry binds {command!r} to missing method {method!r}"
        )


def test_registry_bindings_resolve_to_exact_mixin_function_objects() -> None:
    """Table dispatch must bind the function the old direct call resolved."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    for command, method in GATEWAY_SLASH_HANDLERS.items():
        bound = getattr(runner, method)
        assert bound.__self__ is runner
        assert bound.__func__ is getattr(GatewaySlashCommandsMixin, method), (
            f"{command!r} -> {method!r} resolves to a different function"
        )


def test_registry_values_are_unique() -> None:
    """One plain command binding per handler keeps dispatch unambiguous."""
    values = list(GATEWAY_SLASH_HANDLERS.values())
    assert len(values) == len(set(values))


def test_leaf_mixin_methods_are_disjoint() -> None:
    """MRO order must not decide which leaf implementation wins."""
    seen: dict[str, str] = {}
    for mixin in LEAF_MIXINS:
        for name, value in vars(mixin).items():
            if name.startswith("__") or not callable(value):
                continue
            assert name not in seen, (
                f"{name!r} is defined by {seen[name]} and {mixin.__name__}"
            )
            seen[name] = mixin.__name__


def test_every_leaf_method_is_reachable_through_composed_mixin() -> None:
    for mixin in LEAF_MIXINS:
        for name, value in vars(mixin).items():
            if name.startswith("__") or not callable(value):
                continue
            expected = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
            assert getattr(GatewaySlashCommandsMixin, name) is expected


def test_every_leaf_method_global_reference_resolves() -> None:
    """Selective leaf imports must cover globals used by nested code too."""

    def global_names(code: types.CodeType):
        for instruction in dis.get_instructions(code):
            if instruction.opname == "LOAD_GLOBAL":
                yield instruction.argval
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                yield from global_names(constant)

    for mixin in LEAF_MIXINS:
        for name, descriptor in vars(mixin).items():
            if name.startswith("__") or not callable(descriptor):
                continue
            function = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            missing = sorted(
                global_name
                for global_name in set(global_names(function.__code__))
                if global_name not in function.__globals__
                and not hasattr(builtins, global_name)
            )
            assert not missing, f"{mixin.__name__}.{name}: missing globals {missing}"


def test_no_leaf_module_imports_gateway_run_at_module_scope() -> None:
    """Call-time imports preserve the acyclic module graph."""
    offenders: list[str] = []
    for path in sorted(PKG_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "gateway.run"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("gateway.run"):
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"module-scope gateway.run imports: {offenders}"


def test_importing_package_does_not_import_gateway_run() -> None:
    code = (
        "import sys; import gateway.slash_commands; "
        "raise SystemExit('gateway.run' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-400:]


def test_every_leaf_uses_shared_logger_object() -> None:
    from gateway.slash_commands import _shared

    for _, module_name, _ in pkgutil.iter_modules([str(PKG_DIR)]):
        module = importlib.import_module(f"gateway.slash_commands.{module_name}")
        if hasattr(module, "logger"):
            assert module.logger is _shared.logger
    assert _shared.logger.name == "gateway.run"


def test_every_package_reexport_preserves_object_identity() -> None:
    """Every from-import in __init__ must expose the exact source object."""
    package = importlib.import_module("gateway.slash_commands")
    init_tree = ast.parse((PKG_DIR / "__init__.py").read_text(encoding="utf-8"))

    checked: list[str] = []
    for node in init_tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue
        assert node.module is not None
        source_module = importlib.import_module(node.module)
        for alias in node.names:
            exported = alias.asname or alias.name
            assert getattr(package, exported) is getattr(source_module, alias.name)
            checked.append(exported)
    assert checked, "no package re-exports were checked"


def test_composed_mixin_keeps_async_session_store_annotation() -> None:
    assert "async_session_store" in GatewaySlashCommandsMixin.__annotations__
