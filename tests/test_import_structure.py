"""Structural import tests -- no deferred imports in dispatch methods."""
import ast
import inspect
import textwrap


def test_no_deferred_imports_in_registry_dispatch():
    """tools/registry.py dispatch() must not contain inline imports."""
    from tools.registry import ToolRegistry
    source = textwrap.dedent(inspect.getsource(ToolRegistry.dispatch))
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 0, (
        f"dispatch() has deferred imports that may create circular dependencies: "
        f"{[ast.dump(n) for n in imports]}"
    )


def test_async_bridge_importable():
    """agent/async_bridge.py is independently importable."""
    from agent.async_bridge import run_async
    assert callable(run_async)


def test_model_tools_importable():
    """model_tools.py imports without circular dependency errors."""
    import model_tools
    assert hasattr(model_tools, 'handle_function_call')


def test_registry_importable():
    """tools/registry.py imports without circular dependency errors."""
    from tools.registry import registry
    assert registry is not None


def test_model_tools_reexports_run_async():
    """model_tools._run_async still works for backward compat (e.g. send_message_tool)."""
    from model_tools import _run_async
    assert callable(_run_async)


def test_async_bridge_run_async_works():
    """run_async can actually execute a simple coroutine."""
    import asyncio
    from agent.async_bridge import run_async

    async def _double(x):
        return x * 2

    result = run_async(_double(21))
    assert result == 42
