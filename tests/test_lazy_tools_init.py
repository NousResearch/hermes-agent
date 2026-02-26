"""Verify tools/__init__.py is clean -- no eager imports of tool modules."""
import ast
import pathlib

TOOLS_INIT = pathlib.Path(__file__).resolve().parent.parent / "tools" / "__init__.py"


def test_tools_init_has_no_tool_imports():
    """tools/__init__.py must not eagerly import tool modules.

    Tool discovery is handled by model_tools._discover_tools() which uses
    importlib.import_module with try/except for graceful degradation.
    The __init__.py eager imports are redundant and cause ImportError
    when optional dependencies (firecrawl, fal_client) aren't installed.
    """
    source = TOOLS_INIT.read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("tools."):
                imports.append(f"from {node.module} import ...")
            elif node.level and node.level > 0:
                # Relative imports like "from .web_tools import ..."
                mod = "." * node.level + (node.module or "")
                imports.append(f"from {mod} import ...")
    assert len(imports) == 0, (
        f"tools/__init__.py has eager tool imports that should be removed: {imports}"
    )


def test_tools_package_importable():
    """import tools succeeds even without optional dependencies."""
    import tools

    assert tools.__doc__ is not None


def test_check_file_requirements_removed():
    """check_file_requirements should no longer exist in tools/__init__.py.

    The function was only used by tools/file_tools.py._check_file_reqs()
    which should now import directly from tools.terminal_tool.
    """
    source = TOOLS_INIT.read_text()
    tree = ast.parse(source)
    function_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
    assert "check_file_requirements" not in function_names, (
        "check_file_requirements should be removed from tools/__init__.py"
    )


def test_file_tools_check_reqs_imports_directly():
    """tools/file_tools.py._check_file_reqs() should import from tools.terminal_tool,
    not from tools (the package __init__).
    """
    file_tools_path = TOOLS_INIT.parent / "file_tools.py"
    source = file_tools_path.read_text()
    tree = ast.parse(source)

    # Find the _check_file_reqs function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_check_file_reqs":
            # Walk the function body to find import statements
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom):
                    assert child.module != "tools", (
                        "_check_file_reqs should not import from 'tools' package "
                        "(was: 'from tools import check_file_requirements'). "
                        "Should import directly from tools.terminal_tool instead."
                    )
            break
    else:
        # _check_file_reqs function not found at all -- that's a problem
        raise AssertionError("_check_file_reqs function not found in tools/file_tools.py")


def test_tool_discovery_still_works():
    """model_tools._discover_tools() still discovers tools after __init__.py cleanup."""
    from tools.registry import registry

    # Ensure tools are discovered
    from model_tools import _discover_tools
    _discover_tools()

    names = registry.get_all_tool_names()
    assert len(names) > 5, f"Expected 5+ tools, got {len(names)}: {names}"
    # terminal is a core tool that should always be discoverable
    assert any("terminal" in n for n in names), (
        f"terminal tool not found in discovered tools: {names}"
    )
