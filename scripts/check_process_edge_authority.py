"""Blocking structural check for retired ambient process-authority paths.

This is an AST-based CI checker, not a source-reading behavior test. The typed
broker is the only production module allowed to translate a process intent into
legacy environment helpers during migration. Once a production function opts
in to that broker, every direct subprocess boundary in that function must make
both its environment and stdin policy explicit.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (
    "agent",
    "cron",
    "gateway",
    "hermes_cli",
    "tools",
    "tui_gateway",
)
OWNER_FILES = {
    "tools/environments/local.py",
    "tools/child_process_authority.py",
}
_SUBPROCESS_METHODS = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "run",
}
_NO_POLICY_SUBPROCESS_METHODS = {
    "getoutput",
    "getstatusoutput",
}
_ASYNCIO_SUBPROCESS_METHODS = {
    "create_subprocess_exec",
    "create_subprocess_shell",
}


def _iter_python_files():
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if root.is_dir():
            yield from root.rglob("*.py")
    for name in ("cli.py", "hermes_constants.py"):
        path = REPO_ROOT / name
        if path.is_file():
            yield path


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
    )


def _is_ambient_env_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if _is_os_environ(node):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
        and _is_os_environ(node.func.value)
    ):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and node.args
        and _is_os_environ(node.args[0])
    ):
        return True
    if isinstance(node, ast.Dict):
        return any(
            key is None and _is_os_environ(value)
            for key, value in zip(node.keys, node.values)
        )
    return False


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _is_ambient_environment_access(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"environ", "environb"}
        )
    if isinstance(node, ast.Call):
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and func.attr in {"getenv", "getenvb", "putenv", "unsetenv"}
        )
    return False


def _function_calls_child_env_broker(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call) and _call_name(child) == "build_child_process_env"
        for child in ast.walk(node)
    )


def _imports_process_authority(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "tools.child_process_authority":
                return True
        elif isinstance(node, ast.Import):
            if any(
                alias.name == "tools.child_process_authority" for alias in node.names
            ):
                return True
    return False


def _module_aliases(tree: ast.AST, module: str) -> set[str]:
    aliases = {module}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    aliases.add(alias.asname or module)
    return aliases


def _direct_import_aliases(
    tree: ast.AST,
    module: str,
    names: set[str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != module:
            continue
        for alias in node.names:
            if alias.name in names:
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _spawn_call_name(
    node: ast.Call,
    *,
    subprocess_modules: set[str],
    subprocess_functions: dict[str, str],
    asyncio_modules: set[str],
    asyncio_functions: dict[str, str],
) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in subprocess_modules and func.attr in (
            _SUBPROCESS_METHODS | _NO_POLICY_SUBPROCESS_METHODS
        ):
            return f"subprocess.{func.attr}"
        if (
            func.value.id in asyncio_modules
            and func.attr in _ASYNCIO_SUBPROCESS_METHODS
        ):
            return f"asyncio.{func.attr}"
    if isinstance(func, ast.Name):
        if func.id in subprocess_functions:
            return f"subprocess.{subprocess_functions[func.id]}"
        if func.id in asyncio_functions:
            return f"asyncio.{asyncio_functions[func.id]}"
    return None


def _find_escape_hatches_in_source(source: str, rel: str) -> list[str]:
    if rel in OWNER_FILES:
        return []

    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError:
        return []

    subprocess_modules = _module_aliases(tree, "subprocess")
    subprocess_functions = _direct_import_aliases(
        tree,
        "subprocess",
        _SUBPROCESS_METHODS | _NO_POLICY_SUBPROCESS_METHODS,
    )
    asyncio_modules = _module_aliases(tree, "asyncio")
    asyncio_functions = _direct_import_aliases(
        tree,
        "asyncio",
        _ASYNCIO_SUBPROCESS_METHODS,
    )

    findings: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _function_calls_child_env_broker(function):
            continue
        for node in ast.walk(function):
            if _is_ambient_environment_access(node):
                findings.append(
                    f"{rel}:{getattr(node, 'lineno', '?')}: "
                    "direct ambient environment access beside child-env broker"
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}

        if (
            name == "hermes_subprocess_env"
            and "inherit_credentials" in keywords
            and _is_true(keywords["inherit_credentials"])
        ):
            findings.append(f"{rel}:{node.lineno}: untyped inherit_credentials=True")

        if (
            name == "build_subprocess_env"
            and "scrub_secrets" in keywords
            and _is_false(keywords["scrub_secrets"])
        ):
            findings.append(f"{rel}:{node.lineno}: untyped scrub_secrets=False")

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and node.args
            and _is_os_environ(node.args[0])
        ):
            findings.append(f"{rel}:{node.lineno}: ambient env.update(os.environ)")

    # God-files contain unrelated process surfaces. Once one function adopts
    # the broker, require explicit env/stdin on every spawn in that function,
    # without accidentally declaring every other function migrated too.
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _function_calls_child_env_broker(function):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            spawn_name = _spawn_call_name(
                node,
                subprocess_modules=subprocess_modules,
                subprocess_functions=subprocess_functions,
                asyncio_modules=asyncio_modules,
                asyncio_functions=asyncio_functions,
            )
            if spawn_name is None:
                continue
            if spawn_name.rsplit(".", 1)[-1] in _NO_POLICY_SUBPROCESS_METHODS:
                findings.append(
                    f"{rel}:{node.lineno}: {spawn_name} cannot carry typed env/stdin policy"
                )
                continue
            if "env" not in keywords:
                findings.append(
                    f"{rel}:{node.lineno}: {spawn_name} missing explicit env policy"
                )
            elif _is_ambient_env_expression(keywords["env"]):
                findings.append(
                    f"{rel}:{node.lineno}: {spawn_name} explicitly requests ambient env"
                )
            if "stdin" not in keywords:
                findings.append(
                    f"{rel}:{node.lineno}: {spawn_name} missing explicit stdin policy"
                )

    return findings


def _find_escape_hatches(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return _find_escape_hatches_in_source(source, rel)


def scan_repository() -> list[str]:
    return [
        finding
        for path in _iter_python_files()
        for finding in _find_escape_hatches(path)
    ]


def main() -> int:
    import sys

    findings = scan_repository()
    if not findings:
        print("Process-edge authority check passed.")
        return 0
    print("Production child-process authority bypasses remain:", file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    print(
        "Use tools.child_process_authority with a typed ChildProcessSpec.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
