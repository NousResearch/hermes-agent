#!/usr/bin/env python3
"""Enforce the Phase 7 command and capability-policy ownership hard cut.

Static architecture check, intentionally separate from runtime authorization.
Run: python scripts/check_phase7_boundaries.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETIRED_MODULES = {
    "hermes_cli.commands", "hermes_cli.slash_exec",
    "hermes_cli.toolset_scope", "hermes_cli.commands_platforms",
}
RETIRED_FILES = {
    "hermes_cli/slash_exec.py", "hermes_cli/toolset_scope.py",
    "hermes_cli/commands_platforms.py",
}
TOOL_FACADES = {"hermes_cli.tools_config", "hermes_cli.tools_config_mcp"}
RETIRED_TOOL_NAMES = {
    "_save_platform_tools", "_apply_toolset_change", "_apply_mcp_change",
    "_toolset_configuration_platform", "_CONFIG_ONLY_TOOLSETS", "_cfg_section",
    "_get_platform_tools", "get_platform_tools",
    "_get_plugin_toolset_keys", "get_plugin_toolset_keys",
    "_configurable_keys", "configurable_toolset_keys",
    "_platform_default_toolset", "platform_default_toolset",
    "_coerce_platform_toolsets_value", "coerce_platform_toolsets_value",
    "enabled_mcp_server_names", "_DEFAULT_OFF_TOOLSETS",
    "_RECENTLY_SHIPPED_TOOLSETS", "_warned_invalid_platform_toolsets",
    "_homeassistant_credentials_present", "_xai_credentials_present",
    "_enable_recently_shipped_toolsets", "_configurable_subset_of",
    "_default_off_toolsets", "_platform_default_keys", "_explicit_toolsets",
    "_composite_toolsets", "_enabled_plugin_toolsets", "_context_engine_active",
    "_prune_toolsets_stripped_by_disabled", "_recover_platform_native_toolsets",
    "_merge_mcp_servers", "_warn_all_invalid_platform_toolsets",
    "_TOOLSET_PLATFORM_RESTRICTIONS", "_toolset_allowed_for_platform",
}
RETIRED_TOOL_NAMES |= {
    "_parse_enabled_flag", "parse_enabled_flag", "save_platform_tools",
    "apply_toolset_change", "apply_mcp_change", "toolset_configuration_platform",
    "CONFIG_ONLY_TOOLSETS", "config_section", "_current_platform_tools",
    "_tool_policy", "_tool_settings", "PLATFORM_DEFAULT_TOOLSETS",
    "CONFIGURABLE_TOOLSET_KEYS",
}
SYMBOL_OWNERS = {
    "CommandDef": "commands/__init__.py",
    "COMMAND_REGISTRY": "commands/__init__.py",
    "CommandContext": "commands/execution.py",
    "CommandReply": "commands/execution.py",
}
POLICY_FILES = {
    "tools/platform_policy.py", "tools/toolset_scope.py", "tools/toolset_selection.py",
}
DOMAIN_FORBIDDEN = {"hermes_cli", "cli", "gateway", "tui_gateway", "acp_adapter", "model_tools"}
MODULE_LOADERS = {"import_module", "__import__", "_mod", "_tools_mod", "module_loader"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build",
             "website", "skills", "optional-skills", "apps", "evals", "docs", "MagicMock"}
# Existing frozen external compatibility tests deliberately resolve manifest entries.
EXTERNAL_CONTRACT_TEST = "tests/test_compat_manifest_targets.py"


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        return left + right if left is not None and right is not None else None
    return None


def _import_base(node: ast.ImportFrom, path: str) -> str:
    if not node.level:
        return node.module or ""
    package = path.replace("\\", "/").split("/")[:-1]
    package = package[:len(package) - node.level + 1]
    return ".".join([*package, *filter(None, (node.module or "").split("."))])


def _resolve(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Call):
        callee = (_resolve(node.func, aliases) or "").rsplit(".", 1)[-1]
        if callee in MODULE_LOADERS and node.args:
            module = _literal(node.args[0])
            package = next((_literal(k.value) for k in node.keywords if k.arg == "package"), None)
            if module and module.startswith(".") and package:
                level = len(module) - len(module.lstrip("."))
                base = package.split(".")[:len(package.split(".")) - level + 1]
                return ".".join([*base, module.lstrip(".")]).rstrip(".")
            return module
        if callee in {"getattr", "hasattr", "setattr", "delattr"} and len(node.args) >= 2:
            base, attr = _resolve(node.args[0], aliases), _literal(node.args[1])
            return f"{base}.{attr}" if base and attr else None
    return None


def _bindings(target: ast.AST, value: ast.AST):
    if isinstance(target, ast.Name):
        yield target.id, value
    elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        for child, expression in zip(target.elts, value.elts):
            yield from _bindings(child, expression)


def _dependency_problem(name: str, path: str) -> str | None:
    if any(name == module or name.startswith(module + ".") for module in RETIRED_MODULES):
        return f"retired CLI command/scope dependency: {name}"
    for facade in TOOL_FACADES:
        if name.startswith(facade + "."):
            member = name[len(facade) + 1:].split(".", 1)[0]
            if member in RETIRED_TOOL_NAMES or member == "*":
                return f"CLI-owned capability/settings dependency: {name}"
    if path.startswith("commands/") or path in POLICY_FILES:
        if name.split(".", 1)[0] in DOMAIN_FORBIDDEN:
            return f"shared domain imports application implementation: {name}"
    return None


def scan_source(source: str, path: str) -> list[str]:
    """Return file/line diagnostics for supported Python dependency forms."""
    path = path.replace("\\", "/")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: cannot audit invalid Python: {error.msg}"]
    nodes = list(ast.walk(tree))
    aliases: dict[str, str] = {}
    imported = []
    for node in nodes:
        if isinstance(node, ast.Import):
            for entry in node.names:
                imported.append((node, entry.name))
                aliases[entry.asname or entry.name.split(".")[0]] = (
                    entry.name if entry.asname else entry.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(node, path)
            imported.append((node, base))
            for entry in node.names:
                name = f"{base}.{entry.name}" if base else entry.name
                imported.append((node, name))
                aliases[entry.asname or entry.name] = name
    pending = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                pending.extend(_bindings(target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pending.extend(_bindings(node.target, node.value))
    # Follow simple alias chains and tuple-unpacked module-loader results. This is
    # a dependency lint, not execution or a general Python data-flow interpreter.
    while pending:
        remaining = []
        changed = False
        for name, expression in pending:
            resolved = _resolve(expression, aliases)
            if name not in aliases and resolved and resolved.split(".", 1)[0] in {
                "hermes_cli", "commands", "tools", "gateway", "cli", "model_tools",
                "tui_gateway", "acp_adapter", "importlib",
            }:
                aliases[name] = resolved
                changed = True
            elif name not in aliases:
                remaining.append((name, expression))
        if not changed:
            break
        pending = remaining

    problems = set()
    def flag(node, reason):
        problems.add(f"{path}:{node.lineno}: {reason}")

    for node, name in imported:
        if reason := _dependency_problem(name, path):
            flag(node, reason)
    for node in nodes:
        if isinstance(node, (ast.Attribute, ast.Call)):
            resolved = _resolve(node, aliases)
            if resolved and (reason := _dependency_problem(resolved, path)):
                flag(node, reason)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = SYMBOL_OWNERS.get(node.name)
            if owner and path != owner and not path.startswith("tests/"):
                flag(node, f"{node.name} is owned by {owner}")
            if path in {"hermes_cli/tools_config.py", "hermes_cli/tools_config_mcp.py"}:
                if node.name in RETIRED_TOOL_NAMES - {"_current_platform_tools", "_tool_policy", "_tool_settings"}:
                    flag(node, f"retired runtime/settings implementation: {node.name}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            owner = SYMBOL_OWNERS.get(node.id)
            if owner and path != owner and not path.startswith("tests/"):
                flag(node, f"{node.id} is owned by {owner}")
            if path in {"hermes_cli/tools_config.py", "hermes_cli/tools_config_mcp.py"}:
                if node.id in RETIRED_TOOL_NAMES - {"_current_platform_tools", "_tool_policy", "_tool_settings"}:
                    flag(node, f"retired runtime/settings assignment: {node.id}")
    return sorted(problems)


def python_files(root: Path):
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not name.startswith(".")]
        for name in files:
            if name.endswith(".py"):
                yield Path(directory) / name


def audit(root: Path = ROOT) -> tuple[list[str], int]:
    problems = [f"{name}: retired internal module must stay deleted"
                for name in sorted(RETIRED_FILES) if (root / name).exists()]
    count = 0
    for path in sorted(python_files(root)):
        rel = path.relative_to(root).as_posix()
        if rel == EXTERNAL_CONTRACT_TEST:
            continue
        source = path.read_text(encoding="utf-8-sig")
        count += 1
        # Parse domain owners and potential consumers. Other files cannot
        # contain these dependencies or definitions without naming the boundary.
        if (rel.startswith("commands/") or rel in POLICY_FILES
                or "hermes_cli" in source
                or any(name in source for name in SYMBOL_OWNERS)):
            problems.extend(scan_source(source, rel))
    return sorted(set(problems)), count


def main() -> int:
    problems, count = audit()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Phase 7 ownership boundaries: OK ({count} Python files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
