"""Gateway run.py/config.py cross-file attribute contract validation.

``hermes update``'s stash-restore can leave ``gateway/run.py`` and
``gateway/config.py`` from *different* commits when the stash touches one file
but not the other. The stale ``run.py`` can reference ``config.<attr>`` values
that no longer exist on the current ``GatewayConfig`` — the import-time probe
stays green (the module imports cleanly) and the crash only fires inside
``GatewayRunner.__init__`` when the gateway actually starts.

This module AST-walks ``GatewayRunner`` (and its mixin methods) for
``self.config.<attr>`` / ``config.<attr>`` reads and verifies each attribute
resolves against the current ``GatewayConfig`` dataclass fields + methods. It
never instantiates, so it is side-effect-free and cheap to run as a
post-restore health check.

Real-world case: after a Windows ``hermes update``, every gateway start crashed
with ``AttributeError: 'GatewayConfig' object has no attribute
'default_reset_policy'`` until both files were restored from HEAD manually
(issue #105806).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional


def _parse(path: Path) -> ast.Module:
    with path.open(encoding="utf-8", errors="replace") as f:
        return ast.parse(f.read(), filename=str(path))


def extract_config_attr_reads(run_tree: ast.Module) -> set[str]:
    """All ``self.config.<attr>`` / ``config.<attr>`` attribute reads anywhere
    in ``GatewayRunner`` (including its mixin methods — the class body contains
    every method that runs on the instance)."""
    attrs: set[str] = set()
    for node in ast.walk(run_tree):
        if isinstance(node, ast.ClassDef) and node.name == "GatewayRunner":
            for item in ast.walk(node):
                if isinstance(item, ast.Attribute):
                    if (
                        isinstance(item.value, ast.Attribute)
                        and item.value.attr == "config"
                        and isinstance(item.value.value, ast.Name)
                        and item.value.value.id == "self"
                    ):
                        attrs.add(item.attr)
                    elif isinstance(item.value, ast.Name) and item.value.id == "config":
                        attrs.add(item.attr)
    return attrs


def extract_gateway_config_members(config_tree: ast.Module) -> set[str]:
    """Fields (dataclass annotations / assignments) + methods on GatewayConfig."""
    members: set[str] = set()
    for node in ast.walk(config_tree):
        if isinstance(node, ast.ClassDef) and node.name == "GatewayConfig":
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        members.add(item.target.id)
                    elif isinstance(item.target, ast.Tuple):
                        for e in item.target.elts:
                            if isinstance(e, ast.Name):
                                members.add(e.id)
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            members.add(tgt.id)
                        elif isinstance(tgt, ast.Tuple):
                            for e in tgt.elts:
                                if isinstance(e, ast.Name):
                                    members.add(e.id)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
    return members


def check_gateway_config_contract(
    agent_root: Path,
) -> Optional[list[str]]:
    """Validate the gateway config attribute contract.

    Returns None when the contract is consistent; otherwise the list of
    attribute names that ``run.py`` reads on ``config`` but that do not exist
    on the current ``GatewayConfig`` (i.e. a version mismatch).
    """
    run_py = agent_root / "gateway" / "run.py"
    config_py = agent_root / "gateway" / "config.py"
    if not run_py.exists() or not config_py.exists():
        return None
    try:
        run_tree = _parse(run_py)
        config_tree = _parse(config_py)
    except (SyntaxError, OSError, UnicodeDecodeError):
        # Syntax errors are handled by the existing syntax validation pass.
        return None

    reads = extract_config_attr_reads(run_tree)
    members = extract_gateway_config_members(config_tree)
    missing = sorted(reads - members)
    return missing or None
