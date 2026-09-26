from __future__ import annotations

import ast
from pathlib import Path

from gateway.hermes_tag import BUILTIN_CAPABILITIES, BUILTIN_OBLIGATIONS

PACKAGE = Path(__file__).resolve().parents[3] / "gateway" / "hermes_tag"


def test_every_kernel_python_file_is_below_two_thousand_lines():
    over = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in PACKAGE.glob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 2000
    }
    assert over == {}


def test_additive_kernel_does_not_import_runtime_or_vertical_owners():
    forbidden = {
        "gateway.run",
        "agent.tool_executor",
        "plugins.platforms.slack",
        "plugins.platforms.discord",
        "plugins.platforms.telegram",
        "plugins.platforms.whatsapp",
    }
    violations = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(name == item or name.startswith(f"{item}.") for item in forbidden):
                    violations.append((path.name, name))
    assert violations == []


def test_builtin_contract_cardinality_is_stable():
    assert len(BUILTIN_CAPABILITIES) == 20
    assert len(BUILTIN_OBLIGATIONS) == 28
