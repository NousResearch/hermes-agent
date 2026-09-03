from __future__ import annotations

import ast
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "content_engine"
NAMES = ("x_morning_article", "li_daily_package", "x_quote_scout", "x_thesis_incubator")


def test_canonical_cron_scripts_use_full_governed_chain():
    for name in NAMES:
        path = SCRIPT_DIR / f"{name}.py"
        assert path.is_file(), f"missing canonical source: {path}"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "_call_llm_chain" in source
        assert not any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "_llm_configs"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 0
            for node in ast.walk(tree)
        )
