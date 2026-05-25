from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_DIFF = REPO_ROOT / "scripts" / "lint_diff.py"


def _load_lint_diff():
    spec = importlib.util.spec_from_file_location("lint_diff_under_test", LINT_DIFF)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ty_panic_paths_are_normalized_for_ref_diffs():
    lint_diff = _load_lint_diff()
    base = lint_diff._normalize_ty(
        [
            {
                "check_name": "panic",
                "location": {"path": "", "positions": {"begin": {"line": 1}}},
                "description": "Panicked when checking `/tmp/lint-base/tools/checkpoint_manager.py`: `infer_expression_type_impl(Id(abc123)): cycle`",
            }
        ]
    )
    head = lint_diff._normalize_ty(
        [
            {
                "check_name": "panic",
                "location": {"path": "", "positions": {"begin": {"line": 1}}},
                "description": "Panicked when checking `/home/runner/work/hermes/tools/checkpoint_manager.py`: `infer_expression_type_impl(Id(def456)): cycle`",
            }
        ]
    )

    new, fixed, unchanged = lint_diff._diff(base, head)

    assert new == []
    assert fixed == []
    assert len(unchanged) == 1
    assert unchanged[0]["message"] == "Panicked when checking `tools/checkpoint_manager.py`: `infer_expression_type_impl(Id(<normalized>)): cycle`"


def test_tool_report_totals_follow_stable_unique_keys():
    lint_diff = _load_lint_diff()
    base = [
        {"tool": "ty", "rule": "panic", "path": "", "line": 1, "message": "same"},
    ]
    head = [
        {"tool": "ty", "rule": "panic", "path": "", "line": 1, "message": "same"},
        {"tool": "ty", "rule": "panic", "path": "", "line": 1, "message": "same"},
    ]

    report = lint_diff._tool_report("ty", base, head, True)

    assert "**Unique total:** 1 on HEAD, 1 on base (➖ 0)" in report
    assert "_Raw diagnostic events: 2 on HEAD, 1 on base._" in report
    assert "**🆕 New issues:** none" in report
