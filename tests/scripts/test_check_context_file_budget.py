from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_budget_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "check_context_file_budget.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_check_context_file_budget_under_test",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_passes_under_warning_budget(tmp_path, capsys):
    budget = _load_budget_module()
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("short runtime contract\n", encoding="utf-8")

    status = budget.main([
        str(context_file),
        "--warn-chars",
        "30",
        "--max-chars",
        "40",
    ])

    assert status == 0
    captured = capsys.readouterr()
    assert "chars=23" in captured.out
    assert "ok:" in captured.out
    assert captured.err == ""


def test_warns_between_warning_and_max_without_failing(tmp_path, capsys):
    budget = _load_budget_module()
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("x" * 35, encoding="utf-8")

    status = budget.main([
        str(context_file),
        "--warn-chars",
        "30",
        "--max-chars",
        "40",
    ])

    assert status == 0
    captured = capsys.readouterr()
    assert "chars=35" in captured.out
    assert "warning:" in captured.err
    assert "docs/reference" in captured.err


def test_strict_warning_can_fail_before_max(tmp_path, capsys):
    budget = _load_budget_module()
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("x" * 35, encoding="utf-8")

    status = budget.main([
        str(context_file),
        "--warn-chars",
        "30",
        "--max-chars",
        "40",
        "--strict-warning",
    ])

    assert status == 1
    captured = capsys.readouterr()
    assert "warning:" in captured.err


def test_fails_above_max_with_actionable_agents_md_error(tmp_path, capsys):
    budget = _load_budget_module()
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("x" * 41, encoding="utf-8")

    status = budget.main([
        str(context_file),
        "--warn-chars",
        "30",
        "--max-chars",
        "40",
    ])

    assert status == 1
    captured = capsys.readouterr()
    assert "chars=41" in captured.out
    assert "AGENTS.md is injected into every coding-agent session" in captured.err
    assert "Keep invariants inline" in captured.err
    assert "move explanatory reference material to docs/reference" in captured.err
    assert "short runtime-critical summaries plus links inline" in captured.err


def test_rejects_warning_budget_above_max(tmp_path, capsys):
    budget = _load_budget_module()
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("short\n", encoding="utf-8")

    status = budget.main([
        str(context_file),
        "--warn-chars",
        "50",
        "--max-chars",
        "40",
    ])

    assert status == 2
    captured = capsys.readouterr()
    assert "--warn-chars" in captured.err
