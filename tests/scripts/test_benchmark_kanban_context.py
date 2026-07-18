from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "benchmark_kanban_context.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_kanban_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kanban_context_benchmark_reports_utf8_omissions_and_recovery():
    report = _load_benchmark_module().run_benchmark()

    assert report["full"]["utf8_bytes"] > report["compact"]["utf8_bytes"]
    assert report["savings"]["utf8_bytes"] == (
        report["full"]["utf8_bytes"] - report["compact"]["utf8_bytes"]
    )
    assert report["omissions"] == {
        "comments": 34,
        "parents": 12,
        "prior_attempts": 7,
    }
    assert report["recovery"] == {
        "cli": (
            "hermes kanban context <task_id> --run-id <run_id> "
            "--field partial_summary_full"
        ),
        "model_tool": "kanban_show(task_id=<task_id>, full_context=true)",
    }
    assert all(report["contracts"].values())
