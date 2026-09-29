"""A rep that crashed mid-write must be rewritten, not skipped forever.

`evals/readtool/runner.py` wrote each rep with `Path.write_text` (truncate, then
serialise) and skipped any rep whose file already existed. A crash between those
two steps left truncated JSON that every later run then skipped — and
`report.py` fed that file to an unguarded `json.loads`, so one interrupted rep
poisoned the cell permanently and crashed the whole comparison instead of
degrading.

Both directions are pinned here: a corrupt rep is re-run, a valid one is still
skipped (that skip is the resume behaviour the harness is built on).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

READTOOL_DIR = Path(__file__).resolve().parents[2] / "evals" / "readtool"
sys.path.insert(0, str(READTOOL_DIR))

import runner  # noqa: E402  (after sys.path: the harness imports its siblings by bare name)

LABEL = "unit-label"
SLUG = "unit_model"


def _argv(reps: int) -> list:
    return [
        "runner.py",
        "--model",
        "unit/model",
        "--provider",
        "nous",
        "--reps",
        str(reps),
        "--label",
        LABEL,
        "--tasks",
        runner.TASKS[0].task_id,
    ]


def _out_dir(tmp_path: Path) -> Path:
    return tmp_path / "results" / LABEL / SLUG


def _fake_run_task(calls: list):
    def run_task(task, model, provider, timeout_mult, toolsets):
        calls.append(task.task_id)
        return {
            "task_id": task.task_id,
            "score": 1.0,
            "wall_s": 1.0,
            "api_turns": 1,
            "tool_calls": 1,
            "read_file_calls": 1,
            "total_tokens": 10,
            "error": None,
        }

    return run_task


def test_corrupt_rep_file_is_rewritten_not_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    calls: list = []
    monkeypatch.setattr(runner, "run_task", _fake_run_task(calls))
    monkeypatch.setattr(sys, "argv", _argv(reps=1))

    out_dir = _out_dir(tmp_path)
    out_dir.mkdir(parents=True)
    # Exactly what an interrupt halfway through write_text leaves behind.
    (out_dir / "rep1.json").write_text(
        '{"model": "unit/model", "provider": "nous", "label": "unit-label", "rep": 1,\n "records": [',
        encoding="utf-8",
    )

    assert runner.main() == 0

    assert calls == [runner.TASKS[0].task_id], (
        "the corrupt rep was skipped instead of re-run"
    )
    payload = json.loads((out_dir / "rep1.json").read_text(encoding="utf-8"))
    assert payload["label"] == LABEL
    assert payload["rep"] == 1
    assert [rec["task_id"] for rec in payload["records"]] == [runner.TASKS[0].task_id]


def test_valid_rep_file_is_still_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    calls: list = []
    monkeypatch.setattr(runner, "run_task", _fake_run_task(calls))
    monkeypatch.setattr(sys, "argv", _argv(reps=1))

    out_dir = _out_dir(tmp_path)
    out_dir.mkdir(parents=True)
    existing = {
        "model": "unit/model",
        "provider": "nous",
        "label": LABEL,
        "rep": 1,
        "records": [{"task_id": "earlier-run", "score": 0.5}],
    }
    (out_dir / "rep1.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")

    assert runner.main() == 0

    assert calls == [], "a completed rep was re-run instead of resumed"
    assert json.loads((out_dir / "rep1.json").read_text(encoding="utf-8")) == existing
