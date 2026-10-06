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
import subprocess
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


def test_skip_gate_rejects_json_the_consumer_cannot_read(tmp_path, monkeypatch):
    """Parsing as JSON is not enough — `report.py` does `data["records"]` then `rec.get(...)`.

    Every shape below is valid JSON that the comparison still cannot consume, so the
    gate must re-run the rep instead of resuming it. (Truncated JSON is the other
    test's job: that one fails to parse at all.)
    """
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    calls: list = []
    monkeypatch.setattr(runner, "run_task", _fake_run_task(calls))
    monkeypatch.setattr(sys, "argv", _argv(reps=1))

    out_dir = _out_dir(tmp_path)
    out_dir.mkdir(parents=True)
    rep = out_dir / "rep1.json"
    head = '"model": "unit/model", "provider": "nous", "label": "unit-label", "rep": 1'
    shapes = {
        "top level is not an object": '"just a string"',
        "no records key": "{" + head + "}",
        "records is not a list": "{" + head + ', "records": 42}',
        "records holds non-objects": "{" + head + ', "records": ["oops"]}',
    }

    for name, body in shapes.items():
        rep.write_text(body, encoding="utf-8")
        calls.clear()

        assert runner.main() == 0, name
        assert calls, f"{name}: a rep the comparison cannot read was skipped instead of re-run"
        payload = json.loads(rep.read_text(encoding="utf-8"))
        assert isinstance(payload["records"], list) and all(
            isinstance(rec, dict) for rec in payload["records"]
        ), name


def test_unreadable_rep_does_not_abort_the_run(tmp_path, monkeypatch):
    """A path that exists but cannot be read must not take the whole run down.

    Before the resume gate, `exists()` alone decided the skip, so a `rep1.json`
    that was (say) a directory was left alone and the run continued. The gate's
    read must not turn that into an exception escaping `main()`.
    """
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    calls: list = []
    monkeypatch.setattr(runner, "run_task", _fake_run_task(calls))
    monkeypatch.setattr(sys, "argv", _argv(reps=1))

    out_dir = _out_dir(tmp_path)
    out_dir.mkdir(parents=True)
    (out_dir / "rep1.json").mkdir()  # exists() is True; read_text() raises IsADirectoryError

    assert runner.main() == 0
    assert calls == [], "a path that cannot be read should be left alone, as it was before"


def test_rep_is_ascii_on_disk_and_readable_from_a_c_locale(tmp_path, monkeypatch):
    """A rep must not depend on the reader's locale.

    `report.py` reads each rep with `read_text()` and no `encoding=`. The writer
    this replaced was `json.dumps`, which escapes non-ASCII (`ensure_ascii=True`);
    a writer that emitted raw UTF-8 instead would be unreadable wherever the
    default encoding is not UTF-8 (LC_ALL=C, plausible on the CI base image).
    Both halves are pinned: the bytes on disk stay ASCII, and the reader survives
    raw UTF-8 in a C locale.
    """
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")

    def run_task_cjk(task, model, provider, timeout_mult, toolsets):
        rec = _fake_run_task([])(task, model, provider, timeout_mult, toolsets)
        rec["final_response"] = "分析完成"
        return rec

    monkeypatch.setattr(runner, "run_task", run_task_cjk)
    monkeypatch.setattr(sys, "argv", _argv(reps=1))

    out_dir = _out_dir(tmp_path)
    assert runner.main() == 0
    rep = out_dir / "rep1.json"
    raw = rep.read_bytes()
    assert all(byte < 128 for byte in raw), "the rep on disk must not depend on the reader's locale"
    assert json.loads(raw.decode("utf-8"))["records"][0]["final_response"] == "分析完成"

    # Reader half: a rep carrying raw UTF-8 (written by any other tool) still reads.
    results = tmp_path / "reader"
    model_dir = results / "lbl" / SLUG
    model_dir.mkdir(parents=True)
    (model_dir / "rep1.json").write_bytes(
        json.dumps(
            {"model": "unit/model", "label": "lbl", "rep": 1,
             "records": [{"task_id": "t1", "score": 1.0, "final_response": "分析完成"}]},
            ensure_ascii=False,
        ).encode("utf-8")
    )
    probe = tmp_path / "probe_reader.py"
    probe.write_text(
        "import pathlib, sys\n"
        "sys.path.insert(0, str(pathlib.Path(sys.argv[1])))\n"
        "import report\n"
        "report.RESULTS = pathlib.Path(sys.argv[2])\n"
        "data = report.load_label('lbl', None)\n"
        "assert data, 'no rows read'\n"
        "print(sorted(data))\n",
        encoding="utf-8",
    )
    repo_root = READTOOL_DIR.parents[1]
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONIOENCODING": "ascii",
    }
    proc = subprocess.run(
        [sys.executable, str(probe), str(READTOOL_DIR), str(results)],
        cwd=str(repo_root), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"reader failed under LC_ALL=C: {proc.stderr[-600:]}"

