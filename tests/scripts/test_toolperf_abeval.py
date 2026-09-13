"""CLI contracts for the tool-performance A/B report harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "toolperf_abeval" / "ab_eval.py"


def _write_arm(root: Path, model: str, arm: str, *, tasks: tuple[str, ...], config_digest: str) -> None:
    result_dir = root / "results" / model / arm
    result_dir.mkdir(parents=True)
    import runpy
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        harness = runpy.run_path(str(SCRIPT))
    finally:
        sys.path.pop(0)
    harness["_model_provenance"].__globals__["HOME"] = root / "home"
    evaluator = harness["_evaluator_provenance"]()
    model_provenance = harness["_model_provenance"](model)
    if config_digest != "c" * 64:
        model_provenance["config_digest"] = config_digest
    rows = []
    for task in tasks:
        run_id = f"{task}-r0"
        trace = result_dir / f"{run_id}.atof.jsonl"
        trace.write_text(
            '{"kind":"scope","category":"llm","scope_category":"start","name":"model"}\n'
            '{"kind":"scope","category":"llm","scope_category":"end","name":"model"}\n'
            '{"kind":"scope","category":"tool","scope_category":"start","name":"terminal"}\n'
            '{"kind":"scope","category":"tool","scope_category":"end","name":"terminal",'
            '"metadata":{"status":"ok"},"data":{}}\n',
            encoding="utf-8",
        )
        with trace.open("a") as handle:
            handle.write(json.dumps({"kind": "evaluation", "category": "run",
                                    "scope_category": "end", "run_id": run_id}) + "\n")
        rows.append(
            {
                "run_id": run_id,
                "task": task,
                "rep": 0,
                "wall_s": 1.0,
                "source_sha": "a" * 40 if arm == "baseline" else "b" * 40,
                "model_provenance": model_provenance,
                "evaluator_provenance": evaluator,
                "tail": "",
            }
        )
    (result_dir / "meta.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _run_report(root: Path, model: str) -> subprocess.CompletedProcess[str]:
    env = {"ABEVAL_ROOT": str(root), "ABEVAL_HOME": str(root / "home")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "report", "--models", model],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )


def test_report_rejects_different_arm_model_provenance(tmp_path: Path) -> None:
    model = "test-model"
    root = tmp_path / "workspace"
    _write_arm(root, model, "baseline", tasks=("err_python_env",), config_digest="c" * 64)
    _write_arm(root, model, "fixes", tasks=("err_python_env",), config_digest="f" * 64)

    result = _run_report(root, model)

    assert result.returncode == 1
    assert "different model provenance" in result.stderr or "different model provenance" in result.stdout


def test_report_returns_failure_for_incomplete_battery(tmp_path: Path) -> None:
    model = "test-model"
    root = tmp_path / "workspace"
    (root / "results" / model).mkdir(parents=True)
    (root / "results" / model / "manifest.json").write_text(
        json.dumps({"model": model, "tasks": ["err_python_env", "err_big_output"], "repetitions": 1}),
        encoding="utf-8",
    )
    _write_arm(root, model, "baseline", tasks=("err_python_env",), config_digest="c" * 64)
    _write_arm(root, model, "fixes", tasks=("err_python_env",), config_digest="c" * 64)

    result = _run_report(root, model)

    assert result.returncode == 1
    report = json.loads((root / "results" / model / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "fail"


def test_report_requires_unique_complete_runs_and_current_provenance(tmp_path: Path) -> None:
    model = "test-model"
    for defect in ("duplicate", "truncated", "missing_completion", "config_changed", "evaluator_changed"):
        root = tmp_path / defect
        for arm in ("baseline", "fixes"):
            _write_arm(root, model, arm, tasks=("err_python_env",), config_digest="c" * 64)
        mdir = root / "results" / model
        (mdir / "manifest.json").write_text(json.dumps(
            {"model": model, "tasks": ["err_python_env"], "repetitions": 1}))
        assert _run_report(root, model).returncode == 0
        if defect == "duplicate":
            meta = mdir / "baseline" / "meta.jsonl"
            meta.write_text(meta.read_text() * 2)
        elif defect == "truncated":
            (mdir / "baseline" / "err_python_env-r0.atof.jsonl").write_text('{}\n')
        elif defect == "missing_completion":
            trace = mdir / "baseline" / "err_python_env-r0.atof.jsonl"
            trace.write_text("\n".join(trace.read_text().splitlines()[:-1]) + "\n")
        elif defect == "evaluator_changed":
            for arm in ("baseline", "fixes"):
                meta = mdir / arm / "meta.jsonl"
                row = json.loads(meta.read_text())
                row["evaluator_provenance"]["evaluator_digest"] = "f" * 64
                meta.write_text(json.dumps(row) + "\n")
        else:
            (root / "home").mkdir(exist_ok=True)
            (root / "home" / "config.yaml").write_text("model:\n  provider: changed\n")
        result = _run_report(root, model)
        assert result.returncode != 0
        published = mdir / "report.json"
        assert not published.exists() or json.loads(published.read_text())["status"] == "fail"


def test_run_validates_source_and_resume_before_execution(tmp_path: Path, monkeypatch) -> None:
    import runpy

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.setenv("ABEVAL_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("ABEVAL_HOME", str(tmp_path / "home"))
    harness = runpy.run_path(str(SCRIPT))
    run = harness["run"]
    state = run.__globals__
    source = tmp_path / "source"
    source.mkdir()
    state["_resolve_clean_source"] = lambda path: (source, "a" * 40)
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("model:\n  provider: custom:hyper\nproviders:\n  hyper:\n    base_url: http://localhost:1234\n")
    old = harness["_model_provenance"]("test-model")
    config.write_text(config.read_text().replace("1234", "5678"))
    assert harness["_model_provenance"]("test-model") != old
    config.write_text("model:\n  provider: custom:hyper\ncustom_providers:\n  - name: hyper\n    base_url: http://localhost:1234\n")
    old = harness["_model_provenance"]("test-model")
    config.write_text(config.read_text().replace("1234", "5678"))
    assert harness["_model_provenance"]("test-model") != old
    config.write_text("model: {}\n")
    prior = harness["_model_provenance"]("test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://endpoint.invalid/v1")
    assert harness["_model_provenance"]("test-model") != prior
    (home / ".env").write_text("OPENAI_BASE_URL=https://dotenv.invalid/v1\n")
    dotenv_provenance = harness["_model_provenance"]("test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://changed.invalid/v1")
    assert harness["_model_provenance"]("test-model") == dotenv_provenance
    state["ROOT"] = source / "workspace"
    import pytest
    with pytest.raises(SystemExit, match="outside"):
        run("baseline", "test-model", 1, str(source))
    assert not state["ROOT"].exists()
    state["ROOT"] = tmp_path / "results"
    meta = state["ROOT"] / "results" / "test-model" / "baseline" / "meta.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"run_id": "err_python_env-r0", "source_sha": "a" * 40,
                               "model_provenance": old,
                               "evaluator_provenance": harness["_evaluator_provenance"]()}) + "\n")
    with pytest.raises(SystemExit, match="resume provenance"):
        run("baseline", "test-model", 1, str(source))


def test_interpolated_endpoint_and_interpreter_bind_provenance(tmp_path, monkeypatch):
    import runpy
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        harness = runpy.run_path(str(SCRIPT))
    finally:
        sys.path.pop(0)
    harness["_model_provenance"].__globals__["HOME"] = tmp_path
    (tmp_path / "config.yaml").write_text("providers:\n  hyper:\n    base_url: ${MODEL_HOST}\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_HOST", "https://first.example")
    before = harness["_model_provenance"]("model")
    monkeypatch.setenv("MODEL_HOST", "https://second.example")
    assert harness["_model_provenance"]("model") != before
    assert harness["_evaluator_provenance"]()["interpreter"] == sys.version


def test_run_rejects_checkout_drift_before_recording(tmp_path, monkeypatch):
    import runpy
    import pytest

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.setenv("ABEVAL_ROOT", str(tmp_path / "workspace"))
    harness = runpy.run_path(str(SCRIPT))
    source = tmp_path / "source"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "--allow-empty", "--quiet", "-m", "base"], check=True)
    state = harness["run"].__globals__
    state["TASKS"] = {"probe": "Inspect {WORK}"}
    state["make_sandbox"] = lambda work: None
    state["_model_provenance"] = lambda model: {}
    real_run = subprocess.run

    def execute(command, **kwargs):
        if command[1:3] == ["-m", "hermes_cli.main"]:
            (source / "changed.py").write_text("changed", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "done", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(SystemExit, match="must be clean"):
        harness["run"]("baseline", "model", 1, str(source))
    assert not list((tmp_path / "workspace").rglob("meta.jsonl"))
