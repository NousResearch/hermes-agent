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
    evaluator = {"evaluator_digest": "d" * 64, "battery_digest": "e" * 64}
    model_provenance = {
        "model": model,
        "provider": "test-provider",
        "config_digest": config_digest,
    }
    rows = []
    for task in tasks:
        run_id = f"{task}-r0"
        trace = result_dir / f"{run_id}.atof.jsonl"
        trace.write_text(
            '{"kind":"scope","category":"tool","scope_category":"start","name":"terminal"}\n'
            '{"kind":"scope","category":"tool","scope_category":"end","name":"terminal",'
            '"metadata":{"status":"ok"},"data":{}}\n',
            encoding="utf-8",
        )
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
