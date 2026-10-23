"""self-improvement-loops: fitness.py scores are the weighted sum of the spec, and the history
log turns a second run into a delta against the first. Runs the real script, offline."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "optional-skills/autonomous-ai-agents/self-improvement-loops/scripts/fitness.py")
SPEC = {"name": "latency", "target": "p95 under 200ms",
        "dimensions": [{"name": "speed", "weight": 0.4}, {"name": "accuracy", "weight": 0.6}]}


def _run(spec_path, scores, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), str(spec_path), json.dumps(scores), *extra],
                          capture_output=True, text=True, timeout=30)


@pytest.fixture
def spec_path(tmp_path):
    path = tmp_path / "latency.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    return path


def test_history_log_reports_weighted_score_and_delta(spec_path, tmp_path):
    log = tmp_path / "nested" / "latency.jsonl"
    runs = [{"speed": 0.5, "accuracy": 0.5}, {"speed": 1.0, "accuracy": 0.25}]
    results = []
    for scores in runs:
        proc = _run(spec_path, scores, "--log", str(log))
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout))

    weights = {d["name"]: d["weight"] for d in SPEC["dimensions"]}
    for scores, result in zip(runs, results):
        assert result["score"] == pytest.approx(sum(weights[k] * v for k, v in scores.items()))
    assert results[0]["previous"] is None and results[0]["delta"] is None
    assert results[1]["previous"] == results[0]["score"]
    assert results[1]["delta"] == pytest.approx(results[1]["score"] - results[0]["score"])
    assert len(log.read_text(encoding="utf-8").splitlines()) == len(runs)


@pytest.mark.parametrize("spec_patch, scores", [
    ({"dimensions": [{"name": "speed", "weight": 0.4}, {"name": "accuracy", "weight": 0.4}]},
     {"speed": 1, "accuracy": 1}),
    ({}, {"speed": 1}),
    ({}, {"speed": 1, "accuracy": 1.5}),
    ({}, {"speed": 1, "accuracy": 1, "cost": 0}),
])
def test_invalid_input_is_rejected_without_logging(tmp_path, spec_patch, scores):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({**SPEC, **spec_patch}), encoding="utf-8")
    log = tmp_path / "history.jsonl"
    proc = _run(spec_path, scores, "--log", str(log))
    assert proc.returncode == 2 and proc.stderr.startswith("fitness:")
    assert not log.exists()
