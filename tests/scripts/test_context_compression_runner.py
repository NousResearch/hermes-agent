from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).parents[2] / "scripts" / "compression_eval" / "run_context_compression_eval.py"


def _git_root(path: Path) -> str:
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "--quiet", "-m", "test"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _report(source_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_sha": source_sha,
        "fixture_digest": "b" * 64,
        "compressed_tokens": 1,
        "baseline_tokens": 2,
        "probe_manifest": ["accuracy"],
        "probe_scores": {"accuracy": 1},
        "artifact_trail_preserved": True,
        "continuity_preserved": True,
        "model_provenance": {
            "compression_model": "compression",
            "evaluator_model": "evaluator",
            "provider": "provider",
            "model_config": "config",
        },
        "status": "pass",
    }


def test_runner_resolves_root_strips_separator_and_rejects_stale_report(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    harness = tmp_path / "harness"
    harness.mkdir()
    source_sha = _git_root(root)
    latest = harness / "results" / "latest"
    latest.mkdir(parents=True)
    stale = latest / "report.json"
    stale.write_text(json.dumps(_report(source_sha)), encoding="utf-8")
    writer = harness / "write_report.py"
    writer.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "report = json.loads(os.environ['REPORT'])\n"
        "Path('results/latest/report.json').write_text(json.dumps(report), encoding='utf-8')\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "report.json"
    result = subprocess.run(
        [sys.executable, str(RUNNER),
         "--harness", "harness", "--hermes-root", "hermes", "--output", "out/report.json",
         "--", sys.executable, "write_report.py"],
        cwd=tmp_path, env={**os.environ, "REPORT": json.dumps(_report(source_sha))},
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["source_sha"] == source_sha
    assert list(latest.glob("report.stale.*.json"))


def test_runner_rejects_dirty_evaluated_tree(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    harness = tmp_path / "harness"
    harness.mkdir()
    _git_root(root)
    (root / "dirty.py").write_text("dirty\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--harness", str(harness), "--hermes-root", str(root),
         "--output", str(tmp_path / "out.json"), "--", "true"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must be clean" in result.stderr


def test_runner_rejects_nonfinite_timeout(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/compression_eval/run_context_compression_eval.py",
         "--harness", str(tmp_path), "--hermes-root", str(tmp_path),
         "--output", str(tmp_path / "out.json"), "--timeout-seconds", "nan", "--", "true"],
        cwd=Path(__file__).parents[2], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "finite and positive" in result.stderr


def test_runner_bounds_harness_timeout(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    harness = tmp_path / "harness"
    harness.mkdir()
    _git_root(root)
    (tmp_path / "out.json").write_text('{"status":"pass"}')
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--harness", str(harness), "--hermes-root", str(root),
         "--output", str(tmp_path / "out.json"), "--timeout-seconds", "0.1",
         sys.executable, "-c", "import time; time.sleep(2)"],
        capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "timed out" in result.stderr

    assert not (tmp_path / "out.json").exists()


def test_output_cannot_delete_source_file(tmp_path):
    root = tmp_path / "source"
    _git_root(root)
    harness = tmp_path / "harness"
    harness.mkdir()
    output = root / "README.md"
    before = output.read_bytes()
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--harness", str(harness), "--hermes-root", str(root),
         "--output", str(output), "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert output.read_bytes() == before
