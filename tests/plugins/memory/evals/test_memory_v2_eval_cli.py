"""CLI tests for Memory v2 deterministic eval runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_memory_v2_eval_cli_writes_json_report(tmp_path):
    output_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "tests/plugins/memory/evals/fixtures/local_memory_eval_v1.yaml",
            "--baseline",
            "no_memory",
            "--baseline",
            "raw_fts",
            "--baseline",
            "memory_v2",
            "--workdir",
            str(tmp_path / "work"),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "local_memory_eval_v1"
    assert set(payload["summary"]) == {"no_memory", "raw_fts", "memory_v2"}
    assert payload["summary"]["memory_v2"]["query_count"] == 3
