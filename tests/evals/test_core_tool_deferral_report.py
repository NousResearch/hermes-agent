"""report.py must render every arm the battery actually ran.

`orchestrator.py` takes `--arms=base,pr,control` and stamps `arm` into every
record, but the report iterated a literal ("base", "pr") tuple: records for any
other arm matched no filter, so they were never printed and never reached the
mean-of-task-means aggregate — while the header still counted them (a 3-arm
battery was reported as a 2-arm one, with no warning).

The third arm below is named `alt` on purpose: no allow-list of known arms can
pass, only deriving the set from the records can.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2] / "evals" / "core_tool_deferral"
REPORT = HARNESS / "report.py"

MODEL = "unit-model"
ARMS = ("base", "pr", "alt")
SCORES = {"base": 0.1, "pr": 0.2, "alt": 0.9}


def _write_results(root: Path) -> None:
    out = root / MODEL
    out.mkdir(parents=True)
    for arm in ARMS:
        rec = {
            "arm": arm,
            "model": MODEL,
            "task": "deferral-smoke",
            "rep": 1,
            "score": SCORES[arm],
            "error": None,
            "api_turns": 3,
            "total_tokens": 1000,
            "wall_s": 12.0,
            "bridge_calls": 2,
            "tool_calls_total": 4,
            "tool_counts": {},
            "raw_xml_noise": False,
        }
        (out / f"{arm}__deferral-smoke__rep1.json").write_text(
            json.dumps(rec), encoding="utf-8"
        )


def _run_report(root: Path) -> str:
    env = dict(os.environ, ABDEFER_RESULTS=str(root))
    proc = subprocess.run(
        [sys.executable, str(REPORT), MODEL],
        capture_output=True,
        text=True,
        cwd=str(HARNESS),
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"report.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc.stdout


def test_report_renders_every_arm_present_in_the_results(tmp_path):
    root = tmp_path / "results"
    _write_results(root)
    stdout = _run_report(root)

    missing_rows = [arm for arm in ARMS if f"| {arm} " not in stdout]
    assert not missing_rows, (
        f"arms with no table row: {missing_rows} (header counted "
        f"{len(ARMS)} runs)\n{stdout}"
    )

    aggregate = [
        ln for ln in stdout.splitlines() if ln.startswith("MEAN-OF-TASK-MEANS")
    ]
    aggregated_arms = {ln.split("|")[1].strip() for ln in aggregate}
    assert aggregated_arms == set(ARMS), (
        f"aggregated arms {sorted(aggregated_arms)} != arms present in the results "
        f"{sorted(ARMS)}\n{stdout}"
    )

    # The third arm's numbers must reach the aggregate, not just the table.
    assert f"{SCORES['alt']:.3f}" in stdout, (
        f"alt-arm score did not reach the mean-of-task-means aggregate\n{stdout}"
    )
