"""evals/skills/prompt_growth.py: each learned skill adds one listed row and a positive cost to the next
session's prompt, memory stops at its cap, and two runs in fresh sandboxes agree exactly."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from evals.skills import prompt_growth

ROOT = Path(__file__).resolve().parents[2]
# Ten fresh interpreters (seed, three checkpoints and memory, twice) that each build one skills index.
RUN_TIMEOUT_S = 600


def test_learned_skill_names_stay_unique_past_the_end_of_the_corpus():
    corpus = prompt_growth.learned_corpus()
    assert corpus, "no ungated optional skill to learn from"
    names = []
    for i in range(2 * len(corpus) + 1):
        _, name, text = prompt_growth._nth_skill(corpus, i)
        assert re.search(rf"(?m)^name: {re.escape(name)}$", text), name
        names.append(name)
    assert len(set(names)) == len(names)


def test_each_learned_skill_adds_one_row_and_memory_stops_at_its_cap(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals/skills/prompt_growth.py"), "--checkpoints", "0,2,5", "--repeats", "2",
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["repeats_identical"] == 2
    points = result["points"]
    assert [p["learned"] for p in points] == [0, 2, 5]
    for point in points:
        assert point["index_rows"] - points[0]["index_rows"] == point["learned"]
        assert point["curator_managed_records"] == point["learned"]
    estimates = [p["tokens_hermes_estimate"] for p in points]
    assert estimates == sorted(set(estimates)), estimates
    memory = result["memory"]
    assert all(count > 0 for count in memory["accepted_entries"].values())
    assert set(memory["first_refusal"]) == {"memory", "user"}
    assert (out / "skills_index_0005.txt").read_text(encoding="utf-8").strip()
