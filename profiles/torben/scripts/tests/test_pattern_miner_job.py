from __future__ import annotations

import json
from pathlib import Path

from torben_pattern_miner_job import collect_pattern_events, run_pattern_miner


def test_pattern_miner_job_collects_events_and_writes_review_gated_output(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "torben-open-loops.csv").write_text(
        "\n".join(
            [
                "id,item,state,owner,due,domain,note,created,updated",
                "1,Draft weekly investor update,next-action,eric,,admin,,2026-07-06,2026-07-06",
                "2,Draft weekly investor update,done,eric,,admin,,2026-07-06,2026-07-06",
                "3,Draft weekly investor update,done,eric,,admin,,2026-07-06,2026-07-06",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (state / "torben-capture-confirmations.jsonl").write_text(
        json.dumps({"text": "Captured loop #4 as next-action: Draft weekly investor update"}) + "\n",
        encoding="utf-8",
    )

    events = collect_pattern_events(tmp_path)
    payload = run_pattern_miner(tmp_path)

    assert len(events) == 4
    assert payload["proposals"][0]["status"] == "review_gated"
    written = json.loads((state / "torben-pattern-proposals.json").read_text(encoding="utf-8"))
    assert written["proposals"][0]["support_count"] == 4
