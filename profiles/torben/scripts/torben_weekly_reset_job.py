#!/usr/bin/env python3
"""Cron wrapper for the Torben weekly reset Signal packet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torben_job_contract import torben_home
from torben_open_loops import load_loops
from torben_weekly_reset import build_weekly_packet, load_pattern_proposals, load_pending_decisions, render_packet


JOB_NAME = "torben-weekly-reset"


def build_weekly_reset_from_state(profile_home: Path | None = None) -> dict[str, Any]:
    home = profile_home or torben_home()
    state = home / "state"
    return build_weekly_packet(
        loops=load_loops(state / "torben-open-loops.csv"),
        pending_decisions=load_pending_decisions(state / "torben-pending-decisions.json"),
        pattern_proposals=load_pattern_proposals(state / "torben-pattern-proposals.json"),
    )


def write_weekly_reset_artifacts(packet: dict[str, Any], profile_home: Path | None = None) -> dict[str, str]:
    home = profile_home or torben_home()
    state = home / "state"
    state.mkdir(parents=True, exist_ok=True)
    json_path = state / "torben-weekly-reset-latest.json"
    text_path = state / "torben-weekly-reset-latest.txt"
    text = render_packet(packet)
    json_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    return {"json_path": str(json_path), "text_path": str(text_path), "text": text}


def main() -> int:
    packet = build_weekly_reset_from_state()
    artifacts = write_weekly_reset_artifacts(packet)
    print(artifacts["text"], end="")
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job(JOB_NAME, main))
