# tests/agent/benchmarks/trajectory.py
import json
from pathlib import Path
from typing import Any

TRAJECTORY_DIR = Path(__file__).resolve().parent / "trajectories"


def capture(session_id: str, messages: list[dict],
            engine_state: dict[str, Any]) -> Path:
    """Write a trajectory snapshot. Called from a debug-only hook in
    run_agent.py during a real session. Not enabled in production."""
    out = TRAJECTORY_DIR / f"{session_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 1,
        "messages": messages,
        "engine_state": engine_state,
    }, indent=2, default=str))
    return out


def load(name: str) -> tuple[list[dict], dict[str, Any]]:
    raw = json.loads((TRAJECTORY_DIR / name).read_text())
    return raw["messages"], raw["engine_state"]
