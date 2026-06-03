"""Replayable JSONL storage for eval-lab runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from agent.eval_lab.redaction import redact_secrets
from agent.eval_lab.schemas import EvalScore, TrajectoryGroup


class EvalRunStorage:
    """Persist eval run artifacts below a run directory.

    Default runtime location is ``${HERMES_HOME}/eval_lab/runs/<run_id>``.
    Tests may inject ``base_dir`` to keep writes isolated.
    """

    def __init__(self, run_id: str, base_dir: str | Path | None = None):
        self.run_id = run_id
        root = Path(base_dir) if base_dir is not None else get_hermes_home() / "eval_lab" / "runs"
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def groups_path(self) -> Path:
        return self.run_dir / "trajectory_groups.jsonl"

    @property
    def scores_path(self) -> Path:
        return self.run_dir / "scores.jsonl"

    def write_group(self, group: TrajectoryGroup) -> Path:
        self._append_jsonl(self.groups_path, redact_secrets(group.to_dict()))
        return self.groups_path

    def write_score(self, score: EvalScore) -> Path:
        self._append_jsonl(self.scores_path, redact_secrets(score.to_dict()))
        return self.scores_path

    def load_groups(self) -> list[TrajectoryGroup]:
        return [TrajectoryGroup.from_dict(item) for item in self._read_jsonl(self.groups_path)]

    def load_scores(self) -> list[EvalScore]:
        return [EvalScore.from_dict(item) for item in self._read_jsonl(self.scores_path)]

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Invalid JSONL row {path}:{line_number}: expected object")
            rows.append(item)
        return rows
