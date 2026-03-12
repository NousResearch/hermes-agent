from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EvalTask:
    task_id: str
    lane: str
    prompt: str
    expected: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvalResult:
    task_id: str
    lane: str
    passed: bool
    score: float
    duration_s: float
    output: str
    failure_signature: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvalSuite:
    suite: str
    version: str
    tasks: List[EvalTask]
    holdout_task_ids: List[str]


class EvalHarness:
    """Versioned eval harness for coding/terminal/agentic lanes.

    This is intentionally lightweight and deterministic-first. Callers provide
    a ``runner`` callback that executes a task prompt and returns output text.
    """

    def __init__(self, *, suite_name: str, commit: str = "unknown") -> None:
        self.suite_name = suite_name
        self.commit = commit

    @staticmethod
    def load_suite(path: str | Path) -> EvalSuite:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tasks = [
            EvalTask(
                task_id=t["task_id"],
                lane=t["lane"],
                prompt=t["prompt"],
                expected=t.get("expected"),
                metadata=t.get("metadata") or {},
            )
            for t in raw.get("tasks", [])
        ]
        return EvalSuite(
            suite=raw.get("suite", "unknown"),
            version=str(raw.get("version", "v1")),
            tasks=tasks,
            holdout_task_ids=list(raw.get("holdout_task_ids", [])),
        )

    @staticmethod
    def deterministic_grade(task: EvalTask, output: str) -> tuple[bool, float, Optional[str]]:
        expected = (task.expected or "").strip()
        if not expected:
            # No deterministic target -> treat as informational lane task.
            return True, 1.0, None

        passed = expected.lower() in (output or "").lower()
        if passed:
            return True, 1.0, None
        return False, 0.0, f"missing_expected:{expected[:60]}"

    def run_suite(
        self,
        suite: EvalSuite,
        *,
        runner: Callable[[EvalTask], str],
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        started = time.time()
        results: List[EvalResult] = []

        for task in suite.tasks:
            task_started = time.time()
            output = runner(task)
            passed, score, failure_signature = self.deterministic_grade(task, output)
            results.append(
                EvalResult(
                    task_id=task.task_id,
                    lane=task.lane,
                    passed=passed,
                    score=score,
                    duration_s=round(time.time() - task_started, 4),
                    output=output,
                    failure_signature=failure_signature,
                    metadata=task.metadata or {},
                )
            )

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        lane_scores: Dict[str, List[float]] = {}
        for row in results:
            lane_scores.setdefault(row.lane, []).append(row.score)

        by_lane = {
            lane: {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }
            for lane, scores in lane_scores.items()
        }

        report = {
            "suite": suite.suite,
            "version": suite.version,
            "commit": self.commit,
            "harness": self.suite_name,
            "duration_s": round(time.time() - started, 4),
            "summary": {
                "total": total,
                "passed": passed_count,
                "pass_rate": round((passed_count / total), 4) if total else 0.0,
            },
            "by_lane": by_lane,
            "holdout_task_ids": suite.holdout_task_ids,
            "results": [asdict(r) for r in results],
        }

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        report_file = out_path / f"{suite.suite}_report_{stamp}.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_file"] = str(report_file)
        return report
