"""Manual E2E for the harness contribution (no API keys needed).

Exercises the real path — real SessionDB under a temp HERMES_HOME, real
workspace file work, real verification checks, real persistence + resume —
with a scripted agent standing in for model turns::

    python -m harness.demo
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="harness-demo-home"))
    os.environ["HERMES_HOME"] = str(home)
    workspace = Path(tempfile.mkdtemp(prefix="harness-demo-work"))
    target = workspace / "hello.txt"

    from hermes_state import SessionDB

    from harness.loop import HarnessRunner, StepResult
    from harness.state import Outcome, StepStatus, ToolObservation
    from harness.store import HarnessStore
    from harness.verify import file_contains_check

    store = HarnessStore(SessionDB())
    attempts = {"n": 0}

    def scripted(task, feature, context):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # First turn claims progress but does nothing verifiable.
            return StepResult(
                summary="wrote the file (unverified)",
                status_hint=StepStatus.DONE,
                evidence=["trust me"],
            )
        target.write_text("hello\n", encoding="utf-8")
        check = file_contains_check("hello-present", target, "hello")
        return StepResult(
            summary="wrote hello.txt",
            status_hint=StepStatus.DONE if check.passed else StepStatus.CONTINUE,
            observations=[
                ToolObservation(
                    id=f"obs-{attempts['n']}",
                    tool="write",
                    success=True,
                    summary=f"wrote {target.name}",
                )
            ],
            evidence=[check.detail],
        )

    def make_step(task):
        return scripted

    runner = HarnessRunner(store, make_step)
    task = runner.create(
        "write hello.txt containing hello", ["hello.txt contains hello"]
    )
    print(f"task {task.id} created; Hermes home: {home}")

    first = runner.step(task.id)
    print(f"turn 1 -> {first} (unverified claim must not complete)")
    assert first == Outcome.CONTINUE, first

    # New process, same home: resume from persisted state.
    store.close()
    resumed = HarnessRunner(HarnessStore(SessionDB()), make_step)
    status = resumed.status(task.id)
    print(
        f"resumed: iteration={status['iteration']} observations={status['observations']}"
    )
    assert status["iteration"] == 0  # claim-only turn verified nothing

    final = resumed.run(task.id, max_rounds=5)
    print(f"run -> {final}")
    assert final == Outcome.COMPLETED, final
    assert target.read_text(encoding="utf-8") == "hello\n"
    print(f"verified on disk: {target}")
    print(
        "E2E OK: task, harness, tools, observations, verification, "
        "recovery, persistence, resume, completion"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
