"""Direct, deterministic implementations of the two editor ABI arms.

The Hermes arm imports the production fuzzy matcher.  It deliberately does
not wrap or modify that matcher: this is a meter for the shipped semantics.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from tasks import EXPECTED_OUTCOMES, Edit, Task
from tools.fuzzy_match import IDENTICAL_STRINGS_ERROR, fuzzy_find_and_replace, is_already_applied


def _strict_replace(content: str, edit: Edit) -> tuple[str, str, str]:
    """Approximate SWE-agent str_replace_editor's exact, unique contract."""
    if edit.old == edit.new:
        return content, "rejected", "old and new text are identical"
    matches = content.count(edit.old)
    if matches != 1:
        return content, "rejected", f"exact match count was {matches}, expected 1"
    return content.replace(edit.old, edit.new, 1), "applied", "exact unique match"


def _hermes_replace(content: str, edit: Edit) -> tuple[str, str, str]:
    new_content, match_count, strategy, error = fuzzy_find_and_replace(
        content, edit.old, edit.new, replace_all=False
    )
    if error:
        if is_already_applied(content, edit.old, edit.new):
            return content, "no_change", "target text is already present"
        outcome = "rejected"
        if error == IDENTICAL_STRINGS_ERROR:
            outcome = "no_change"
        return content, outcome, error
    if match_count == 0:
        if edit.new in content:
            return content, "no_change", "target text is already present"
        return content, "rejected", "no fuzzy match"
    return new_content, "applied", f"matched with {strategy}"


def _apply_task(arm: str, root: Path, task: Task) -> dict:
    replace = _strict_replace if arm == "str_replace" else _hermes_replace
    outcomes: list[str] = []
    reasons: list[str] = []
    for edit in task.edits:
        path = root / edit.path
        content = path.read_text(encoding="utf-8")
        updated, outcome, reason = replace(content, edit)
        outcomes.append(outcome)
        reasons.append(reason)
        if outcome == "applied":
            path.write_text(updated, encoding="utf-8")
        else:
            break
    outcome = "applied" if outcomes and all(item == "applied" for item in outcomes) else outcomes[-1]
    expected = EXPECTED_OUTCOMES[arm][task.task_id]
    return {
        "task_id": task.task_id,
        "capability": task.capability,
        "outcome": outcome,
        "edits_attempted": len(outcomes),
        "edits_requested": len(task.edits),
        "reason": "; ".join(reasons),
        "expected": expected,
        "passed": outcome == expected,
    }


def run_arm(arm: str, workspace, tasks: tuple[Task, ...]) -> list[dict]:
    """Run one ABI arm against fresh copies of every task fixture."""
    if arm not in {"str_replace", "hermes_patch"}:
        raise ValueError(f"unknown arm: {arm}")
    root = Path(workspace.name)
    results = []
    for task in tasks:
        # Each task gets an independent fixture tree, avoiding cross-task state.
        with tempfile.TemporaryDirectory(prefix=f"edittool-{task.task_id}-") as isolated:
            shutil.copytree(root, isolated, dirs_exist_ok=True)
            results.append(_apply_task(arm, Path(isolated), task))
    return results
