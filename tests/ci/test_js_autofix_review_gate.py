"""The js-autofix bot PR must require human review to land on main.

The generate-patch job runs `npm run fix`, which executes repo code and every
installed eslint plugin — the workflow's own header names that process hostile.
The privileged apply job trusts only the patch artifact, and its file gate
verifies paths by extension, so hostile output written inside an allowed path
(a rogue plugin emitting into src/*.ts or a config file) passes it. Enabling
auto-merge on the bot PR is therefore the difference between "untrusted code
ran on an ephemeral runner" and "untrusted content landed on main with no
human review".
"""
from pathlib import Path
import re

import hermes_yaml as yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/js-autofix.yml"

# Auto-merge entry points that would land the bot PR without review.
AUTO_MERGE = re.compile(r"gh\s+pr\s+merge[^\n]*--auto|--auto[^\n]*--squash")


def _steps():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            yield job_name, step


def test_no_step_enables_auto_merge_on_the_bot_pr():
    offenders = [
        (job_name, step.get("name", "<unnamed>"))
        for job_name, step in _steps()
        if AUTO_MERGE.search(str(step.get("run", "")))
    ]
    assert offenders == [], (
        "steps enabling auto-merge on the bot PR remove the human review gate: "
        + ", ".join(f"{job}/{name}" for job, name in offenders)
    )


def test_extension_gate_and_review_note_stay_in_sync():
    """The produce-patch step keeps its allowed-extension filter and the
    no-auto-merge rationale is documented where the PR is created."""
    produce = next(
        step
        for _, step in _steps()
        if step.get("id") == "produce-patch"
    )
    assert "grep -vE" in produce["run"], "the allowed-extension filter is missing"

    pr_step = next(
        step
        for job, step in _steps()
        if job == "apply-patch" and "gh pr create" in str(step.get("run", ""))
    )
    run = pr_step["run"]
    assert "--auto" not in run
    assert "untrusted" in run, "the PR step must document why review is required"
