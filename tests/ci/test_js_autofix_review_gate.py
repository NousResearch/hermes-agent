"""The js-autofix bot PR must require human review to land on main.

The generate-patch job runs `npm run fix`, which executes repo code and every
installed eslint plugin; the workflow's own header names that process hostile.
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

# Anything that lands the bot patch on main without a human merging the PR:
# an explicit merge invocation (auto or immediate), the REST or GraphQL merge
# endpoints, or a direct push to main that skips the PR entirely.
BYPASS = re.compile(
    r"gh\s+pr\s+merge"
    r"|gh\s+api[^\n]*(?:pulls/\S+/merge|mergePullRequest|branches/\S+/merge)"
    r"|mergePullRequest"
    r"|git\s+push[^\n]*\bmain\b"
)


def _steps():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            yield job_name, step


def test_no_step_can_land_the_patch_without_review():
    offenders = [
        (job_name, step.get("name", "<unnamed>"))
        for job_name, step in _steps()
        if BYPASS.search(str(step.get("run", "")))
    ]
    assert offenders == [], (
        "steps that merge the bot PR or push to main bypass the human review "
        "gate: " + ", ".join(f"{job}/{name}" for job, name in offenders)
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


def test_apply_step_revalidates_paths_on_the_privileged_side():
    """The gate in generate-patch cannot be the only filter: that job runs
    the repo's own code, so a hostile run controls both the artifact and its
    vetting. The apply job must enumerate the staged diff itself and refuse
    anything outside plain JS/TS/JSON content changes."""
    apply_step = next(
        step
        for job, step in _steps()
        if job == "apply-patch" and "git apply" in str(step.get("run", ""))
    )
    run = apply_step["run"]
    assert "git diff --cached --name-status" in run, (
        "apply-side gate must enumerate staged paths"
    )
    assert "git diff --cached --summary" in run, (
        "apply-side gate must reject mode changes, renames, symlinks, gitlinks"
    )
    for denied in (".github/*", "package.json", "package-lock.json", "eslint.config.*"):
        assert denied in run, f"apply-side gate lost the {denied} denial"
