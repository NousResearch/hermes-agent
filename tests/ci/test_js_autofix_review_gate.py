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
# an explicit merge invocation in any gh flag spelling (auto or immediate),
# the REST merge endpoints (pulls/N/merge, repos/N/merges, git/refs writes),
# the GraphQL merge mutations, or a direct push to main that skips the PR.
BYPASS = re.compile(
    r"gh\s+(?:-{1,2}\S+(?:\s+\S+)?\s+)*pr\s+merge"
    r"|gh\s+api[^\n]*(?:pulls/\S+/merge|/merges\b|git/refs|"
    r"mergePullRequest|enablePullRequestAutoMerge|enqueuePullRequest|mergeBranch)"
    r"|mergePullRequest|enablePullRequestAutoMerge|enqueuePullRequest|mergeBranch"
    r"|git\s+push[^\n]*\bmain\b"
)


def _normalized(text: str) -> str:
    """Collapse shell line continuations so a split command cannot hide."""
    return text.replace("\\\n", " ")


def _steps():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            yield job_name, step


def test_no_step_can_land_the_patch_without_review():
    offenders = [
        (job_name, step.get("name", "<unnamed>"))
        for job_name, step in _steps()
        if any(
            BYPASS.search(_normalized(str(step.get(field, ""))))
            for field in ("run", "with")
        )
    ]
    assert offenders == [], (
        "steps that merge the bot PR or push to main bypass the human review "
        "gate: " + ", ".join(f"{job}/{name}" for job, name in offenders)
    )


def test_bypass_tripwire_catches_known_merge_vectors():
    """Positive controls: the regex must actually fire on the merge shapes
    it exists to catch, and must not fire on the legitimate commands the
    workflow runs."""
    for text in (
        'gh pr merge "$PR_NUM" --auto --squash',
        'gh pr merge "$PR" --squash',
        'gh -R nous/hermes pr merge 42 --squash',
        'gh api -X PUT repos/x/pulls/1/merge',
        'gh api -X POST repos/x/y/merges -f base=main',
        'gh api -X PATCH repos/x/y/git/refs/heads/main',
        'gh api graphql -f query="mutation{mergePullRequest(...)}"',
        'gh api graphql -f query="mutation{enablePullRequestAutoMerge(...)}"',
        'gh api graphql -f query="mutation{enqueuePullRequest(...)}"',
        'git push origin HEAD:main',
        'git push origin main',
        'git push --force origin HEAD:refs/heads/main',
        'gh pr merge \\\n  --auto --squash "$PR_NUM"',
    ):
        assert BYPASS.search(_normalized(text)), text
    for text in (
        'git push --force origin HEAD:"$BOT_BRANCH"',
        'gh api "repos/x/branches/main" --jq .commit.sha',
        'gh pr list --head "$BOT_BRANCH" --state open',
        'gh pr close "$PR_NUM" --delete-branch',
        'gh pr create --head "$BOT_BRANCH" --base main --title t --body b',
        'gh pr checks "$PR_NUM" --json bucket',
    ):
        assert not BYPASS.search(_normalized(text)), text


def test_extension_gate_and_review_note_stay_in_sync():
    """The produce-patch step keeps its allowed-extension filter and the
    no-auto-merge rationale is documented where the PR is created."""
    produce = next(
        step
        for _, step in _steps()
        if step.get("id") == "produce-patch"
    )
    assert "--allowed" in produce["run"], "the allowed-extension filter is missing"

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
    vetting. The apply job must verify the staged diff through the checked-in
    gate script, and the produce job must derive its exclusions from the same
    script so the two sides cannot drift apart."""
    apply_step = next(
        step
        for job, step in _steps()
        if job == "apply-patch" and "git apply" in str(step.get("run", ""))
    )
    assert "autofix_path_gate.py --staged" in apply_step["run"], (
        "apply-side gate must verify the staged diff via the shared script"
    )

    produce = next(step for _, step in _steps() if step.get("id") == "produce-patch")
    assert "autofix_path_gate.py --deny-globs" in produce["run"], (
        "produce-side exclusions must come from the shared gate script"
    )
    assert "autofix_path_gate.py --allowed" in produce["run"], (
        "produce-side extension check must use the shared gate script"
    )

    script = (ROOT / "scripts/ci/autofix_path_gate.py").read_text(encoding="utf-8")
    for denied in ("package.json", "package-lock.json", "npm-shrinkwrap.json",
                   "eslint.config", ".github", "tsconfig"):
        assert denied in script, f"gate script lost the {denied} denial"
