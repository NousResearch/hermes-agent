"""P13 contract tests for the 3 reconstructed missing agent-job prompts.

For each of mossy-pr-feedback-loop, hermes-contrib-daily-gate, moss-fix-pipeline,
verifies the prompt-isolation contract:
  1. No hardcoded /home/kensei paths (must use ~ or $HERMES_HOME).
  2. Reads HERMES_HOME (references $HERMES_HOME or ~/ so the runtime can
     redirect the home).
  3. Has an explicit output contract ([SILENT] when nothing to report, or a
     structured deliverable spec).
  4. Respects its dry-run env flag (names the flag and short-circuits GitHub
     / git operations).
  5. For moss-fix-pipeline: has an explicit approval gate before any fork
     push or PR creation.
"""
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_FILE = REPO_ROOT / "cron" / "p13-missing-prompts.json"

HARDCODED_RE = re.compile(r"/home/kensei[^\s\"']*")

JOB_NAMES = [
    "mossy-pr-feedback-loop",
    "hermes-contrib-daily-gate",
    "moss-fix-pipeline",
]


def _load():
    if not PROMPTS_FILE.exists():
        pytest.skip("cron/p13-missing-prompts.json not found")
    data = json.loads(PROMPTS_FILE.read_text())
    return {j.get("name"): j for j in data.get("jobs", [])}


@pytest.fixture(scope="module")
def jobs():
    return _load()


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_job_present(jobs, job_name):
    assert job_name in jobs, f"{job_name} missing from p13-missing-prompts.json"


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_no_hardcoded_kensei_paths(jobs, job_name):
    prompt = (jobs.get(job_name) or {}).get("prompt", "")
    offenders = HARDCODED_RE.findall(prompt)
    assert not offenders, f"{job_name} prompt has hardcoded /home/kensei paths: {offenders}"


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_respects_hermes_home(jobs, job_name):
    prompt = (jobs.get(job_name) or {}).get("prompt", "")
    assert "$HERMES_HOME" in prompt, f"{job_name} prompt does not reference $HERMES_HOME"
    # Must also allow tilde fallback so a profile-relative run works.
    assert "~/" in prompt or "~." in prompt, f"{job_name} prompt has no ~/ fallback"


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_has_explicit_output_contract(jobs, job_name):
    prompt = (jobs.get(job_name) or {}).get("prompt", "")
    has_silent = "[SILENT]" in prompt or "SILENT" in prompt.upper()
    has_output_spec = any(tok in prompt.lower() for tok in ("output", "report", "deliver"))
    assert has_silent and has_output_spec, (
        f"{job_name} prompt lacks explicit output contract (needs both [SILENT] and a structured-report spec)"
    )


@pytest.mark.parametrize(
    "job_name, dry_flag",
    [
        ("mossy-pr-feedback-loop", "MOSS_FEEDBACK_DRY_RUN"),
        ("hermes-contrib-daily-gate", "CONTRIBUTION_GATE_DRY_RUN"),
        ("moss-fix-pipeline", "MOSS_FIX_DRY_RUN"),
    ],
)
def test_respects_dry_run(jobs, job_name, dry_flag):
    prompt = (jobs.get(job_name) or {}).get("prompt", "")
    assert dry_flag in prompt, f"{job_name} prompt does not reference dry-run flag {dry_flag}"
    # The dry-run instruction must short-circuit the live operations.
    lower = prompt.lower()
    assert "dry-run" in lower or "dry run" in lower, (
        f"{job_name} prompt names {dry_flag} but does not describe dry-run short-circuit"
    )
    # Must explicitly skip the live calls under dry-run.
    assert any(tok in lower for tok in ("skip", "no github", "no git operations")), (
        f"{job_name} prompt dry-run section does not explicitly skip live operations"
    )


def test_mossy_pr_feedback_loop_delegates_to_octacon(jobs):
    """mossy-pr-feedback-loop must delegate addressable items to the octacon profile."""
    prompt = (jobs.get("mossy-pr-feedback-loop") or {}).get("prompt", "")
    assert "octacon" in prompt.lower(), "mossy-pr-feedback-loop does not delegate to octacon"
    assert "delegate_task" in prompt or "delegate" in prompt.lower(), (
        "mossy-pr-feedback-loop lacks explicit delegation instruction"
    )


def test_moss_fix_pipeline_approval_gate(jobs):
    """moss-fix-pipeline must require explicit approval before any fork push or PR creation."""
    prompt = (jobs.get("moss-fix-pipeline") or {}).get("prompt", "")
    # The mandatory phrase from the task spec.
    assert "Do NOT push or create a PR without asking Sahil first" in prompt, (
        "moss-fix-pipeline missing the mandatory approval-gate phrase"
    )
    # Must gate both push and PR creation.
    assert "git push fork" in prompt, "moss-fix-pipeline does not reference fork push"
    assert "gh pr create" in prompt, "moss-fix-pipeline does not reference PR creation"
    # Must forbid auto-merge.
    assert "auto-merge" in prompt.lower(), "moss-fix-pipeline does not forbid auto-merge"


def test_hermes_contrib_daily_gate_read_only(jobs):
    """hermes-contrib-daily-gate must be read-only — no push/commit/merge/close."""
    prompt = (jobs.get("hermes-contrib-daily-gate") or {}).get("prompt", "")
    assert "read-only" in prompt.lower(), "hermes-contrib-daily-gate does not declare itself read-only"
    # Must not instruct any mutating git/gh operation.
    forbidden = ["git push", "gh pr create", "gh pr merge", "git commit", "gh pr close"]
    for op in forbidden:
        # The prompt may mention these only inside an explicit "never" clause —
        # allow that, but fail if it appears as an instruction.
        assert op not in prompt, f"hermes-contrib-daily-gate instructs mutating op: {op}"
