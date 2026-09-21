"""P13 prompt-isolation contract tests for agent-job prompts.

For each agent job (kensei-triage-investigator, memory-promotion-daily,
kensei-librarian-daily, denji-skill-audit), verifies the prompt contract:
- No hardcoded /home/kensei paths (must use ~ or $HERMES_HOME so a
  profile/checkout-relative run works).
- Respects HERMES_HOME (references the home via $HERMES_HOME or ~/ so
  the runtime can redirect it).
- Has an explicit output contract ([SILENT] when there is nothing to
  report, or a structured deliverable spec).
"""
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "cron" / "jobs.snapshot.json"

HARDCODED_RE = re.compile(r"/home/kensei[^\s\"']*")


def _jobs():
    d = json.loads(SNAPSHOT.read_text())
    return {j.get("name"): j for j in d.get("jobs", [])}


@pytest.fixture(scope="module")
def jobs():
    return _jobs()


@pytest.mark.parametrize(
    "job_name",
    ["kensei-triage-investigator", "memory-promotion-daily", "kensei-librarian-daily", "denji-skill-audit"],
)
def test_prompt_has_no_hardcoded_kensei_paths(jobs, job_name):
    """No prompt may reference /home/kensei/... — it must use ~ or
    $HERMES_HOME so the runtime can redirect the home."""
    job = jobs.get(job_name)
    assert job is not None, f"{job_name} missing from snapshot"
    prompt = job.get("prompt") or ""
    offenders = HARDCODED_RE.findall(prompt)
    assert not offenders, (
        f"{job_name} prompt has hardcoded /home/kensei paths: {offenders}"
    )


@pytest.mark.parametrize(
    "job_name",
    ["kensei-triage-investigator", "memory-promotion-daily", "kensei-librarian-daily", "denji-skill-audit"],
)
def test_prompt_respects_hermes_home_or_tilde(jobs, job_name):
    """Any path reference must use ~ or $HERMES_HOME (not a hardcoded
    absolute home), so the runtime can redirect the home."""
    job = jobs.get(job_name)
    prompt = (job.get("prompt") or "")
    # Must reference the home via ~ or $HERMES_HOME if it references
    # .hermes / brain / wiki at all.
    references_home = any(
        tok in prompt
        for tok in (".hermes", "brain", "wiki", "runbooks")
    )
    if not references_home:
        pytest.skip(f"{job_name} does not reference the home dir")
    uses_redirectable = ("~/" in prompt) or ("$HERMES_HOME" in prompt)
    assert uses_redirectable, (
        f"{job_name} prompt references home but uses no ~ or $HERMES_HOME redirect"
    )


@pytest.mark.parametrize(
    "job_name",
    ["kensei-triage-investigator", "memory-promotion-daily", "kensei-librarian-daily", "denji-skill-audit"],
)
def test_prompt_has_explicit_output_contract(jobs, job_name):
    """Each prompt must specify an explicit output contract: [SILENT]
    when there is nothing to report, or a structured deliverable spec."""
    job = jobs.get(job_name)
    prompt = job.get("prompt") or ""
    # Must mention [SILENT] (the cron silence contract) OR an explicit
    # output/deliverable instruction.
    has_silent = "[SILENT]" in prompt or "SILENT" in prompt.upper()
    has_output_spec = any(
        tok in prompt.lower()
        for tok in ("output", "respond", "deliver", "produce", "return")
    )
    assert has_silent or has_output_spec, (
        f"{job_name} prompt has no explicit output contract "
        "(no [SILENT] and no output/deliverable instruction)"
    )
