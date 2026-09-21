"""P13 prompt-isolation contract tests for blog agent-job prompts.

For each blog agent job (pr-to-blog-daily, blog-stream-daily,
blog-backlog-pregen) defined in cron/p13-blog-prompts.json, verifies:
- no_agent is false (agent-driven, not no-agent shell)
- deliver is "local"
- no hardcoded /home/kensei paths (uses $HOME/$HERMES_HOME/$BLOG_CONTENT_ROOT)
- references HERMES_HOME (env-overridable home resolution)
- respects BLOG_DAILY_DRY_RUN=1 (dry-run short-circuit)
- has an explicit [SILENT] output contract
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_JSON = REPO_ROOT / "cron" / "p13-blog-prompts.json"

HARDCODED_RE = re.compile(r"/home/kensei[^\s\"']*")
JOB_NAMES = ["pr-to-blog-daily", "blog-stream-daily", "blog-backlog-pregen"]


def _entries() -> dict[str, dict]:
    d = json.loads(PROMPTS_JSON.read_text())
    return {e["name"]: e for e in d.values()}


@pytest.fixture(scope="module")
def entries() -> dict[str, dict]:
    return _entries()


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_job_present_in_prompts_json(entries, job_name):
    e = entries.get(job_name)
    assert e is not None, f"{job_name} missing from cron/p13-blog-prompts.json"


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_job_is_agent_driven_not_no_agent(entries, job_name):
    e = entries[job_name]
    assert e.get("no_agent") is False, (
        f"{job_name} must have no_agent=false (agent job, not shell)"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_job_delivers_locally(entries, job_name):
    e = entries[job_name]
    assert e.get("deliver") == "local", (
        f"{job_name} deliver must be 'local' (no Discord on success)"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_prompt_has_no_hardcoded_kensei_paths(entries, job_name):
    prompt = entries[job_name].get("prompt") or ""
    offenders = HARDCODED_RE.findall(prompt)
    assert not offenders, (
        f"{job_name} prompt has hardcoded /home/kensei paths: {offenders}"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_prompt_references_hermes_home(entries, job_name):
    prompt = entries[job_name].get("prompt") or ""
    assert "$HERMES_HOME" in prompt, (
        f"{job_name} prompt must read $HERMES_HOME for env-overridable home"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_prompt_respects_dry_run(entries, job_name):
    prompt = entries[job_name].get("prompt") or ""
    assert "BLOG_DAILY_DRY_RUN" in prompt, (
        f"{job_name} prompt must respect BLOG_DAILY_DRY_RUN=1"
    )
    assert "SILENT" in prompt.upper(), (
        f"{job_name} dry-run path must end with [SILENT]"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_prompt_has_silent_output_contract(entries, job_name):
    prompt = entries[job_name].get("prompt") or ""
    assert "[SILENT]" in prompt, (
        f"{job_name} prompt must declare [SILENT] success output contract"
    )
    assert "ERROR:" in prompt, (
        f"{job_name} prompt must declare structured ERROR: failure contract"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_prompt_runs_foreground_not_background(entries, job_name):
    prompt = entries[job_name].get("prompt") or ""
    lower = prompt.lower()
    assert "foreground" in lower, (
        f"{job_name} prompt must instruct foreground execution (no 300s timeout)"
    )
    assert "no background" in lower or "no detach" in lower, (
        f"{job_name} prompt must explicitly forbid background detach"
    )


@pytest.mark.parametrize("job_name", JOB_NAMES)
def test_prompt_uses_redirectable_paths(entries, job_name):
    prompt = entries[job_name].get("prompt") or ""
    # Must reference BLOG_CONTENT_ROOT or $HOME for the content root
    assert "$BLOG_CONTENT_ROOT" in prompt or "$HOME" in prompt, (
        f"{job_name} prompt must use $BLOG_CONTENT_ROOT or $HOME for content root"
    )
