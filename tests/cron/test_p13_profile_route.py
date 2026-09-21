"""Prompt-isolation contract tests for the 3 profile-route agent jobs.

Each agent job (no_agent: false) must:
- carry no hardcoded /home/kensei paths (portable across hosts/profiles)
- reference $HERMES_HOME so the prompt resolves paths at runtime
- declare an explicit output contract ([SILENT] or structured report)
- carry explicit profile-routing metadata (profile + profile_routing)

The rewritten prompts live in cron/p13-profile-route-prompts.json; the
profile-store routing utility is scripts/cron_profile_router.py.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROMPTS_FILE = REPO / "cron" / "p13-profile-route-prompts.json"
ROUTER_SCRIPT = REPO / "scripts" / "cron_profile_router.py"

JOB_NAMES = {
    "content-engine-personal-llm",
    "MrHermagi Daily Lesson",
    "wesker-ops-daily",
}

EXPECTED_PROFILES = {
    "content-engine-personal-llm": "ceecee",
    "MrHermagi Daily Lesson": "mrhermagi",
    "wesker-ops-daily": "wesker",
}


def _load_prompts() -> dict[str, dict]:
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    return {job["name"]: job for job in data["jobs"]}


def _router_module():
    spec = importlib.util.spec_from_file_location("cron_profile_router", ROUTER_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompts_file_exists_and_loads():
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    names = {job["name"] for job in data["jobs"]}
    assert JOB_NAMES <= names


@pytest.mark.parametrize("job_name", sorted(JOB_NAMES))
def test_job_is_agent_job(job_name):
    job = _load_prompts()[job_name]
    assert job["no_agent"] is False


@pytest.mark.parametrize("job_name", sorted(JOB_NAMES))
def test_prompt_has_no_hardcoded_home_paths(job_name):
    prompt = _load_prompts()[job_name]["prompt"]
    assert "/home/kensei" not in prompt, f"{job_name} prompt still hardcodes /home/kensei"


@pytest.mark.parametrize("job_name", sorted(JOB_NAMES))
def test_prompt_references_portable_env_var(job_name):
    """Each prompt must reference a portable env var (HERMES_HOME or
    BLOG_CONTENT_ROOT) instead of a hardcoded host path. CeeCee's content
    engine lives in a separate repo, so it routes via BLOG_CONTENT_ROOT;
    the other two route via HERMES_HOME."""
    prompt = _load_prompts()[job_name]["prompt"]
    assert "HERMES_HOME" in prompt or "BLOG_CONTENT_ROOT" in prompt, (
        f"{job_name} prompt references no portable env var"
    )


# Jobs that are silent on success (no output means healthy/done).
SILENT_JOBS = {"content-engine-personal-llm", "wesker-ops-daily"}
# Jobs that always deliver structured output (never silent).
STRUCTURED_JOBS = {"MrHermagi Daily Lesson"}


@pytest.mark.parametrize("job_name", sorted(JOB_NAMES))
def test_prompt_has_output_contract(job_name):
    """Each prompt must declare an explicit output contract: either [SILENT]
    on success (ops/content jobs that stay quiet when healthy) or a structured
    delivery contract (MrHermagi always delivers a titled lesson + HTML)."""
    prompt = _load_prompts()[job_name]["prompt"]
    if job_name in SILENT_JOBS:
        assert "[SILENT]" in prompt, f"{job_name} prompt has no [SILENT] contract"
    else:
        assert job_name in STRUCTURED_JOBS
        # Structured jobs must name a concrete delivery format, not be open-ended.
        assert "DELIVERY FORMAT" in prompt or "TITLE FORMAT" in prompt, (
            f"{job_name} prompt has no structured delivery contract"
        )


@pytest.mark.parametrize("job_name", sorted(JOB_NAMES))
def test_prompt_has_profile_routing_metadata(job_name):
    job = _load_prompts()[job_name]
    assert job.get("profile") == EXPECTED_PROFILES[job_name]
    routing = job.get("profile_routing")
    assert isinstance(routing, dict)
    assert routing.get("target_profile") == EXPECTED_PROFILES[job_name]
    assert "cron_store" in routing and "$HERMES_HOME" in routing["cron_store"]


@pytest.mark.parametrize("job_name", sorted(JOB_NAMES))
def test_router_returns_profile_cron_store(job_name, tmp_path):
    module = _router_module()
    profile = EXPECTED_PROFILES[job_name]
    store = module.get_profile_cron_store(job_name, hermes_home=tmp_path)
    assert store == tmp_path / "profiles" / profile / "cron" / "jobs.json"


def test_router_unknown_job_raises():
    module = _router_module()
    with pytest.raises(KeyError):
        module.get_profile_for_job("nonexistent-job")


def test_router_requires_hermes_home(monkeypatch):
    module = _router_module()
    monkeypatch.delenv("HERMES_HOME", raising=False)
    with pytest.raises(ValueError, match="HERMES_HOME"):
        module.get_profile_cron_store("wesker-ops-daily")


def test_router_maps_all_three_jobs():
    module = _router_module()
    for job_name, profile in EXPECTED_PROFILES.items():
        assert module.get_profile_for_job(job_name) == profile
