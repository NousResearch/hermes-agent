"""Skills preloaded into a cron prompt count as already viewed for that run.

A cron job's ``skills`` are loaded into the prompt before the agent starts. If the agent
then calls ``skill_view`` on the same unchanged skill, it should get the dedup stub, not
the full SKILL.md a second time.
"""

import json

import pytest


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with one planted skill; returns (scheduler, skill_view_with_bump)."""
    hermes_home = tmp_path / ".hermes"
    skill_dir = hermes_home / "skills" / "digest-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: digest-skill\ndescription: test\n---\n\nStep one: compile the digest.\n",
        encoding="utf-8",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "sources.md").write_text("Source list.\n", encoding="utf-8")
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(hermes_home / "skill-bundles"))

    import tools.skills_tool as skills_tool
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", hermes_home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", hermes_home)
    skills_tool.reset_skill_view_dedup()
    import agent.skill_bundles as skill_bundles
    skill_bundles._bundles_cache = {}
    skill_bundles._bundles_cache_mtime = None

    import cron.scheduler as scheduler
    yield scheduler, skills_tool._skill_view_with_bump
    skills_tool.reset_skill_view_dedup()


def _view(view_fn, task_id: str, file_path=None) -> dict:
    args = {"name": "digest-skill"}
    if file_path:
        args["file_path"] = file_path
    return json.loads(view_fn(args, task_id=task_id))


def test_skill_view_after_preload_returns_dedup_stub(cron_env):
    scheduler, view = cron_env
    job = {"id": "digest", "prompt": "Run the digest.", "skills": ["digest-skill"]}
    prompt = scheduler._build_job_prompt(job, run_task_id="cron:digest:run1")
    assert "Step one: compile the digest." in prompt

    repeat = _view(view, "cron:digest:run1")
    assert repeat.get("dedup") is True
    assert "content" not in repeat

    # Scoped to this run, and only to SKILL.md itself.
    assert "Step one" in _view(view, "cron:digest:run2").get("content", "")
    assert "Source list." in _view(view, "cron:digest:run1", "references/sources.md").get("content", "")


def test_run_job_preloads_under_the_agent_task_id(cron_env, monkeypatch):
    """The id the prompt builder records under must be the task_id the agent's tools run with."""
    import sys
    scheduler, _ = cron_env
    from cron import scheduler_delivery
    observed: dict = {}

    class FakeAgent:
        def __init__(self, **_kw):
            pass

        def run_conversation(self, *_a, task_id=None, **_kw):
            observed["agent_task_id"] = task_id
            return {"final_response": "done", "messages": [{"role": "assistant", "content": "done"}]}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake_mod = type(sys)("run_agent")
    fake_mod.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **_kw: {
        "provider": "test", "api_key": "k", "base_url": "http://test.local",
        "api_mode": "chat_completions"})

    def fake_build(job, prerun_script=None, **kw):
        observed["prompt_task_id"] = kw.get("run_task_id")
        return "hi"

    monkeypatch.setattr(scheduler, "_build_job_prompt", fake_build)
    monkeypatch.setattr(scheduler_delivery, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(scheduler, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(scheduler, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    success, *_ = scheduler.run_job({"id": "digest", "name": "digest", "schedule_display": "manual"})
    assert success is True
    assert observed["prompt_task_id"]
    assert observed["prompt_task_id"] == observed["agent_task_id"]
