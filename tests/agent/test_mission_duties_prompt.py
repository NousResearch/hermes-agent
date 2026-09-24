"""Mission duty guidance follows the owning profile and the available scheduler."""

from types import SimpleNamespace

from agent.system_prompt import MISSION_DUTIES_GUIDANCE, build_system_prompt_parts


def _agent(home, tools):
    return SimpleNamespace(
        _session_db=SimpleNamespace(db_path=str(home / "state.db")),
        load_soul_identity=True,
        skip_context_files=True,
        valid_tool_names=tools,
        platform="cron",
        provider="",
        model="",
        _memory_store=None,
        _memory_manager=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _environment_probe=False,
        _bot_mode_protocol=False,
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _execution_guidance=False,
        _kanban_worker_guidance="",
        pass_session_id=False,
        session_id="mission-test",
    )


def test_profile_mission_guidance_is_scoped_to_home_and_scheduler(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "SOUL.md").write_text("Summarize Gmail into Discord.", encoding="utf-8")
    (b / "SOUL.md").write_text("Answer math questions.", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(b))
    monkeypatch.setattr("agent.system_prompt._skills_prompt", lambda agent: "")
    monkeypatch.setattr("agent.system_prompt._coding_parts", lambda agent: ([], [], []))
    monkeypatch.setattr("agent.system_prompt._post_workspace_parts", lambda agent: [])
    monkeypatch.setattr("agent.system_prompt._auto_load_parts", lambda agent: [])
    monkeypatch.setattr("agent.prompt_builder.build_environment_hints", lambda: "")

    for home, other in ((a, b), (b, a), (a, b)):
        prompt = build_system_prompt_parts(_agent(home, {"cronjob_manage"}))["stable"]
        assert (home / "SOUL.md").read_text(encoding="utf-8") in prompt
        assert (other / "SOUL.md").read_text(encoding="utf-8") not in prompt
        assert MISSION_DUTIES_GUIDANCE in prompt
        assert "paused=true" in prompt and "explicit user consent" in prompt
        assert "next user interaction" in prompt

    without_cron = build_system_prompt_parts(_agent(a, set()))["stable"]
    assert MISSION_DUTIES_GUIDANCE not in without_cron
    (a / "SOUL.md").unlink()
    without_mission = build_system_prompt_parts(_agent(a, {"cronjob_manage"}))["stable"]
    assert MISSION_DUTIES_GUIDANCE not in without_mission
