"""Frozen launch skills reach both gateway execution modes via the stable field."""

import json

import pytest
from dataclasses import replace
from types import SimpleNamespace

from gateway.session_policy import build_policy
from gateway.session_worker_construct import construct_inputs, construct_kwargs
from agent.system_prompt import build_system_prompt_parts
from tests.agent.test_system_prompt import _make_agent


def test_worker_keeps_skills_separate_from_personality(tmp_path):
    policy = replace(build_policy({'cwd': str(tmp_path)}, {}), skills_prompt="SKILL BYTES")
    runner = SimpleNamespace(
        config=None, _prefill_messages=None, _refresh_fallback_model=lambda: None,
        _get_system_prompt_for_channel=lambda *a: "PERSONALITY",
    )
    authority = SimpleNamespace(runner=runner, profile_id=str(tmp_path))
    inputs = construct_inputs(authority, policy, "chat")
    result = construct_kwargs({'policy': {'request_json': json.dumps({'construct_v1': inputs})}})
    agent = _make_agent(**result)
    parts = build_system_prompt_parts(agent)
    assert "SKILL BYTES" in parts['stable']
    assert "PERSONALITY" not in "\n\n".join(parts.values())
    assert agent.ephemeral_system_prompt == "PERSONALITY"


def test_in_process_constructor_forwards_stable_skills(tmp_path, monkeypatch):
    from gateway.session_execution import GatewaySessionAgentMixin

    policy = replace(build_policy({'cwd': str(tmp_path)}, {}), skills_prompt="SKILL BYTES")
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda *a: policy)
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda cfg: {})
    monkeypatch.setattr("gateway.session_execution.api_runtime_locked", lambda: True)
    source = SimpleNamespace(**dict.fromkeys(
        ('user_id', 'user_id_alt', 'user_name', 'chat_id', 'chat_name', 'chat_type', 'thread_id')))
    constructor = lambda **kwargs: _make_agent(**kwargs)
    host = SimpleNamespace(
        _ctx=SimpleNamespace(source=source, AIAgent=constructor, user_config={}, enabled_toolsets=[],
                             disabled_toolsets=[], session_id="session", session_key="key"),
        _runner=SimpleNamespace(_prefill_messages=None, _service_tier=None, _session_db=None),
    )
    agent = GatewaySessionAgentMixin._build_fresh_agent(host, {'model': 'm', 'runtime': {}}, 'cli',
                                              'PERSONALITY', 10, None, {}, False)
    parts = build_system_prompt_parts(agent)
    assert "SKILL BYTES" in parts['stable']
    assert "PERSONALITY" not in "\n\n".join(parts.values())
    assert agent.ephemeral_system_prompt == "PERSONALITY"


@pytest.mark.parametrize("surface", ["gateway", "tui", "oneshot"])
@pytest.mark.parametrize("skill_tools", [False, True])
def test_real_gateway_skill_preload_deduplicates_auto_load(tmp_path, monkeypatch, surface, skill_tools):
    from gateway.session_policy import render_launch_skills
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    skill = home / "skills" / "overlap"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: overlap\ndescription: Shared startup skill.\n---\n\nUNIQUE OVERLAPPING SKILL BODY", encoding="utf-8")
    auto = home / "skills" / "auto-only"
    auto.mkdir()
    (auto / "SKILL.md").write_text(
        "---\nname: auto-only\ndescription: Other auto skill.\n---\n\nOTHER AUTO SKILL BODY", encoding="utf-8")
    # Quoted markers in an active skill are prose, not metadata for other loads.
    with (skill / "SKILL.md").open("a") as stream:
        stream.write('\nExample: with the "auto-only" skill preloaded')
    (home / "config.yaml").write_text("skills:\n  auto_load: [overlap, auto-only]\n", encoding="utf-8")
    if surface == "gateway":
        prompt = render_launch_skills({'skills': ['overlap', 'overlap']}, "dedup-session")
    elif surface == "oneshot":
        from hermes_cli.oneshot import _build_preloaded_skills_prompt
        prompt = _build_preloaded_skills_prompt("overlap,overlap")
    else:
        from tui_gateway import server
        monkeypatch.setenv("HERMES_TUI_SKILLS", "overlap,overlap")
        _, prompt = server._startup_system_prompt({}, "dedup-session")
    agent = _make_agent(preloaded_skills_prompt=prompt, session_id="dedup-session",
                        valid_tool_names={'skill_view', 'skills_list'} if skill_tools else set())
    parts = build_system_prompt_parts(agent)
    assert parts['stable'].count("UNIQUE OVERLAPPING SKILL BODY") == 1
    assert parts['stable'].count("OTHER AUTO SKILL BODY") == int(skill_tools)
    from agent.review_engine import collect_parent_loaded_skills
    assert collect_parent_loaded_skills(agent, []) == ["overlap"]


def test_auto_load_keeps_distinct_skill_with_same_display_name(monkeypatch):
    from agent.skill_commands import build_preloaded_skills_prompt
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    for directory, content in [("a/one", "AUTO PATH BODY"), ("b/two", "EXPLICIT PATH BODY")]:
        skill = home / "skills" / directory
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: dup-demo\ndescription: Duplicate display name.\n---\n\n{content}", encoding="utf-8")
    (home / "config.yaml").write_text("skills:\n  auto_load: [a/one]\n", encoding="utf-8")
    prompt, _, _ = build_preloaded_skills_prompt(["b/two"], task_id="distinct-paths")
    parts = build_system_prompt_parts(_make_agent(
        preloaded_skills_prompt=prompt, valid_tool_names={"skill_view"}))
    assert parts['stable'].count("AUTO PATH BODY") == 1
    assert parts['stable'].count("EXPLICIT PATH BODY") == 1


def test_flagless_resume_rebuild_keeps_persisted_skill_after_file_removal(monkeypatch):
    from agent.skill_commands import build_preloaded_skills_prompt
    from agent.system_prompt import build_system_prompt, invalidate_system_prompt
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tests.agent.test_launch_skills_resume import _agent, _restore

    home = get_hermes_home()
    skill = home / "skills" / "retained-file"
    skill.mkdir(parents=True)
    source = skill / "SKILL.md"
    source.write_text("---\nname: retained-file\ndescription: Persisted bytes.\n---\n\nORIGINAL SKILL CONTENT", encoding="utf-8")
    prompt, _, _ = build_preloaded_skills_prompt(["retained-file"], task_id="launch-session")
    with SessionDB(db_path=home / "state.db") as db:
        db.create_session("launch-session", source="cli")
        original = _agent(db, prompt)
        _restore(original)
        source.unlink()
        resumed = _agent(db)
        _restore(resumed)
        assert resumed.preloaded_skills_prompt == prompt
        invalidate_system_prompt(resumed)
        assert build_system_prompt(resumed).count("ORIGINAL SKILL CONTENT") == 1
