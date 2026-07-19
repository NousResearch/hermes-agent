from gateway.run import GatewayRunner


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._profile_default_skills_prompt_cache = {}
    runner._profile_default_skills_cache_sessions = {}
    return runner


def test_reset_renders_profile_default_skill_for_new_session(monkeypatch, tmp_path):
    import agent.skill_commands as skill_commands

    loaded_skill = {
        "content": "Session: ${HERMES_SESSION_ID}",
        "raw_content": "Session: ${HERMES_SESSION_ID}",
    }
    monkeypatch.setattr(
        skill_commands,
        "_load_skill_payload",
        lambda _identifier, task_id=None: (
            loaded_skill,
            tmp_path,
            "profile-skill",
        ),
    )
    monkeypatch.setattr(
        skill_commands,
        "_load_skills_config",
        lambda: {"template_vars": True},
    )

    runner = _make_runner()
    session_key = "agent:main:discord:dm:42"
    old_prompt = runner._get_profile_default_skills_prompt(
        session_key=session_key,
        session_id="session-old",
        default_skills=["profile-skill"],
    )
    runner._evict_profile_default_skills_prompt(session_id="session-old")
    new_prompt = runner._get_profile_default_skills_prompt(
        session_key=session_key,
        session_id="session-new",
        default_skills=["profile-skill"],
    )

    assert "Session: session-old" in old_prompt
    assert "Session: session-new" in new_prompt
    assert "Session: session-old" not in new_prompt
    assert {key[0] for key in runner._profile_default_skills_prompt_cache} == {
        "session-new"
    }


def test_auto_reset_evicts_prior_prompts_for_stable_route():
    runner = _make_runner()
    runner._profile_default_skills_prompt_cache = {
        ("session-old", ("profile-skill",), ""): "old",
        ("session-new", ("profile-skill",), ""): "new",
    }
    runner._profile_default_skills_cache_sessions = {
        "agent:main:discord:dm:42": {"session-old", "session-new"}
    }

    runner._evict_profile_default_skills_prompt(
        session_key="agent:main:discord:dm:42",
        keep_session_id="session-new",
    )

    assert list(runner._profile_default_skills_prompt_cache) == [
        ("session-new", ("profile-skill",), "")
    ]
