from types import SimpleNamespace

from agent.system_prompt import build_system_prompt


def _agent_with_tools(valid_tool_names):
    return SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=set(valid_tool_names),
        provider="openrouter",
        model="test-model",
        platform="cli",
        memory_provider=None,
        _memory_store=None,
        _memory_nudge_interval=0,
        _turns_since_memory=0,
        workdir=None,
        skip_memory=True,
        user_id=None,
        user_name=None,
        chat_id=None,
        chat_name=None,
        chat_type=None,
        thread_id=None,
        session_id="test-session",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        _cached_system_prompt=None,
        ephemeral_system_prompt=None,
        tools=[],
        memory_context="",
        user_profile="",
        status_callback=None,
        _tool_use_enforcement=False,
        is_tui=False,
        supports_session_search=False,
        supports_session_search_exact=False,
        _context_files_prompt_cache=None,
        _skills_system_prompt_cache=None,
        _soul_md_cache=None,
        service_tier=None,
        request_overrides=None,
        user_profile_store=None,
        _memory_manager=None,
        _cached_environment_hints="",
        pass_session_id=False,
        _memory_enabled=False,
        _user_profile_enabled=False,
    )


def _stub_ra():
    return SimpleNamespace(
        load_soul_md=lambda: None,
        build_nous_subscription_prompt=lambda valid_tool_names: "",
        build_skills_system_prompt=lambda **kwargs: "",
        build_environment_hints=lambda: "",
        build_context_files_prompt=lambda *args, **kwargs: "",
        get_toolset_for_tool=lambda *args, **kwargs: None,
    )


def test_guidance_blocks_compact_mode_shortens_system_prompt(monkeypatch):
    import agent.system_prompt as sp

    monkeypatch.setattr(sp, "_ra", _stub_ra)
    monkeypatch.setattr(sp, "_compact_guidance_blocks_enabled", lambda: False)
    full = build_system_prompt(_agent_with_tools({"memory", "session_search", "skill_manage", "web_search", "search_router"}))

    monkeypatch.setattr(sp, "_compact_guidance_blocks_enabled", lambda: True)
    compact = build_system_prompt(_agent_with_tools({"memory", "session_search", "skill_manage", "web_search", "search_router"}))

    assert len(compact) < len(full)
    assert "Save only durable facts with `memory`" in compact
    assert "For Hermes Agent config/setup/use/troubleshooting" in compact
    assert "Save durable facts using the memory tool" in full


def test_guidance_blocks_compact_mode_preserves_key_tool_rules(monkeypatch):
    import agent.system_prompt as sp

    monkeypatch.setattr(sp, "_ra", _stub_ra)
    monkeypatch.setattr(sp, "_compact_guidance_blocks_enabled", lambda: True)
    prompt = build_system_prompt(_agent_with_tools({"memory", "session_search", "skill_manage", "web_search", "search_router"}))

    assert "`hermes-agent`" in prompt
    assert "`memory`" in prompt
    assert "`session_search`" in prompt
    assert "`skill_manage`" in prompt
    assert "`search_router`" in prompt
    assert "`web_search`" in prompt


def test_guidance_blocks_default_mode_keeps_full_copy(monkeypatch):
    import agent.system_prompt as sp

    monkeypatch.setattr(sp, "_ra", _stub_ra)
    monkeypatch.setattr(sp, "_compact_guidance_blocks_enabled", lambda: False)
    prompt = build_system_prompt(_agent_with_tools({"memory"}))

    assert "Save durable facts using the memory tool" in prompt
    assert "Save only durable facts with `memory`" not in prompt
