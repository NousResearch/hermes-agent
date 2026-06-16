from agent.agent_init import _default_codex_reasoning_replay_enabled


def test_openai_codex_gpt55_reasoning_replay_is_opt_in_by_default():
    assert _default_codex_reasoning_replay_enabled(
        {}, provider="openai-codex", model="gpt-5.5"
    ) is False


def test_non_codex_routes_keep_reasoning_replay_default():
    assert _default_codex_reasoning_replay_enabled(
        {}, provider="openrouter", model="anthropic/claude-sonnet-4"
    ) is True


def test_agent_config_can_reenable_codex_reasoning_replay():
    assert _default_codex_reasoning_replay_enabled(
        {"agent": {"codex_reasoning_replay_enabled": True}},
        provider="openai-codex",
        model="gpt-5.5",
    ) is True


def test_codex_config_can_disable_other_codex_reasoning_replay():
    assert _default_codex_reasoning_replay_enabled(
        {"codex": {"reasoning_replay_enabled": False}},
        provider="openai-codex",
        model="gpt-5.4",
    ) is False
