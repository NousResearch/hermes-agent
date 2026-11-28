"""Blocked built-in memory reaches the agent's user-visible notice rail (#156389)."""

from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("stage", ["startup", "compression"])
def test_memory_block_is_reported_at_each_prompt_load(tmp_path, monkeypatch, stage):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("memory:\n  provider: ''\n", encoding="utf-8")
    memories = tmp_path / "memories"
    memories.mkdir()
    path = memories / "USER.md"
    blocked = "He likes to edit his notes and keeps a CLAUDE.md for each project."
    path.write_text(blocked if stage == "startup" else "A clean profile.", encoding="utf-8")
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", Mock())
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 204_800)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *_a, **_k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *_a, **_k: {})
    from run_agent import AIAgent
    from agent.system_prompt import invalidate_system_prompt

    notices = []
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test-model",
        platform="discord", quiet_mode=True, skip_context_files=True,
        notice_callback=notices.append if stage == "compression" else None,
    )
    try:
        if stage == "startup":
            assert notices == []
            agent.notice_callback = notices.append
            agent._replay_startup_warnings()
        if stage == "compression":
            assert not any("USER.md" in notice.text for notice in notices)
            path.write_text(blocked, encoding="utf-8")
            invalidate_system_prompt(agent)
        warnings = [notice for notice in notices if "USER.md" in notice.text]
        assert len(warnings) == 1
        assert warnings[0].level == "warn" and "unchanged" in warnings[0].text
        assert blocked not in warnings[0].text
        assert path.read_text(encoding="utf-8") == blocked
    finally:
        agent.close()
