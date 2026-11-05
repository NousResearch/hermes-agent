"""A vetoed prompt's submit-time row is rewritten to the withheld placeholder when the agent writes through a
storage interface handle, as the gateway, one-shot, cron and TUI agents do."""

from __future__ import annotations

from agent.turn_context import PRE_LLM_BLOCK_WITHHELD_PROMPT, _withhold_blocked_prompt

SECRET = "CUSTOMER-ID-4242"


def test_a_submit_time_row_is_rewritten_to_the_placeholder_through_the_handle(tmp_path, monkeypatch):
    """The TUI and Desktop store the user row at submit time; the veto rewrites that row in place, so the stored
    content is the placeholder and the typed text is kept only in the row's display metadata."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from session_rules.turns import withheld_prompt_text
    from persistence.conversation.messages import NewMessage
    from persistence.conversation.sessions import NewSession
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent

    typed = f"look up {SECRET}"
    store = open_persistence(profile_scope(home=tmp_path), eager=True)
    try:
        store.sessions.create(NewSession(session_id="s-block", source="tui"))
        row_id = store.messages.append("s-block", NewMessage(role="user", content=typed))
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:9/v1", provider="openai-compat", model="test-model",
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="tui", persistence=store, session_id="s-block",
        )
        try:
            _withhold_blocked_prompt(agent, {"role": "user", "content": typed, "_row_id": row_id}, typed)
        finally:
            agent.close()
        (stored,) = [message.as_openai_message() for message in store.messages.read_conversation("s-block")]
        assert stored["content"] == PRE_LLM_BLOCK_WITHHELD_PROMPT
        assert withheld_prompt_text(stored) == typed
    finally:
        store.close()
