"""The prompt-cache clock survives a restart: it rides the token batch into ``sessions`` and a
resumed agent reads it back (sessions stored before the column fall back to the transcript)."""

from types import SimpleNamespace

from agent.prompt_cache_clock import CacheEvent, cache_window, event_to_json
from hermes_state import SessionDB


def _resumed_agent(db, session_id):
    return SimpleNamespace(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", model="anthropic/claude-sonnet-4.6",
        api_mode="chat_completions", session_id=session_id, _cache_ttl="1h", _use_prompt_caching=True,
        _session_db=db, context_compressor=SimpleNamespace(compression_count=0),
    )


def test_resume_restores_the_stored_window(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        event = CacheEvent(route=("openrouter", "openrouter.ai", "anthropic/claude-sonnet-4.6"),
                           route_class="openrouter_claude", requested_ttl="1h", sent_at=1_000.0, read_tokens=0,
                           write_tokens=4800, tier="1h", confidence="declared", reason="openrouter_honours_ttl",
                           session_id="s1", compression_count=3)
        db.queue_token_counts("s1", input_tokens=10, cache_write_tokens=4800, api_call_count=1,
                              prompt_cache_clock=event_to_json(event))
        db.queue_token_counts("s1", input_tokens=5, api_call_count=1)  # a later batch without an event keeps it
        assert db.flush_token_counts()

        window = cache_window(_resumed_agent(db, "s1"))  # a fresh process: its compressor starts at 0
        assert (window.expires_at, window.tier, window.confidence) == (4_600.0, "1h", "declared")

        # No stored clock (a pre-column session): the newest assistant row anchors an assumed window,
        # but only when the session ever touched the cache (a failed turn's error row is no evidence).
        db.create_session("old", "cli")
        db.append_message("old", "user", "hi")
        db.append_message("old", "assistant", "hello", timestamp=2_000.0)
        assert cache_window(_resumed_agent(db, "old")) is None
        db.update_token_counts("old", cache_read_tokens=4000, api_call_count=1)
        window = cache_window(_resumed_agent(db, "old"))
        assert (window.expires_at, window.tier, window.confidence) == (5_600.0, "1h", "assumed")
    finally:
        db.close()
