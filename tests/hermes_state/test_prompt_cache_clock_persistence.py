"""The prompt-cache clock survives a restart: it rides the token batch into ``sessions`` and a
resumed agent reads it back (sessions stored before the column fall back to the transcript). The
saved event stays bound to the context generation it was recorded at, and a resumed agent folds
its first response onto the saved event, never onto nothing."""

from types import SimpleNamespace

from agent.prompt_cache_clock import (
    CacheEvent, cache_window, event_to_json, record_attempt_sent, record_cache_response, stored_cache_window,
)
from agent.usage_pricing import normalize_usage
from hermes_state import SessionDB

_OR = ("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6")


def _resumed_agent(db, session_id, route=_OR, api_mode="chat_completions"):
    return SimpleNamespace(
        provider=route[0], base_url=route[1], model=route[2], api_mode=api_mode, session_id=session_id,
        _cache_ttl="1h", _use_prompt_caching=True, _session_db=db,
        context_compressor=SimpleNamespace(compression_count=0),  # a fresh process: its compressor starts at 0
    )


def _save(db, session_id, event):
    db.queue_token_counts(session_id, input_tokens=10, cache_write_tokens=event.write_tokens, api_call_count=1,
                          prompt_cache_clock=event_to_json(event))
    assert db.flush_token_counts()


def _event(route=_OR, **overrides):
    fields = dict(route=route, destination=route, route_class="openrouter_claude", requested_ttl="1h", sent_at=1_000.0,
                  read_tokens=0, write_tokens=4800, tier="1h", confidence="declared", reason="openrouter_honours_ttl",
                  coverage={"1h": 4_600.0}, guaranteed_until=4_600.0, estimated_until=4_600.0, session_id="s1",
                  generation=0)
    return CacheEvent(**{**fields, **overrides})


def test_resume_restores_the_stored_window(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        _save(db, "s1", _event())
        db.queue_token_counts("s1", input_tokens=5, api_call_count=1)  # a later batch without an event keeps it
        assert db.flush_token_counts()
        window = cache_window(_resumed_agent(db, "s1"))
        assert (window.guaranteed_until, window.tier, window.confidence) == (4_600.0, "1h", "declared")

        # No stored clock (a pre-column session): the newest assistant row anchors an assumed end, never
        # a guarantee, and only when the session ever touched the cache (an error row is no evidence).
        db.create_session("old", "cli")
        db.append_message("old", "user", "hi")
        db.append_message("old", "assistant", "hello", timestamp=2_000.0)
        assert cache_window(_resumed_agent(db, "old")) is None
        db.update_token_counts("old", cache_read_tokens=4000, api_call_count=1)
        window = cache_window(_resumed_agent(db, "old"))
        assert (window.guaranteed_until, window.estimated_until, window.confidence) == (None, 5_600.0, "assumed")
    finally:
        db.close()


def test_a_compaction_committed_before_the_restart_keeps_the_saved_clock_invalid(tmp_path, monkeypatch):
    """In-place /compress archives rows without touching the saved clock. The event keeps the
    generation it was recorded at, so a reopened session (whose compressor count restarts at 0)
    cannot revive the pre-compaction window."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        for i in range(4):
            db.append_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
        assert db.prompt_cache_generation("s1") == 0
        _save(db, "s1", _event(generation=0))
        assert cache_window(_resumed_agent(db, "s1")).confidence == "declared"

        db.archive_and_compact("s1", [{"role": "user", "content": "[summary]"}])
        assert db.prompt_cache_generation("s1") > 0
        window = cache_window(_resumed_agent(db, "s1"))
        assert (window.confidence, window.reason) == ("unknown", "compacted")
        # The agentless /context read sees the same answer from the column alone.
        stored = stored_cache_window(db.get_session("s1")["prompt_cache_clock"], session_id="s1",
                                     generation=db.prompt_cache_generation("s1"))
        assert (stored["confidence"], stored["reason"]) == ("unknown", "compacted")
    finally:
        db.close()


def test_a_resumed_first_pure_read_keeps_the_saved_proven_tier(tmp_path, monkeypatch):
    """The saved event is restored before the first response is folded in: a pure read on a resumed
    agent keeps the proven 12h coverage (and its fixed deadline) instead of overwriting it with 30m."""
    import agent.prompt_cache_clock as clock

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    route = ("openai-api", "https://api.openai.com/v1", "gpt-5.6-luna")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        _save(db, "s1", _event(route=route, route_class="openai_ttl", requested_ttl="12h", tier="12h",
                               confidence="proven", reason="openai_write_tier", coverage={"12h": 44_200.0},
                               guaranteed_until=44_200.0, estimated_until=87_400.0))
        agent = _resumed_agent(db, "s1", route, api_mode="codex_responses")
        monkeypatch.setattr(clock.time, "time", lambda: 2_000.0)
        record_attempt_sent(agent, {"input": [], "extra_body": {"prompt_cache_options": {"ttl": "12h"}}})
        read = normalize_usage({"input_tokens": 4000, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 3640}},
                               provider=route[0], api_mode="codex_responses")
        event = record_cache_response(agent, read, model=route[2], provider=route[0], base_url=route[1],
                                      api_mode="codex_responses")
        assert (event.tier, event.confidence, event.guaranteed_until) == ("12h", "proven", 44_200.0)
    finally:
        db.close()


def test_a_failed_store_read_never_loses_the_reply_and_reports_unknown(tmp_path, monkeypatch):
    """The clock is advisory: a SQLite error on the restore or generation read inside
    record_response_usage leaves the response recorded (session counters move) and the window unknown."""
    import sqlite3

    from agent import turn_usage
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(api_key="k", base_url=_OR[1], provider=_OR[0], api_mode="chat_completions", model=_OR[2],
                    session_id="s1", platform="cli", session_db=db, quiet_mode=True, skip_context_files=True,
                    skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    try:
        def broken(*_a, **_k):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(db, "get_prompt_cache_clock", broken)
        monkeypatch.setattr(db, "prompt_cache_generation", broken)
        record_attempt_sent(agent, {"messages": [{"role": "system", "content": [
            {"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]})
        usage = SimpleNamespace(prompt_tokens=5000, completion_tokens=3, total_tokens=5003, completion_tokens_details=None,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=4000, cache_write_tokens=0))
        turn_usage.record_response_usage(agent, SimpleNamespace(usage=usage, id=None, model=_OR[2]),
                                         messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                         api_duration=0.1, compression_attempts=0, max_compression_attempts=3)
        assert agent.session_cache_read_tokens == 4000
        window = cache_window(agent)
        assert (window.confidence, window.reason) == ("unknown", "storage_unavailable")
    finally:
        agent.close()
        db.close()


def test_a_managed_worker_clock_carries_the_owner_generation(tmp_path, monkeypatch):
    """A worker agent's session_db is a RuntimeSessionStore: its event must carry the owner's durable
    generation, so the owner's agentless read sees the fresh window as warm, not compacted."""
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        for i in range(4):
            db.append_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
        db.archive_and_compact("s1", [{"role": "user", "content": "[summary]"}])
        owner_generation = db.prompt_cache_generation("s1")
        assert owner_generation > 0
        scope = dict(epoch=begin_runtime_epoch(db, instance_id="fixture"), execution_id="w", session_id="s1",
                     generation=0)
        register_worker_execution(db, **scope, kind="compute", adoption_secret="secret")
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / "outbox")
        try:
            agent = _resumed_agent(store, "s1")
            record_attempt_sent(agent, {"messages": [{"role": "system", "content": [
                {"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]})
            write = normalize_usage({"prompt_tokens": 5000, "completion_tokens": 3,
                                     "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4800}},
                                    provider=_OR[0], api_mode="chat_completions")
            event = record_cache_response(agent, write, model=_OR[2], provider=_OR[0], base_url=_OR[1],
                                          api_mode="chat_completions")
            assert event.generation == owner_generation
            store.queue_token_counts("s1", input_tokens=10, cache_write_tokens=4800, api_call_count=1,
                                     prompt_cache_clock=event_to_json(event))
        finally:
            store.close()
        stored = stored_cache_window(db.get_session("s1")["prompt_cache_clock"], session_id="s1",
                                     generation=db.prompt_cache_generation("s1"))
        assert (stored["confidence"], stored["tier"]) == ("declared", "1h")
    finally:
        db.close()
