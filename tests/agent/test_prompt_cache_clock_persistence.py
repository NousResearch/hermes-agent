"""The prompt-cache clock survives a restart on the agent's storage interface handle: it rides the usage
queue into the session row and a resumed agent reads it back through ``usage.prompt_cache_clock`` (sessions
stored before the clock fall back to the transcript). The saved event stays bound to the context generation
it was recorded at (``usage.prompt_cache_generation``), and a resumed agent folds its first response onto the
saved event, never onto nothing."""

from types import SimpleNamespace

from agent.prompt_cache_clock import (
    CacheEvent, cache_window, event_from_json, event_to_json, record_attempt_sent, record_cache_response, stored_cache_window,
)
from agent.usage_pricing import normalize_usage
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session, seed_usage

_OR = ("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6")


def _open(tmp_path):
    return open_persistence(profile_scope(home=tmp_path), eager=True)


def _resumed_agent(db, session_id, route=_OR, api_mode="chat_completions"):
    """An agent resumed on *db*: a writer's handle is also its interface; a worker's store has none."""
    from agent.runtime_session_store import RuntimeSessionStore
    return SimpleNamespace(
        provider=route[0], base_url=route[1], model=route[2], api_mode=api_mode, session_id=session_id,
        _cache_ttl="1h", _use_prompt_caching=True, _session_db=db,
        persistence=None if isinstance(db, RuntimeSessionStore) else db,
        context_compressor=SimpleNamespace(compression_count=0),  # a fresh process: its compressor starts at 0
    )


def _save(db, session_id, event):
    seed_usage(db, session_id, input_tokens=10, cache_write_tokens=event.write_tokens, api_call_count=1,
               prompt_cache_clock=event_to_json(event))


def _event(route=_OR, **overrides):
    fields = dict(route=route, destination=route, route_class="openrouter_claude", requested_ttl="1h", sent_at=1_000.0,
                  read_tokens=0, write_tokens=4800, tier="1h", confidence="declared", reason="openrouter_honours_ttl",
                  coverage={"1h": 4_600.0}, guaranteed_until=4_600.0, estimated_until=4_600.0, session_id="s1",
                  generation=0)
    return CacheEvent(**{**fields, **overrides})


def test_resume_restores_the_stored_window(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _open(tmp_path)
    try:
        seed_session(db, "s1", "cli")
        _save(db, "s1", _event())
        seed_usage(db, "s1", input_tokens=5, api_call_count=1)  # a later delta without an event keeps it
        window = cache_window(_resumed_agent(db, "s1"))
        assert (window.guaranteed_until, window.tier, window.confidence) == (4_600.0, "1h", "declared")

        # No stored clock (a pre-column session): the newest assistant row anchors an assumed end, never
        # a guarantee, and only when the session ever touched the cache (an error row is no evidence).
        seed_session(db, "old", "cli")
        seed_messages(db, "old", {"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello", "timestamp": 2_000.0})
        assert cache_window(_resumed_agent(db, "old")) is None
        seed_usage(db, "old", cache_read_tokens=4000, api_call_count=1)
        window = cache_window(_resumed_agent(db, "old"))
        assert (window.guaranteed_until, window.estimated_until, window.confidence) == (None, 5_600.0, "assumed")
    finally:
        db.close()


def test_a_compaction_committed_before_the_restart_keeps_the_saved_clock_invalid(tmp_path, monkeypatch):
    """In-place /compress archives rows without touching the saved clock. The event keeps the
    generation it was recorded at, so a reopened session (whose compressor count restarts at 0)
    cannot revive the pre-compaction window."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _open(tmp_path)
    try:
        seed_session(db, "s1", "cli")
        seed_messages(db, "s1", *({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(4)))
        assert db.usage.prompt_cache_generation("s1") == 0
        _save(db, "s1", _event(generation=0))
        assert cache_window(_resumed_agent(db, "s1")).confidence == "declared"

        db.messages.compact("s1", [NewMessage.from_live({"role": "user", "content": "[summary]"})])
        assert db.usage.prompt_cache_generation("s1") > 0
        window = cache_window(_resumed_agent(db, "s1"))
        assert (window.confidence, window.reason) == ("unknown", "compacted")
        # The agentless /context read sees the same answer from the column alone.
        stored = stored_cache_window(db.sessions.get("s1").prompt_cache_clock, session_id="s1",
                                     generation=db.usage.prompt_cache_generation("s1"))
        assert (stored["confidence"], stored["reason"]) == ("unknown", "compacted")
    finally:
        db.close()


def test_a_resumed_first_pure_read_keeps_the_saved_proven_tier(tmp_path, monkeypatch):
    """The saved event is restored before the first response is folded in: a pure read on a resumed
    agent keeps the proven 12h coverage (and its fixed deadline) instead of overwriting it with 30m."""
    import agent.prompt_cache_clock as clock

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    route = ("openai-api", "https://api.openai.com/v1", "gpt-5.6-luna")
    db = _open(tmp_path)
    try:
        seed_session(db, "s1", "cli")
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
    """The clock is advisory: a store error on the restore or generation read inside
    record_response_usage leaves the response recorded (session counters move) and the window unknown."""
    import sqlite3

    from agent import turn_usage
    from persistence.sqlite.conversation import usage as sqlite_usage
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _open(tmp_path)
    agent = AIAgent(api_key="k", base_url=_OR[1], provider=_OR[0], api_mode="chat_completions", model=_OR[2],
                    session_id="s1", platform="cli", session_db=db, persistence=db, quiet_mode=True,
                    skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    try:
        def broken(*_a, **_k):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(sqlite_usage, "get_prompt_cache_clock", broken)
        monkeypatch.setattr(sqlite_usage, "prompt_cache_generation", broken)
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
    """A worker agent's session_db is a RuntimeSessionStore: the saved event must carry the owner's durable
    generation (stamped in the usage receipt, no extra read through the write journal), so the owner's
    agentless read sees the fresh window as warm, not compacted."""
    from agent.runtime_session_store import RuntimeSessionStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _open(tmp_path)
    try:
        seed_session(db, "s1", "cli")
        seed_messages(db, "s1", *({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(4)))
        db.messages.compact("s1", [NewMessage.from_live({"role": "user", "content": "[summary]"})])
        owner_generation = db.usage.prompt_cache_generation("s1")
        assert owner_generation > 0
        scope = dict(epoch=db.runtime.begin_epoch(instance_id="fixture"), execution_id="w", session_id="s1",
                     generation=0)
        db.worker_executions.register(**scope, kind="compute", adoption_secret="secret")
        store = RuntimeSessionStore(lambda method, **p: db.worker_executions.persist_operation(**p), scope,
                                    tmp_path / "outbox")
        try:
            agent = _resumed_agent(store, "s1")
            record_attempt_sent(agent, {"messages": [{"role": "system", "content": [
                {"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]})
            write = normalize_usage({"prompt_tokens": 5000, "completion_tokens": 3,
                                     "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4800}},
                                    provider=_OR[0], api_mode="chat_completions")
            event = record_cache_response(agent, write, model=_OR[2], provider=_OR[0], base_url=_OR[1],
                                          api_mode="chat_completions")
            store.queue_token_counts("s1", input_tokens=10, cache_write_tokens=4800, api_call_count=1,
                                     prompt_cache_clock=event_to_json(event))
            assert store.failure is None
            assert event_from_json(db.sessions.get("s1").prompt_cache_clock).generation == owner_generation
            stored = stored_cache_window(db.sessions.get("s1").prompt_cache_clock, session_id="s1",
                                         generation=db.usage.prompt_cache_generation("s1"))
            assert (stored["confidence"], stored["tier"]) == ("declared", "1h")
            # A prune in the same worker archives in place without moving compression_count: the
            # window it folded is gone, and the next response must not carry its coverage.
            store.archive_and_compact("s1", [{"role": "user", "content": "[pruned]"}])
            assert (cache_window(agent).confidence, cache_window(agent).reason) == ("unknown", "compacted")
        finally:
            store.close()
    finally:
        db.close()
