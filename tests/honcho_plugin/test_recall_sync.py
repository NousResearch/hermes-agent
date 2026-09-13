"""Opt-in current-query alignment and bounded late-worker isolation."""
import threading
import time

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig

class RecallManager:
    def __init__(self):
        self.queries = []
        self.prompts = []
        self.queued = []
        self.pending_context = {}

    def set_context_result(self, session, context):
        self.pending_context[session] = context

    def pop_context_result(self, session):
        return self.pending_context.pop(session, None)

    def get_prefetch_context(self, session, query, **kwargs):
        self.queries.append((session, query))
        return {"representation": f"base:{query}"}

    def dialectic_query(self, session, prompt, **kwargs):
        self.prompts.append((session, prompt, kwargs))
        return f"dialectic:{prompt}"

    def prefetch_context(self, session, query):
        self.queued.append((session, query))


def make_provider(**options):
    provider = HonchoMemoryProvider()
    cfg = HonchoClientConfig(timeout=1, **options)
    cfg.recall_sync = True
    provider._config = cfg
    provider._manager = RecallManager()
    provider._session_key = "session-a"
    provider._session_initialized = True
    for name in ("recall_sync", "recall_mode", "injection_frequency", "context_cadence",
                 "dialectic_cadence", "dialectic_depth", "dialectic_depth_levels"):
        setattr(provider, f"_{name}", getattr(cfg, name))
    return provider


def test_two_queries_never_consume_previous_query_caches(tmp_path, monkeypatch):
    from plugins.memory.honcho import cli

    config_provider = HonchoMemoryProvider()
    path = tmp_path / "honcho.json"
    config_provider.save_config({"recallSync": True}, str(tmp_path))
    assert HonchoClientConfig.from_global_config(host="hermes", config_path=path).recall_sync
    config_provider.save_config({"hosts": {"hermes": {"recallSync": False}}}, str(tmp_path))
    assert not HonchoClientConfig.from_global_config(host="hermes", config_path=path).recall_sync
    assert not HonchoClientConfig().recall_sync
    monkeypatch.setattr(cli, "_prompt", lambda label, default=None, **kw: default or "")
    host = {"recallSync": False}
    cli._setup_tuning({"recallSync": True}, host)
    assert host["recallSync"] is False
    provider = make_provider()
    provider._base_context_cache = "STALE BASE"
    provider._prefetch_result = "STALE DIALECTIC"
    for turn, query in enumerate(("Plan the garden", "Debug the compiler"), 1):
        provider.on_turn_start(turn, query)
        result = provider.prefetch(query)
        assert f"base:{query}" in result
        assert "STALE" not in result
        if turn == 2:
            assert "Plan the garden" not in result
        assert query in provider._manager.prompts[-1][1]
        provider.queue_prefetch(query)
    assert provider._manager.queries == [("session-a", "Plan the garden"), ("session-a", "Debug the compiler")]
    assert provider._manager.queued == []
    assert provider._base_context_cache == "STALE BASE"
    assert provider._prefetch_result == "STALE DIALECTIC"
    assert provider._manager.pending_context == {}


def test_timeout_keeps_single_flight_and_late_result_cannot_publish():
    provider = make_provider()
    provider._config.timeout = 0.02
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(session, query, **kwargs):
        calls.append(query)
        entered.set()
        assert release.wait(3)
        return {"representation": "LATE OLD QUERY"}

    provider._manager.get_prefetch_context = blocked
    provider._base_context_cache = "STALE"
    provider.on_turn_start(1, "Plan the garden")
    try:
        started = time.monotonic()
        assert provider.prefetch("Plan the garden") == ""
        assert entered.wait(2)
        worker = provider._recall_sync_thread
        provider.on_turn_start(2, "Debug the compiler")
        assert provider.prefetch("Debug the compiler") == ""
        assert time.monotonic() - started < 2
        assert provider._recall_sync_thread is worker and worker.is_alive()
        assert calls == ["Plan the garden"]
        assert provider._last_context_turn == provider._last_dialectic_turn == -999
    finally:
        release.set()
        provider._recall_sync_thread.join(2)
    assert provider._base_context_cache == "STALE"
    assert provider._prefetch_result == ""
    assert provider._manager.pending_context == {}
    assert provider._manager.prompts == []
    assert provider._last_context_turn == provider._last_dialectic_turn == -999
    provider._manager.get_prefetch_context = lambda session, query, **kwargs: {"card": query}
    assert "Debug the compiler" in provider.prefetch("Debug the compiler")

    # Turn numbers can repeat after reset/rewind; only the operation generation owns results.
    for invalidate in (lambda: provider.on_turn_start(2, "A different query"),
                       lambda: provider.on_session_switch("session-b")):
        provider = make_provider()
        provider.on_turn_start(2, "Plan the garden")

        def superseded(session, query, **kwargs):
            invalidate()
            return {"card": "old generation"}

        provider._manager.get_prefetch_context = superseded
        assert provider.prefetch("Plan the garden") == ""
        assert provider._last_context_turn == provider._last_dialectic_turn == -999
