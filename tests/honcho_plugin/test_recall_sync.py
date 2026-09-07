"""Current-query recall contracts through the real Honcho provider (no SDK/network)."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


def test_recall_sync_opt_in_roundtrip_and_host_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = HonchoMemoryProvider()
    path = tmp_path / "honcho.json"
    provider.save_config({"recallSync": True}, str(tmp_path))
    assert HonchoClientConfig.from_global_config(host="hermes", config_path=path).recall_sync is True
    provider.save_config({"hosts": {"hermes": {"recallSync": False}}}, str(tmp_path))
    assert HonchoClientConfig.from_global_config(host="hermes", config_path=path).recall_sync is False
    assert json.loads(path.read_text())["recallSync"] is True
    assert HonchoClientConfig().recall_sync is False


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

    def get_prefetch_context(self, session, query):
        self.queries.append((session, query))
        return {"representation": f"base:{query}"}

    def dialectic_query(self, session, prompt, **kwargs):
        self.prompts.append((session, prompt, kwargs))
        return f"dialectic:{prompt}"

    def prefetch_context(self, session, query):
        self.queued.append((session, query))


def make_provider(**options):
    provider = HonchoMemoryProvider()
    cfg = HonchoClientConfig(recall_sync=True, timeout=1, **options)
    provider._config = cfg
    provider._manager = RecallManager()
    provider._session_key = "session-a"
    provider._session_initialized = True
    for name in ("recall_sync", "recall_mode", "injection_frequency", "context_cadence",
                 "dialectic_cadence", "dialectic_depth", "dialectic_depth_levels"):
        setattr(provider, f"_{name}", getattr(cfg, name))
    return provider


def test_two_queries_never_consume_previous_query_caches():
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

    def blocked(session, query):
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
    provider._manager.get_prefetch_context = lambda session, query: {"card": query}
    assert "Debug the compiler" in provider.prefetch("Debug the compiler")


def test_cadences_are_independent_and_gaps_do_not_reuse_context():
    provider = make_provider(context_cadence=3, dialectic_cadence=2)
    results = []
    for turn in range(1, 5):
        query = f"Discuss project number {turn}"
        provider.on_turn_start(turn, query)
        results.append(provider.prefetch(query))
    assert results[1] == ""
    assert "base:" not in results[2] and "dialectic:" in results[2]
    assert "base:" in results[3] and "dialectic:" not in results[3]
    assert provider._last_context_turn == 4 and provider._last_dialectic_turn == 3


def test_first_turn_base_preserves_query_specific_dialectic_depth():
    provider = make_provider(injection_frequency="first-turn", dialectic_depth=3,
                             dialectic_depth_levels=["minimal", "medium", "high"])
    provider._query_rewrite_enabled = True
    provider._query_rewriter = lambda query: ""
    provider._manager.dialectic_query = lambda session, prompt, **kw: (
        provider._manager.prompts.append((session, prompt, kw)) or "short evidence")
    provider.on_turn_start(2, "Debug the compiler")
    result = provider.prefetch("Debug the compiler")
    assert result == "short evidence"
    assert provider._manager.queries == []
    assert [entry[2]["reasoning_level"] for entry in provider._manager.prompts] == ["minimal", "medium", "high"]
    assert all("Debug the compiler" in entry[1] for entry in provider._manager.prompts)


@pytest.mark.parametrize("changed", ["manager", "session", "turn"])
def test_owner_change_discards_result_and_bookkeeping(changed):
    provider = make_provider()
    provider.on_turn_start(1, "Plan the garden")

    def changed_owner(session, query):
        if changed == "manager":
            provider._manager = RecallManager()
        elif changed == "session":
            provider._session_key = "session-b"
        else:
            provider.on_turn_start(2, "Debug the compiler")
        return {"card": "old owner"}

    provider._manager.get_prefetch_context = changed_owner
    assert provider.prefetch("Plan the garden") == ""
    assert provider._last_context_turn == provider._last_dialectic_turn == -999


def test_exception_and_trivial_prompt_do_not_reuse_cache():
    provider = make_provider()
    provider._base_context_cache = "STALE"
    provider._prefetch_result = "STALE"
    def fail(*args):
        raise RuntimeError("offline")
    provider._manager.get_prefetch_context = fail
    assert provider.prefetch("Debug the compiler") == ""
    assert provider.prefetch("thanks") == ""
    assert provider._last_context_turn == provider._last_dialectic_turn == -999


def test_default_queue_and_tools_mode_unchanged():
    provider = make_provider()
    provider._recall_sync = False
    provider._config.recall_sync = False
    provider._last_dialectic_turn = 1
    provider.on_turn_start(1, "Plan the garden")
    provider._base_context_cache = "legacy cached context"
    provider._prefetch_result = "legacy pending dialectic"
    assert provider.prefetch("Plan the garden") == "legacy cached context\n\nlegacy pending dialectic"
    provider.queue_prefetch("Plan the garden")
    assert provider._manager.queued == [("session-a", "Plan the garden")]
    provider._recall_sync = True
    provider._recall_mode = "tools"
    assert provider.prefetch("Debug the compiler") == ""
    assert provider._manager.queries == []


@pytest.mark.parametrize("timeout", [None, 0, -1, float("inf"), float("nan")])
def test_invalid_timeout_has_finite_default(timeout, monkeypatch):
    provider = make_provider()
    provider._config.timeout = timeout
    waits = []
    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: waits.append(timeout))
    provider.prefetch("Debug the compiler")
    assert waits and all(0 <= wait <= 5 for wait in waits)


def test_setup_preserves_explicit_host_opt_out(monkeypatch):
    from plugins.memory.honcho import cli
    monkeypatch.setattr(cli, "_prompt", lambda label, default=None, **kw: default or "")
    host = {"recallSync": False}
    cli._setup_tuning({"recallSync": True}, host)
    assert host["recallSync"] is False


def test_runtime_auth_notice_survives_failed_sync_recall():
    provider = make_provider()
    notices = ["authorization revoked"]
    provider._manager.pop_auth_notice = lambda: notices.pop() if notices else None
    def fail(*args, **kwargs):
        raise RuntimeError("authorization revoked")
    provider._manager.dialectic_query = fail
    result = provider.prefetch("Debug the compiler")
    assert "[Honcho memory status]" in result
    assert "authorization revoked" in result
    assert provider.prefetch("Plan the garden") == ""


def test_setup_and_initialize_propagate_flag_without_prewarm(tmp_path, monkeypatch):
    from plugins.memory.honcho import cli, client, session
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_prompt", lambda label, default=None, **kw:
                        "y" if label.startswith("Wait for current-query") else default or "")
    host = {}
    cli._setup_tuning({}, host)
    assert host["recallSync"] is True
    provider = HonchoMemoryProvider()
    provider.save_config({"baseUrl": "http://example.invalid", "hosts": {"hermes": host}}, str(tmp_path))
    cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=tmp_path / "honcho.json")
    monkeypatch.setattr(client.HonchoClientConfig, "from_global_config", lambda: cfg)
    monkeypatch.setattr(client, "get_honcho_client", lambda cfg: object())
    messages, saved = [], []
    record = SimpleNamespace(messages=["existing"], add_message=lambda role, content: messages.append((role, content)))
    manager = RecallManager()
    manager.get_or_create = lambda key: record
    manager.save = lambda value: saved.append(value)
    monkeypatch.setattr(session, "HonchoSessionManager", lambda **kw: manager)
    provider.initialize("session-a")
    provider._init_thread.join(2)
    assert provider._session_initialized and provider._recall_sync
    assert provider._prefetch_thread is None and manager.prompts == []
    provider.sync_turn("Plan the garden", "Start with soil")
    provider._sync_thread.join(2)
    assert messages == [("user", "Plan the garden"), ("assistant", "Start with soil")]
    assert saved == [record]


def test_session_context_mixin_forwards_each_query_through_controlled_http():
    """Real read mixin with a controlled HTTP seam, not an SDK/hosted E2E."""
    import httpx
    from plugins.memory.honcho.session_context import SessionContextMixin
    requests = []
    def respond(request):
        requests.append(request)
        query = request.url.params.get("search_query", "identity")
        return httpx.Response(200, json={"representation": query, "peer_card": ["fact"]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        class Peer:
            def context(self, **kwargs):
                return SimpleNamespace(**http.get("https://honcho.invalid/context", params=kwargs).json())

        manager = SessionContextMixin()
        manager._cache = {"session-a": SimpleNamespace(honcho_session_id="id", user_peer_id="user", assistant_peer_id="ai")}
        manager._sessions_cache = {}
        manager._authed_call = lambda label, fn: fn()
        manager._get_or_create_peer = lambda peer: Peer()
        manager._resolve_observer_target = lambda session, peer: ("ai", "user")
        provider = make_provider(dialectic_cadence=10)
        provider._manager = manager
        provider._last_dialectic_turn = 0
        for turn, query in enumerate(("Plan the garden", "Debug the compiler"), 1):
            provider.on_turn_start(turn, query)
            assert query in provider.prefetch(query)
        assert [r.url.params["search_query"] for r in requests if "search_query" in r.url.params] == [
            "Plan the garden", "Debug the compiler"]


def test_initialization_wait_and_lock_contention_stay_bounded():
    provider = make_provider()
    provider._config.timeout = 0.02
    provider._session_initialized = False
    provider._manager = None
    provider._lazy_init_kwargs = {}
    release = threading.Event()
    provider._init_thread = threading.Thread(target=lambda: release.wait(3), daemon=True)
    provider._init_thread.start()
    provider._init_lock.acquire()
    try:
        started = time.monotonic()
        assert provider.prefetch("Debug the compiler") == ""
        assert time.monotonic() - started < 2
        assert provider._recall_sync_thread is None
    finally:
        provider._init_lock.release()
        release.set()
        provider._init_thread.join(2)


@pytest.mark.parametrize("phase", ["rewrite", "dialectic"])
def test_slow_dialectic_pipeline_has_one_deadline_and_no_late_bookkeeping(phase):
    provider = make_provider()
    provider._config.timeout = 0.02
    provider._query_rewrite_enabled = True
    release = threading.Event()
    def blocked(*args, **kwargs):
        assert release.wait(3)
        return "late result"
    if phase == "rewrite":
        provider._query_rewriter = blocked
    else:
        provider._manager.dialectic_query = blocked
    try:
        started = time.monotonic()
        assert provider.prefetch("Debug the compiler") == ""
        assert time.monotonic() - started < 2
    finally:
        release.set()
        provider._recall_sync_thread.join(2)
    assert provider._last_context_turn == provider._last_dialectic_turn == -999
    assert provider._base_context_cache is None and provider._prefetch_result == ""
    assert provider._manager.pending_context == {}
    if phase == "rewrite":
        assert provider._manager.prompts == []


def test_dialectic_exception_discards_partial_base_without_advancing():
    provider = make_provider()
    def fail(session, prompt, **kwargs):
        assert kwargs["raise_errors"] is True
        raise RuntimeError("dialectic unavailable")
    provider._manager.dialectic_query = fail
    assert provider.prefetch("Debug the compiler") == ""
    assert provider._last_context_turn == provider._last_dialectic_turn == -999
