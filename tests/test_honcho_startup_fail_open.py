"""Regression tests for Honcho startup fail-open behavior."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider


class _FakeHonchoConfig(SimpleNamespace):
    def resolve_session_name(self, **kwargs):
        return "test-session"


def _configured_hybrid_config() -> _FakeHonchoConfig:
    return _FakeHonchoConfig(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        init_on_session_start=False,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-directory",
        prefetch_generic_context=True,
    )


def _configured_tools_config(*, init_on_session_start: bool = False) -> _FakeHonchoConfig:
    cfg = _configured_hybrid_config()
    cfg.recall_mode = "tools"
    cfg.init_on_session_start = init_on_session_start
    return cfg


def test_honcho_hybrid_initialize_returns_without_waiting_for_session_init(monkeypatch):
    """Slow Honcho session creation must not block agent startup."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def slow_session_init(self, cfg, session_id, **kwargs):
        started.set()
        release.wait(timeout=5)
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    start = time.perf_counter()
    provider.initialize("session-1", platform="cli")
    elapsed = time.perf_counter() - start

    try:
        assert elapsed < 0.5
        assert started.wait(timeout=1)
        assert provider._session_key == "test-session"
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)


def test_honcho_background_init_rechecks_state_after_lock_race():
    """Startup should not spawn/crash if init completes while waiting for lock."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()
    provider._lazy_init_kwargs = {"platform": "cli"}
    provider._lazy_init_session_id = "session-1"

    class RacingLock:
        def __enter__(self):
            provider._session_initialized = True
            provider._lazy_init_kwargs = None
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    provider._init_lock = RacingLock()

    provider._start_session_init_background()

    assert provider._init_thread is None
    assert provider._session_initialized is True


def test_honcho_prefetch_returns_without_waiting_for_first_context_fetch():
    """First-turn context injection must fail open when Honcho is slow."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.timeout = 0.1
    fetch_started = threading.Event()

    class SlowManager:
        def get_prefetch_context(self, session_key, user_message=None):
            fetch_started.set()
            time.sleep(5)
            return {"representation": "late"}

        def prefetch_context(self, session_key, user_message=None):
            fetch_started.set()

        def pop_context_result(self, session_key):
            return {}

    provider._config = cfg
    provider._manager = SlowManager()
    provider._session_key = "test-session"
    provider._session_initialized = True
    provider._turn_count = 1

    start = time.perf_counter()
    result = provider.prefetch("what do you know about me?")
    elapsed = time.perf_counter() - start

    assert result == ""
    assert elapsed < 0.5
    assert fetch_started.is_set()



def test_honcho_sync_turn_does_not_start_network_write_before_session_init():
    """Session-end sync must not create a blocking writer before init finishes."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    get_started = threading.Event()
    background_started = threading.Event()
    release_init = threading.Event()

    class SlowManager:
        def get_or_create(self, session_key):
            get_started.set()
            time.sleep(5)
            return SimpleNamespace()

        def _flush_session(self, session):
            pass

    provider._config = cfg
    provider._manager = SlowManager()
    provider._session_key = "test-session"
    provider._session_initialized = False
    provider._start_session_init_background = background_started.set
    provider._init_thread = threading.Thread(
        target=lambda: release_init.wait(timeout=5), daemon=True
    )
    provider._init_thread.start()

    try:
        provider.sync_turn("hello", "world")

        assert provider._sync_thread is None
        assert background_started.is_set()
        assert not get_started.wait(timeout=0.1)
    finally:
        release_init.set()
        provider._init_thread.join(timeout=1)


def test_honcho_sync_turn_waits_for_full_background_startup(monkeypatch):
    """Manager assignment alone is not readiness while background init continues."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    session_created = threading.Event()
    migration_started = threading.Event()
    release_migration = threading.Event()
    get_calls = []

    class StartupManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_create(self, session_key):
            get_calls.append(session_key)
            session_created.set()
            return SimpleNamespace(messages=[])

        def migrate_memory_files(self, session_key, mem_dir):
            migration_started.set()
            release_migration.wait(timeout=5)

        def prefetch_context(self, session_key, user_message=None):
            pass

        def _flush_session(self, session):
            pass

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )
    monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda cfg: object())
    monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", StartupManager)

    provider.initialize("session-1", platform="cli")
    try:
        assert session_created.wait(timeout=1)
        assert migration_started.wait(timeout=1)
        assert provider._manager is not None
        assert provider._session_initialized is False

        provider.sync_turn("hello", "world")

        assert provider._sync_thread is None
        assert get_calls == ["test-session"]
    finally:
        release_migration.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1)

    assert provider._session_initialized is True


def test_honcho_system_prompt_advertises_active_while_background_init_runs(monkeypatch):
    """Prompt metadata should not require a completed network session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = threading.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def slow_session_init(self, cfg, session_id, **kwargs):
        release.wait(timeout=5)
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    provider.initialize("session-1", platform="cli")
    try:
        prompt = provider.system_prompt_block()
        assert "Honcho Memory" in prompt
        assert "hybrid mode" in prompt
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)


def test_honcho_tools_eager_init_still_ready_on_return(monkeypatch):
    """tools + initOnSessionStart=true keeps its ready-on-return contract."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def fake_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    provider.initialize("session-1", platform="cli")

    assert provider._session_initialized is True
    assert provider._manager is not None
    assert provider._init_thread is None


def test_honcho_tools_eager_init_failure_does_not_leave_ready_manager(monkeypatch):
    """Failed eager tools startup must not leave hooks seeing a ready session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def failing_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        raise RuntimeError("boom")

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", failing_session_init)

    provider.initialize("session-1", platform="cli")
    assert provider._session_initialized is False
    assert provider._manager is None

    background_started = threading.Event()
    provider._start_session_init_background = background_started.set
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers safe Honcho startup")

    assert provider._sync_thread is None
    assert not background_started.is_set()

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))
    assert "could not be initialized" in result["error"]
    assert provider._manager is None


def test_honcho_tools_lazy_hooks_do_not_prestart_background_init(monkeypatch):
    """tools lazy mode lets the first tool call own session initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=False)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider.initialize("session-1", platform="cli")
    background_started = threading.Event()
    provider._start_session_init_background = background_started.set

    provider.prefetch("what do you know?")
    provider.queue_prefetch("what do you know?")
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert not background_started.is_set()
    assert provider._session_initialized is False

    class ToolManager:
        def get_peer_card(self, session_key, peer="user"):
            return ["ready"]

    init_calls = []

    def fake_session_init(self, cfg, session_id, **kwargs):
        init_calls.append(session_id)
        self._manager = ToolManager()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))

    assert result == {"result": ["ready"]}
    assert init_calls == ["session-1"]
    assert not background_started.is_set()
