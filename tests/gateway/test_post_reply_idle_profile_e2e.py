"""The idle watcher binds the actual profile home before opening its SessionDB."""

import asyncio
import threading
from pathlib import Path

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED
from gateway.session import SessionStore
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_due_worker_compacts_each_configured_profile(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    import hermes_state
    from gateway import run
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_identity import RoutingIdentity

    root = tmp_path / "hermes"
    other = root / "profiles" / "other"
    root.mkdir(parents=True)
    other.mkdir(parents=True)
    config = 'compression:\n  post_reply_idle:\n    channels:\n      - platform: signal\n        chat_id: chat\n        after_seconds: 300\n'
    (root / "config.yaml").write_text(config)
    (other / "config.yaml").write_text(config)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setattr(run, "_multiplex_profile_homes", lambda cfg: [("default", root), ("other", other)])

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner.session_store = SessionStore(root / "sessions", runner.config)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._queued_events = {}
    runner._running_agents = {}
    runner._is_session_running = lambda key: False
    async def no_compression(key):
        return False
    runner._session_has_compression_in_flight = no_compression
    runner._resolve_session_agent_runtime = lambda **kwargs: ("test-model", {})
    runner._evict_cached_agent = lambda key: None
    runner._track_deferred_agent_worker = lambda *args: None
    async def cleanup(*args, **kwargs):
        pass
    runner._cleanup_agent_resources_off_loop = cleanup
    async def build_agent(model, runtime, entry):
        from gateway.run import _seed_hygiene_system_prompt
        from run_agent import AIAgent

        db = runner._session_db._db
        agent = AIAgent(
            model=model, provider="openai-api", base_url="http://127.0.0.1:9/v1", api_key="dummy",
            session_id=entry.session_id, session_db=db, quiet_mode=True,
            skip_memory=True, enabled_toolsets=["memory"], skip_context_files=True,
        )
        _seed_hygiene_system_prompt(agent, db.get_session(entry.session_id))
        agent.context_compressor._summarize_window = lambda *args, **kwargs: "Retained the earlier task."
        agent.context_compressor.tail_token_budget = 100
        return agent, db
    runner._hmwa_hygiene_build_agent = build_agent

    sessions = []
    try:
        for name, home in [("default", root), ("other", other)]:
            async with run._async_profile_runtime_scope(home):
                source = SessionSource(platform=Platform.SIGNAL, chat_id="chat", chat_type="group",
                                       profile=None if name == "default" else name)
                source._identity = RoutingIdentity(name, name, home, home)
                entry = runner.session_store.get_or_create_session(source)
                db = runner._session_db._db
                db.update_system_prompt(entry.session_id, "pinned prompt")
                for i in range(10):
                    db.append_message(entry.session_id, role="user" if i % 2 == 0 else "assistant",
                                      content="message " * 150)
                generation = db.invalidate_post_reply_idle(entry.session_id)
                assert db.arm_post_reply_idle(entry.session_id, entry.session_key, time.time() - 1,
                                              expected_generation=generation)
                sessions.append((home, entry.session_id))
        # A fresh watcher finds both jobs from state.db, without an in-memory timer.
        await runner._process_due_post_reply_idle()
        for home, session_id in sessions:
            async with run._async_profile_runtime_scope(home):
                active = runner._session_db._db.get_messages(session_id)
                assert len(active) < 10
                assert any("Retained the earlier task." in str(m["content"]) for m in active)
                assert runner._session_db._db.get_session(session_id)["system_prompt"] == "pinned prompt"
    finally:
        runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_multiplex_idle_scan_uses_each_real_profile_store(tmp_path, monkeypatch):
    import hermes_state
    from gateway import run

    root = tmp_path / "hermes"
    other = root / "profiles" / "other"
    root.mkdir(parents=True)
    other.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n")
    (other / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setattr(run, "_multiplex_profile_homes", lambda config: [("default", root), ("other", other)])

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner.session_store = SessionStore(root / "sessions", runner.config)
    seen = []

    async def observe():
        handle = runner._session_db
        seen.append((Path(get_hermes_home()), Path(handle._db.db_path)))

    runner._post_reply_idle_claim_one = observe
    try:
        await runner._process_due_post_reply_idle()
        async with run._async_profile_runtime_scope(root):
            await observe()
        assert seen == [(root, root / "state.db"), (other, other / "state.db"),
                        (root, root / "state.db")]
        assert Path(get_hermes_home()) == root
    finally:
        runner.close_all_session_db_handles()
