"""B8 regressions: honcho unit tests must never touch the network.

Real incident: an async manager built with a late MagicMock swap wrote test
messages (session cli-test, hello/hi) into the production workspace of a live
local Honcho. These tests pin the isolation contract.
"""

import json
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import session as session_module
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _session(**kw) -> HonchoSession:
    return HonchoSession(
        key=kw.get("key", "cli:isolation"),
        user_peer_id="eri",
        assistant_peer_id="hermes",
        honcho_session_id=kw.get("sid", "cli-isolation"),
        messages=kw.get("messages", []),
    )


class TestConstructorRace:
    def test_fake_factory_before_constructor_gets_all_calls(self, monkeypatch):
        """Regression 1: fake factory installed BEFORE the constructor receives
        every flush; no real transport is ever touched (network guard active)."""
        fake = MagicMock()
        monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: fake)
        cfg = HonchoClientConfig(write_frequency="async", api_key="test-key", enabled=True)
        mgr = HonchoSessionManager(honcho=fake, config=cfg)
        try:
            sess = _session()
            sess.add_message("user", "hello")
            sess.add_message("assistant", "hi")
            mgr.save(sess)
            deadline = time.time() + 3.0
            while not fake.method_calls and time.time() < deadline:
                time.sleep(0.05)
            assert fake.method_calls, "fake client never received the flush"
        finally:
            mgr.shutdown()

    def test_construction_spawns_no_thread_and_no_network(self):
        """Constructing a manager (async mode) does nothing externally."""
        cfg = HonchoClientConfig(write_frequency="async", api_key="test-key", enabled=True)
        mgr = HonchoSessionManager(config=cfg)
        try:
            assert mgr._async_thread is None
        finally:
            mgr.shutdown()


class TestAmbientProductionConfig:
    def test_ambient_live_config_produces_zero_requests(self, tmp_path, monkeypatch):
        """Regression 2: ambient HERMES_HOME with a live URL must not leak
        requests - hygiene (factory injection) keeps the suite green."""
        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "honcho.json").write_text(json.dumps({
            "baseUrl": "http://localhost:8000",
            "workspace": "iris_curated_v1",
            "hosts": {"hermes": {"apiKey": "live-looking-key", "saveMessages": True}},
        }))
        monkeypatch.setenv("HERMES_HOME", str(home))
        fake = MagicMock()
        monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: fake)
        cfg = HonchoClientConfig(write_frequency="async", api_key="live-looking-key", enabled=True)
        mgr = HonchoSessionManager(honcho=fake, config=cfg)
        try:
            sess = _session(sid="cli-test")
            sess.add_message("user", "hello")
            mgr.save(sess)
            mgr.flush_all()
        finally:
            mgr.shutdown()
        # teardown-assert конфтеста дополнительно проверит network_attempts == []

    @pytest.mark.expect_network_attempts
    def test_guard_blocks_real_connections(self, network_attempts):
        """The guard itself works: a raw connection attempt is recorded+raised."""
        with pytest.raises(RuntimeError, match="network disabled"):
            socket.create_connection(("127.0.0.1", 8000), timeout=1)
        assert network_attempts == [("127.0.0.1", 8000)]


class TestThreadCleanup:
    def test_shutdown_leaves_no_writer_threads(self, monkeypatch):
        """Regression 3: after shutdown - queue drained, thread stopped."""
        fake = MagicMock()
        monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: fake)
        cfg = HonchoClientConfig(write_frequency="async", api_key="test-key", enabled=True)
        mgr = HonchoSessionManager(honcho=fake, config=cfg)
        sess = _session()
        sess.add_message("user", "bye")
        mgr.save(sess)
        mgr.shutdown()
        assert mgr._async_queue.empty()
        assert not any(
            t.name == "honcho-async-writer" and t.is_alive() for t in threading.enumerate()
        )
