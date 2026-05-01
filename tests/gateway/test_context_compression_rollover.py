"""Tests for gateway rollover after repeated lossy context compression."""

import threading
from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="user-1")


def _runner(tmp_path, max_compressions=3):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        max_context_compressions_before_reset=max_compressions,
    )
    runner.session_store = SessionStore(tmp_path, runner.config)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


class TestContextCompressionRollover:
    def test_rolls_session_when_compression_limit_reached(self, tmp_path):
        runner = _runner(tmp_path, max_compressions=3)
        entry = runner.session_store.get_or_create_session(_source())
        old_session_id = entry.session_id
        runner.session_store.update_session(
            entry.session_key,
            last_prompt_tokens=123,
            compression_count=3,
        )
        old_agent = MagicMock()
        runner._agent_cache[entry.session_key] = (old_agent, "sig")
        runner._cleanup_agent_resources = MagicMock()

        new_entry, did_rollover = runner._rollover_session_if_compression_limit_reached(entry)

        assert did_rollover is True
        assert new_entry.session_key == entry.session_key
        assert new_entry.session_id != old_session_id
        assert new_entry.compression_count == 0
        assert new_entry.last_prompt_tokens == 0
        assert new_entry.is_fresh_reset is False
        runner._cleanup_agent_resources.assert_called_once_with(old_agent)
        assert entry.session_key not in runner._agent_cache

    def test_does_not_roll_before_limit(self, tmp_path):
        runner = _runner(tmp_path, max_compressions=3)
        entry = runner.session_store.get_or_create_session(_source())
        runner.session_store.update_session(entry.session_key, compression_count=2)

        same_entry, did_rollover = runner._rollover_session_if_compression_limit_reached(entry)

        assert did_rollover is False
        assert same_entry.session_id == entry.session_id

    def test_zero_disables_compression_rollover(self, tmp_path):
        runner = _runner(tmp_path, max_compressions=0)
        entry = runner.session_store.get_or_create_session(_source())
        runner.session_store.update_session(entry.session_key, compression_count=99)

        same_entry, did_rollover = runner._rollover_session_if_compression_limit_reached(entry)

        assert did_rollover is False
        assert same_entry.session_id == entry.session_id

    def test_syncs_persisted_count_into_agent_before_turn(self, tmp_path):
        runner = _runner(tmp_path, max_compressions=3)
        entry = runner.session_store.get_or_create_session(_source())
        runner.session_store.update_session(entry.session_key, compression_count=2)
        agent = MagicMock()
        agent.context_compressor.compression_count = 0

        runner._sync_agent_compression_count(entry, agent)

        assert agent.context_compressor.compression_count == 2
