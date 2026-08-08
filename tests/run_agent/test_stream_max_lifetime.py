"""Total-lifetime cap for streaming responses (port of QwenLM/qwen-code#8602).

The stale-stream detector only bounds the gap BETWEEN chunks and resets on
every chunk, so a drip-fed stream — a gateway trickling keep-alive-shaped
chunks, or a model crawling through one runaway generation — defeats it
indefinitely. These tests cover the lifetime cap added to
``interruptible_streaming_api_call``:

- a stream that keeps yielding events but never completes is killed once the
  cap elapses, even though the stale detector never fires;
- the cap counts the kill in the same cross-turn stale-streak breaker;
- ``0`` disables the cap (a healthy stream completes normally);
- config/env resolution: config.yaml ``agent.stream_max_lifetime`` is the
  default, ``HERMES_STREAM_MAX_LIFETIME`` overrides it.

The harness mirrors tests/run_agent/test_stream_stale_circuit_breaker.py.
"""

import threading
import time

import pytest
from unittest.mock import MagicMock

from types import SimpleNamespace


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    return agent


def _good_stream_cm():
    """Context manager whose stream yields no events and returns a valid message."""
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter([]))
    msg = MagicMock()
    msg.content = []
    msg.stop_reason = "end_turn"
    msg.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    stream.get_final_message = MagicMock(return_value=msg)
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestStreamMaxLifetime:
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_drip_fed_stream_killed_at_lifetime_cap(self, monkeypatch):
        """A stream that keeps yielding events (so the stale detector never
        fires) must still be killed once the total-lifetime cap elapses."""
        # Stale detector armed but never tripped: events arrive every 50ms.
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "5")
        monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "5")
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 0
        killed = threading.Event()

        def _drip_gen():
            # Keep-alive-shaped events forever, until the watchdog aborts us.
            while not killed.wait(timeout=0.05):
                ev = MagicMock()
                ev.type = "ping"
                yield ev
            raise ConnectionError("socket aborted by lifetime watchdog")

        def _stream_side_effect(*args, **kwargs):
            cm = MagicMock()
            stream = MagicMock()
            stream.__iter__ = MagicMock(return_value=_drip_gen())
            cm.__enter__ = MagicMock(return_value=stream)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        # The lifetime watchdog aborts the request-local client's sockets from
        # the poll thread; simulate the socket shutdown waking the read.
        agent._abort_request_anthropic_client = lambda *a, **k: killed.set()

        start = time.time()
        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})
        elapsed = time.time() - start

        # Killed by the cap — far sooner than any stale-detector trip could
        # have fired against a stream that never went quiet, and the kill is
        # counted in the cross-turn breaker.
        assert elapsed < 60
        assert agent._consecutive_stale_streams >= 1

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_zero_disables_cap(self, monkeypatch):
        """stream_max_lifetime=0 disables the cap; a healthy stream completes."""
        monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "0")

        agent = _make_anthropic_agent()
        agent._anthropic_client.messages.stream.return_value = _good_stream_cm()

        resp = agent._interruptible_streaming_api_call({})
        assert resp is not None
        assert agent._consecutive_stale_streams == 0


class TestResolveStreamMaxLifetime:
    def test_env_overrides_default(self, monkeypatch):
        from agent.chat_completion_helpers import _resolve_stream_max_lifetime

        monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "42")
        assert _resolve_stream_max_lifetime(MagicMock()) == 42.0

    def test_config_value_is_default(self, monkeypatch):
        import agent.chat_completion_helpers as cch

        monkeypatch.delenv("HERMES_STREAM_MAX_LIFETIME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"agent": {"stream_max_lifetime": 777}},
        )
        assert cch._resolve_stream_max_lifetime(MagicMock()) == 777.0

    def test_default_when_unset(self, monkeypatch):
        import agent.chat_completion_helpers as cch

        monkeypatch.delenv("HERMES_STREAM_MAX_LIFETIME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: {"agent": {}}
        )
        assert cch._resolve_stream_max_lifetime(MagicMock()) == 1800.0

    def test_negative_clamped_to_zero(self, monkeypatch):
        from agent.chat_completion_helpers import _resolve_stream_max_lifetime

        monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "-5")
        assert _resolve_stream_max_lifetime(MagicMock()) == 0.0

    def test_bool_config_value_ignored(self, monkeypatch):
        import agent.chat_completion_helpers as cch

        monkeypatch.delenv("HERMES_STREAM_MAX_LIFETIME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"agent": {"stream_max_lifetime": True}},
        )
        assert cch._resolve_stream_max_lifetime(MagicMock()) == 1800.0
