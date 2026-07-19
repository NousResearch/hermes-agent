"""Direct-Anthropic watchdog must not release the TLS socket FD from a stranger thread (#67142).

Issue #29507 gave the OpenAI request path an owner-thread contract: the outer
stale/interrupt watchdog (a *stranger* thread relative to the worker driving the
request) only ``shutdown(SHUT_RDWR)``s the sockets, and the owning worker thread
performs ``client.close()`` on its way out. Calling ``close()`` from the
watchdog thread raced the worker's still-live SSL BIO; when the kernel recycled
the just-freed TLS socket FD into an unrelated ``open()`` (e.g.
``cron/executions.db``), a pending 24-byte TLS record was flushed into that
file's header and corrupted a SQLite database.

The direct-Anthropic path (``api_mode == "anthropic_messages"``) still closed
``_anthropic_client`` from the outer stale/interrupt watchdog on ``origin/main``,
so the same corruption reproduced on an Anthropic cron request.

These tests prove the two properties the fix must hold:

1. The outer stale/interrupt watchdog never calls ``_anthropic_client.close()``
   from a stranger thread — it aborts sockets via ``_abort_anthropic_client``.
2. The worker still unblocks promptly and performs the close+rebuild from its
   OWN thread (preserving the #28161 no-hang cleanup).

Plus a socket-level test that ``_abort_anthropic_client`` shuts sockets down
without releasing the FD.

Fixes #67142
"""

import socket
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-8",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
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


class _FakeSock:
    """Minimal socket stand-in that records shutdown/close separately."""

    def __init__(self):
        self.shutdown_calls = []
        self.closed = False

    def shutdown(self, how):
        self.shutdown_calls.append(how)

    def close(self):
        self.closed = True


def _install_fake_pool(anthropic_client, sock):
    """Wire ``anthropic_client`` so ``_iter_pool_sockets`` finds ``sock``."""
    stream = SimpleNamespace(_sock=sock)
    conn = SimpleNamespace(_connection=SimpleNamespace(_network_stream=stream))
    pool = SimpleNamespace(_connections=[conn])
    transport = SimpleNamespace(_pool=pool)
    anthropic_client._client = SimpleNamespace(_transport=transport)


# ---------------------------------------------------------------------------
# Socket-level abort contract
# ---------------------------------------------------------------------------


class TestAbortAnthropicClient:
    def test_abort_shuts_down_sockets_without_releasing_fd(self):
        """``_abort_anthropic_client`` shuts sockets down but never closes them
        or the client — the FD release belongs to the owning worker thread."""
        agent = _make_anthropic_agent()
        sock = _FakeSock()
        _install_fake_pool(agent._anthropic_client, sock)

        agent._abort_anthropic_client(reason="unit")

        # FIN sent so the worker's blocked recv/send unwinds …
        assert sock.shutdown_calls == [socket.SHUT_RDWR]
        # … but the FD is NOT released from this (stranger) thread.
        assert sock.closed is False
        agent._anthropic_client.close.assert_not_called()

    def test_abort_is_safe_when_no_client(self):
        agent = _make_anthropic_agent()
        agent._anthropic_client = None
        # Must not raise.
        agent._abort_anthropic_client(reason="unit")


# ---------------------------------------------------------------------------
# Streaming path ownership contract
# ---------------------------------------------------------------------------


class TestStreamingWatchdogOwnership:
    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_stale_stream_aborts_from_stranger_thread_and_closes_from_worker(
        self, monkeypatch
    ):
        """Stale-stream detector (stranger thread) aborts; the worker thread
        performs the Anthropic close()+rebuild."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")

        agent = _make_anthropic_agent()
        main_tid = threading.get_ident()
        unblock = threading.Event()
        attempt = [0]

        abort_threads = []
        close_threads = []
        rebuild_threads = []

        def _stream_side_effect(*args, **kwargs):
            attempt[0] += 1
            if attempt[0] == 1:
                cm = MagicMock()
                stream = MagicMock()

                def _blocking_gen():
                    unblock.wait(timeout=5.0)
                    raise httpx.ConnectError("dropped after abort")
                    yield  # generator so iteration triggers the wait

                stream.__iter__ = MagicMock(return_value=_blocking_gen())
                cm.__enter__ = MagicMock(return_value=stream)
                cm.__exit__ = MagicMock(return_value=False)
                return cm
            return _good_stream_cm()

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        agent._anthropic_client.close.side_effect = (
            lambda *a, **k: close_threads.append(threading.get_ident())
        )
        agent._abort_anthropic_client = MagicMock(
            side_effect=lambda *a, **k: (
                abort_threads.append(threading.get_ident()),
                unblock.set(),
            )
        )

        def _rebuild(*a, **k):
            rebuild_threads.append(threading.get_ident())

        with patch.object(agent, "_rebuild_anthropic_client", side_effect=_rebuild):
            with patch.object(agent, "_replace_primary_openai_client") as mock_replace:
                agent._interruptible_streaming_api_call({})

        # Never the OpenAI primary client on an Anthropic-native config.
        mock_replace.assert_not_called()

        # Property 1: the stale watchdog aborted from the stranger (main) thread
        # and never released the FD there.
        assert main_tid in abort_threads
        assert main_tid not in close_threads

        # Property 2: the worker unblocked and performed close()+rebuild from
        # its OWN thread.
        assert close_threads, "worker never closed the Anthropic client"
        assert all(tid != main_tid for tid in close_threads)
        assert rebuild_threads and all(tid != main_tid for tid in rebuild_threads)

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_interrupt_aborts_from_stranger_thread(self, monkeypatch):
        """An interrupt during an Anthropic stream aborts from the poll thread
        rather than closing the client there."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "600")

        agent = _make_anthropic_agent()
        main_tid = threading.get_ident()
        started = threading.Event()
        unblock = threading.Event()
        abort_threads = []
        close_threads = []

        def _stream_side_effect(*args, **kwargs):
            cm = MagicMock()
            stream = MagicMock()

            def _blocking_gen():
                started.set()
                unblock.wait(timeout=5.0)
                raise httpx.ConnectError("dropped after abort")
                yield

            stream.__iter__ = MagicMock(return_value=_blocking_gen())
            cm.__enter__ = MagicMock(return_value=stream)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        agent._anthropic_client.close.side_effect = (
            lambda *a, **k: close_threads.append(threading.get_ident())
        )
        agent._abort_anthropic_client = MagicMock(
            side_effect=lambda *a, **k: (
                abort_threads.append(threading.get_ident()),
                unblock.set(),
            )
        )

        def _request_interrupt():
            started.wait(timeout=5.0)
            agent._interrupt_requested = True

        trigger = threading.Thread(target=_request_interrupt, daemon=True)
        trigger.start()

        with patch.object(agent, "_rebuild_anthropic_client"):
            with pytest.raises(InterruptedError):
                agent._interruptible_streaming_api_call({})

        trigger.join(timeout=5.0)

        # The interrupt watchdog aborted from the stranger (poll) thread and
        # never released the FD there.
        assert main_tid in abort_threads
        assert main_tid not in close_threads


# ---------------------------------------------------------------------------
# Non-streaming path ownership contract
# ---------------------------------------------------------------------------


class TestNonStreamingWatchdogOwnership:
    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_stale_call_aborts_from_stranger_thread_and_closes_from_worker(
        self, monkeypatch
    ):
        """Non-streaming stale detector aborts from the stranger thread; the
        worker performs the Anthropic close()+rebuild from its own thread."""
        agent = _make_anthropic_agent()
        main_tid = threading.get_ident()
        unblock = threading.Event()
        abort_threads = []
        close_threads = []
        rebuild_threads = []

        def _blocking_create(api_kwargs):
            unblock.wait(timeout=5.0)
            raise httpx.ConnectError("dropped after abort")

        agent._anthropic_messages_create = MagicMock(side_effect=_blocking_create)
        agent._compute_non_stream_stale_timeout = MagicMock(return_value=0.1)
        agent._anthropic_client.close.side_effect = (
            lambda *a, **k: close_threads.append(threading.get_ident())
        )
        agent._abort_anthropic_client = MagicMock(
            side_effect=lambda *a, **k: (
                abort_threads.append(threading.get_ident()),
                unblock.set(),
            )
        )

        def _rebuild(*a, **k):
            rebuild_threads.append(threading.get_ident())

        with patch.object(agent, "_rebuild_anthropic_client", side_effect=_rebuild):
            with pytest.raises(Exception):
                agent._interruptible_api_call({})

        # Property 1: abort from the stranger (main) thread, never close there.
        assert main_tid in abort_threads
        assert main_tid not in close_threads

        # Property 2: worker closed + rebuilt from its own thread.
        assert close_threads and all(tid != main_tid for tid in close_threads)
        assert rebuild_threads and all(tid != main_tid for tid in rebuild_threads)
