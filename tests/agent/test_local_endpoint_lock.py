"""Tests for local_endpoint_lock (#108596): concurrent requests against the same
local/self-hosted backend must queue instead of contending for the same GPU slot.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from agent.model_metadata import local_endpoint_lock


def _stub_agent(base_url):
    notices = []
    return SimpleNamespace(
        base_url=base_url,
        _interrupt_requested=False,
        _emit_wait_notice=lambda text: notices.append(text),
        _touch_activity=lambda desc: None,
    ), notices


class TestLocalEndpointLock:
    def test_cross_thread_contention_serializes(self):
        """A second thread calling the SAME local backend must wait for the
        first to finish — never run concurrently inside the guarded section."""
        agent, _ = _stub_agent("http://localhost:11434")
        active = {"count": 0, "max_seen": 0}
        active_lock = threading.Lock()

        def hold(duration):
            with local_endpoint_lock(agent, agent.base_url):
                with active_lock:
                    active["count"] += 1
                    active["max_seen"] = max(active["max_seen"], active["count"])
                time.sleep(duration)
                with active_lock:
                    active["count"] -= 1

        t1 = threading.Thread(target=hold, args=(0.3,))
        t2 = threading.Thread(target=hold, args=(0.0,))
        t1.start()
        time.sleep(0.05)  # let t1 acquire first
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert active["max_seen"] == 1, "both threads ran inside the lock concurrently"

    def test_losing_thread_gets_wait_notice_and_clears_it(self):
        agent, notices = _stub_agent("http://127.0.0.1:8080")

        def hold():
            with local_endpoint_lock(agent, agent.base_url):
                time.sleep(0.3)

        t1 = threading.Thread(target=hold)
        t1.start()
        time.sleep(0.05)
        with local_endpoint_lock(agent, agent.base_url):
            pass
        t1.join(timeout=5)

        assert any("another request is already using the local backend" in n for n in notices)
        assert notices[-1] == ""  # cleared once acquired

    def test_same_thread_reentry_does_not_deadlock(self):
        """Codex streaming re-enters the non-streaming entry point on the SAME
        thread; a plain Lock would self-deadlock here. RLock must not block and
        must not fire a wait notice for the nested acquire."""
        agent, notices = _stub_agent("http://localhost:11434")

        done = {"ok": False}

        def nested():
            with local_endpoint_lock(agent, agent.base_url):
                with local_endpoint_lock(agent, agent.base_url):
                    done["ok"] = True

        t = threading.Thread(target=nested)
        t.start()
        t.join(timeout=2)

        assert done["ok"] is True
        assert notices == []

    def test_non_local_endpoint_never_serializes(self):
        """Cloud backends must not be gated at all — two concurrent calls run
        with no coordination."""
        agent, _ = _stub_agent("https://api.openai.com/v1")
        active = {"count": 0, "max_seen": 0}
        active_lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=5)

        def hold():
            with local_endpoint_lock(agent, agent.base_url):
                with active_lock:
                    active["count"] += 1
                    active["max_seen"] = max(active["max_seen"], active["count"])
                barrier.wait()
                with active_lock:
                    active["count"] -= 1

        t1 = threading.Thread(target=hold)
        t2 = threading.Thread(target=hold)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert active["max_seen"] == 2, "non-local calls were unexpectedly serialized"

    def test_interrupt_while_waiting_raises(self):
        """A request queued behind the lock must abort promptly when the agent
        is interrupted, instead of waiting out the holder indefinitely."""
        agent, _ = _stub_agent("http://localhost:11434")

        def hold():
            with local_endpoint_lock(agent, agent.base_url):
                time.sleep(2.0)

        t = threading.Thread(target=hold)
        t.start()
        time.sleep(0.05)
        agent._interrupt_requested = True
        with pytest.raises(InterruptedError):
            with local_endpoint_lock(agent, agent.base_url):
                pass
        agent._interrupt_requested = False
        t.join(timeout=5)
