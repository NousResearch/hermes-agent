"""cron.background_review: opt-in end-of-turn skill/memory review for cron agents.

Cron defaults to ``skip_background_review=True``. The review fork is a daemon thread
started when the turn ends; cron finalizes the session and closes the agent (whose
``_active_children`` include the fork) right after, and the restart-safe external worker then
exits. So enabling the flag alone is not enough — run_job must wait (bounded) for the review
before teardown.
"""

from __future__ import annotations

import threading
import time
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import run_agent  # import at collection time (bootstrap reads the real home; the I/O guard is not armed yet)
from cron.scheduler import (
    _await_cron_background_review,
    _cron_background_review_enabled,
    _cron_background_review_wait_seconds,
    run_job,
)

_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({}, False),
        ({"cron": None}, False),
        ({"cron": {}}, False),
        ({"cron": {"background_review": False}}, False),
        ({"cron": {"background_review": "false"}}, False),
        ({"cron": {"background_review": True}}, True),
        ({"cron": {"background_review": "true"}}, True),
        ("not-a-dict", False),
    ],
)
def test_enabled_flag_defaults_off(cfg, expected):
    assert _cron_background_review_enabled(cfg) is expected


def test_wait_seconds_default_and_override():
    assert _cron_background_review_wait_seconds({}) == 300.0
    assert _cron_background_review_wait_seconds({"cron": {"background_review_wait_seconds": 12}}) == 12.0
    assert _cron_background_review_wait_seconds({"cron": {"background_review_wait_seconds": "bad"}}) == 300.0


class _Agent:
    def __init__(self, skip=False):
        self.skip_background_review = skip
        self._background_review_thread: Optional[threading.Thread] = None
        self._background_review_run = None


def test_await_joins_running_review_thread():
    agent = _Agent()
    done = threading.Event()

    def _review():
        time.sleep(0.2)
        done.set()

    agent._background_review_thread = threading.Thread(target=_review, daemon=True)
    agent._background_review_thread.start()
    _await_cron_background_review(agent, "job", 5.0)
    assert done.is_set()


def test_await_is_noop_when_review_disabled():
    agent = _Agent(skip=True)
    agent._background_review_thread = MagicMock()
    _await_cron_background_review(agent, "job", 5.0)
    agent._background_review_thread.join.assert_not_called()


def test_await_cancels_review_after_timeout():
    agent = _Agent()
    release = threading.Event()
    agent._background_review_thread = threading.Thread(target=release.wait, daemon=True)
    agent._background_review_thread.start()
    try:
        with patch("agent.background_review.cancel_background_review_for_live_turn") as cancel:
            started = time.monotonic()
            _await_cron_background_review(agent, "job", 0.05)
            assert time.monotonic() - started < 2.0
        cancel.assert_called_once_with(agent)
    finally:
        release.set()


def _run_job_with_review(tmp_path, config_text):
    """Drive run_job with a fake AIAgent whose turn spawns a slow review thread."""
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    events = []
    constructed = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.skip_background_review = kwargs.get("skip_background_review", False)
            self._background_review_thread = None
            self._background_review_run = None

        def run_conversation(self, *_a, **_kw):
            if not self.skip_background_review:
                def _review():
                    time.sleep(0.3)
                    events.append("review_done")

                self._background_review_thread = threading.Thread(target=_review, daemon=True)
                self._background_review_thread.start()
            return {"final_response": "ok"}

        def close(self):
            events.append("agent_closed")

    def _finalize(*_a, **_kw):
        events.append("session_finalized")

    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
         patch("cron.scheduler._finalize_cron_session", side_effect=_finalize), \
         patch("run_agent.AIAgent", FakeAgent):
        success, _output, final_response, error = run_job({"id": "bg-review-job", "name": "t", "prompt": "hi"})
    return success, final_response, error, events, constructed


def test_run_job_default_keeps_review_skipped(tmp_path):
    success, final_response, error, events, constructed = _run_job_with_review(tmp_path, "model: test-model\n")
    assert success is True and final_response == "ok" and error is None
    assert constructed["skip_background_review"] is True
    assert "review_done" not in events


def test_run_job_waits_for_review_before_finalize_and_teardown(tmp_path):
    success, final_response, error, events, constructed = _run_job_with_review(
        tmp_path, "model: test-model\ncron:\n  background_review: true\n")
    assert success is True and final_response == "ok" and error is None
    assert constructed["skip_background_review"] is False
    assert events.index("review_done") < events.index("session_finalized")
    assert events.index("review_done") < events.index("agent_closed")


def test_run_job_wait_zero_restores_race(tmp_path):
    """Sensitivity check: with the wait disabled the review is still running at teardown."""
    _success, _final, _error, events, _constructed = _run_job_with_review(
        tmp_path, "model: test-model\ncron:\n  background_review: true\n  background_review_wait_seconds: 0\n")
    assert "session_finalized" in events and "agent_closed" in events
    assert "review_done" not in events[: events.index("agent_closed") + 1]


def test_spawn_stores_started_review_thread_handle():
    """Behavioral contract the cron wait relies on: spawning stores a live thread on the agent."""
    agent = types.SimpleNamespace(_maybe_requeue_preempted_review=lambda *a, **k: None)
    finished = threading.Event()

    def _target():
        finished.set()

    with patch("agent.background_review.spawn_background_review_thread",
               return_value=(_target, "prompt")):
        run_agent.AIAgent._spawn_background_review_now(agent, messages_snapshot=[])  # type: ignore[arg-type]
    thread = agent._background_review_thread
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=5.0)
    assert finished.is_set()
    assert not thread.is_alive()
