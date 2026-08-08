"""Regression tests for the reasoning-only stale detector (#78807).

A model that emits only reasoning tokens (``reasoning_content`` /
``thinking``) with no visible output resets the ordinary stream stale
detector on every chunk, so a genuinely stuck reasoning stream can run
until the HTTP timeout (up to 1800s).  These tests drive the REAL
streaming path (``chat_completion_helpers.interruptible_streaming_api_call``) with
fake chunk streams and assert that:

* a reasoning-only stream is killed once the
  ``agent.reasoning_only_stale_timeout`` window is exceeded,
* the threshold is read from config.yaml (``agent`` section),
* ``0`` disables the check,
* the window is anchored at the FIRST reasoning chunk (TTFT latency
  does not eat the budget),
* the model's reasoning stale floor extends the unconfigured default,
  while explicit config wins over the floor,
* a reasoning-only kill is NOT retried (byte-identical retries would be
  killed again),
* reasoning-then-content and content-only streams are never killed.

The kill is observed through the same seam a user would see: the poll
loop aborts the request client with reason ``reasoning_only_stale_kill``
and emits a status message.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_agent(tmp_path, monkeypatch, *, reasoning_timeout=0.3, configure_key=True):
    """Real AIAgent wired to a fake client, with a tiny reasoning threshold."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Keep the no-chunk stale detectors far away so any kill is
    # unambiguously the reasoning-only path.
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "60")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    if configure_key:
        (tmp_path / "config.yaml").write_text(
            f"agent:\n  reasoning_only_stale_timeout: {reasoning_timeout}\n",
            encoding="utf-8",
        )
    else:
        (tmp_path / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="deepseek/deepseek-v4-flash",
        provider="deepseek",
        api_key="sk-dummy",
        base_url="https://api.deepseek.com",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0)
    return agent


def _reasoning_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(
                reasoning_content="Let me think about this problem carefully...",
                content=None,
                tool_calls=None,
            ),
            finish_reason=None,
        )],
        model="deepseek/deepseek-v4-flash",
        usage=None,
    )


def _content_chunk(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(
                reasoning_content=None,
                content=text,
                tool_calls=None,
            ),
            finish_reason=None,
        )],
        model="deepseek/deepseek-v4-flash",
        usage=None,
    )


class _FakeStream:
    """Iterable chunk stream whose lifetime is controlled by the test.

    ``abort_flag`` is set by the patched ``_abort_request_openai_client``
    when the poll loop kills the connection; from then on ``__next__``
    raises like a real aborted socket read, which drives the worker's
    exception handling exactly like production.
    """

    def __init__(self, chunk_factory, abort_flag, stop_event):
        self._chunk_factory = chunk_factory
        self._abort_flag = abort_flag
        self._stop = stop_event
        self.response = None  # raw httpx response surface

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop.is_set():
            raise StopIteration
        if self._abort_flag["aborted"]:
            raise ConnectionError("connection aborted (reasoning-only stale kill)")
        time.sleep(0.01)
        return self._chunk_factory()

    def close(self):
        pass


def _wire_fake_client(agent, monkeypatch, chunk_factory, abort_flag, stop_event):
    """Patch the agent so ``interruptible_streaming_api_call`` streams from a fake.

    Returns ``(aborts, statuses, create_calls)`` — the recorded abort
    reasons, buffered status messages, and the number of stream-create
    calls (to assert no-retry behavior).
    """
    aborts: list[str] = []
    statuses: list[str] = []
    create_calls: list = []

    def _create(**kw):
        create_calls.append(kw)
        return _FakeStream(chunk_factory, abort_flag, stop_event)

    dummy_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda **k: dummy_client,
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: (aborts.append(reason), abort_flag.__setitem__("aborted", True)),
    )
    monkeypatch.setattr(
        agent,
        "_buffer_status",
        lambda msg: statuses.append(msg),
    )
    return aborts, statuses, create_calls


def _call_kwargs() -> dict:
    return {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }


class TestReasoningOnlyStaleKill:
    def test_reasoning_only_stream_is_killed_after_threshold(self, tmp_path, monkeypatch):
        """A stream emitting only reasoning chunks is killed, without retry."""
        from agent import chat_completion_helpers as h

        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=0.3)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        aborts, statuses, create_calls = _wire_fake_client(
            agent, monkeypatch, _reasoning_chunk, abort_flag, stop_event
        )

        # The stream never produces content; the reasoning-only kill aborts
        # it and the worker exits WITHOUT retrying, so the call surfaces the
        # connection error.
        with pytest.raises(Exception):
            h.interruptible_streaming_api_call(agent, _call_kwargs())

        assert "reasoning_only_stale_kill" in aborts, (
            f"expected reasoning-only kill, aborts={aborts}"
        )
        assert any("has been reasoning" in s for s in statuses), (
            f"expected user-facing reasoning status, statuses={statuses}"
        )
        # Byte-identical retries would be killed again: exactly ONE stream
        # attempt must be made.
        assert len(create_calls) == 1, (
            f"reasoning-only kill must not retry, create_calls={len(create_calls)}"
        )

    def test_threshold_read_from_config_yaml(self, tmp_path, monkeypatch):
        """config.yaml ``agent.reasoning_only_stale_timeout`` controls the kill."""
        from agent import chat_completion_helpers as h

        # Long threshold: reasoning-only stream must NOT be killed quickly.
        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=600.0)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, _reasoning_chunk, abort_flag, stop_event
        )

        result_box: list = []
        errors: list = []

        def _run():
            try:
                result_box.append(h.interruptible_streaming_api_call(agent, _call_kwargs()))
            except Exception as exc:  # pragma: no cover - assertion path
                errors.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(1.2)  # far beyond the 0.3s used by the kill test
        assert "reasoning_only_stale_kill" not in aborts, (
            f"long threshold must not kill early, aborts={aborts}"
        )
        stop_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "call should complete after the stream ends"

    def test_zero_disables_the_check(self, tmp_path, monkeypatch):
        """``agent.reasoning_only_stale_timeout: 0`` disables the kill."""
        from agent import chat_completion_helpers as h

        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=0)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, _reasoning_chunk, abort_flag, stop_event
        )

        def _run():
            h.interruptible_streaming_api_call(agent, _call_kwargs())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert "reasoning_only_stale_kill" not in aborts, (
            f"disabled check must never kill, aborts={aborts}"
        )
        stop_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "call should complete after the stream ends"

    def test_ttft_does_not_eat_reasoning_budget(self, tmp_path, monkeypatch):
        """The window is anchored at the first reasoning chunk, not attempt start."""
        from agent import chat_completion_helpers as h

        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=0.3)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        state = {"reasoning_started": None}
        abort_times: list = []

        def _slow_reasoning_chunk_factory():
            if state["reasoning_started"] is None:
                # Simulate TTFT: the provider is silent for 0.5s before the
                # first chunk (a reasoning chunk) arrives.
                time.sleep(0.5)
                state["reasoning_started"] = time.time()
            return _reasoning_chunk()

        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, _slow_reasoning_chunk_factory, abort_flag, stop_event
        )

        def _record_abort(client, reason=None):
            abort_times.append(time.time())
            aborts.append(reason)
            abort_flag["aborted"] = True

        monkeypatch.setattr(agent, "_abort_request_openai_client", _record_abort)

        with pytest.raises(Exception):
            h.interruptible_streaming_api_call(agent, _call_kwargs())

        assert state["reasoning_started"] is not None
        assert abort_times, "reasoning-only kill should have fired"
        # The kill must come ~0.3s AFTER reasoning started, not 0.3s after
        # the attempt start (which would be ~0.2s before reasoning began).
        elapsed_after_reasoning = abort_times[0] - state["reasoning_started"]
        assert elapsed_after_reasoning >= 0.25, (
            f"TTFT must not eat the reasoning budget; "
            f"kill {elapsed_after_reasoning:.2f}s after reasoning started "
            f"(threshold 0.3s)"
        )

    def test_reasoning_floor_extends_unconfigured_default(self, tmp_path, monkeypatch):
        """Without explicit config, the model's reasoning floor raises the default."""
        from agent import chat_completion_helpers as h
        from agent import reasoning_timeouts

        agent = _make_agent(tmp_path, monkeypatch, configure_key=False)
        # deepseek-v4-flash's real floor is 600s; patch it down to a value
        # that is still ABOVE the 300s default so the test stays fast while
        # proving the floor extends the default (300s would have killed).
        monkeypatch.setattr(reasoning_timeouts, "get_reasoning_stale_timeout_floor", lambda model: 600.0)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, _reasoning_chunk, abort_flag, stop_event
        )

        def _run():
            h.interruptible_streaming_api_call(agent, _call_kwargs())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(1.5)  # far beyond the 300s default scaled down for test speed
        assert "reasoning_only_stale_kill" not in aborts, (
            f"reasoning floor must extend the default, aborts={aborts}"
        )
        stop_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "call should complete after the stream ends"

    def test_explicit_config_wins_over_reasoning_floor(self, tmp_path, monkeypatch):
        """Explicit config is NOT raised by the model's reasoning floor."""
        from agent import chat_completion_helpers as h
        from agent import reasoning_timeouts

        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=0.3)
        monkeypatch.setattr(reasoning_timeouts, "get_reasoning_stale_timeout_floor", lambda model: 600.0)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, _reasoning_chunk, abort_flag, stop_event
        )

        with pytest.raises(Exception):
            h.interruptible_streaming_api_call(agent, _call_kwargs())

        assert "reasoning_only_stale_kill" in aborts, (
            f"explicit config must win over the floor, aborts={aborts}"
        )


class TestNoFalsePositive:
    def test_reasoning_then_content_is_not_killed(self, tmp_path, monkeypatch):
        """A stream that reasons briefly then produces content is left alone."""
        from agent import chat_completion_helpers as h

        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=0.3)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        state = {"content_since": None}

        def _chunk_factory():
            # Reasoning for the first 150ms, then real content — the kill
            # window (0.3s) must never fire because content resets the
            # content timer.
            now = time.time()
            if state["content_since"] is None:
                state["content_since"] = now + 0.15
            if now < state["content_since"]:
                return _reasoning_chunk()
            return _content_chunk("hello")

        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, _chunk_factory, abort_flag, stop_event
        )
        response_box: list = []

        def _run():
            response_box.append(h.interruptible_streaming_api_call(agent, _call_kwargs()))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert "reasoning_only_stale_kill" not in aborts, (
            f"reasoning-then-content must not be killed, aborts={aborts}"
        )
        stop_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "call should complete after the stream ends"

    def test_content_only_stream_is_not_killed(self, tmp_path, monkeypatch):
        """A stream with only content (no reasoning) never trips the check."""
        from agent import chat_completion_helpers as h

        agent = _make_agent(tmp_path, monkeypatch, reasoning_timeout=0.1)
        abort_flag = {"aborted": False}
        stop_event = threading.Event()
        aborts, _, _ = _wire_fake_client(
            agent, monkeypatch, lambda: _content_chunk("hello"), abort_flag, stop_event
        )
        response_box: list = []

        def _run():
            response_box.append(h.interruptible_streaming_api_call(agent, _call_kwargs()))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(0.8)  # several kill windows pass
        assert "reasoning_only_stale_kill" not in aborts, (
            f"content-only stream must never be killed, aborts={aborts}"
        )
        stop_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "call should complete after the stream ends"
