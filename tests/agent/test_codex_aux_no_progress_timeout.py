"""Progress-aware deadlines for the Codex auxiliary Responses stream.

Regression tests for the masoria report (Aug 2026): each compression
attempt sat the FULL 300s absolute timeout on a dead Codex stream before
falling back, and repeated attempts stacked into a 20+ minute
"Summarizing thread" stall.

New contract for ``_CodexCompletionsAdapter.create``:

1. No first token within the 60s no-progress window -> fail fast
   (``no-progress timeout`` in the message) so the fallback chain runs
   after ~60s, not 300s.
2. A live stream re-arms the window on every substantive event: a slow
   summary that keeps producing tokens is never killed by the old
   absolute ``total_timeout``.
3. A mid-stream stall (tokens seen, then silence) dies one no-progress
   window after the last token (``stalled`` in the message).
4. The compression critical-path retry gate distinguishes the two: a
   cheap first-token failure still gets the same-provider retry; a
   full-budget stall skips straight to fallback (#54465 semantics).
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter, call_llm


def _content_event(text="tok"):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _keepalive_event():
    return SimpleNamespace(type="response.in_progress")


def _make_adapter(event_iter):
    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=lambda **_kwargs: event_iter),
        close=lambda: None,
    )
    return _CodexCompletionsAdapter(real_client, "gpt-5.6-sol")


def _consume(stream, *, model, on_event):
    del model
    for event in stream:
        on_event(event)
    return SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="summary")],
        )],
        usage=None,
    )


class TestNoProgressFailFast:
    def test_dead_stream_fails_at_no_progress_window_not_total_timeout(self):
        """Keepalive-only stream dies at the (patched) no-progress window,
        long before the 300s-style total timeout."""

        def _zombie():
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                time.sleep(0.02)
                yield _keepalive_event()

        adapter = _make_adapter(_zombie())
        start = time.monotonic()
        with (
            patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.3),
            patch("agent.codex_runtime._consume_codex_event_stream", _consume),
            pytest.raises(TimeoutError, match="no-progress timeout"),
        ):
            adapter.create(
                messages=[{"role": "user", "content": "summarize"}],
                timeout=300,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"fail-fast took {elapsed:.1f}s"

    def test_live_stream_outlives_the_old_absolute_total_timeout(self):
        """Tokens arriving inside the window keep the stream alive past
        ``timeout`` — the old code killed this call at total_timeout."""

        def _slow_but_alive():
            # 8 tokens, 0.1s apart: total ~0.8s, well past timeout=0.4.
            for _ in range(8):
                time.sleep(0.1)
                yield _content_event()

        adapter = _make_adapter(_slow_but_alive())
        with (
            patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.4),
            patch("agent.codex_runtime._consume_codex_event_stream", _consume),
        ):
            response = adapter.create(
                messages=[{"role": "user", "content": "summarize"}],
                timeout=0.4,
            )
        assert response.choices[0].message.content == "summary"

    def test_mid_stream_stall_raises_stalled_timeout(self):
        def _stalls_after_two_tokens():
            yield _content_event()
            yield _content_event()
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                time.sleep(0.02)
                yield _keepalive_event()

        adapter = _make_adapter(_stalls_after_two_tokens())
        start = time.monotonic()
        with (
            patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.3),
            patch("agent.codex_runtime._consume_codex_event_stream", _consume),
            pytest.raises(TimeoutError, match="stalled: no new output"),
        ):
            adapter.create(
                messages=[{"role": "user", "content": "summarize"}],
                timeout=300,
            )
        assert time.monotonic() - start < 5.0

    def test_hard_ceiling_bounds_a_token_drip(self):
        """A degenerate one-token-per-window drip still terminates at the
        _aux_stream_total_ceiling backstop."""

        def _dripper():
            while True:
                time.sleep(0.05)
                yield _content_event()

        adapter = _make_adapter(_dripper())
        with (
            patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 5.0),
            patch("agent.auxiliary_client._aux_stream_total_ceiling",
                  return_value=0.3),
            patch("agent.codex_runtime._consume_codex_event_stream", _consume),
            pytest.raises(TimeoutError, match="hard ceiling"),
        ):
            adapter.create(
                messages=[{"role": "user", "content": "summarize"}],
                timeout=300,
            )

    def test_watchdog_timer_fires_while_blocked_before_first_event(self):
        """responses.create() itself can block with zero bytes; the re-armable
        watchdog must mark the timeout (without releasing FDs from its own
        stranger thread — #29507) and the owning thread must surface the
        no-progress TimeoutError and perform the close on unwind."""
        release = threading.Event()

        def _blocked_create(**_kwargs):
            release.wait(timeout=30.0)
            return iter([])

        closed_by = []
        real_client = SimpleNamespace(
            base_url="https://chatgpt.com/backend-api/codex",
            responses=SimpleNamespace(create=_blocked_create),
            close=lambda: closed_by.append(threading.get_ident()),
        )
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.6-sol")
        owner_result: dict = {}
        try:
            with (
                patch("agent.auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.3),
                patch("agent.auxiliary_client._evict_cached_client_instance"),
            ):
                def _run():
                    owner_result["tid"] = threading.get_ident()
                    try:
                        adapter.create(
                            messages=[{"role": "user", "content": "x"}],
                            timeout=300,
                        )
                    except Exception as exc:  # noqa: BLE001
                        owner_result["exc"] = exc

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                # Watchdog fires within the patched window; it must NOT
                # close() from its own thread (FD ownership, #29507).
                time.sleep(1.0)
                assert not closed_by, f"stranger-thread close: {closed_by}"
                release.set()
                t.join(timeout=5.0)
            assert isinstance(owner_result.get("exc"), TimeoutError)
            assert "no-progress timeout" in str(owner_result["exc"])
            # The OWNER released the FDs on unwind.
            assert closed_by == [owner_result["tid"]], closed_by
        finally:
            release.set()


class TestCompressionRetryGate:
    """First-token failures retry same-provider; stalls skip to fallback."""

    def _run_call_llm(self, primary_error, second_response=None):
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        if second_response is not None:
            primary_client.chat.completions.create.side_effect = [
                primary_error, second_response,
            ]
        else:
            primary_client.chat.completions.create.side_effect = primary_error

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="fallback"),
                finish_reason="stop",
            )],
            model="fb", usage=None,
        )

        with (
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(primary_client, "gpt-5.6-sol")),
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("auto", "gpt-5.6-sol", None, None, None)),
            patch("agent.auxiliary_client._try_configured_fallback_chain",
                  return_value=(None, None, "")),
            patch("agent.auxiliary_client._try_main_fallback_chain",
                  return_value=(None, None, "")),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=(fallback_client, "fb", "openrouter")) as mock_fb,
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )
        return result, primary_client, mock_fb

    def test_no_progress_timeout_retries_same_provider(self):
        err = TimeoutError(
            "Codex auxiliary Responses stream produced no output within "
            "60.0s (no-progress timeout, 60.2s elapsed)"
        )
        good = SimpleNamespace(
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="retried"),
                finish_reason="stop",
            )],
            model="gpt-5.6-sol", usage=None,
        )
        result, primary, mock_fb = self._run_call_llm(err, second_response=good)
        assert result.choices[0].message.content == "retried"
        assert primary.chat.completions.create.call_count == 2
        assert not mock_fb.called

    def test_stalled_timeout_skips_same_provider_retry(self):
        err = TimeoutError(
            "Codex auxiliary Responses stream stalled: no new output for "
            "60.0s (247.3s elapsed)"
        )
        result, primary, mock_fb = self._run_call_llm(err)
        assert result.choices[0].message.content == "fallback"
        assert primary.chat.completions.create.call_count == 1
        assert mock_fb.called

    def test_hard_ceiling_timeout_skips_same_provider_retry(self):
        err = TimeoutError(
            "Codex auxiliary Responses stream exceeded 600.0s hard ceiling"
        )
        result, primary, mock_fb = self._run_call_llm(err)
        assert result.choices[0].message.content == "fallback"
        assert primary.chat.completions.create.call_count == 1
        assert mock_fb.called
