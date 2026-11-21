"""Provider ``Retry-After`` handling in the retry loop: honoured up to a 600-second cap (#26293),
also on retryable 5xx responses; a rate limit that reopens past the cap takes the short backoff
unless ``fallback.min_switch_reset_seconds`` keeps the turn on the primary."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_run_agent import _make_tool_defs


@pytest.fixture()
def agent(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestRetryAfterCap:
    """The loop honors provider cooldowns up to a 600-second ceiling.

    This covers rate-limit headers (#26293) and retryable 5xx responses.
    """

    @staticmethod
    def _retryable_error(status_code, headers, body=None):
        """A provider error carrying optional Retry-After surfaces."""
        message = (
            "Error code: 429 - Rate limit exceeded."
            if status_code == 429
            else f"Error code: {status_code} - origin response timeout"
        )

        class _ProviderError(Exception):
            def __init__(self):
                super().__init__(message)
                self.status_code = status_code
                self.response = SimpleNamespace(headers=headers)
                if body is not None:
                    self.body = body

        return _ProviderError()

    def _drive_once(self, agent, error, status_marker):
        """Raise ``error`` from the API call and capture the backoff status the
        loop chose. Interrupt during the backoff sleep so the test doesn't
        actually wait, and return the status string reporting the wait."""

        def _fake_api_call(api_kwargs):
            raise error

        agent._interruptible_api_call = _fake_api_call
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None

        captured = []
        original_buffer = agent._buffer_status
        original_emit = agent._emit_status

        def _capture_status(msg, *args, **kwargs):
            captured.append((msg, "buffer"))
            # Break out of the backoff sleep immediately rather than blocking
            # for the full Retry-After window.
            if status_marker in msg:
                agent._interrupt_requested = True
            return original_buffer(msg, *args, **kwargs)

        def _capture_emit(msg):
            captured.append((msg, "emit"))
            if status_marker in msg:
                agent._interrupt_requested = True
            return original_emit(msg)

        agent._buffer_status = _capture_status
        agent._emit_status = _capture_emit
        agent.run_conversation("hello")
        return next(((m, s) for m, s in captured if status_marker in m), ("", ""))

    def test_retry_after_under_cap_is_honored(self, agent):
        # 300s > old 120s cap but < new 600s cap → used verbatim.
        error = self._retryable_error(429, {"retry-after": "300"})
        status, _ = self._drive_once(agent, error, "Waiting")
        assert "Waiting 300.0s" in status

    @pytest.mark.parametrize(
        ("headers", "body", "expected_wait", "expected_surface"),
        [
            # Long cooldowns (> 60s) surface immediately...
            ({"Retry-After": "120"}, {}, "120.0", "emit"),
            ({}, {"status": 524, "retry_after": 120}, "120.0", "emit"),
            ({}, {"status": 524, "error": {"retry_after": 120}}, "120.0", "emit"),
            # Above the 600s ceiling → capped, never used verbatim.
            ({"Retry-After": "3600"}, {}, "600.0", "emit"),
            # ...short cooldowns keep the buffered status line.
            ({"Retry-After": "30"}, {}, "30.0", "buffer"),
            # No cooldown on header or body → falls through to jittered
            # backoff (patched to 0.0 by the conftest fixture), no crash.
            ({}, {"status": 524}, "0.0", "buffer"),
        ],
        ids=(
            "header",
            "problem-detail-body",
            "nested-problem-detail-body",
            "over-cap-is-capped",
            "short-cooldown-is-buffered",
            "no-cooldown-falls-back",
        ),
    )
    def test_retry_after_on_cloudflare_524_is_honored(
        self, agent, headers, body, expected_wait, expected_surface
    ):
        """A retryable 5xx must not bypass the provider's cooldown."""
        error = self._retryable_error(524, headers, body)
        status, surface = self._drive_once(agent, error, "Retrying in")
        assert f"Retrying in {expected_wait}s" in status
        assert surface == expected_surface

    def test_rate_limit_reset_past_the_cap_is_not_slept_out(self, agent):
        """A 429 whose Retry-After lies past the cap names a spent quota window (e.g. a weekly
        plan limit, ~1.9 days). Sleeping the 600s cap cannot reach it, so the loop takes the
        short backoff instead of stalling the turn — unlike a 5xx, whose over-cap value stays
        capped (see ``over-cap-is-capped``)."""
        error = self._retryable_error(429, {"retry-after": "164160"})
        status, _ = self._drive_once(agent, error, "Waiting")
        assert status, "the rate-limit wait status must still be reported"
        assert "Waiting 600.0s" not in status
        # jittered_backoff is patched to 0.0 by the conftest fast-backoff fixture.
        assert "Waiting 0.0s" in status

    def test_rate_limit_past_the_cap_is_waited_when_the_primary_is_kept(self, agent):
        """fallback.min_switch_reset_seconds above the reset keeps the turn on the primary so the
        retry backoff rides out the window; dropping the over-cap value there would exhaust the
        retries long before the reset and end the turn with no fallback tried."""
        error = self._retryable_error(429, {"retry-after": "1000"})
        config = {"fallback": {"min_switch_reset_seconds": 3600}}
        with patch("hermes_cli.config.load_config", return_value=config):
            status, _ = self._drive_once(agent, error, "Waiting")
        assert "Waiting 600.0s" in status

    def test_primary_is_kept_only_when_the_provider_reset_is_under_the_threshold(self, agent):
        """The keep-the-primary check reads the same reset the fallback walk reads: a body reset
        a day out wins over a 1000 s header, so the turn is not held on the primary for 600 s
        waits only to switch once the retries run out."""
        error = self._retryable_error(
            429, {"retry-after": "1000"}, body={"error": {"resets_in_seconds": 86400}},
        )
        config = {"fallback": {"min_switch_reset_seconds": 3600}}
        with patch("hermes_cli.config.load_config", return_value=config):
            status, _ = self._drive_once(agent, error, "Waiting")
        assert "Waiting 600.0s" not in status
        assert "Waiting 0.0s" in status
