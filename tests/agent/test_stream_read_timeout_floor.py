"""Stream read timeout must never preempt the stale-stream detector.

Reasoning models (e.g. Opus) routinely pause mid-stream for minutes during
extended thinking.  The stale-stream detector is deliberately scaled up to
tolerate this (180s base, raised to 240s/300s for large contexts).  The httpx
socket read timeout, however, defaulted to a flat 120s for cloud providers and
fired *first* — tearing down a healthy reasoning stream before the stale
detector (which owns retry + diagnostics) could act.

These tests pin the invariant: for a cloud provider on the default read
timeout, the httpx socket read timeout is floored at the stale-stream timeout
so it can never fire before the detector.  They mirror the inline logic in
``agent/chat_completion_helpers.py`` (the real builder lives deep inside a
worker thread, so — like ``test_local_stream_timeout.py`` — the resolution is
reproduced here rather than driven end-to-end).
"""

import os

import pytest

from agent.model_metadata import is_local_endpoint


def _resolve_stale_timeout(base_url, est_tokens, stale_base=180.0):
    """Mirror of the stale-stream detector resolution."""
    if stale_base == 180.0 and base_url and is_local_endpoint(base_url):
        return float("inf")  # detector disabled for local providers
    if est_tokens > 100_000:
        return max(stale_base, 300.0)
    if est_tokens > 50_000:
        return max(stale_base, 240.0)
    return stale_base


def _resolve_read_timeout(base_url, stale_timeout, base_timeout=1800.0):
    """Mirror of the httpx socket read-timeout builder (cloud branch)."""
    read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
    if read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
        read_timeout = base_timeout
    elif (
        read_timeout == 120.0
        and stale_timeout is not None
        and stale_timeout != float("inf")
        and stale_timeout > read_timeout
    ):
        read_timeout = stale_timeout
    return read_timeout


CLOUD_URLS = [
    "https://api.githubcopilot.com",
    "https://api.openai.com",
    "https://openrouter.ai/api",
    "https://api.anthropic.com",
]


class TestCloudReadTimeoutFloor:
    @pytest.fixture(autouse=True)
    def _clear_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("HERMES_STREAM_READ_TIMEOUT", raising=False)
            yield

    @pytest.mark.parametrize("base_url", CLOUD_URLS)
    @pytest.mark.parametrize("est_tokens", [0, 10_000, 60_000, 150_000])
    def test_read_timeout_never_below_stale(self, base_url, est_tokens):
        """Core invariant: the socket read timeout >= the stale detector."""
        stale = _resolve_stale_timeout(base_url, est_tokens)
        read = _resolve_read_timeout(base_url, stale)
        assert read >= stale

    @pytest.mark.parametrize("base_url", CLOUD_URLS)
    def test_small_context_floored_to_stale_base(self, base_url):
        """Reported case: ~120s timeouts on Copilot are raised to the 180s base."""
        stale = _resolve_stale_timeout(base_url, est_tokens=37_000)
        read = _resolve_read_timeout(base_url, stale)
        assert read == 180.0

    @pytest.mark.parametrize("base_url", CLOUD_URLS)
    def test_large_context_tracks_scaled_stale(self, base_url):
        """Big contexts scale the stale detector; the read timeout follows."""
        assert _resolve_read_timeout(base_url, _resolve_stale_timeout(base_url, 60_000)) == 240.0
        assert _resolve_read_timeout(base_url, _resolve_stale_timeout(base_url, 150_000)) == 300.0

    def test_user_override_is_respected(self):
        """An explicit HERMES_STREAM_READ_TIMEOUT is never overridden by the floor."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("HERMES_STREAM_READ_TIMEOUT", "90")
            stale = _resolve_stale_timeout("https://api.githubcopilot.com", est_tokens=0)
            assert _resolve_read_timeout("https://api.githubcopilot.com", stale) == 90.0


class TestLocalUnaffected:
    @pytest.fixture(autouse=True)
    def _clear_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("HERMES_STREAM_READ_TIMEOUT", raising=False)
            yield

    def test_local_still_raised_to_base(self):
        """Local providers keep their existing behavior (raise to base timeout)."""
        stale = _resolve_stale_timeout("http://localhost:11434", est_tokens=0)
        assert stale == float("inf")  # detector disabled for local
        read = _resolve_read_timeout("http://localhost:11434", stale)
        assert read == 1800.0  # not clamped by inf

    def test_stale_none_falls_back_to_default(self):
        """If the stale value is unresolved, the read timeout keeps its default."""
        assert _resolve_read_timeout("https://api.githubcopilot.com", None) == 120.0
