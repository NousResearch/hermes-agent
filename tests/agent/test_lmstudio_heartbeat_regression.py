"""End-to-end regression anchor for #24510 -- "Cannot connect to LM Studio".

The bug shape (per the discord traceback in the issue):

    WARNING discord.gateway: Shard ID None heartbeat blocked for
        more than 20 seconds.
    Loop thread traceback (most recent call last):
      File ".../gateway/run.py", line 6849, in
        _handle_message_with_agent
          _hyg_context_length = get_model_context_length(...)
      File ".../agent/model_metadata.py", line 1359, in
        get_model_context_length
          local_ctx = _query_local_context_length(...)
      File ".../agent/model_metadata.py", line 987, in
        _query_local_context_length
          server_type = detect_local_server_type(...)
      File ".../agent/model_metadata.py", line 455, in
        detect_local_server_type
          r = client.get(f"{server_url}/v1/props")

The function:
  - Issues up to 4 sync httpx.Client.get() probes per call
  - Has no caching, so EVERY message redoes them
  - Runs inside the asyncio event loop on the gateway hot path
  - Therefore eats the Discord heartbeat every message

This module ships ONE focused test that drives the gateway hot path
shape (10 messages in a row, all hitting the same local base_url) and
asserts that no more than ONE round of probes is issued -- the cache
must absorb the rest.

Why this proves the bug is fixed: on upstream/main this test fails
with ``mock_factory.call_count >= 10`` (one fresh httpx.Client per
"message" -> the asyncio loop blocks N times, exactly the #24510
symptom).  After the fix it's exactly 1.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_detect_cache():
    """Drop the module-level cache before/after each test.

    Defensive about the helper missing so that on upstream/main
    (pre-fix) the test still RUNS and fails with the actual
    bug-shape AssertionError rather than an ImportError -- the
    point of a regression anchor is to surface the bug, not to
    crash before the assertion fires."""
    from agent import model_metadata
    clear = getattr(model_metadata, "clear_local_server_type_cache", None)
    if clear is not None:
        clear()
    yield
    if clear is not None:
        clear()


def _make_lm_studio_200_client():
    """An ``httpx.Client`` mock that always returns 200 -- LM Studio
    happy path."""
    resp = MagicMock()
    resp.status_code = 200
    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)
    client.get.return_value = resp
    return client


class TestLmStudioHeartbeatRegression:
    """#24510 anchor: the gateway must not re-probe the local server
    on every message."""

    def test_repeated_calls_on_hot_path_collapse_to_single_probe_round(self):
        """Simulate a Discord shard taking 10 messages in quick
        succession against a local LM Studio base_url.  Without the
        cache, this triggers 10 rounds of probes -- each round
        construct a new httpx.Client, do up to 4 sync GETs, and block
        the asyncio loop.  With the cache, exactly ONE httpx.Client
        is constructed and the rest hit the dict.

        Asserting on ``httpx.Client`` construction count is the
        cleanest signal because every probe path sets up a fresh
        client; one client = one probe round.
        """
        from agent.model_metadata import detect_local_server_type

        client = _make_lm_studio_200_client()
        with patch("httpx.Client", return_value=client) as mock_factory:
            # Drive 10 "messages" through the same hot path.
            results = [
                detect_local_server_type(
                    "http://localhost:1234/v1", api_key="lm-token"
                )
                for _ in range(10)
            ]

        assert results == ["lm-studio"] * 10
        assert mock_factory.call_count == 1, (
            "#24510 regression: detect_local_server_type re-probed "
            f"{mock_factory.call_count} times across 10 sequential calls "
            "to the same base_url -- the gateway hot path is back to "
            "blocking the asyncio loop on every message and Discord "
            "will drop the heartbeat exactly as the bug report shows."
        )

    def test_legacy_lm_studio_install_does_not_silently_misdetect(self):
        """The other half of #24510: a user on LM Studio < 0.4.0 who
        only has /api/v0/models.  Pre-fix, detection returned None
        and the gateway then ran context-length code paths that
        don't speak LM Studio -- producing the user-visible HTTP
        400.  Post-fix, /api/v0/models is also probed and detection
        succeeds."""
        from agent.model_metadata import detect_local_server_type

        seen_paths: list[str] = []

        def fake_get(url, *a, **kw):
            seen_paths.append(url)
            # Legacy LM Studio: only /api/v0/models responds.
            if "/api/v0/models" in url:
                resp = MagicMock()
                resp.status_code = 200
                return resp
            resp = MagicMock()
            resp.status_code = 404
            return resp

        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = fake_get

        with patch("httpx.Client", return_value=client):
            result = detect_local_server_type("http://localhost:1234/v1")

        assert result == "lm-studio", (
            "#24510 regression: a legacy LM Studio install only exposes "
            "/api/v0/models -- the probe sequence must include it or "
            "the gateway falls through to non-LM-Studio code paths and "
            "users see HTTP 400 errors as in the bug report."
        )
        # Sanity: we tried v1 first AND we did try v0.
        assert any("/api/v1/models" in p for p in seen_paths)
        assert any("/api/v0/models" in p for p in seen_paths)
