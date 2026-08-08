"""Adversarial verifier probes for the mattermost-ws-401-classify fix
(commit fdd1a11ac5).

The implementer's test_ws_auth_retry.py covers:
  - WSServerHandshakeError(status=401) stops the loop
  - a transient RuntimeError whose message contains "401" now retries
  - the pre-existing _closing early-return path is untouched

This file probes boundary/edge cases the implementer's tests did NOT
cover, per the independent-verifier mandate to go beyond what the
implementer thought to test:

  1. WSServerHandshakeError(status=403) — the other structured-check
     status value — must still stop the loop (only 401 was tested).
  2. WSServerHandshakeError with a non-auth status (e.g. 500) must NOT
     stop the loop — the structured check must not over-match.
  3. A transient exception containing "unauthorized" (not "401"/"403")
     must retry now that the substring fallback is fully removed —
     the implementer only exercised the "401" substring variant.
  4. Multiple consecutive transient errors must all retry (not just
     one) — proves the removed code path isn't silently reintroduced
     via some other mechanism after N attempts.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from plugins.platforms.mattermost.adapter import MattermostAdapter


def _make_adapter(closing: bool = False) -> MattermostAdapter:
    adapter = MattermostAdapter.__new__(MattermostAdapter)
    adapter._closing = closing
    return adapter


class TestMattermostWSAuthRetryBoundaryProbes:
    def test_403_handshake_stops_reconnect(self):
        """status=403 (the other half of the structured check's {401, 403}
        set) must also stop the loop. The implementer only tested 401."""
        exc = aiohttp.WSServerHandshakeError(
            request_info=MagicMock(),
            history=(),
            status=403,
            message="Forbidden",
            headers=MagicMock(),
        )

        adapter = _make_adapter()
        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            raise exc

        adapter._ws_connect_and_listen = fake_connect

        asyncio.run(adapter._ws_loop())

        assert call_count == 1

    def test_non_auth_handshake_status_does_not_stop_reconnect(self):
        """A WSServerHandshakeError with a non-auth status (500) is a
        structured exception of the RIGHT TYPE but the WRONG status —
        it must NOT be classified as a permanent auth failure. This
        guards against an overly broad isinstance-only check that
        forgets to gate on .status."""
        exc = aiohttp.WSServerHandshakeError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Internal Server Error",
            headers=MagicMock(),
        )

        adapter = _make_adapter()
        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                adapter._closing = True
            raise exc

        adapter._ws_connect_and_listen = fake_connect

        with patch("asyncio.sleep", new=AsyncMock()):
            asyncio.run(adapter._ws_loop())

        assert call_count == 2

    def test_unauthorized_substring_no_longer_stops_reconnect(self):
        """Before the fix, a transient error whose message contained the
        word 'unauthorized' (not digits) would ALSO trip the removed
        substring fallback. The implementer's regression test only
        covered the '401' digit-substring case; this proves the
        'unauthorized' word variant is equally fixed."""
        adapter = _make_adapter()
        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                adapter._closing = True
            raise RuntimeError(
                "upstream proxy replied: request unauthorized by WAF rule, retry"
            )

        adapter._ws_connect_and_listen = fake_connect

        with patch("asyncio.sleep", new=AsyncMock()):
            asyncio.run(adapter._ws_loop())

        assert call_count == 2

    def test_repeated_transient_errors_all_retry(self):
        """Guards against a fix that only relaxes classification for the
        FIRST occurrence (e.g. some hidden retry-budget/counter that
        starts rejecting after N attempts). Runs 5 consecutive
        transient errors and confirms every one retries."""
        adapter = _make_adapter()
        call_count = 0
        target_attempts = 5

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            if call_count >= target_attempts:
                adapter._closing = True
            raise RuntimeError("403 seen in unrelated proxy diagnostic body")

        adapter._ws_connect_and_listen = fake_connect

        with patch("asyncio.sleep", new=AsyncMock()):
            asyncio.run(adapter._ws_loop())

        assert call_count == target_attempts
