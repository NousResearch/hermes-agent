"""Unavailability failures surface at ERROR, not WARNING (#88868).

The builtin memory provider keeps working when the OpenViking server is
down, so WARNING-level swallows let the outage persist invisibly for days.
Connection-level failures (the server could not be reached at all) on the
two highest-impact paths — session commit and the system prompt block —
must additionally log at ERROR so they land in errors.log.
"""

import logging
from unittest.mock import MagicMock

import httpx
import pytest

import plugins.memory.openviking as openviking_plugin
from plugins.memory.openviking import OpenVikingMemoryProvider, _OpenVikingHTTPError


class TestUnavailabilityClassifier:
    def test_connection_errors_are_unavailability(self):
        for exc in (
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("timed out"),
            httpx.PoolTimeout("pool exhausted"),
            ConnectionRefusedError(61),
        ):
            assert openviking_plugin._is_unavailability_error(exc), exc

    def test_http_answers_are_not_unavailability(self):
        """A server that ANSWERED (even with 5xx) is reachable — keep the
        per-call failure at WARNING and do not double-log at ERROR."""
        for exc in (
            _OpenVikingHTTPError("boom", status_code=500),
            _OpenVikingHTTPError("denied", status_code=403),
        ):
            assert not openviking_plugin._is_unavailability_error(exc), exc


class TestEscalationPaths:
    @pytest.fixture
    def provider(self):
        prov = OpenVikingMemoryProvider()
        prov._client = MagicMock()
        return prov

    def test_commit_failure_escalates_when_unreachable(self, provider, caplog):
        provider._client.post.side_effect = httpx.ConnectError("refused")

        with caplog.at_level(logging.DEBUG, logger=openviking_plugin.__name__):
            committed = provider._commit_session(
                "s1", 2, context="test", clear_missing=True
            )

        assert committed is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "unreachable server must log at ERROR"
        assert any("OpenViking unavailable" in r.getMessage() for r in errors)

    def test_commit_http_failure_stays_warning(self, provider, caplog):
        provider._client.post.side_effect = _OpenVikingHTTPError(
            "boom", status_code=500
        )

        with caplog.at_level(logging.DEBUG, logger=openviking_plugin.__name__):
            committed = provider._commit_session(
                "s1", 2, context="test", clear_missing=True
            )

        assert committed is False
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_prompt_block_degradation_escalates_when_unreachable(
        self, provider, caplog, monkeypatch
    ):
        monkeypatch.setattr(provider, "_ensure_client", lambda: True)
        provider._client.get.side_effect = httpx.ConnectError("refused")

        with caplog.at_level(logging.DEBUG, logger=openviking_plugin.__name__):
            block = provider.system_prompt_block()

        assert block  # fallback prompt text still returned
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors
        assert any("OpenViking unavailable" in r.getMessage() for r in errors)
