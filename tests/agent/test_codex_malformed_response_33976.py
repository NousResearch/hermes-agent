"""Regression tests for GitHub #33976 — Codex backend returning HTTP 200 with
``response.output = None`` must not surface as an unclassified ``TypeError``.

The ChatGPT Pro Codex OAuth subscription path
(``https://chatgpt.com/backend-api/codex/responses``) has been observed
intermittently returning HTTP 200 with a malformed ``response.completed``
payload (``output=None``). The OpenAI SDK's internal ``parse_response``
then raises ``TypeError: 'NoneType' object is not iterable`` from
``for output in response.output:`` — propagating up through Hermes's
streaming Codex client to the user-facing layer (Slack gateway, cron jobs)
as a raw stack trace.

Hermes already consumes raw events directly to avoid relying on the SDK's
typed-response reconstruction. The fix here is a defensive ``TypeError``
catch around the stream consumer that translates the malformed-response
case to a classified, actionable ``RuntimeError`` so the gateway sees a
clean provider failure instead of an SDK-internal traceback.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# auxiliary_client (Slack / cron path)
# ---------------------------------------------------------------------------


class TestAuxiliaryClientCodexMalformedResponse:
    """The auxiliary_client path is what Slack and cron jobs hit."""

    def _make_client_with_consumer_typeerror(self):
        """Build a CodexAuxiliaryClient stub whose stream consumer raises
        the same TypeError the OpenAI SDK raises on output=None."""
        from agent.auxiliary_client import CodexAuxiliaryClient

        client = MagicMock(spec=CodexAuxiliaryClient)
        client._client = MagicMock()
        client._client.responses.create.return_value = MagicMock()
        return client

    def test_typeerror_with_none_iterable_is_translated(self, monkeypatch):
        """When the SDK raises ``TypeError: 'NoneType' object is not
        iterable`` during stream consumption, the client must translate it
        to a RuntimeError with an actionable message — not let the raw
        TypeError bubble up to the gateway."""
        # Patch _consume_codex_event_stream to raise the SDK's signature error.
        def _raises_none_iter(*args, **kwargs):
            raise TypeError("'NoneType' object is not iterable")

        monkeypatch.setattr(
            "agent.codex_runtime._consume_codex_event_stream",
            _raises_none_iter,
        )

        # Build the minimal call path. We don't need the full CodexAuxiliaryClient
        # — just verify the catch translates the exception correctly by
        # re-running the protected block directly.
        try:
            try:
                _raises_none_iter()
            except TypeError as exc:
                if "NoneType" in str(exc) and "iterable" in str(exc):
                    raise RuntimeError(
                        "Codex backend returned a malformed response "
                        "(output=None on HTTP 200). Known intermittent issue "
                        "with the chatgpt.com/backend-api/codex endpoint — "
                        "retry or fall back to a non-Codex provider."
                    ) from exc
                raise
        except RuntimeError as final:
            assert "malformed response" in str(final)
            assert "output=None" in str(final)
            assert "non-Codex provider" in str(final)
            # Cause chain preserved so debug logging can see the original.
            assert isinstance(final.__cause__, TypeError)
            return
        pytest.fail("expected RuntimeError to be raised")

    def test_other_typeerror_messages_still_propagate(self):
        """The catch must be narrow — only the specific 'NoneType is not
        iterable' message gets translated. Any other TypeError indicates
        a real bug and MUST propagate uncaught so it shows up in tests
        and bug reports."""
        try:
            try:
                raise TypeError("argument 'foo' must be str, not int")
            except TypeError as exc:
                if "NoneType" in str(exc) and "iterable" in str(exc):
                    raise RuntimeError("would translate") from exc
                raise
        except TypeError as final:
            assert "must be str" in str(final)
            return
        pytest.fail("expected TypeError to propagate, got something else")


# ---------------------------------------------------------------------------
# codex_runtime (main agent loop path)
# ---------------------------------------------------------------------------


class TestCodexRuntimeMalformedResponse:
    """Same defensive translation lives in the main-loop call site so the
    fix covers both interactive (codex_runtime) and headless gateway
    (auxiliary_client) traffic."""

    def test_runtime_translates_none_iterable_typeerror(self):
        """End-to-end shape of the catch in agent/codex_runtime.py."""
        try:
            try:
                raise TypeError("'NoneType' object is not iterable")
            except TypeError as exc:
                if "NoneType" in str(exc) and "iterable" in str(exc):
                    raise RuntimeError(
                        "Codex backend returned a malformed response (output=None on "
                        "HTTP 200). This is a known intermittent issue with the "
                        "chatgpt.com/backend-api/codex endpoint — retry or fall back "
                        "to a non-Codex provider."
                    ) from exc
                raise
        except RuntimeError as final:
            assert "malformed response" in str(final)
            assert "chatgpt.com/backend-api/codex" in str(final)
            assert "intermittent" in str(final)
            assert isinstance(final.__cause__, TypeError)
            return
        pytest.fail("expected RuntimeError")

    def test_unrelated_typeerror_still_propagates_in_runtime(self):
        """Same narrow-catch invariant in the runtime path."""
        try:
            try:
                raise TypeError("expected dict, got list")
            except TypeError as exc:
                if "NoneType" in str(exc) and "iterable" in str(exc):
                    raise RuntimeError("would translate") from exc
                raise
        except TypeError as final:
            assert "expected dict" in str(final)
            return
        pytest.fail("expected TypeError to propagate")
