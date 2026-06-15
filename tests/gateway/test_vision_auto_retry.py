"""Regression coverage for #28972 — Discord auto-vision returning
``success: false`` on first call and forcing the agent to issue a
duplicate ``vision_analyze`` manually.

The fix adds a bounded inline retry inside
``GatewayRunner._enrich_message_with_vision``:

* default 1 retry (configurable via ``auxiliary.vision.auto_retries``
  in config.yaml)
* exponential backoff capped at 3 s
* skip the retry budget when the failure signature is permanent
  (image too large, insufficient credits, model doesn't support
  vision, etc.) so we don't waste API calls on errors that won't
  go away on a second try

These tests exercise the helper directly and the happy/sad/retry
paths through the public ``_enrich_message_with_vision`` entry
point.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Sequence
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def gateway_runner():
    """Minimal stub that binds the retry-related class members + the
    public entry point so we can exercise the real logic without
    constructing a full ``GatewayRunner`` instance (which transitively
    pulls in adapter setup, session store, etc.)."""
    from gateway.run import GatewayRunner

    class _Stub:
        _enrich_message_with_vision = GatewayRunner._enrich_message_with_vision
        _vision_analyze_with_auto_retry = GatewayRunner._vision_analyze_with_auto_retry
        _vision_auto_retry_count = staticmethod(GatewayRunner._vision_auto_retry_count)
        _vision_failure_is_retryable = staticmethod(GatewayRunner._vision_failure_is_retryable)
        _VISION_AUTO_RETRY_COUNT_DEFAULT = GatewayRunner._VISION_AUTO_RETRY_COUNT_DEFAULT
        _VISION_AUTO_RETRY_INITIAL_BACKOFF_S = GatewayRunner._VISION_AUTO_RETRY_INITIAL_BACKOFF_S
        _VISION_AUTO_RETRY_MAX_BACKOFF_S = GatewayRunner._VISION_AUTO_RETRY_MAX_BACKOFF_S
        _VISION_NONRETRYABLE_HINTS = GatewayRunner._VISION_NONRETRYABLE_HINTS

    return _Stub()


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Replace the retry sleep with a no-op so test wall-time stays
    sub-second even when the helper backs off between attempts.
    The retry logic itself doesn't depend on actual elapsed time —
    only on awaiting the sleep — so this patch is invisible to it."""
    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("gateway.run.asyncio.sleep", _instant_sleep)


def _run(coro):
    """Drive an awaitable to completion under a fresh event loop.

    A fresh loop is cleaner than the global default — pytest-asyncio
    isn't a hard dependency of the gateway test suite, and using
    ``asyncio.run`` would close stray helper loops created by other
    tests in the same worker."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _vision_responses(*responses: Dict[str, Any]) -> AsyncMock:
    """Build an AsyncMock that returns the given JSON-serialised
    responses one per call.  Avoids the noise of repeating
    ``json.dumps`` at every callsite."""
    return AsyncMock(side_effect=[json.dumps(r) for r in responses])


_UNSET = object()


def _patch_auto_retries(monkeypatch, value=_UNSET) -> None:
    """Point ``hermes_cli.config.load_config`` at a synthetic config so
    ``_vision_auto_retry_count`` resolves ``auxiliary.vision.auto_retries``
    deterministically.  ``value=_UNSET`` leaves the key absent (exercises
    the default-fallback path); any other value is written verbatim so the
    parse/clamp logic can be tested with ints, strings, garbage, etc."""
    vision: Dict[str, Any] = {}
    if value is not _UNSET:
        vision["auto_retries"] = value
    cfg = {"auxiliary": {"vision": vision}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)


# ---------------------------------------------------------------------------
# _vision_auto_retry_count — config.yaml (auxiliary.vision.auto_retries)
# ---------------------------------------------------------------------------


class TestVisionAutoRetryCountResolver:
    def test_default_when_key_unset(self, monkeypatch, gateway_runner):
        _patch_auto_retries(monkeypatch)
        assert gateway_runner._vision_auto_retry_count() == 1

    def test_explicit_zero_opts_out(self, monkeypatch, gateway_runner):
        # Operators on metered APIs can disable the safety net
        # entirely without losing the gateway entrypoint.
        _patch_auto_retries(monkeypatch, 0)
        assert gateway_runner._vision_auto_retry_count() == 0

    def test_explicit_high_value(self, monkeypatch, gateway_runner):
        _patch_auto_retries(monkeypatch, 3)
        assert gateway_runner._vision_auto_retry_count() == 3

    def test_negative_clamped_to_zero(self, monkeypatch, gateway_runner):
        # A negative budget would be nonsensical (and would crash
        # the ``range(retries + 1)`` loop with 0 iterations).
        _patch_auto_retries(monkeypatch, -5)
        assert gateway_runner._vision_auto_retry_count() == 0

    def test_unparsable_falls_back_to_default(self, monkeypatch, gateway_runner):
        # A bad config value must NOT silently disable the safety
        # net (which would re-introduce the very bug we're fixing).
        _patch_auto_retries(monkeypatch, "off")
        assert gateway_runner._vision_auto_retry_count() == 1

    def test_string_value_tolerated(self, monkeypatch, gateway_runner):
        # YAML may hand us a quoted/whitespace-padded scalar.
        _patch_auto_retries(monkeypatch, "  2  ")
        assert gateway_runner._vision_auto_retry_count() == 2

    def test_default_when_config_load_fails(self, monkeypatch, gateway_runner):
        # A broken config load must fall back to the default rather
        # than crash the gateway's vision-enrichment path.
        def _boom(*_a, **_k):
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        assert gateway_runner._vision_auto_retry_count() == 1


# ---------------------------------------------------------------------------
# _vision_failure_is_retryable — error signature classification
# ---------------------------------------------------------------------------


class TestVisionFailureRetryability:
    @pytest.mark.parametrize("error_text", [
        "transient API failure: 502 Bad Gateway",
        "Read timeout after 30s",
        "Empty content returned from provider",
        "connection reset",
        "",
    ])
    def test_transient_signatures_retry(self, gateway_runner, error_text):
        result = {"success": False, "error": error_text, "analysis": ""}
        assert gateway_runner._vision_failure_is_retryable(result)

    @pytest.mark.parametrize("error_text", [
        "Image too large for vision API",
        "Insufficient credits or payment required",
        "openai/gpt-4-mini does not support vision",
        "content_policy violation",
        "Invalid image source. Provide an HTTP/HTTPS URL or a valid local file path.",
        "Blocked unsafe URL (SSRF protection)",
        "Interrupted",
        "PermissionError: blocked",
    ])
    def test_permanent_signatures_do_not_retry(self, gateway_runner, error_text):
        result = {"success": False, "error": error_text, "analysis": ""}
        assert not gateway_runner._vision_failure_is_retryable(result)

    def test_case_insensitive_match(self, gateway_runner):
        # The hints table is lowercase; the matcher must lowercase
        # the haystack to catch real-world capitalised error text.
        result = {"success": False, "error": "Image TOO LARGE for the API"}
        assert not gateway_runner._vision_failure_is_retryable(result)

    def test_match_against_analysis_field_too(self, gateway_runner):
        # ``vision_analyze_tool`` sometimes leaves the user-facing
        # explanation in ``analysis`` rather than ``error``.  Both
        # fields must participate in the classification.
        result = {
            "success": False,
            "error": "",
            "analysis": (
                "model gpt-4-mini does not support vision, the request "
                "was rejected by the server."
            ),
        }
        assert not gateway_runner._vision_failure_is_retryable(result)

    def test_empty_result_is_not_retryable(self, gateway_runner):
        # A defensive belt-and-braces guard: missing dict → caller
        # already failed, no point retrying with no diagnosis.
        assert not gateway_runner._vision_failure_is_retryable({})
        assert not gateway_runner._vision_failure_is_retryable(None)


# ---------------------------------------------------------------------------
# _vision_analyze_with_auto_retry — the retry loop itself
# ---------------------------------------------------------------------------


class TestVisionAnalyzeAutoRetry:
    def _patch_vision_tool(self, mock: AsyncMock):
        """Patch the import inside ``_vision_analyze_with_auto_retry``."""
        return patch("tools.vision_tools.vision_analyze_tool", new=mock)

    def test_first_call_success_no_retry(self, gateway_runner):
        # Happy path — no retry budget consumed.
        mock = _vision_responses(
            {"success": True, "analysis": "A cat on a keyboard."},
        )
        with self._patch_vision_tool(mock):
            result = _run(gateway_runner._vision_analyze_with_auto_retry(
                "/tmp/img.jpg", "describe this",
            ))
        assert result["success"] is True
        assert "cat on a keyboard" in result["analysis"]
        assert mock.call_count == 1

    def test_transient_failure_then_success(self, gateway_runner):
        # The exact #28972 scenario: first call fails transiently,
        # second call against the same path succeeds.
        mock = _vision_responses(
            {"success": False, "error": "transient 502 from provider"},
            {"success": True, "analysis": "A photo of a sunset."},
        )
        with self._patch_vision_tool(mock):
            result = _run(gateway_runner._vision_analyze_with_auto_retry(
                "/tmp/img.jpg", "describe this",
            ))
        assert result["success"] is True
        assert mock.call_count == 2

    def test_permanent_failure_short_circuits_retries(self, gateway_runner, monkeypatch):
        # Even with a generous retry budget, a permanent error should
        # only trigger one call — retrying "image too large" would
        # waste API quota for no chance of recovery.
        _patch_auto_retries(monkeypatch, 5)
        mock = _vision_responses(
            {"success": False, "error": "Image too large for vision API"},
        )
        with self._patch_vision_tool(mock):
            result = _run(gateway_runner._vision_analyze_with_auto_retry(
                "/tmp/img.jpg", "describe this",
            ))
        assert result["success"] is False
        assert mock.call_count == 1, (
            "Permanent failures must not consume retry budget — "
            "see _VISION_NONRETRYABLE_HINTS."
        )

    def test_all_attempts_fail_returns_last_result(self, gateway_runner, monkeypatch):
        _patch_auto_retries(monkeypatch, 2)
        mock = _vision_responses(
            {"success": False, "error": "transient 503"},
            {"success": False, "error": "transient 503"},
            {"success": False, "error": "transient 503 (final)"},
        )
        with self._patch_vision_tool(mock):
            result = _run(gateway_runner._vision_analyze_with_auto_retry(
                "/tmp/img.jpg", "describe this",
            ))
        assert result["success"] is False
        # The CALLER sees the *last* attempt's diagnostic so logs
        # reflect what we actually tried last.
        assert "final" in result["error"]
        assert mock.call_count == 3   # 1 initial + 2 retries

    def test_retries_disabled_via_config(self, gateway_runner, monkeypatch):
        # Explicit opt-out path — restores the legacy single-shot
        # behaviour for operators on metered APIs.
        _patch_auto_retries(monkeypatch, 0)
        mock = _vision_responses(
            {"success": False, "error": "transient 503"},
            {"success": True, "analysis": "would have succeeded"},
        )
        with self._patch_vision_tool(mock):
            result = _run(gateway_runner._vision_analyze_with_auto_retry(
                "/tmp/img.jpg", "describe this",
            ))
        assert result["success"] is False
        assert mock.call_count == 1   # no retry consumed

    def test_exception_propagates(self, gateway_runner):
        # The outer ``_enrich_message_with_vision`` owns the
        # ``except Exception`` branch.  The helper itself must NOT
        # swallow exceptions, otherwise the "something went wrong"
        # fallback never fires.
        mock = AsyncMock(side_effect=RuntimeError("provider timeout"))
        with self._patch_vision_tool(mock):
            with pytest.raises(RuntimeError, match="provider timeout"):
                _run(gateway_runner._vision_analyze_with_auto_retry(
                    "/tmp/img.jpg", "describe this",
                ))


# ---------------------------------------------------------------------------
# _enrich_message_with_vision — public entry point, end-to-end
# ---------------------------------------------------------------------------


class TestEnrichMessageWithVisionRetry:
    """Drive the public entry point so the retry logic is exercised
    through the same call path the gateway uses in production."""

    def test_28972_repro_first_call_fails_second_succeeds(self, gateway_runner):
        """The exact #28972 reproduction recipe: Discord-cached image
        fails transient on the first vision call, succeeds on the
        second.  After the fix, the agent receives the happy-path
        descriptor rather than the kawaii fallback, and never has to
        issue a manual ``vision_analyze`` round-trip."""
        mock = _vision_responses(
            {"success": False, "error": "empty content from vision provider"},
            {"success": True, "analysis": "A red apple on a wooden table."},
        )
        with patch("tools.vision_tools.vision_analyze_tool", new=mock):
            out = _run(gateway_runner._enrich_message_with_vision(
                "what is this?", ["/tmp/discord_img.jpg"],
            ))

        assert "red apple" in out
        assert "couldn't quite see it" not in out, (
            "After the retry succeeded, the kawaii fallback must "
            "not appear — it's the marker the agent uses to decide "
            "it needs to call vision_analyze manually (#28972)."
        )
        assert mock.call_count == 2

    def test_kawaii_fallback_still_fires_when_all_attempts_fail(self, gateway_runner):
        # Regression: the retry must not erase the existing fallback
        # behaviour for the (rare) case where every attempt fails.
        mock = _vision_responses(
            {"success": False, "error": "transient"},
            {"success": False, "error": "transient (final)"},
        )
        with patch("tools.vision_tools.vision_analyze_tool", new=mock):
            out = _run(gateway_runner._enrich_message_with_vision(
                "what is this?", ["/tmp/discord_img.jpg"],
            ))
        assert "couldn't quite see it" in out
        assert "/tmp/discord_img.jpg" in out

    def test_first_call_succeeds_no_retry_overhead(self, gateway_runner):
        # On the common happy path the retry layer adds zero extra
        # API calls — important for users on metered providers.
        mock = _vision_responses(
            {"success": True, "analysis": "A bowl of soup."},
        )
        with patch("tools.vision_tools.vision_analyze_tool", new=mock):
            out = _run(gateway_runner._enrich_message_with_vision(
                "lunch", ["/tmp/img.jpg"],
            ))
        assert "bowl of soup" in out
        assert mock.call_count == 1

    def test_permanent_failure_does_not_retry(self, gateway_runner, monkeypatch):
        _patch_auto_retries(monkeypatch, 5)
        mock = _vision_responses(
            {"success": False, "error": "Insufficient credits"},
        )
        with patch("tools.vision_tools.vision_analyze_tool", new=mock):
            out = _run(gateway_runner._enrich_message_with_vision(
                "what?", ["/tmp/img.jpg"],
            ))
        assert "couldn't quite see it" in out
        assert mock.call_count == 1

    def test_exception_path_still_uses_something_went_wrong_fallback(
        self, gateway_runner,
    ):
        # The retry helper deliberately doesn't swallow exceptions
        # so the outer try/except in ``_enrich_message_with_vision``
        # can still produce its distinct "something went wrong"
        # message (semantically different from "couldn't see it").
        mock = AsyncMock(side_effect=RuntimeError("provider explosion"))
        with patch("tools.vision_tools.vision_analyze_tool", new=mock):
            out = _run(gateway_runner._enrich_message_with_vision(
                "?", ["/tmp/img.jpg"],
            ))
        assert "something went wrong" in out
        assert "/tmp/img.jpg" in out

    def test_per_image_retry_budget_isolated(self, gateway_runner):
        # When two images are attached and only the FIRST is
        # transient, the second image's per-image retry budget must
        # be untouched (no global counter, no shared state).
        mock = _vision_responses(
            # Image 1: transient → success on retry
            {"success": False, "error": "transient"},
            {"success": True, "analysis": "Image one: a dog."},
            # Image 2: succeeds first try
            {"success": True, "analysis": "Image two: a cat."},
        )
        with patch("tools.vision_tools.vision_analyze_tool", new=mock):
            out = _run(gateway_runner._enrich_message_with_vision(
                "two pics", ["/tmp/a.jpg", "/tmp/b.jpg"],
            ))
        assert "Image one: a dog" in out
        assert "Image two: a cat" in out
        assert "couldn't quite see it" not in out
        # 2 calls for image 1 (transient + retry) + 1 for image 2 = 3
        assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


class TestRetryConstantsShape:
    """Lock down the public-ish class constants so a drive-by tuning
    change can't silently disable the safety net (which would
    reintroduce #28972)."""

    def test_default_retry_count_is_at_least_one(self, gateway_runner):
        # The whole point of the fix is "at least one retry by
        # default".  A regression to 0 would silently re-open the bug.
        assert gateway_runner._VISION_AUTO_RETRY_COUNT_DEFAULT >= 1

    def test_initial_backoff_is_positive_and_capped(self, gateway_runner):
        assert gateway_runner._VISION_AUTO_RETRY_INITIAL_BACKOFF_S > 0
        assert gateway_runner._VISION_AUTO_RETRY_INITIAL_BACKOFF_S <= 5.0
        assert gateway_runner._VISION_AUTO_RETRY_MAX_BACKOFF_S >= (
            gateway_runner._VISION_AUTO_RETRY_INITIAL_BACKOFF_S
        )

    def test_nonretryable_hints_are_lowercase(self, gateway_runner):
        # The matcher lowercases the haystack, not the hints, so any
        # uppercase character in the hint table would make it
        # silently dead.
        for hint in gateway_runner._VISION_NONRETRYABLE_HINTS:
            assert hint == hint.lower(), (
                f"Hint {hint!r} contains uppercase characters — it "
                f"will never match against the lowercased haystack."
            )

    def test_nonretryable_hints_cover_known_permanent_errors(self, gateway_runner):
        # Smoke-check that the hints table actually catches the
        # error messages emitted by tools.vision_tools.  If any of
        # these slip through, "permanent" errors would burn through
        # the retry budget pointlessly.
        permanent_samples = [
            "Image too large for vision API: base64 payload is 23 MB",
            "Insufficient credits or payment required",
            "gpt-4-mini does not support vision",
            "Only real image files are supported for vision analysis.",
            "Blocked unsafe URL (SSRF protection)",
            "Interrupted",
        ]
        for sample in permanent_samples:
            result = {"success": False, "error": sample, "analysis": ""}
            assert not gateway_runner._vision_failure_is_retryable(result), (
                f"Sample error {sample!r} should be classified as "
                f"permanent but the matcher thinks it's retryable."
            )
