"""Regression tests for #31977 — MiniMax base_url misconfiguration warning.

Issue #31977 (sub-issue 3) reports user confusion: typing
``https://api.minimax.io/v1/chat/completions`` into ``model.base_url``
returns 401 with no useful body, because ``api.minimax.io`` (the
*Global* MiniMax endpoint) only speaks the Anthropic-compatible API
under ``/anthropic`` — the OpenAI-compatible chat-completions API is
only available on the *China* hosts (``api.minimaxi.com``, legacy
``api.minimax.chat``).

The fix adds a one-time hint logged from
``_detect_api_mode_for_url`` whenever a base URL with the bad combo
shows up.  These tests pin both the trigger conditions (so a future
refactor can't silently mute the warning) and the no-false-positive
contract for the supported endpoints.
"""

from __future__ import annotations

import inspect
import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_minimax_warned_set():
    """Clear the once-per-URL warning cache between tests.

    The helper de-duplicates warnings using a module-level ``set``, so
    leaving entries between tests would mask trigger-detection failures
    in subsequent cases.
    """
    from hermes_cli import runtime_provider

    runtime_provider._MINIMAX_BASE_URL_WARNED.clear()
    yield
    runtime_provider._MINIMAX_BASE_URL_WARNED.clear()


# ---------------------------------------------------------------------------
# Trigger conditions: warn loudly for the broken combinations
# ---------------------------------------------------------------------------


class TestMinimaxBaseUrlWarningTriggers:
    """The warning fires for the specific URLs that return 401 in the wild."""

    @pytest.mark.parametrize(
        "bad_url",
        [
            "https://api.minimax.io/v1/chat/completions",
            "https://api.minimax.io/v1",
            "https://api.minimax.io/v1/",
            "https://api.minimax.io/v2/chat/completions",
            "https://api.minimax.io/openai/v1",
            "http://api.minimax.io/v1/chat/completions",  # plain http
        ],
    )
    def test_warning_fires_for_chat_completions_path_on_global_endpoint(
        self, bad_url: str, caplog: pytest.LogCaptureFixture,
    ):
        from hermes_cli.runtime_provider import _warn_if_minimax_base_url_misconfigured

        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider"):
            _warn_if_minimax_base_url_misconfigured(bad_url)

        assert any(
            "31977" in rec.getMessage() and "api.minimax.io" in rec.getMessage()
            for rec in caplog.records
        ), f"expected #31977 warning for {bad_url}, got: {[r.getMessage() for r in caplog.records]}"

    def test_warning_fires_through_detect_api_mode_for_url(
        self, caplog: pytest.LogCaptureFixture,
    ):
        """Calling ``_detect_api_mode_for_url`` (the runtime hot path) also warns.

        Direct callers of the helper get the warning, but most real call
        sites go through ``_detect_api_mode_for_url`` (auxiliary client,
        delegation, model switch, etc.).  Keep the trigger wired there
        so a typo in any of those call sites still surfaces the hint.
        """
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider"):
            _detect_api_mode_for_url("https://api.minimax.io/v1/chat/completions")

        assert any("31977" in rec.getMessage() for rec in caplog.records)

    def test_warning_dedupes_per_url(self, caplog: pytest.LogCaptureFixture):
        """Identical URL warns once, not on every runtime resolution.

        ``resolve_runtime_provider`` and friends are called many times
        per session — auxiliary client setup, ``/model`` switch,
        delegation child task, etc.  Without dedup the user would see
        the same hint repeated on every API turn.
        """
        from hermes_cli.runtime_provider import _warn_if_minimax_base_url_misconfigured

        url = "https://api.minimax.io/v1/chat/completions"
        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider"):
            for _ in range(10):
                _warn_if_minimax_base_url_misconfigured(url)

        warnings = [r for r in caplog.records if "31977" in r.getMessage()]
        assert len(warnings) == 1, (
            f"expected exactly 1 warning, got {len(warnings)}"
        )

    def test_distinct_urls_each_get_their_own_warning(
        self, caplog: pytest.LogCaptureFixture,
    ):
        """Two different bad URLs should each surface independently."""
        from hermes_cli.runtime_provider import _warn_if_minimax_base_url_misconfigured

        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider"):
            _warn_if_minimax_base_url_misconfigured("https://api.minimax.io/v1")
            _warn_if_minimax_base_url_misconfigured("https://api.minimax.io/v2")

        warnings = [r for r in caplog.records if "31977" in r.getMessage()]
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# No false positives: known-good MiniMax endpoints stay warning-free
# ---------------------------------------------------------------------------


class TestMinimaxBaseUrlWarningNoFalsePositives:
    """The supported endpoints must NEVER trigger the warning."""

    @pytest.mark.parametrize(
        "good_url",
        [
            # Anthropic transport on the Global endpoint — the canonical setup.
            "https://api.minimax.io/anthropic",
            "https://api.minimax.io/anthropic/",
            "http://api.minimax.io/anthropic",
            # Bare host with no path — user is mid-config, no commitment yet.
            "https://api.minimax.io",
            "https://api.minimax.io/",
            # China endpoints — both Anthropic and OpenAI-compatible paths
            # should pass without warning.
            "https://api.minimaxi.com/v1",
            "https://api.minimaxi.com/v1/chat/completions",
            "https://api.minimaxi.com/anthropic",
            # Legacy China host — still works for OpenAI-compatible /v1.
            "https://api.minimax.chat/v1",
            "https://api.minimax.chat/v1/chat/completions",
            # Unrelated providers — should never trigger this hint.
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "https://openrouter.ai/api/v1",
            "https://api.deepseek.com/v1",
            # Empty / None / whitespace.
            "",
            "   ",
        ],
    )
    def test_no_warning_for_supported_or_unrelated_urls(
        self, good_url: str, caplog: pytest.LogCaptureFixture,
    ):
        from hermes_cli.runtime_provider import _warn_if_minimax_base_url_misconfigured

        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider"):
            _warn_if_minimax_base_url_misconfigured(good_url)

        offending = [r for r in caplog.records if "31977" in r.getMessage()]
        assert not offending, (
            f"unexpected #31977 warning for {good_url!r}: "
            f"{[r.getMessage() for r in offending]}"
        )

    def test_lookalike_host_does_not_trigger_warning(
        self, caplog: pytest.LogCaptureFixture,
    ):
        """``api.minimax.io.evil`` must NOT match — substring guards only.

        ``base_url_hostname`` already resists that lookalike attack
        (the docstring explicitly calls it out).  This test pins the
        invariant for the MiniMax warning helper specifically so a
        future ``"api.minimax.io" in base_url`` shortcut would be loud.
        """
        from hermes_cli.runtime_provider import _warn_if_minimax_base_url_misconfigured

        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider"):
            _warn_if_minimax_base_url_misconfigured(
                "https://api.minimax.io.evil/v1/chat/completions"
            )
            _warn_if_minimax_base_url_misconfigured(
                "https://attacker.example/api.minimax.io/v1"
            )

        offending = [r for r in caplog.records if "31977" in r.getMessage()]
        assert not offending


# ---------------------------------------------------------------------------
# Source-level guards: keep the wiring intact under future refactors
# ---------------------------------------------------------------------------


class TestSourceGuards:
    """Make accidental removal of the warning loud at code review."""

    def test_detect_api_mode_for_url_calls_warning_helper(self):
        """The runtime hot path still routes through the warning helper."""
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        src = inspect.getsource(_detect_api_mode_for_url)
        assert "_warn_if_minimax_base_url_misconfigured" in src

    def test_warning_helper_references_issue_number(self):
        """The log message must cite #31977 so users can find context."""
        from hermes_cli.runtime_provider import _warn_if_minimax_base_url_misconfigured

        src = inspect.getsource(_warn_if_minimax_base_url_misconfigured)
        assert "31977" in src
        # Also references the three valid endpoints so a refactor doesn't
        # silently drop the actionable advice.
        assert "/anthropic" in src
        assert "api.minimaxi.com" in src
        assert "api.minimax.chat" in src
