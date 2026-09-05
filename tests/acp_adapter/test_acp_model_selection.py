"""Regression tests for ACP ``provider:model`` resolution.

``_resolve_model_selection`` ran name-based provider detection whenever the
parsed provider equalled the session's current provider.  An explicit
``anthropic:<model>`` request sent while the session was already on
``anthropic`` was therefore indistinguishable from a bare model id, so
detection could reroute it to a different provider while the agent kept the
Anthropic ``base_url`` -- every request then failed with HTTP 404.

Observed via Buzz Desktop, which launches Hermes over ACP with a fully
qualified ``anthropic:<model>`` and got a provider the caller never asked
for.  These tests pin the contract: an explicit prefix wins, bare ids still
get detection.
"""

from __future__ import annotations

from unittest.mock import patch

from acp_adapter.server import HermesACPAgent


def _resolve(raw: str, current: str) -> tuple[str, str]:
    return HermesACPAgent._resolve_model_selection(raw, current)


class TestExplicitProviderPrefixWins:
    def test_explicit_prefix_matching_current_provider_skips_detection(self):
        """The bug: ``anthropic:x`` on an anthropic session must stay anthropic
        even when detection would claim ``x`` for someone else."""

        def _detect(_model, _current):
            return ("opencode-zen", _model)

        with patch("hermes_cli.models.detect_provider_for_model", side_effect=_detect):
            assert _resolve("anthropic:some-model", "anthropic") == (
                "anthropic",
                "some-model",
            )

    def test_explicit_prefix_switching_provider_is_honoured(self):
        assert _resolve("nous:hermes-3", "anthropic") == ("nous", "hermes-3")

    def test_explicit_prefix_with_slash_in_model_id(self):
        assert _resolve("openrouter:anthropic/claude-sonnet-4.5", "anthropic") == (
            "openrouter",
            "anthropic/claude-sonnet-4.5",
        )

    def test_detection_never_consulted_for_explicit_prefix(self):
        with patch("hermes_cli.models.detect_provider_for_model") as detect:
            _resolve("anthropic:some-model", "anthropic")
            detect.assert_not_called()


class TestBareModelIdStillDetected:
    def test_bare_model_id_uses_detection(self):
        """No caller-stated provider -> detection still gets to reroute."""

        def _detect(_model, _current):
            return ("opencode-zen", _model)

        with patch("hermes_cli.models.detect_provider_for_model", side_effect=_detect):
            assert _resolve("some-model", "anthropic") == ("opencode-zen", "some-model")

    def test_bare_model_id_falls_back_to_current_provider(self):
        with patch("hermes_cli.models.detect_provider_for_model", return_value=None):
            assert _resolve("some-model", "anthropic") == ("anthropic", "some-model")


class TestResolutionIsTotal:
    def test_whitespace_is_stripped(self):
        assert _resolve("  nous:hermes-3  ", "anthropic") == ("nous", "hermes-3")

    def test_import_failure_falls_back_to_raw_model(self):
        """A broken detection import must not take the session down."""
        with patch(
            "hermes_cli.models.parse_model_input",
            side_effect=RuntimeError("boom"),
        ):
            assert _resolve("some-model", "anthropic") == ("anthropic", "some-model")
