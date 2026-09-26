"""Regression tests for the #102203 buffered fix.

When a transform_llm_output hook is registered, the CLI must NOT stream
token-by-token — a mutating transform is applied AFTER streaming, so any
post-hoc print (suffix or banner) lands after already-rendered bytes and
garbles the terminal (see review of the ba0969505a approach on PR #102239).

The fix gates ``stream_delta_callback`` via
``CLIAgentSetupMixin._resolve_stream_delta_callback``, wired from
``_init_agent``. Tests drive THAT method (not a copy of its expression), so
a refactor that drops the hook gate breaks the suite.

Pinned against origin/main 63279301bc: the tests must FAIL on main (the
resolver method does not exist there).
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


def _make_cli(*, streaming_enabled: bool) -> CLIAgentSetupMixin:
    cli = CLIAgentSetupMixin()
    cli.streaming_enabled = streaming_enabled
    cli._stream_delta = lambda *_a, **_k: None
    return cli


class TestTransformHookGate:
    """The predicate behind the gate."""

    def test_gate_true_when_hook_registered(self):
        cli = _make_cli(streaming_enabled=True)
        with patch("hermes_cli.plugins.has_hook", return_value=True):
            assert cli._transform_llm_output_hook_active() is True

    def test_gate_false_without_hook(self):
        cli = _make_cli(streaming_enabled=True)
        with patch("hermes_cli.plugins.has_hook", return_value=False):
            assert cli._transform_llm_output_hook_active() is False

    def test_gate_fail_open_on_plugin_error(self):
        cli = _make_cli(streaming_enabled=True)
        with patch("hermes_cli.plugins.has_hook", side_effect=RuntimeError("boom")):
            assert cli._transform_llm_output_hook_active() is False


class TestResolveStreamDeltaCallback:
    """The actual wiring decision — invoked from _init_agent at line ~581."""

    def test_hook_active_suppresses_streaming(self):
        cli = _make_cli(streaming_enabled=True)
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            assert cli._resolve_stream_delta_callback() is None

    def test_no_hook_streams_normally(self):
        cli = _make_cli(streaming_enabled=True)
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=False
        ):
            assert cli._resolve_stream_delta_callback() is cli._stream_delta

    def test_streaming_disabled_still_none(self):
        cli = _make_cli(streaming_enabled=False)
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=False
        ):
            assert cli._resolve_stream_delta_callback() is None

    def test_hook_active_with_streaming_disabled_still_none(self):
        cli = _make_cli(streaming_enabled=False)
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            assert cli._resolve_stream_delta_callback() is None
