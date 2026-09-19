"""Regression tests for the #102203 buffered fix — per-token hook gate.

When a transform_llm_output hook is registered, the CLI must NOT stream
token-by-token: a mutating transform is applied AFTER streaming, so any
post-hoc print (suffix or banner) lands after already-rendered bytes and
garbles the terminal (see PR #102216 Chunk5 review).

The wiring decision happens in
``CLIAgentSetupMixin._resolve_stream_delta_callback`` (called once from
``_init_agent``), which now returns a per-token wrapper
(``_stream_delta_with_hook_gate``) instead of a baked yes/no value. That
means a hook registered AFTER ``_init_agent`` — plugin discovery running
lazily, ``/plugins enable``, programmatic registration — still suppresses
streaming from the moment it goes live; we don't need to re-init the agent.

Invariants the tests pin:

- ``_resolve_stream_delta_callback`` no longer consults the hook; it only
  honours ``streaming_enabled``.
- The wrapper forwards ``text=None`` unconditionally: this marks a turn
  boundary and MUST reach ``_stream_delta`` so its ``_flush_stream()`` /
  ``_reset_stream_state()`` path runs even when a hook suppressed every
  visible token of a prior turn.
- The wrapper drops visible text ONLY when the hook predicate is active.
- The wrapper delegates to the predicate via ``self.`` and uses the
  ``_stream_delta`` attribute dynamically, so tests that patch either on the
  class see the same behaviour as production.

Pinned against origin/main 63279301bc: the tests must FAIL on main (the
resolver and wrapper do not exist there).
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
    """The predicate behind the gate — consulted per token by the wrapper."""

    def test_gate_true_when_hook_registered(self):
        cli = _make_cli(streaming_enabled=True)
        with patch("hermes_cli.plugins.has_mutating_hook", return_value=True):
            assert cli._transform_llm_output_hook_active() is True

    def test_gate_false_without_hook(self):
        cli = _make_cli(streaming_enabled=True)
        with patch("hermes_cli.plugins.has_mutating_hook", return_value=False):
            assert cli._transform_llm_output_hook_active() is False

    def test_gate_fail_open_on_plugin_error(self):
        cli = _make_cli(streaming_enabled=True)
        with patch(
            "hermes_cli.plugins.has_mutating_hook", side_effect=RuntimeError("boom")
        ):
            assert cli._transform_llm_output_hook_active() is False


class TestResolveStreamDeltaCallback:
    """The wiring decision — must not bake the hook verdict into the agent."""

    def test_returns_wrapper_when_streaming_enabled(self):
        cli = _make_cli(streaming_enabled=True)
        cb = cli._resolve_stream_delta_callback()
        # Identity: the SAME function that runs the per-token gate.
        assert cb == cli._stream_delta_with_hook_gate

    def test_returns_none_when_streaming_disabled(self):
        cli = _make_cli(streaming_enabled=False)
        assert cli._resolve_stream_delta_callback() is None

    def test_hook_state_at_init_does_not_change_returned_callback(self):
        """Late-registration safety: the resolver must NOT consult the hook.

        A hook registered after ``_init_agent`` must not need the agent to be
        re-initialised for the gate to engage. The wrapper converges on the
        right behaviour the moment the hook shows up.
        """
        cli = _make_cli(streaming_enabled=True)
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=False
        ):
            cb_no_hook = cli._resolve_stream_delta_callback()
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            cb_with_hook = cli._resolve_stream_delta_callback()
        assert cb_no_hook == cb_with_hook  # same wrapper either way


class TestStreamDeltaWithHookGate:
    """The per-token gate itself."""

    def test_forwards_text_when_no_hook(self):
        cli = _make_cli(streaming_enabled=True)
        seen: list = []
        cli._stream_delta = seen.append
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=False
        ):
            cli._stream_delta_with_hook_gate("hello")
        assert seen == ["hello"]

    def test_drops_text_when_hook_active(self):
        cli = _make_cli(streaming_enabled=True)
        seen: list = []
        cli._stream_delta = seen.append
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            cli._stream_delta_with_hook_gate("hello")
        assert seen == []

    def test_none_always_forwarded_regardless_of_hook(self):
        """Turn-boundary None must flush/reset even with a hook active."""
        cli = _make_cli(streaming_enabled=True)
        seen: list = []
        cli._stream_delta = seen.append
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            cli._stream_delta_with_hook_gate(None)
        assert seen == [None]

    def test_hook_late_registration_engages_gate(self):
        """The motivating case: hook arrives between agent init and turn N."""
        cli = _make_cli(streaming_enabled=True)
        seen: list = []
        cli._stream_delta = seen.append
        cb = cli._resolve_stream_delta_callback()
        # First turn: no hook → text flows.
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=False
        ):
            cb("early-token")
        # Late register (simulated by flipping the predicate mid-session).
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            cb("late-token")
        assert seen == ["early-token"]

    def test_safe_when_stream_delta_is_none(self):
        """No display callback wired → wrapper becomes a no-op, not a crash."""
        cli = _make_cli(streaming_enabled=True)
        cli._stream_delta = None
        with patch.object(
            CLIAgentSetupMixin, "_transform_llm_output_hook_active", return_value=True
        ):
            # Must not raise even when both _stream_delta is None AND hook is on.
            cli._stream_delta_with_hook_gate("anything")
            cli._stream_delta_with_hook_gate(None)
