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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


def _make_cli(*, streaming_enabled: bool):
    """A real HermesCLI shell, bound to only what _init_agent reads.

    ``HermesCLI.__new__`` (not CLIAgentSetupMixin()) is deliberate: the
    constructor call is preceded by ``self._agent_status_print(...)``, which
    lives on cli_stream_mixin. A bare mixin raises AttributeError there,
    ``_init_agent`` swallows it in its own ``except Exception``, returns
    False — and the test would assert against a half-initialised agent.
    Recipe follows tests/hermes_cli/test_resume_quiet_stderr.py.
    """
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.streaming_enabled = streaming_enabled
    cli._stream_delta = lambda *_a, **_k: None
    # Blockers: keep _init_agent moving without running real side effects.
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli.finalize_preloaded_skills = lambda: None
    # Session/restore state _init_agent reads before the constructor.
    cli.agent = None
    cli.model = "test/model"
    cli.session_id = "s1"
    cli.max_turns = 10
    cli.verbose = False
    cli.enabled_toolsets = None
    cli.disabled_toolsets = None
    cli.tool_progress_mode = "all"
    cli.reasoning_config = None
    cli.service_tier = None
    cli.system_prompt = None
    cli.prefill_messages = None
    cli.checkpoints_enabled = False
    cli.checkpoint_max_snapshots = 0
    cli.checkpoint_max_total_size_mb = 0
    cli.checkpoint_max_file_size_mb = 0
    cli.pass_session_id = False
    cli.ignore_rules = True
    cli._inline_diffs_enabled = False
    # MagicMock, not None: None sends _init_agent down
    # hermes_state_registry.acquire() and builds a real SQLite DB (~0.9s).
    cli._session_db = MagicMock()
    cli._resumed = False
    cli.conversation_history = []
    cli._providers_only = None
    cli._providers_ignore = None
    cli._providers_order = None
    cli._provider_sort = None
    cli._provider_require_params = False
    cli._provider_data_collection = False
    cli._openrouter_min_coding_score = None
    cli._fallback_model = None
    cli._single_query_mode = False
    cli._clarify_callback = None
    cli._connection_callback = None
    cli._on_tool_progress = None
    cli._on_tool_start = None
    cli._on_tool_complete = None
    cli._on_tool_gen_start = None
    cli._on_notice = None
    cli._on_notice_clear = None
    cli._on_reaction = None
    cli._on_thinking = None
    cli._current_reasoning_callback = lambda: None
    # Read after the constructor: _init_agent applies a pending /title intent
    # once the agent exists.
    cli._pending_title = None
    return cli


_RUNTIME = {
    "api_key": "k",
    "base_url": "https://x/v1",
    "provider": "custom",
    "requested_provider": "custom",
    "api_mode": "chat_completions",
    "command": None,
    "args": None,
    "credential_pool": None,
}


@contextmanager
def _patched_init_agent(capture, *, has_hook):
    """Patch the two module-level seams _init_agent dereferences.

    Everything else is stubbed on the instance by ``_make_cli`` — no
    ``create=True`` needed, because the blockers are real HermesCLI methods.
    """
    with (
        patch("run_agent.AIAgent", capture),
        patch("hermes_cli.plugins.has_hook", return_value=has_hook),
        patch("cli._prepare_deferred_agent_startup", return_value=None),
    ):
        yield


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

    @pytest.mark.parametrize(
        "has_hook, expect_streaming",
        [(False, True), (True, False)],
        ids=["no_hook_streams", "hook_suppresses_streaming"],
    )
    def test_init_agent_wires_resolver_into_constructor(self, has_hook, expect_streaming):
        """The AIAgent(...) call-site must pass the RESOLVER's value.

        The unit tests below invoke ``_resolve_stream_delta_callback``
        directly, so they stay green even if ``_init_agent`` stops calling
        it and re-inlines ``self._stream_delta if self.streaming_enabled
        else None``. Drive the real ``_init_agent`` and read the kwarg the
        constructor was handed. The ``has_hook=True`` row is the one that
        separates the two implementations: only the resolver returns None
        for a registered transform_llm_output hook.
        """
        cli = _make_cli(streaming_enabled=True)

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with _patched_init_agent(_capture, has_hook=has_hook):
            started = cli._init_agent(runtime_override=_RUNTIME)

        # _init_agent swallows its own exceptions and returns False, so the
        # constructor can be reached by a half-failed init. Both guards bite.
        assert started is True, "_init_agent bailed before completing"
        assert "stream_delta_callback" in captured, (
            "_init_agent never reached the AIAgent(...) construction; the "
            "wiring assertion below would be vacuous"
        )
        expected = cli._stream_delta if expect_streaming else None
        assert captured["stream_delta_callback"] is expected, (
            f"has_hook={has_hook}: constructor must receive {expected!r}, got "
            f"{captured['stream_delta_callback']!r}"
        )

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
