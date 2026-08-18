"""Plugin Relay marks must be bounded — never an unbounded native call.

The nemo_relay plugin's ``_Runtime.run_in_session`` wrapper serves every
mark/event the plugin emits (turn start/end, approvals, subagent marks) and
runs synchronously on the agent's conversation thread. The host's
``run_in_session`` default (``timeout=None``) is an unbounded native call: a
wedged native Relay pipeline blocked the agent between API calls with zero
activity ticks until the cron 600s inactivity kill fired (observed live
2026-08-15 — two cron jobs and a gateway chat session died with
``last_activity=API call #N completed``).

The wrapper now always passes ``timeout=relay_runtime._SCOPE_OP_TIMEOUT`` to
the host, and a breach flags ``scope_errored`` so ``close_session`` skips the
ATIF export for the wedged session.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from agent import relay_runtime as core_relay_runtime


def _plugin_module():
    mod = sys.modules.get("plugins.observability.nemo_relay")
    if mod is None:
        mod = importlib.import_module("plugins.observability.nemo_relay")
    return mod


def _make_runtime_and_state(host):
    plugin = _plugin_module()
    runtime = object.__new__(plugin._Runtime)
    runtime.host = host
    state = plugin._SessionState(session_id="s-bound")
    state.relay_session = SimpleNamespace(session_id="s-bound")
    return plugin, runtime, state


def test_plugin_marks_pass_bounded_timeout_to_host():
    """Every wrapper dispatch carries the core scope-op budget."""
    seen = {}

    class _Host:
        def run_in_session(self, session, callback, *args, timeout=None, **kwargs):
            seen["timeout"] = timeout
            return callback(*args, **kwargs)

    _plugin, runtime, state = _make_runtime_and_state(_Host())
    result = runtime.run_in_session(state, lambda: "ok")

    assert result == "ok"
    assert seen["timeout"] == core_relay_runtime._SCOPE_OP_TIMEOUT
    assert state.scope_errored is False


def test_timeout_breach_flags_session_and_skips_atif_export():
    """A wedged native call costs one span and disables the session export."""

    class _WedgedHost:
        def run_in_session(self, session, callback, *args, timeout=None, **kwargs):
            raise TimeoutError(
                f"Relay scope operation exceeded {timeout}s (session=s-bound)"
            )

    plugin, runtime, state = _make_runtime_and_state(_WedgedHost())

    with pytest.raises(TimeoutError):
        runtime.run_in_session(state, lambda: "never")

    assert state.scope_errored is True

    # export_atif must now skip without touching the exporter.
    class _ExplodingExporter:
        def export_json(self):  # pragma: no cover - must not be called
            raise AssertionError("export ran for a scope-errored session")

    state.atif_exporter = _ExplodingExporter()
    runtime.settings = plugin._Settings(
        atif_enabled=True, atif_output_directory="/tmp/never-used"
    )
    runtime.export_atif(state)  # no raise, no export


def test_non_timeout_errors_still_flag_session():
    """The pre-existing generic error path keeps its scope_errored contract."""

    class _BrokenHost:
        def run_in_session(self, session, callback, *args, timeout=None, **kwargs):
            raise RuntimeError("scope handle is not at the top of the stack")

    _plugin, runtime, state = _make_runtime_and_state(_BrokenHost())

    with pytest.raises(RuntimeError):
        runtime.run_in_session(state, lambda: "never")

    assert state.scope_errored is True
