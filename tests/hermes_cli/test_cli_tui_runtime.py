"""Tests for classic-TUI runtime hooks.

Focus:
- _tui_after_turn() invokes the active memory provider's runtime health probe.
- True / False / None / exception are mapped to MemoryHealthState correctly.
- Failed probes enter the cooldown window and are not immediately retried.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.memory_health import get_health_state, reset_health_state
from cli import HermesCLI


@pytest.fixture(autouse=True)
def _clean_memory_health_state():
    """Reset the process-wide memory health singleton around every test."""
    reset_health_state()
    yield
    reset_health_state()


def _make_cli_for_after_turn(provider):
    """Build the minimum HermesCLI object required by _tui_after_turn()."""
    cli = HermesCLI.__new__(HermesCLI)

    cli._agent_running = True
    cli._pet_reasoning = False
    cli._spinner_text = ""
    cli._last_scrollback_tool = ""
    cli._tool_start_time = 0.0
    cli._pending_tool_info = {}
    cli._interactive_turn = True
    cli._last_turn_interrupted = False

    cli._app = SimpleNamespace(invalidate=Mock())

    cli._pet_react_turn_end = Mock()
    cli._turn_summary_emit = Mock()
    cli._drain_interrupt_queue_to_pending_input = Mock()
    cli._maybe_continue_goal_after_turn = Mock()
    cli._maybe_complete_loop_tick_after_turn = Mock()
    cli._drain_process_notifications = Mock()

    cli._voice_mode = False
    cli._voice_continuous = False
    cli._voice_recording = False

    cli.agent = SimpleNamespace(
        _memory_manager=SimpleNamespace(
            get_provider=Mock(return_value=provider),
        )
    )

    return cli


class TestTuiAfterTurnMemoryHealth:
    """Runtime health-probe wiring in _tui_after_turn()."""

    def test_probe_true_marks_memory_healthy(self):
        """A successful runtime probe marks the active provider healthy."""
        hs = get_health_state()
        hs.active_provider = "test_provider"

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=True),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_called_once()
        assert hs.health == "healthy"
        assert hs.reason == ""

    def test_probe_false_marks_memory_unavailable(self):
        """A failed runtime probe marks the active provider unavailable."""
        hs = get_health_state()
        hs.active_provider = "test_provider"

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=False),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_called_once()
        assert hs.health == "unavailable"
        assert hs.reason == "test_provider backend unreachable"
        assert hs.probe_cooldown_active() is True

    def test_probe_none_keeps_existing_state(self):
        """Providers without runtime probing support must leave health unchanged."""
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_healthy()

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=None),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_called_once()
        assert hs.health == "healthy"
        assert hs.reason == ""

    def test_probe_exception_marks_memory_unavailable(self):
        """A probe exception marks the active provider unavailable."""
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_healthy()

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(side_effect=RuntimeError("connection refused")),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_called_once()
        assert hs.health == "unavailable"
        assert hs.reason == "test_provider probe failed"
        assert hs.probe_cooldown_active() is True

    def test_failed_probe_enters_cooldown(self):
        """A failed probe must not run again during the 30-second cooldown."""
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_unavailable("backend unreachable")
        hs.record_probe_failure()

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=True),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_not_called()
        assert hs.health == "unavailable"

    def test_provider_without_probe_health_is_left_unchanged(self):
        """Providers that use the default probe_health() -> None must remain unchanged."""
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_unavailable("previous failure")

        provider = SimpleNamespace(
            name="test_provider",
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        assert hs.health == "unavailable"
        assert hs.reason == "previous failure"

    def test_no_active_provider_does_not_probe(self):
        """No active external provider means _tui_after_turn() must not probe anything."""
        hs = get_health_state()
        assert hs.active_provider == ""

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=True),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_not_called()
        assert hs.health == "unknown"

    def test_probe_true_recovers_from_unavailable(self):
        """§26 auto-recovery: a successful probe restores healthy — and a
        failure that did NOT arm the cooldown (operation failure) must not
        suppress the recovery probe."""
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_unavailable("backend unreachable")  # operation failure: no cooldown
        assert hs.probe_cooldown_active() is False

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=True),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        provider.probe_health.assert_called_once()  # recovery probe ran
        assert hs.health == "healthy"
        assert hs.reason == ""


class TestAfterTurnUiRefresh:
    """Review: the status bar repaints only on UI events — a probe state
    change must invalidate immediately, without a keypress."""

    def test_state_change_refreshes_status_ui(self):
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_healthy()

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=False),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        assert hs.health == "unavailable"
        # >= 2: the pre-existing turn-level refresh (before the probe) plus
        # the state-change refresh (after it) — the bar must repaint after
        # the probe updates state, not only before.
        assert cli._app.invalidate.call_count >= 2

    def test_no_state_change_does_not_refresh(self):
        hs = get_health_state()
        hs.active_provider = "test_provider"
        hs.mark_healthy()

        provider = SimpleNamespace(
            name="test_provider",
            probe_health=Mock(return_value=True),
        )
        cli = _make_cli_for_after_turn(provider)

        HermesCLI._tui_after_turn(cli)

        assert hs.health == "healthy"
        # Exactly 1: only the pre-existing turn-level refresh — no extra
        # repaint when the probe leaves health unchanged.
        assert cli._app.invalidate.call_count == 1


class TestTuiStartupPrewarmMemoryHealth:
    """Startup prewarm writes configured_provider only — sentinel-filtered, no probe."""

    def _run_prewarm(self, memory_section):
        from unittest.mock import patch as _patch
        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {}
        cli._console_print = Mock()
        with _patch("hermes_cli.config.load_config", return_value={"memory": memory_section}):
            HermesCLI._tui_startup_prewarm_and_warnings(cli, None)

    def test_prewarm_ignores_core_sentinel_provider(self):
        """Frozen §16: config provider "builtin" must not render as external."""
        hs = get_health_state()
        self._run_prewarm({"provider": "builtin"})
        assert hs.configured_provider == ""
        assert hs.indicator_text() == ""

    def test_prewarm_sets_external_configured_provider(self):
        """External provider is shown from startup with health unknown (§9:
        prewarm never probes — the agent does not exist yet)."""
        hs = get_health_state()
        self._run_prewarm({"provider": "hindsight"})
        assert hs.configured_provider == "hindsight"
        assert hs.health == "unknown"
        assert hs.indicator_text() == "hindsight ○ Connecting"