"""Regression tests for the idle-path double Ctrl+C guard (PR #90874).

First Ctrl+C at an empty prompt (no agent running, no text/images) must
arm a 2s timer instead of exiting; a second press within the window
force-exits. The idle and agent-running state machines are independent,
and the disarm contract is symmetric: every intervening Ctrl+C press
disarms the other path's timer (cancel/clear branches disarm both), so
only consecutive same-path presses arm-then-exit.

``_handle_ctrl_c`` was extracted from the ``run()`` keybinding closure so
the branches are reachable from unit tests with a mocked ``event.app``.

The complementary half of the contract — a new agent run starts both
timers clean via ``_disarm_ctrl_c_timers()`` at the single
``_agent_running = True`` site in ``run()`` — lives in the run loop; the
helper itself is pinned by ``test_disarm_ctrl_c_timers_disarms_both`` and
the call site is covered by inspection. Likewise the ``@kb.add('c-c')``
binding site (a one-line delegate to ``_handle_ctrl_c``) is covered by
inspection: the tests drive the method directly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cli():
    """Build a HermesCLI instance with prompt_toolkit stubbed out.

    Returns ``(cli_module, cli_instance)``. The module is returned so tests
    can patch module-level names (``request_hard_interrupt``,
    ``threading``) on the exact object the instance uses. Note: ``reload``
    re-executes cli.py in place, so the stubbed prompt_toolkit bindings
    stick to that module object for the rest of the session — this is the
    established pattern of ``test_cli_interrupt_drain_regression.py`` /
    ``test_cli_steer_busy_path.py`` and is NOT a guarantee of isolation;
    patching by name would additionally hit a *fresh* module the instance
    never closes over, which is why we hold the module object itself.
    """
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod, _cli_mod.HermesCLI()


@pytest.fixture()
def cli():
    """(cli_module, HermesCLI) with the __init__ timer contract pinned.

    Asserts rather than assigns: if a timer attribute were ever dropped
    from ``__init__``, the real keybinding path would raise AttributeError
    on the first Ctrl+C — the fixture must fail here, not mask it.
    """
    mod, c = _make_cli()
    assert c._last_ctrl_c_time == float("-inf")  # never-armed sentinel
    assert c._last_idle_ctrl_c_time == float("-inf")
    assert c._should_exit is False
    # Branch-guard attributes must come from __init__ too: if one were
    # dropped, the real keybinding path AttributeErrors on the first
    # Ctrl+C while these tests stay green (tests below assign to them).
    assert c._voice_recording is False
    assert c._voice_recorder is None
    assert c._voice_continuous is False
    assert c._slash_confirm_state is None
    assert c._model_picker_state is None
    assert c._sudo_state is None
    assert c._secret_state is None
    assert c._approval_state is None
    assert c._clarify_state is None
    assert c._agent_running is False
    assert c.agent is None
    assert c._voice_lock is not None
    assert c._attached_images == []
    return mod, c


def _fake_event(text: str = ""):
    app = SimpleNamespace(
        current_buffer=SimpleNamespace(text=text, reset=MagicMock()),
        invalidate=MagicMock(),
        exit=MagicMock(),
    )
    return SimpleNamespace(app=app)


def _press(c, t: float, text: str = ""):
    """Run one Ctrl+C press at fake time ``t``; returns the fake event."""
    event = _fake_event(text)
    with patch("time.monotonic", return_value=t):
        c._handle_ctrl_c(event)
    return event


class TestIdleDoublePress:
    """The empty-prompt state machine: arm -> arm-then-exit within 2s."""

    def test_single_idle_press_arms_timer_without_exiting(self, cli):
        mod, c = cli
        with patch.object(mod, "_cprint") as cprint:
            event = _press(c, t=1000.0)

        assert c._last_idle_ctrl_c_time == 1000.0
        assert c._should_exit is False
        event.app.exit.assert_not_called()
        event.app.invalidate.assert_called_once()
        # The user-visible arm hint is the point of the PR: without it a
        # first press would look like a silent no-op.
        cprint.assert_called_once()
        assert "Press Ctrl+C again to exit." in cprint.call_args.args[0]

    def test_second_idle_press_within_window_force_exits(self, cli, capsys):
        _, c = cli
        _press(c, t=1000.0)
        event = _press(c, t=1001.0)

        assert c._should_exit is True
        event.app.exit.assert_called_once()
        event.app.invalidate.assert_not_called()
        assert "Force exiting..." in capsys.readouterr().out

    def test_idle_press_at_exactly_2s_rearms_instead_of_exiting(self, cli):
        """The window is strict (< 2.0s): a press at t+2.0s must not exit.

        The literals are deliberately exact: 1000.0 and 1002.0 are both
        exactly representable in IEEE-754, so the difference is exactly
        2.0 and the strict-inequality check is exercised without float
        flakiness.
        """
        _, c = cli
        _press(c, t=1000.0)
        event = _press(c, t=1002.0)

        assert c._should_exit is False
        assert c._last_idle_ctrl_c_time == 1002.0
        event.app.exit.assert_not_called()
        event.app.invalidate.assert_called_once()

    def test_idle_press_just_under_2s_force_exits(self, cli):
        """The other side of the boundary: 1001.999 - 1000.0 < 2.0 exits."""
        _, c = cli
        _press(c, t=1000.0)
        event = _press(c, t=1001.999)

        assert c._should_exit is True
        event.app.exit.assert_called_once()

    def test_boot_window_first_idle_press_arms_not_exits(self, cli):
        """time.monotonic() starts at boot: within the first 2s of uptime
        the clock is in [0, 2.0). The never-armed sentinel is -inf, so the
        very first press must ARM — with a 0 sentinel it would force-exit."""
        _, c = cli
        event = _press(c, t=0.5)  # fresh instance, boot window

        assert c._should_exit is False
        assert c._last_idle_ctrl_c_time == 0.5
        event.app.exit.assert_not_called()

        # And the double-press still works inside the boot window.
        event = _press(c, t=1.0)
        assert c._should_exit is True
        event.app.exit.assert_called_once()


class TestInterveningPressDisarms:
    """Any non-empty-prompt Ctrl+C press must disarm the other path's timer.

    Cancel/clear branches disarm both timers; the arm branches (idle arm,
    agent interrupt) arm their own path and disarm only the other. Either
    way, a stale arm from one path can never combine with a later press on
    the other path into an accidental force-exit.
    """

    def test_disarm_ctrl_c_timers_disarms_both(self, cli):
        """The helper is the only place both timers are disarmed together
        after construction (__init__ sets the initial sentinels) — pinned
        so a future edit can't drop one half of the pair."""
        _, c = cli
        c._last_ctrl_c_time = 999.5
        c._last_idle_ctrl_c_time = 999.5

        c._disarm_ctrl_c_timers()

        assert c._last_ctrl_c_time == float("-inf")
        assert c._last_idle_ctrl_c_time == float("-inf")

    def test_buffer_clear_disarms_both_timers(self, cli):
        _, c = cli
        c._last_idle_ctrl_c_time = 999.5  # stale idle arm
        c._last_ctrl_c_time = 999.7  # stale agent arm
        event = _press(c, t=1000.0, text="typed text")  # buffer-clear branch

        assert c._last_idle_ctrl_c_time == float("-inf")
        assert c._last_ctrl_c_time == float("-inf")
        event.app.current_buffer.reset.assert_called_once()  # "like bash"

    def test_images_only_clear_disarms_both_timers_and_clears_images(self, cli):
        _, c = cli
        c._last_idle_ctrl_c_time = 999.5
        c._last_ctrl_c_time = 999.7
        c._attached_images = [Path("img.png")]
        event = _press(c, t=1000.0)  # empty buffer, images present

        assert c._attached_images == []
        assert c._last_idle_ctrl_c_time == float("-inf")
        assert c._last_ctrl_c_time == float("-inf")
        event.app.current_buffer.reset.assert_called_once()

    def test_slash_confirm_cancel_disarms_both_timers(self, cli):
        _, c = cli
        c._last_idle_ctrl_c_time = 999.5
        c._last_ctrl_c_time = 999.7
        c._slash_confirm_state = {"prompt": "confirm?"}
        with patch.object(c, "_submit_slash_confirm_response") as submit:
            _press(c, t=1000.0)

        submit.assert_called_once_with("cancel")
        assert c._last_idle_ctrl_c_time == float("-inf")
        assert c._last_ctrl_c_time == float("-inf")

    def test_model_picker_cancel_disarms_both_timers(self, cli):
        _, c = cli
        c._last_idle_ctrl_c_time = 999.5
        c._last_ctrl_c_time = 999.7
        c._model_picker_state = {"open": True}
        with patch.object(c, "_close_model_picker") as close:
            _press(c, t=1000.0)

        close.assert_called_once()
        assert c._last_idle_ctrl_c_time == float("-inf")
        assert c._last_ctrl_c_time == float("-inf")

    def test_voice_cancel_disarms_both_timers_and_uses_daemon_thread(self, cli):
        mod, c = cli
        c._last_idle_ctrl_c_time = 999.5
        c._last_ctrl_c_time = 999.7
        c._voice_recording = True
        c._voice_continuous = True
        c._voice_recorder = MagicMock()
        with patch.object(mod, "threading") as mock_threading:
            _press(c, t=1000.0)

        assert c._last_idle_ctrl_c_time == float("-inf")
        assert c._last_ctrl_c_time == float("-inf")
        # "Don't block the event loop" intent: cancel runs on a daemon thread.
        mock_threading.Thread.assert_called_once_with(
            target=c._voice_recorder.cancel, daemon=True
        )
        mock_threading.Thread.return_value.start.assert_called_once()

    @pytest.mark.parametrize(
        "state_attr",
        ["_sudo_state", "_secret_state", "_approval_state", "_clarify_state"],
    )
    def test_overlay_clear_without_agent_disarms_both_timers(self, cli, state_attr):
        _, c = cli
        c._last_idle_ctrl_c_time = 999.5
        c._last_ctrl_c_time = 999.7
        setattr(c, state_attr, {"response_queue": MagicMock()})
        with patch.object(c, "_clear_active_overlays_for_interrupt") as clear:
            _press(c, t=1000.0)

        clear.assert_called_once()
        assert c._last_idle_ctrl_c_time == float("-inf")
        assert c._last_ctrl_c_time == float("-inf")

    def test_agent_interrupt_press_disarms_idle_timer(self, cli):
        mod, c = cli
        _press(c, t=1000.0)  # idle arm
        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(mod, "request_hard_interrupt") as hard_interrupt:
            _press(c, t=1000.5)  # agent-interrupt press
        assert c._last_idle_ctrl_c_time == float("-inf")  # disarmed by the interrupt
        hard_interrupt.assert_called_once_with(c.agent)

        # Back at the empty prompt, a quick press must ARM, not exit.
        c._agent_running = False
        c.agent = None
        event = _press(c, t=1001.0)
        assert c._should_exit is False
        assert c._last_idle_ctrl_c_time == 1001.0
        event.app.exit.assert_not_called()

    def test_overlay_with_running_agent_clears_and_interrupts(self, cli):
        """#14026 fall-through: one press clears overlays AND interrupts."""
        mod, c = cli
        c._sudo_state = {"response_queue": MagicMock()}
        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(c, "_clear_active_overlays_for_interrupt") as clear, patch.object(
            mod, "request_hard_interrupt"
        ) as hard_interrupt:
            event = _press(c, t=1000.0)

        clear.assert_called_once()
        hard_interrupt.assert_called_once_with(c.agent)
        assert c._last_idle_ctrl_c_time == float("-inf")  # agent branch disarms idle
        assert c._last_ctrl_c_time == 1000.0  # interrupt armed agent path
        assert c._should_exit is False
        event.app.exit.assert_not_called()

        # Second press inside the window (overlay still present, agent still
        # running): must force-exit, not re-interrupt.
        with patch.object(c, "_clear_active_overlays_for_interrupt") as clear2, patch.object(
            mod, "request_hard_interrupt"
        ) as hard_interrupt2:
            event = _press(c, t=1000.5)

        clear2.assert_called_once()
        hard_interrupt2.assert_not_called()
        assert c._should_exit is True
        event.app.exit.assert_called_once()


class TestAgentPathIndependence:
    """The agent-running double-press machine is untouched and separate."""

    def test_agent_path_force_exit_uses_own_timer(self, cli):
        mod, c = cli
        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(mod, "request_hard_interrupt"):
            _press(c, t=1000.0)  # first press: interrupt
        assert c._last_ctrl_c_time == 1000.0
        assert c._last_idle_ctrl_c_time == float("-inf")

        event = _press(c, t=1001.0)  # second press: force exit
        assert c._should_exit is True
        event.app.exit.assert_called_once()

    def test_agent_path_press_at_exactly_2s_interrupts_instead_of_exiting(self, cli):
        """Agent window is strict (< 2.0s) too: a press at t+2.0s must
        interrupt, not force-exit (mirror of the idle boundary test)."""
        mod, c = cli
        c._agent_running = True
        c.agent = MagicMock()
        c._last_ctrl_c_time = 998.0  # armed exactly 2.0s ago
        with patch.object(mod, "request_hard_interrupt") as hard_interrupt:
            event = _press(c, t=1000.0)

        hard_interrupt.assert_called_once_with(c.agent)
        assert c._should_exit is False
        event.app.exit.assert_not_called()

    def test_boot_window_first_agent_press_interrupts_not_exits(self, cli):
        """Boot-window variant for the agent path: with a 0 sentinel the
        first press against a fresh run inside 2s of boot would force-exit."""
        mod, c = cli
        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(mod, "request_hard_interrupt") as hard_interrupt:
            event = _press(c, t=0.5)

        hard_interrupt.assert_called_once_with(c.agent)
        assert c._should_exit is False
        event.app.exit.assert_not_called()

    def test_idle_arm_does_not_leak_into_agent_path(self, cli):
        mod, c = cli
        _press(c, t=1000.0)  # idle arm
        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(mod, "request_hard_interrupt") as hard_interrupt:
            _press(c, t=1000.5)  # agent first press: must INTERRUPT, not exit

        hard_interrupt.assert_called_once_with(c.agent)
        assert c._should_exit is False
        assert c._last_ctrl_c_time == 1000.5

    def test_idle_press_disarms_stale_agent_arm(self, cli):
        """Symmetric leak: interrupt -> agent ends -> idle press -> new run.

        A stale agent arm must be disarmed by the idle press, so the first
        Ctrl+C against the NEW agent run interrupts instead of force-exiting
        the session.
        """
        mod, c = cli
        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(mod, "request_hard_interrupt"):
            _press(c, t=1000.0)  # interrupt agent, agent timer = 1000.0
        assert c._last_ctrl_c_time == 1000.0

        c._agent_running = False
        c.agent = None
        _press(c, t=1000.5)  # idle arm: must disarm the stale agent arm
        assert c._last_ctrl_c_time == float("-inf")

        c._agent_running = True
        c.agent = MagicMock()
        with patch.object(mod, "request_hard_interrupt") as hard_interrupt:
            event = _press(c, t=1001.0)  # first press on the new run

        assert c._should_exit is False
        hard_interrupt.assert_called_once_with(c.agent)  # interrupt, NOT exit
        event.app.exit.assert_not_called()
