from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


class _Buffer:
    def __init__(self):
        self.reset = MagicMock()


class _App:
    def __init__(self):
        self.current_buffer = _Buffer()
        self.invalidate = MagicMock()
        self.exit = MagicMock()


def _event():
    return SimpleNamespace(app=_App())


def _cli(**overrides):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = MagicMock()
    cli._agent_running = False
    cli._secret_state = None
    cli._sudo_state = None
    cli._approval_state = None
    cli._clarify_state = None
    cli._slash_confirm_state = None
    cli._model_picker_state = None
    cli._last_ctrl_c_time = 0
    cli._should_exit = False
    for key, value in overrides.items():
        setattr(cli, key, value)
    return cli


def test_escape_interrupts_running_agent_through_shared_prompt_path():
    cli = _cli(_agent_running=True)
    event = _event()

    handled = cli._handle_escape_interrupt(event)

    assert handled is True
    cli.agent.interrupt.assert_called_once_with()
    event.app.invalidate.assert_called_once_with()
    event.app.exit.assert_not_called()


def test_escape_does_not_interrupt_when_modal_prompt_has_priority():
    cli = _cli(_agent_running=True, _secret_state={"response_queue": MagicMock()})
    event = _event()

    handled = cli._handle_escape_interrupt(event)

    assert handled is False
    cli.agent.interrupt.assert_not_called()
    event.app.invalidate.assert_not_called()


def test_escape_is_noop_when_idle():
    cli = _cli(_agent_running=False)
    event = _event()

    handled = cli._handle_escape_interrupt(event)

    assert handled is False
    cli.agent.interrupt.assert_not_called()
    event.app.exit.assert_not_called()
    event.app.current_buffer.reset.assert_not_called()
