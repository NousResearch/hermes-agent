"""PoC: duplicate submissions while a turn is unanswered are dropped silently."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli import cli_duplicate_submit
from tests.hermes_cli.test_cli_init import _make_cli


@pytest.fixture
def cli():
    cli = _make_cli()
    cli._attached_images = []
    cli._agent_running = False
    cli._pending_input = MagicMock()
    cli._inline_pastes = MagicMock()
    cli._tui_enter_inline_command = MagicMock(return_value=False)
    cli._tui_enter_while_busy = MagicMock()
    cli._dup_unanswered = None
    cli._dup_last_idle = None
    return cli


def _submit(cli, text):
    buf = MagicMock()
    buf.text = text
    cli._tui_submit_buffer(SimpleNamespace(app=MagicMock()), buf)
    return buf


def _start_turn(cli, text, monkeypatch):
    """Run only the PoC's turn-start seeding; the real turn (super()) is stubbed out."""
    mro = type(cli).__mro__
    parent = mro[mro.index(cli_duplicate_submit.CLIDuplicateSubmitMixin) + 1]
    monkeypatch.setattr(parent, "_tui_process_one_input", lambda self, user_input: None, raising=False)
    cli._tui_process_one_input(text)
    cli._agent_running = True


def test_resending_the_running_prompt_is_dropped(cli, monkeypatch):
    _start_turn(cli, "fix the bug", monkeypatch)
    buf = _submit(cli, "  fix the bug ")
    cli._tui_enter_while_busy.assert_not_called()
    buf.reset.assert_called_once_with(append_to_history=True)


def test_repeated_redirect_is_routed_once(cli, monkeypatch):
    _start_turn(cli, "first", monkeypatch)
    for _ in range(5):
        _submit(cli, "STOP, do X instead")
    assert cli._tui_enter_while_busy.call_count == 1


def test_different_text_while_busy_is_routed(cli, monkeypatch):
    _start_turn(cli, "first", monkeypatch)
    _submit(cli, "a")
    _submit(cli, "b")
    assert cli._tui_enter_while_busy.call_count == 2


def test_next_turn_resets_so_resending_after_the_answer_works(cli, monkeypatch):
    _start_turn(cli, "continue", monkeypatch)
    cli._agent_running = False
    _start_turn(cli, "something else", monkeypatch)
    _submit(cli, "continue")
    assert cli._tui_enter_while_busy.call_count == 1


@pytest.mark.parametrize("text", ["/status", "!ls"])
def test_local_dispatch_is_never_dropped(cli, monkeypatch, text):
    _start_turn(cli, "x", monkeypatch)
    _submit(cli, text)
    _submit(cli, text)
    assert cli._pending_input.put.call_count == 2


def test_images_are_never_dropped(cli, monkeypatch):
    _start_turn(cli, "x", monkeypatch)
    cli._attached_images = ["/tmp/a.png"]
    assert not cli._dup_should_drop("x")


def test_idle_double_enter_is_dropped_but_later_resend_is_not(cli, monkeypatch):
    clock = iter([100.0, 100.5, 110.0])
    monkeypatch.setattr(cli_duplicate_submit.time, "monotonic", lambda: next(clock))
    _submit(cli, "weiter")
    _submit(cli, "weiter")  # 0.5 s later: double Enter
    _submit(cli, "weiter")  # 10 s later: intentional resend
    assert cli._pending_input.put.call_count == 2
