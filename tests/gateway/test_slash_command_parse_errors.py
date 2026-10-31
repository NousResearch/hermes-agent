"""A quote in the argument of a messaging command is a named parse error, not a crash.

The bound-session side already classifies this input (``226d854e8d`` for ``/skills`` and
``/kanban``, ``f2fe6f5ef3`` for ``/memory``), and ``/resume`` returns
``gateway.resume.parse_error`` from ``shlex.split``.  The messaging handlers parsed the
same raw text with an unguarded ``shlex.split``, so an unclosed quote escaped as a
``ValueError`` to the message loop's catch-all and the user got a generic internal error
with no command output (#140803).
"""
import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.config import Platform


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="u1"),
        message_id="m1",
    )


def _runner() -> GatewayRunner:
    # The handlers read nothing off the runner before they parse the argument.
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_kanban_with_unclosed_quote_replies_with_a_parse_error(monkeypatch):
    import hermes_cli.kanban as kanban_cli

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("/kanban reached the CLI with an unparseable argument")

    monkeypatch.setattr(kanban_cli, "run_slash", _must_not_run)

    reply = await _runner()._handle_kanban_command(_event("/kanban show it's"))

    assert "Could not parse" in reply
    assert "`/kanban`" in reply
    assert "No closing quotation" in reply


@pytest.mark.asyncio
async def test_curator_with_unclosed_quote_replies_with_a_parse_error(monkeypatch):
    import hermes_cli.curator as curator_cli

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("/curator reached the CLI with an unparseable argument")

    monkeypatch.setattr(curator_cli, "cli_main", _must_not_run)

    reply = await _runner()._handle_curator_command(_event("/curator it's"))

    assert "Could not parse" in reply
    assert "`/curator`" in reply
    assert "No closing quotation" in reply
