"""A refused side task (/btw, /bg) must read as one line, not as a crash.

The gateway refuses a side task whose socket holds no subscription for the session
(``permission_denied``). Before, that refusal escaped ``_handle_btw_command`` /
``_handle_background_command`` as an unhandled ``GatewayRPCError``: the prompt_toolkit key
handler printed a full traceback and the CLI looked broken. The refusal is the gateway's
answer about the session, so it is reported in words.
"""

from __future__ import annotations

from unittest.mock import patch

# Imported at module scope (like the other CLI tests): the module import must happen
# before the test run's home guard installs.
from cli import HermesCLI

from hermes_cli.gateway_client import GatewayRPCError


def _make_cli():
    """Bare HermesCLI (no __init__): the side-task handlers read only these attributes."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "local-side"
    cli._gateway_session_open = True
    cli._background_task_counter = 0
    return cli


class _RefusingConnection:
    """A gateway connection whose side verbs are refused (no subscription for the session)."""

    final = False

    def call_tagged(self, *_args, **_kwargs):
        raise GatewayRPCError("permission_denied")


def _rendered(mock_print):
    return " ".join(str(arg) for call in mock_print.call_args_list for arg in call.args)


def _refuse_every_side_verb(monkeypatch):
    from hermes_cli import gateway_side

    monkeypatch.setattr(gateway_side, "connection_for", lambda _cli: _RefusingConnection())
    monkeypatch.setattr(gateway_side, "start", lambda *a, **k: (_ for _ in ()).throw(
        GatewayRPCError("permission_denied")))


def test_btw_refusal_prints_one_line_instead_of_raising(monkeypatch):
    _refuse_every_side_verb(monkeypatch)
    cli = _make_cli()

    with patch("cli._cprint") as mock_print:
        cli._handle_btw_command("/btw what did I just say?")

    rendered = _rendered(mock_print)
    assert "/btw" in rendered
    assert "no longer attached" in rendered  # what it means and what to do
    assert "Traceback" not in rendered


def test_bg_refusal_prints_one_line_instead_of_raising(monkeypatch):
    _refuse_every_side_verb(monkeypatch)
    cli = _make_cli()

    with patch("cli._cprint") as mock_print:
        cli._handle_background_command("/bg do work")

    rendered = _rendered(mock_print)
    assert "/bg" in rendered
    assert "no longer attached" in rendered
    assert "Traceback" not in rendered
