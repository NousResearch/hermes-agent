"""A forwarded slash command's refusal is printed as the owner explained it.

``forward`` special-cased only ``unsupported_command`` and printed the parity text, dropping the
owner's ``data.detail`` — so every refusal either read as a generic "not available yet" or (for any
other code) reached the panel as ``Error: permission_denied``. The owner now names where a command
runs; the CLI prints that (#153927).
"""
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_chat_commands import forward
from hermes_cli.gateway_client import GatewayRPCError


class _Client:
    def __init__(self, error):
        self.error = error

    async def rpc(self, *args, **kwargs):
        raise self.error


def _view(error):
    return SimpleNamespace(client=_Client(error), session_id="local-s")


@pytest.mark.asyncio
async def test_the_owner_hint_is_printed_instead_of_the_bare_refusal(capsys):
    hint = "/review runs in the telegram chat that owns this session, not from this viewer"
    error = GatewayRPCError("unsupported_command",
                            data={"reason": "unsupported_command", "detail": hint})
    await forward(_view(error), "review", "the PR as a whole")
    printed = capsys.readouterr().out.strip()
    assert printed == hint, printed
    assert "unsupported_command" not in printed


@pytest.mark.asyncio
async def test_a_refusal_without_a_hint_keeps_the_parity_text(capsys):
    await forward(_view(GatewayRPCError("unsupported_command")), "review", "")
    printed = capsys.readouterr().out
    assert "/review is not available on the shared gateway yet" in printed, printed


@pytest.mark.asyncio
async def test_an_unknown_command_without_a_hint_still_says_unknown(capsys):
    await forward(_view(GatewayRPCError("unsupported_command")), "no-such-command", "")
    assert "Unknown command: /no-such-command" in capsys.readouterr().out
