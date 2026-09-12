"""Stale-only denial bookkeeping contracts; no consumers or payloads."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from agent.i18n import t
from tools import approval


@pytest.mark.parametrize("args,preserve", [
    ("0" * 32, True),
    ("0" * 32 + " --reason not authorized", True),
    ("unknown-id", True),
    ("unknown-id --reason not authorized", True),
    ("", False),
    ("--reason not authorized", False),
    ("all", False),
    ("ALL --reason not authorized", False),
])
def test_stale_ui_selector_scope(tmp_path, monkeypatch, args, preserve):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(approval, "_gateway_queues", {})

    class Slash(GatewaySlashCommandsMixin):
        def _session_key_for_source(self, source):
            return "fixture-stale-only"

        async def _deliver_approval_confirmation(self, *args):
            raise AssertionError("no confirmation for an absent request")

    slash = Slash()
    record = {"command": "inert obsolete prompt", "metadata": {"keep": [1, 2]}}
    other = {"command": "other session prompt"}
    registry = {"fixture-stale-only": record, "other-session": other}
    before = deepcopy(registry)
    slash._pending_approvals = registry
    event = SimpleNamespace(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="fixture", user_id="fixture"),
        get_command_args=lambda: args,
    )
    reply = asyncio.run(slash._handle_deny_command(event))
    assert slash._pending_approvals is registry
    assert registry["other-session"] is other
    assert not approval._gateway_queues
    if preserve:
        assert registry == before
        assert registry["fixture-stale-only"] is record
        if args.startswith("0" * 32):
            assert reply == t("gateway.deny.no_pending")
        else:
            assert reply == "Usage: /deny [all|exact-request-id] [--reason text]"
    else:
        assert registry == {"other-session": other}
        assert reply == t("gateway.deny.stale")
