"""A chat message refused by a restart's after-turn wait is answered with the drain notice.

While ``request_restart`` holds the gateway in ``draining`` for an in-flight turn (up to
``agent.restart_after_turn_timeout``), the session authority's admission gate refuses every new
messaging input with ``runtime_draining`` (#152410). That refusal must reach the sender as the same
"Gateway is restarting" notice the in-process path gives, not escape ``_handle_message`` as an
exception the adapter turns into the generic "Something went wrong… /retry" turn error. Real
GatewayRunner, DiscordAdapter and SessionAuthority over SQLite.
"""

import pytest

from tests.gateway.test_wake_during_restart_drain_authority import admissions, authority_route


@pytest.mark.asyncio
async def test_message_during_restart_wait_gets_the_drain_notice_not_a_turn_error(monkeypatch):
    from agent.i18n import t
    from gateway.platforms.event import MessageEvent

    executed = []
    runner, _adapter, authority, source = await authority_route(monkeypatch, executed)
    runner._restart_requested = True
    runner._draining = True  # request_restart: refusing new turns while the in-flight one finishes

    reply = await runner._handle_message(MessageEvent(text="are you there?", source=source, message_id="B"))

    assert reply == t("gateway.busy.drain_rejected_new_work", action=runner._status_action_gerund())
    assert executed == [] and admissions(runner, authority, source) == []  # nothing half-admitted
