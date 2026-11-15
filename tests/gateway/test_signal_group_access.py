"""Signal group access.

``signal.group_allowed_chats`` (config.yaml) lets every member of a listed group talk to the bot even
with a ``SIGNAL_ALLOWED_USERS`` sender allowlist, as on Telegram. It takes the raw group ids that
``SIGNAL_GROUP_ALLOWED_USERS`` lists, while Signal emits ``chat_id="group:<id>"`` (raw id in
``chat_id_alt``). With ``require_mention`` a reply to one of the bot's own messages, or a
``/command``, addresses the bot.
"""

import json
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, load_gateway_config

OWNER, MEMBER = "+15550001111", "+15550002222"
ACCOUNT, ACCOUNT_UUID = "+15551234567", "aaaaaaaa-1111-4222-8333-444444444444"
MEMBER_UUID = "bbbbbbbb-1111-4222-8333-444444444444"


def _gateway(monkeypatch, tmp_path, group_allowed_chats=None):
    """A runner and its Signal adapter built from a config.yaml; every group is monitored."""
    from gateway.run import GatewayRunner
    from plugins.platforms.signal.adapter import SignalAdapter

    home = tmp_path / ".hermes"
    home.mkdir()
    lines = ["signal:", "  enabled: true", "  http_url: http://localhost:8080", f'  account: "{ACCOUNT}"']
    if group_allowed_chats is not None:
        lines.append(f"  group_allowed_chats: {json.dumps(group_allowed_chats)}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", OWNER)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    config = load_gateway_config()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    adapter = SignalAdapter(config.platforms[Platform.SIGNAL])
    adapter.gateway_runner = runner
    adapter.set_authorization_check(lambda *args, **kwargs: True)  # the runner wires one at startup
    runner.adapters = {Platform.SIGNAL: adapter}
    return runner, adapter


async def _group_events(adapter, data_message, group_id="g1"):
    """Events the adapter dispatches for one group message from a member."""
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    await adapter._handle_envelope({"envelope": {
        "sourceNumber": MEMBER, "sourceUuid": MEMBER_UUID, "sourceName": "Member", "timestamp": 1700000000000,
        "dataMessage": {"groupInfo": {"groupId": group_id}, **data_message}}})
    return events


async def _member_is_authorized(runner, adapter, group_id="g1"):
    events = await _group_events(adapter, {"message": "hello"}, group_id)
    assert events, "the adapter dropped a message from a monitored group"
    return runner._is_user_authorized(events[0].source)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_chats", [["g1"], ["other", "g1"], ["*"]])
async def test_a_member_of_a_listed_group_is_authorized(monkeypatch, tmp_path, allowed_chats):
    assert await _member_is_authorized(*_gateway(monkeypatch, tmp_path, allowed_chats))


@pytest.mark.asyncio
async def test_a_member_of_an_unlisted_group_is_refused(monkeypatch, tmp_path):
    assert not await _member_is_authorized(*_gateway(monkeypatch, tmp_path, ["other"]))


@pytest.mark.asyncio
async def test_monitoring_a_group_does_not_authorize_its_members(monkeypatch, tmp_path):
    """``SIGNAL_GROUP_ALLOWED_USERS`` picks the groups the adapter listens to, nothing more."""
    assert not await _member_is_authorized(*_gateway(monkeypatch, tmp_path))

