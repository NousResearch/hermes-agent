"""Bot Chat retains canonical addressing and refuses arbitrary extra recipient fields."""
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


@pytest.mark.parametrize('option', [
    {'deliver_extra': {'chat_id': 'other'}},
    {'deliver_extra': {'session_id': 'other'}},
    {'deliver': 'bot-chat:other/session'},
])
def test_static_and_dynamic_routes_refuse_unsupported_bot_chat_options(option):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'valid-secret'}))
    route = {'deliver': 'bot-chat', **option}
    with pytest.raises(ValueError, match='bot-chat'):
        adapter._validate_route('fixture', route)
    assert not adapter._dynamic_route_allowed('fixture', route)


def test_unrelated_destination_keeps_existing_behavior():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'valid-secret'}))
    route = {'deliver': 'telegram', 'deliver_extra': {'chat_id': 'other'}}
    adapter._validate_route('fixture', route)
    assert adapter._dynamic_route_allowed('fixture', route)


def test_native_creators_refuse_bot_chat_target_before_writing(capsys):
    from argparse import Namespace
    from hermes_cli.webhook import _cmd_subscribe
    from tools.mcp_tool_events import _subscribe_args_error

    args = Namespace(name='fixture', deliver='bot-chat', deliver_chat_id='other')
    assert _cmd_subscribe(args) == 1
    assert 'bot-chat' in capsys.readouterr().out
    assert 'bot-chat' in _subscribe_args_error({'deliver': 'bot-chat', 'deliver_chat_id': 'other'})


@pytest.mark.asyncio
async def test_web_creator_refuses_explicit_target(monkeypatch):
    from fastapi import HTTPException
    from hermes_cli.web_models import WebhookCreate
    from hermes_cli.web_routers import ops

    monkeypatch.setattr('hermes_cli.webhook._is_webhook_enabled', lambda: True)
    body = WebhookCreate(name='fixture', deliver='bot-chat', deliver_chat_id='other')
    with pytest.raises(HTTPException) as exc:
        await ops.create_webhook(body)
    assert exc.value.status_code == 400
    assert 'bot-chat' in exc.value.detail
