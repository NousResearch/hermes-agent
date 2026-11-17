"""Platform-neutral mode control and fail-closed work correlation."""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session_parallel import binding_of, route_input
from tests.gateway.test_parallel_native_ingress import (
    MESSAGING_PLATFORMS, DiscordFixture, admissions, event, setup_owner,
)
from tests.waiting import async_wait_until


@pytest.mark.parametrize('platform', MESSAGING_PLATFORMS, ids=lambda p: p.value)
@pytest.mark.asyncio
async def test_busy_parallel_enables_messaging_without_platform_allowlist(monkeypatch, platform):
    import cli
    saved = []
    monkeypatch.setattr(cli, 'save_config_value', lambda key, value: saved.append((key, value)) or True)
    runner = GatewayRunner(GatewayConfig())
    monkeypatch.setattr(runner, '_busy_profile_name_for_source', lambda source: '')
    adapter = DiscordFixture()
    adapter.platform = platform
    await runner._handle_busy_command(event(adapter, 'enable', '/busy parallel'))
    assert saved == [('display.busy_input_mode', 'parallel')]
    assert runner._effective_busy_input_mode(adapter.build_source(chat_id='room', user_id='alice')) == 'parallel'


@pytest.mark.parametrize('platform', MESSAGING_PLATFORMS, ids=lambda p: p.value)
@pytest.mark.asyncio
async def test_missing_message_identity_refuses_instead_of_merging_work(monkeypatch, platform):
    async def produce(*args):
        pytest.fail('an unidentifiable request must not execute')

    runner, authority, adapter = await setup_owner(monkeypatch, produce, platform=platform)
    try:
        incoming = event(adapter, '', 'request')
        notice = await route_input(runner, incoming)
        assert 'stable message ID' in notice
        assert binding_of(incoming.source) is None
        assert admissions(authority) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_equal_message_ids_on_different_messengers_do_not_share_work(monkeypatch):
    gate = asyncio.Event()

    async def produce(*args):
        await gate.wait()
        return 'DONE'

    runner, authority, telegram = await setup_owner(monkeypatch, produce, platform=Platform.TELEGRAM)
    slack = DiscordFixture()
    slack.platform = Platform.SLACK
    slack.gateway_runner = runner
    runner.adapters[Platform.SLACK] = slack
    runner._wire_adapter_handlers(slack)
    originals = {}
    try:
        for adapter in [telegram, slack]:
            original = event(adapter, 'same-id', 'request', chat='same-room')
            originals[adapter.platform] = original
            await adapter.handle_message(original)
        await async_wait_until(lambda: len(admissions(authority)) == 2, what='separate messenger work')
        assert len({row['target_session_id'] for row in admissions(authority)}) == 2
        await slack.handle_message(event(slack, 'follow', 'follow slack', reply='same-id', chat='same-room'))
        await async_wait_until(lambda: len(admissions(authority)) == 3, what='slack follow-up admitted')
        follow = next(row for row in admissions(authority) if row['request_id'] == 'follow')
        route = runner._session_key_for_source(originals[Platform.SLACK].source)
        assert follow['target_session_id'] == runner.session_store.peek_session_id(route)
    finally:
        gate.set()
        await async_wait_until(lambda: not telegram._background_tasks and not slack._background_tasks, what='messenger cleanup')
        authority.db.close()

