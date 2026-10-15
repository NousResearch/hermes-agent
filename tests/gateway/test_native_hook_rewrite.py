"""Native hook rewrite qualification for #139045; incident attribution is unconfirmed."""
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authorities import SessionAuthorities
from gateway.session_envelope import prepare_native
from hermes_state_runtime import RuntimeStoreError


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token='fixture'), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id='fixture')

    async def get_chat_info(self, chat_id):
        return {'id': chat_id}


@pytest.mark.asyncio
@pytest.mark.parametrize('action,text,expected', [
    ('allow', None, 'hello'),
    ('rewrite', 'rewritten', 'rewritten'),
    ('rewrite', '', ''),
    ('rewrite', None, 'hello'),
    ('skip', None, None),
    ('raise', None, 'hello'),
])
@pytest.mark.parametrize('profile', [None, 'secondary'])
async def test_primary_native_hook_text_reaches_envelope(tmp_path, monkeypatch, action, text, expected, profile):
    home = tmp_path / 'home'
    secondary = home / 'profiles' / 'secondary'
    secondary.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr('hermes_constants.get_process_hermes_home', lambda: home)
    (home / '.env').write_text('TELEGRAM_ALLOWED_USERS=fixture-user\n')
    (secondary / '.env').write_text('TELEGRAM_ALLOWED_USERS=fixture-user\n')
    config = GatewayConfig(multiplex_profiles=True)
    config._runtime_profile_homes = [('default', home), ('secondary', secondary)]
    runner = GatewayRunner(config)
    registry = SessionAuthorities(home)
    for name, root in config._runtime_profile_homes:
        registry.add(root, SimpleNamespace(profile_id=str(root)), name=name)
    runner.session_authorities = registry
    runner.session_authority = registry.launch
    adapter = Adapter()
    adapter.gateway_runner = runner
    if profile is None:
        runner.adapters[Platform.TELEGRAM] = adapter
    else:
        runner._profile_adapters = {profile: {Platform.TELEGRAM: adapter}}

    async def hook(name, **kwargs):
        if action == 'raise':
            raise ValueError('fixture hook failure')
        return [{'action': action, 'text': text}]

    monkeypatch.setattr('hermes_cli.lifecycle.ainvoke_hook', hook)
    seen = []

    async def capture(event):
        admitted = await runner._hm_admit_event(event)
        if admitted is None:
            assert expected is None
            return
        assert admitted[0] is event
        assert getattr(event, '_turn_marker_handoff') is True
        try:
            seen.append(await prepare_native(runner, admitted[0]))
        except RuntimeStoreError as exc:
            seen.append({'error': exc.reason})

    runner._handle_message = capture
    if profile is None:
        runner._wire_adapter_handlers(adapter)
    else:
        runner._configure_profile_adapter(adapter, profile, Platform.TELEGRAM)
    event = MessageEvent(text='hello', source=adapter.build_source(
        chat_id='fixture-chat', chat_type='dm', user_id='fixture-user'), message_id='one')
    event._turn_marker_handoff = True
    await adapter._message_handler(event)
    if expected is None:
        assert seen == []
        return
    assert seen and 'error' not in seen[0], seen
    assert seen[0]['text'] == expected
    owner_home = secondary if profile else home
    assert seen[0]['native_text_v1']['provenance']['runtime_home'] == str(owner_home.resolve())
    # A rewritten event outside the owning callback still has no admission grant.
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await prepare_native(runner, event)
