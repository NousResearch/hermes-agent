"""All-or-nothing eligibility uses the running runtime, not a future configured model."""

import pytest


from agent.content_steering import ContentSteering
from gateway.session_queued_controls import queued_controls
from hermes_state_runtime import get_session_admission
from tests.gateway.test_native_image_transfer import owner

pytest_plugins = ["tests.gateway.test_native_image_steering"]


@pytest.mark.asyncio
@pytest.mark.parametrize('restriction', [
    'mixed', 'invalid', 'missing-snapshot', 'queue', 'nonvision', 'app-server', 'text-policy',
    'no-control', 'preadoption', 'closed', 'explicit-queue',
])
async def test_unavailable_images_keep_the_complete_admission(setup, monkeypatch, restriction):
    runner, agent, images, _adapter, _key, event, authority, _running = await owner(setup, monkeypatch)
    agent._content_steering = ContentSteering()
    if restriction != 'closed':
        agent._content_steering.open()
    if restriction == 'missing-snapshot':
        from pathlib import Path
        import gateway.run_image_steering as native
        prepare = native._prepare
        def missing(authority, row, agent):
            Path(row['payload']['native_text_v1']['media'][-1]['path']).unlink()
            return prepare(authority, row, agent)
        monkeypatch.setattr(native, '_prepare', missing)
    elif restriction in ('mixed', 'invalid'):
        images[-1].write_bytes(b'not an image')
        event.media_types[-1] = 'application/pdf' if restriction == 'mixed' else ''
    actions = {
        'queue': lambda: setattr(runner, '_busy_input_mode', 'queue'),
        'nonvision': lambda: monkeypatch.setattr(agent, '_model_supports_vision', lambda: False),
        'app-server': lambda: setattr(agent, 'api_mode', 'codex_app_server'),
        'text-policy': lambda: monkeypatch.setattr(runner, '_decide_image_input_mode', lambda **kw: 'text'),
        'no-control': lambda: setattr(event, 'allow_gateway_control', False),
        'preadoption': authority.adopted.clear,
        'explicit-queue': lambda: setattr(event, '_fifo_only', True),
    }
    actions.get(restriction, lambda: None)()
    receipt = await authority.admit_native(event)
    row = get_session_admission(authority.db, admission_id=receipt.admission_id)
    assert row['status'] == 'queued'
    assert row['payload']['text'] == event.text
    assert len(row['payload']['native_text_v1']['media']) == 2
    assert queued_controls(authority, row) == [], 'public text-only controls must not accept native envelopes'
    assert not agent.withdraw_steer(receipt.admission_id)
    authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['skip', 'rewrite'])
async def test_producer_policy_precedes_transfer_and_retries_do_not_rewrite(setup, monkeypatch, action):
    from gateway.platforms.webhook_ingress import admit_producer
    _runner, agent, _images, adapter, _key, event, authority, _running = await owner(setup, monkeypatch)
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    calls = []
    monkeypatch.setattr('hermes_cli.lifecycle.has_hook', lambda name: name == 'pre_gateway_dispatch')
    async def hook(name, **kwargs):
        calls.append(name)
        return [{'action': action, 'text': 'Use the replacement geometry'}]
    monkeypatch.setattr('hermes_cli.lifecycle.ainvoke_hook', hook)
    receipt = await admit_producer(adapter, event)
    if action == 'skip':
        assert receipt is None
    else:
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert row['payload']['text'] == 'Use the replacement geometry'
        assert row['outcome'] == 'steering'
        again = await admit_producer(adapter, event)
        assert again.admission_id == receipt.admission_id
        assert agent.withdraw_steer(receipt.admission_id)
        assert not agent.withdraw_steer(receipt.admission_id)
    assert calls == ['pre_gateway_dispatch']
    authority.db.close()


@pytest.mark.asyncio
async def test_explicit_queue_preserves_pixels_without_transfer(setup, monkeypatch):
    from hermes_state_runtime import list_session_admissions
    runner, agent, _images, _adapter, _key, event, authority, _running = await owner(setup, monkeypatch)
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    event.text = '/queue Inspect these later'
    await runner._handle_message(event)
    rows = list_session_admissions(authority.db, session_id=agent.session_id, pending_only=False)
    assert len(rows) == 2
    row = rows[-1]
    assert row['status'] == 'queued' and row['payload']['text'] == 'Inspect these later'
    assert len(row['payload']['native_text_v1']['media']) == 2
    assert not agent.withdraw_steer(row['admission_id'])
    authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['llamacpp', 'custom'])
async def test_preparation_transcodes_for_receiving_runtime_and_restores_ambient(setup, monkeypatch, provider):
    import base64
    from io import BytesIO
    from PIL import Image
    from agent.auxiliary_client import scoped_runtime_main, _runtime_main_value
    from gateway.run_image_steering import _prepare
    from gateway.session_envelope import prepare_native

    runner, agent, images, _adapter, _key, event, authority, _running = await owner(setup, monkeypatch)
    webp = images[0].with_suffix('.webp')
    Image.new('RGB', (8, 8), 'blue').save(webp, lossless=True)
    event.media_urls = [str(webp)]
    event.media_types = ['image/webp']
    agent.provider = provider
    agent.model = 'vision-model'
    agent.base_url = 'http://127.0.0.1:12345/v1'
    monkeypatch.setattr(agent, '_model_supports_vision', lambda: True)
    monkeypatch.setattr('hermes_cli.local_runtime.growth.is_managed_endpoint',
                        lambda url: url == agent.base_url)
    payload = await prepare_native(runner, event)
    ambient = {'provider': 'openai', 'model': 'ambient-model', 'base_url': 'https://example.invalid/v1'}
    try:
        with scoped_runtime_main(ambient):
            content = _prepare(authority, {'payload': payload}, agent)
            assert {key: _runtime_main_value(key) for key in ambient} == ambient
        assert content is not None
        url = content[1]['image_url']['url']
        assert url.startswith('data:image/png;base64,')
        with Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as decoded:
            assert decoded.format == 'PNG'
            assert decoded.size == (8, 8)
            assert decoded.getpixel((0, 0)) == (0, 0, 255)
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['native', 'text'])
async def test_preparation_uses_source_profile_policy_and_running_model(setup, monkeypatch, mode):
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from gateway.run_image_steering import _prepare
    from gateway.session_envelope import prepare_native
    from gateway.config import Platform
    runner, agent, _images, adapter, _key, event, authority, _running = await owner(setup, monkeypatch)
    root = get_hermes_home()
    home = root / 'profiles' / 'studio'
    home.mkdir(parents=True)
    (root / 'config.yaml').write_text('agent:\n  image_input_mode: text\n')
    (home / 'config.yaml').write_text(f'model:\n  supports_vision: true\nagent:\n  image_input_mode: {mode}\n')
    # Capture first, then route the retained event to the served secondary.
    payload = await prepare_native(runner, event)
    payload['native_text_v1']['source']['profile'] = 'studio'
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = (('default', root), ('studio', home))
    runner._profile_adapters['studio'] = {Platform.SLACK: adapter}
    runner.adapters.clear()
    runner._configure_profile_adapter(adapter, 'studio', Platform.SLACK)
    monkeypatch.setattr(Path, 'home', lambda: root.parent)
    content = _prepare(authority, {'payload': payload}, agent)
    assert bool(content) is (mode == 'native')
    if content:
        assert 'studio' in content[0]['text']
    authority.db.close()
