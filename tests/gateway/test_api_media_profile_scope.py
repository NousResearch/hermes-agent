"""Startup collection of retained API images pairs each profile's media root with its own state.db."""
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


def _image(data):
    return [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}]


def _orphan(data):
    """An API image file no admission holds (its chat was deleted while the owner was down)."""
    import hashlib
    from gateway.session_ingress_media import _media_root
    digest = hashlib.sha256(data).hexdigest()
    path = _media_root() / digest / f'api_{digest[:32]}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.asyncio
async def test_restart_recovery_keeps_every_profiles_held_images_and_collects_each_profiles_orphans(
        tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import _profile_runtime_scope
    from gateway.session import SessionStore
    from gateway.session_api_media import commit_api_images
    from gateway.session_api_turn import recover_api_turns
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input, settle_session_input

    launch = tmp_path / '.hermes'
    beta = launch / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    for home in (launch, beta):
        (home / 'config.yaml').write_text('{}')
    monkeypatch.setenv('HERMES_HOME', str(launch))
    store = SessionStore(config=GatewayConfig(), sessions_dir=launch / 'sessions')
    registry = SessionAuthorities(launch, multiplexed=True)
    runner = SimpleNamespace(_draining=False, session_store=store, session_authorities=registry,
                             config=SimpleNamespace(multiplex_profiles=True))
    authorities = {}
    for name, home in (('default', launch), ('beta', beta)):
        db = SessionDB(home / 'state.db')
        authorities[name] = SessionAuthority(runner, profile_id=str(home), instance_id='first', db=db,
                                             epoch=begin_runtime_epoch(db, instance_id='first'))
        registry.add(home, authorities[name], name=name)
    store._db = authorities['default'].db
    runner.session_authority = authorities['default']
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner._adapter_for_source = lambda source: adapter

    def completed_image_turn(authority, data):
        # The committed shape ``admit_api_turn`` writes (bytes captured under the active scope).
        content = _image(data)
        authority.db.create_session('chat', source='api_server')
        row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='api', session_id='chat',
            request_id='image', payload={'text': content, 'api_turn_v1': {
                'history': [], 'settings': {}, 'media': commit_api_images(content)}})
        started = claim_session_input(authority.db, epoch=authority.epoch, session_id='chat')
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=started['generation'], outcome='completed')
        # A completed API turn's image stays as history context for later turns.
        return Path(row['payload']['api_turn_v1']['media'][0]['path'])

    try:
        launch_held = completed_image_turn(authorities['default'], PNG + b'launch')
        launch_orphan = _orphan(PNG + b'launch-orphan')
        with _profile_runtime_scope(beta, hydrate_secrets=False):
            beta_orphan = _orphan(PNG + b'beta-orphan')
        assert beta_orphan.is_relative_to(beta) and launch_held.is_relative_to(launch / 'cache')

        recover_api_turns(adapter)  # the gateway's unscoped startup pass over every served profile

        assert launch_held.read_bytes() == PNG + b'launch', 'a secondary profile collected the launch image'
        assert not launch_orphan.exists()
        assert not beta_orphan.exists(), "the secondary profile's own media root was never examined"
        with _profile_runtime_scope(beta, hydrate_secrets=False):
            beta_held = completed_image_turn(authorities['beta'], PNG + b'beta')
        recover_api_turns(adapter)
        assert beta_held.read_bytes() == PNG + b'beta'
        assert launch_held.read_bytes() == PNG + b'launch'
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        for authority in authorities.values():
            authority.db.close()
