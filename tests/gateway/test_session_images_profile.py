"""image.attach_bytes accepts its declared ``profile`` selector for the owner's own home (dokterdok N73)."""
import base64
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nL8AAAAASUVORK5CYII=')


@pytest.mark.asyncio
async def test_upload_profile_selector_names_own_home_or_is_refused(tmp_path, monkeypatch):
    # The owner is the `ops` profile; `profile: "ops"` is what Bot Mode's group turns send.
    root = tmp_path / 'root'
    home = root / 'profiles' / 'ops'
    home.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: root)
    with SessionDB(db_path=home / 'state.db') as db:
        db.create_session('s', source='test')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(home),
            instance_id='current', db=db, epoch=begin_runtime_epoch(db, instance_id='current'))
        owner = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
        live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=Platform.LOCAL,
            user_id=owner.actor.subject), 'route')
        owner.subscriptions['s'] = 'test-viewer'
        live.subscribers['test-viewer'] = owner.actor
        params = {'session_id': 's', 'content_base64': base64.b64encode(PNG).decode(), 'filename': 'shot.png'}

        own = await owner.dispatch({'id': 1, 'method': 'image.attach_bytes', 'params': {**params, 'profile': 'ops'}})
        assert own.get('result', {}).get('mime') == 'image/png', own
        foreign = await owner.dispatch({'id': 2, 'method': 'image.attach_bytes', 'params': {**params, 'profile': 'other'}})
        assert foreign['error']['message'] == 'profile_mismatch', foreign
        assert len(list((home / 'cache/images').iterdir())) == 1
        await owner.close()
