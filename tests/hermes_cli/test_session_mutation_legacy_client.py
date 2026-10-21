"""A pre-fence client (Desktop 0.22.2) gets a named upgrade refusal it can render (#139742).

0.22.2's session PATCH/DELETE carry no ``request_id``/``expected_revision``. The owner never
commits an unfenced write (the revision boundary holds on every transport), so it refuses with
``client_upgrade_required``, worded so 0.22.2's own toast summariser (``readableError`` ->
``rpcOutOfSync``, ``apps/desktop/src/store/notifications.ts`` at v0.22.2) shows "The app and the
backend are on different versions. Update both." with its Update action instead of a raw code.
"""
import re

import httpx
import pytest
from fastapi import FastAPI

from hermes_cli.web_routers.sessions import manage_router
from hermes_state import SessionDB

AUTH = {'Authorization': 'Bearer legacy-token'}
# Verbatim from v0.22.2: notifications.ts ``readableError`` and gateway-rpc.ts ``isOutOfSyncRpcParams``.
_DETAIL_RE = re.compile(r'"detail"\s*:\s*"([^"]+)"')
_OUT_OF_SYNC_RE = re.compile(r'out of sync \(different versions\)', re.IGNORECASE)


def _app(tmp_path, monkeypatch):
    from hermes_cli import web_server
    monkeypatch.setattr(web_server, '_SESSION_TOKEN', 'legacy-token')
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    return app


def _renders_as_upgrade(response):
    # Electron's httpStatusError builds ``f"{status}: {body}"``; the renderer reads it back.
    message = f'{response.status_code}: {response.text}'
    detail = _DETAIL_RE.search(message).group(1)
    return detail.startswith('client_upgrade_required') and bool(_OUT_OF_SYNC_RE.search(detail))


@pytest.mark.asyncio
async def test_identityless_legacy_writes_get_a_renderable_upgrade_refusal(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('old-desktop', source='desktop')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost',
                                     headers=AUTH) as client:
            # The exact 0.22.2 shapes: setSessionArchived / renameSession / deleteSession.
            for response in (await client.patch('/api/sessions/old-desktop', json={'archived': True}),
                             await client.patch('/api/sessions/old-desktop', json={'title': 'New'}),
                             await client.delete('/api/sessions/old-desktop')):
                assert response.status_code == 409, response.text
                assert _renders_as_upgrade(response), response.text
            # A current client that sent only half the identity is a client bug, not an old app.
            partial = await client.patch('/api/sessions/old-desktop', json={'request_id': 'r', 'title': 'X'})
            assert (partial.status_code, partial.json()['detail']) == (409, 'mutation_identity_required')
        row = db.get_session('old-desktop')
        assert row is not None and row['title'] is None and not row.get('archived')
        assert row['runtime_revision'] == 0
