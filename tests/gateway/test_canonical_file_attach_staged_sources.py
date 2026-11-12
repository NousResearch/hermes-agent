"""A browser pick reaches the session only from the exact upload it names, through the owner's ``file.attach``.

The Webapp upload route publishes a browser file into its profile's ``uploads/`` and hands the renderer
a ``staged_upload`` receipt (``hermes_cli/staged_uploads.py``). The browser is an attach client of the
gateway, so that receipt arrives at ``gateway/session_ancillary.py::attach_file``, which copies it into
the session's ``attachments/``. The receipt is only a claim: a deleted source profile, another
backend's install, a missing source or an oversized file must never be copied, and a failed copy must
leave the upload in place for a retry.
"""
import io
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import uuid

import pytest
import pytest_asyncio



async def _authority_with_old_desktop_chat(tmp_path, monkeypatch, cwd=None):
    """This home's session authority over a store holding one ended pre-gateway Desktop chat."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'gui': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('old-desktop', source='desktop', model='old-model', cwd=str(cwd or tmp_path))
    db.append_message('old-desktop', role='user', content='OLD-DESKTOP')
    db.end_session('old-desktop', 'session_switch')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')


def _attach(authority, tickets, purpose, subject):
    """Exactly gateway/run_api.py: redeem the minted host-scoped ticket and bind its operator."""
    from gateway.session_controls import AuthorityConnection
    ticket = tickets.mint(profile_id=authority.profile_id, subject=subject, purpose=purpose, scope='host')
    grant = tickets.redeem(ticket, profile_id=None, purpose=purpose)
    identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                'native_bootstrap': True, 'profile_scope': grant.get('scope', 'profile')}
    return AuthorityConnection(authority, object(), identity, operator=True)


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    """The profile's owner with one adopted chat open on a native connection, serving a sibling profile."""
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import install_identity

    monkeypatch.setattr(install_identity, '_INSTALL_ID_CACHE', {'root': None, 'value': None})
    # The chat runs in a workspace of its own: a pick from the profile's uploads/ is copied, never referenced.
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    authority = await _authority_with_old_desktop_chat(tmp_path, monkeypatch, cwd=workspace)
    worker = tmp_path / 'profiles' / 'worker'
    worker.mkdir(parents=True)
    (worker / 'config.yaml').write_text('{}\n', encoding='utf-8')
    # A multiplexer serves its sibling profiles; a host-scoped grant may take a pick from one.
    from gateway.session_authorities import SessionAuthorities
    served = SessionAuthorities(tmp_path, multiplexed=True)
    served.add(tmp_path, authority)
    served.add(worker, SimpleNamespace(profile_id=str(worker), retired=False))
    authority.runner.session_authorities = served
    host = _attach(authority, TicketStore('first', frozenset({authority.profile_id})), 'interactive', 'uid:1000')
    resumed = await host.dispatch({'id': 'resume', 'method': 'session.resume', 'params': {'session_id': 'old-desktop'}})
    assert 'result' in resumed, resumed

    def upload(home=tmp_path, payload=b'browser payload'):
        # Exactly the Webapp upload route: publish into the profile home, return its receipt.
        from hermes_cli.staged_uploads import publish_staged_upload, staged_upload_descriptor
        path = publish_staged_upload(io.BytesIO(payload), home, 'notes.txt')
        return staged_upload_descriptor(path, home)

    async def attach(receipt, connection=host, **extra):
        reply = await connection.dispatch({'id': 'attach', 'method': 'file.attach', 'params': {
            'session_id': 'old-desktop', 'name': 'notes.txt', 'staged_upload': receipt, **extra}})
        return reply.get('result') or reply.get('error', {}).get('message')

    return SimpleNamespace(home=tmp_path, worker=worker, attachments=tmp_path / 'attachments', upload=upload,
                           attach=attach, authority=authority, install_identity=install_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['own', 'sibling'])
async def test_a_staged_pick_is_copied_into_the_session_cache_without_retransmission(owner, source):
    receipt = owner.upload(owner.home if source == 'own' else owner.worker)
    attached = await owner.attach(receipt)
    assert isinstance(attached, dict) and attached['attached'] is True and attached['uploaded'] is True, attached
    stored = Path(attached['path'])
    # The cache directory container/SSH terminal backends mount.
    from tools.credential_files import _CACHE_DIRS
    assert stored.parent == owner.attachments and stored.parent.name in {rel for rel, _ in _CACHE_DIRS}
    assert stored.read_bytes() == Path(receipt['path']).read_bytes() == b'browser payload'
    assert attached['ref_text'].startswith('@file:')


@pytest.mark.asyncio
async def test_a_staged_pick_attaches_while_another_profile_is_starting(owner):
    """Hot-serve reserves a starting profile's registry slot before its authority exists. A host-scoped
    pick from a served profile still attaches meanwhile."""
    starting = owner.home / 'profiles' / 'starting'
    starting.mkdir()
    owner.authority.runner.session_authorities.add(starting, None, name='starting')  # as serve_profile_runtime does
    attached = await owner.attach(owner.upload())
    assert isinstance(attached, dict) and attached['attached'] is True, attached


@pytest.mark.asyncio
async def test_a_deleted_source_profile_does_not_vouch_for_its_receipt(owner):
    """The rule every named-home write follows (``assert_named_profile_home_live``): a profile whose
    deletion is recorded, or whose home is gone, has no live uploads, even while its files remain."""
    from hermes_constants import mark_named_profile_deleted
    receipt = owner.upload(owner.worker)
    source = Path(receipt['path'])
    mark_named_profile_deleted(owner.worker)
    assert isinstance(await owner.attach(receipt), str)  # refused
    assert not owner.attachments.exists()
    assert source.read_bytes() == b'browser payload'
    shutil.rmtree(owner.worker)
    assert isinstance(await owner.attach(receipt), str)  # refused
    assert not owner.attachments.exists()


@pytest.mark.asyncio
async def test_another_backend_install_does_not_vouch_for_the_receipt(owner):
    receipt = owner.upload()
    # A replacement backend at the same path persists a different install identity.
    (owner.home / 'install_id').write_text(uuid.uuid4().hex + '\n', encoding='utf-8')
    owner.install_identity._INSTALL_ID_CACHE.clear()
    assert isinstance(await owner.attach(receipt), str)  # refused
    assert not owner.attachments.exists()
    assert Path(receipt['path']).is_file()


@pytest.mark.asyncio
async def test_a_failed_copy_keeps_the_upload_for_a_retry(owner):
    receipt = owner.upload()
    # The session's attachments/ cannot be created: a file holds its name.
    owner.attachments.write_bytes(b'')
    assert isinstance(await owner.attach(receipt), str)  # refused
    owner.attachments.unlink()
    assert Path(receipt['path']).read_bytes() == b'browser payload'
    recovered = await owner.attach(receipt)
    assert isinstance(recovered, dict) and recovered['attached'] is True, recovered
    assert Path(recovered['path']).read_bytes() == b'browser payload'


@pytest.mark.asyncio
async def test_a_missing_upload_never_falls_back_to_supplied_bytes(owner):
    receipt = owner.upload()
    Path(receipt['path']).unlink()
    refused = await owner.attach(receipt, path=receipt['path'], data_url='data:text/plain;base64,d3Jvbmc=')
    assert isinstance(refused, str), refused
    assert not owner.attachments.exists()


@pytest.mark.asyncio
async def test_an_upload_that_grew_past_the_browser_limit_is_not_copied(owner):
    from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
    receipt = owner.upload()
    os.truncate(receipt['path'], WEBAPP_ATTACHMENT_MAX_BYTES + 1)
    assert isinstance(await owner.attach(receipt), str)  # refused
    assert not owner.attachments.exists()


@pytest_asyncio.fixture(params=['home-folder', 'profile-home'])
async def home_in_workspace(request, tmp_path, monkeypatch):
    """A chat whose workspace holds the profile home: the user's home folder (where sessions start
    unless ``terminal.cwd`` says otherwise; a default profile home is ``~/.hermes``) or the profile
    home itself."""
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import install_identity

    monkeypatch.setattr(install_identity, '_INSTALL_ID_CACHE', {'root': None, 'value': None})
    home = tmp_path / '.hermes'
    home.mkdir()
    workspace = tmp_path if request.param == 'home-folder' else home
    authority = await _authority_with_old_desktop_chat(home, monkeypatch, cwd=workspace)
    host = _attach(authority, TicketStore('first', frozenset({authority.profile_id})), 'interactive', 'uid:1000')
    resumed = await host.dispatch({'id': 'resume', 'method': 'session.resume', 'params': {'session_id': 'old-desktop'}})
    assert 'result' in resumed, resumed

    def upload(payload):
        from hermes_cli.staged_uploads import publish_staged_upload, staged_upload_descriptor
        path = publish_staged_upload(io.BytesIO(payload), home, 'notes.txt')
        return staged_upload_descriptor(path, home)

    async def attach(params):
        reply = await host.dispatch({'id': 'attach', 'method': 'file.attach', 'params': {
            'session_id': 'old-desktop', 'name': 'notes.txt', **params}})
        return reply.get('result') or reply.get('error', {}).get('message')

    return SimpleNamespace(home=home, upload=upload, attach=attach)


@pytest.mark.asyncio
@pytest.mark.parametrize('by', ['receipt', 'path'])
async def test_a_pick_in_a_workspace_holding_the_profile_home_outlives_upload_pruning(home_in_workspace, by):
    """A staged upload is a temporary copy the upload route prunes after 7 days, never a workspace
    file: a pick is copied into ``attachments/`` even when the session's workspace holds the profile
    home, so its ``@file:`` ref still names the file after the next upload prunes ``uploads/``. That
    holds for the receipt and for the staged path an install without an id attaches instead."""
    receipt = home_in_workspace.upload(b'what the user picked')
    params = {'staged_upload': receipt} if by == 'receipt' else {'path': receipt['path']}
    attached = await home_in_workspace.attach(params)
    assert isinstance(attached, dict) and attached['attached'] is True, attached
    stored = Path(attached['path'])
    assert attached['uploaded'] is True and stored.parent == (home_in_workspace.home / 'attachments').resolve(), attached

    # Eight days on, the user picks another file in the browser: the upload prunes stale picks.
    week_ago = time.time() - 8 * 24 * 60 * 60
    os.utime(receipt['path'], (week_ago, week_ago))
    home_in_workspace.upload(b'another pick')

    assert not Path(receipt['path']).exists()
    assert stored.read_bytes() == b'what the user picked'
