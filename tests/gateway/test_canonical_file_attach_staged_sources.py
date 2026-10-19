"""A browser pick reaches the session only from the exact upload it names, through the owner's ``file.attach``.

The Webapp upload route publishes a browser file under its profile's current generation and hands the
renderer a ``staged_upload`` receipt (``hermes_cli/staged_uploads.py``). The browser is an attach client
of the gateway, so that receipt arrives at ``gateway/session_ancillary.py::attach_file``, which copies it
into the session's ``attachments/``. The receipt is only a claim: a profile deleted and recreated under
the same name, another backend's install, a missing source or an oversized file must never be copied,
and a failed copy must leave the upload in place for a retry.
"""
import io
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest
import pytest_asyncio

from tests.gateway.test_delegated_attach import _attach, _authority_with_old_desktop_chat


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
        # Exactly the Webapp upload route: publish under the leased generation, return its receipt.
        from hermes_cli.profile_incarnation import ensure_profile_incarnation
        from hermes_cli.staged_uploads import publish_staged_upload, staged_upload_descriptor
        incarnation = ensure_profile_incarnation(home)
        path = publish_staged_upload(io.BytesIO(payload), home, incarnation, 'notes.txt')
        return staged_upload_descriptor(path, home, incarnation)

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
async def test_a_profile_recreated_under_the_same_name_does_not_vouch_for_the_old_receipt(owner):
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation
    receipt = owner.upload(owner.worker)
    source = Path(receipt['path'])
    shutil.rmtree(owner.worker)
    owner.worker.mkdir(parents=True)
    write_fresh_profile_incarnation(owner.worker)
    # A restore or clone that kept the old upload file name puts different bytes at the same path.
    source.parent.mkdir()
    source.write_bytes(b'replacement generation')
    assert isinstance(await owner.attach(receipt), str)  # refused
    assert not owner.attachments.exists()
    assert source.read_bytes() == b'replacement generation'


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
async def test_a_failed_copy_keeps_the_upload_for_a_retry(owner, monkeypatch):
    from tui_gateway import server
    receipt = owner.upload()
    write_temp = server._write_attachment_temp

    def fail_target(path, data):
        if path == owner.attachments:
            raise PermissionError('temporary target failure')
        return write_temp(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(server, '_write_attachment_temp', fail_target)
        assert isinstance(await owner.attach(receipt), str)  # refused
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
