"""Files working-copy custody is private, verified, and profile-local."""
import errno
import hashlib
import os

import pytest

from gateway import hosted_room_input_custody as custody
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.session_hosted_attachments import submission_payload
from gateway.session_ingress_media import _media_root, restore_native_media
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.hosted_document_custody_helpers import local_documents


@pytest.mark.parametrize('fallback', [False, True], ids=['private-hardlink', 'portable-copy'])
@pytest.mark.parametrize('changed', ['backing', 'alias'])
def test_working_copies_never_link_canonical_blobs_or_repair_changed_bytes(tmp_path, monkeypatch, fallback, changed):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    rpc, bound = local_documents(tmp_path, transferred=False)
    store = HostedRoomAttachmentStore(rpc.authority.db.db_path)
    source_blobs = [path for path in store.blob_root.rglob('*') if path.is_file()]
    assert source_blobs
    if fallback:
        def cross_device(*args):
            raise OSError(errno.EXDEV, 'fixture different filesystem')
        monkeypatch.setattr(custody.os, 'link', cross_device)
    payload = submission_payload(rpc, 'read', bound)
    digest = hashlib.sha256(b'A' * 2048).hexdigest()
    alias = _media_root() / digest / '0.txt'
    backing = custody._root(store.db_path) / digest / '0.txt'
    reference = {'path': str(alias), 'sha256': digest, 'size': 2048}
    assert str(alias) in payload['text']
    assert backing.read_bytes() == alias.read_bytes() == b'A' * 2048
    assert os.path.samefile(backing, alias) is not fallback
    assert all(not os.path.samefile(path, source) for path in (alias, backing) for source in source_blobs)
    assert custody.holds_native_reference(store.db_path, reference)
    assert not custody.holds_native_reference(tmp_path / 'profiles' / 'other' / 'state.db', reference)

    damaged = backing if changed == 'backing' else alias
    damaged.write_bytes(b'changed private working copy')
    with pytest.raises(RuntimeStoreError):
        submission_payload(rpc, 'read', bound)
    assert damaged.read_bytes() == b'changed private working copy'
    saved = store.read(room_id='room', attachment_id=bound[0]['attachment_id'],
        event_id='source', recipient_member_id='member')
    assert saved.data == b'A' * 2048
    if changed == 'alias' or not fallback:
        with pytest.raises(RuntimeStoreError):
            restore_native_media([reference])


@pytest.mark.parametrize('redirect', ['backing', 'alias'])
def test_working_copy_symlink_refuses_materialization_without_modifying_target(tmp_path, monkeypatch, redirect):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    rpc, bound = local_documents(tmp_path, transferred=False)
    digest = hashlib.sha256(b'A' * 2048).hexdigest()
    root = custody._root(rpc.authority.db.db_path) if redirect == 'backing' else _media_root()
    target = root / digest / '0.txt'
    target.parent.mkdir(parents=True)
    outside = tmp_path / 'not-an-input.txt'
    outside.write_bytes(b'A' * 2048)
    target.symlink_to(outside)
    with pytest.raises(RuntimeStoreError):
        submission_payload(rpc, 'read', bound)
    assert target.is_symlink()
    assert outside.read_bytes() == b'A' * 2048
