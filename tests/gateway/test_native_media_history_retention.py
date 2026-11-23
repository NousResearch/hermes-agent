"""A settled native input stays on disk while the transcript still tells a follow-up turn where it
is saved (dokterdok N28), in whichever spelling the note used, and is bounded by main's 24 h media
age once settled, as main's document cache cleanup bounded the staging copy the note used to name."""
import os
import time

import pytest

from gateway.session_ingress_media import _media_root, capture_native_media, release_admission_media
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _settled_native_document(tmp_path, monkeypatch, note):
    from gateway.platforms.base import get_document_cache_dir
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    staged = get_document_cache_dir() / 'doc_1_report.pdf'
    staged.write_bytes(b'%PDF-1.4 retained')
    media = capture_native_media([staged])
    retained = media[0]['path']
    db = storage_handle(tmp_path)
    seed_session(db, 's', 'telegram')
    seed_messages(db, 's', {'role': 'user', 'content': note(retained)})
    epoch = db.runtime.begin_epoch(instance_id='owner')
    row, _created = db.admissions.admit(epoch=epoch, principal_id='p', session_id='s', request_id='m1',
                                        payload={'text': 'page 3?', 'native_text_v1': {'media': media}})
    claim = db.admissions.claim(epoch=epoch, session_id='s')
    db.admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=claim.generation,
                         outcome='completed')
    return db, row, retained


@pytest.mark.parametrize('note', [
    lambda path: f"[The user sent a document: 'report.pdf'. It is saved at: {path}.]",
    # A docker/modal backend's agent-visible spelling of the same retained file.
    lambda path: "It is saved at: /root/.hermes/cache/documents/" + path.split('/documents/', 1)[1],
], ids=['host-path', 'agent-visible-path'])
def test_settled_native_document_named_by_the_transcript_survives_settlement(tmp_path, monkeypatch, note):
    import gateway.session_ingress_media as media_module
    monkeypatch.setattr(media_module, '_last_sweep', {}, raising=False)
    db, row, retained = _settled_native_document(tmp_path, monkeypatch, note)
    try:
        release_admission_media(storage_handle(tmp_path), row.admission_id)
        assert os.path.exists(retained), 'the follow-up turn is told to open a path settlement deleted'
        # Settled and older than main's media age: collected whatever the transcript says.
        old = time.time() - 25 * 3600
        os.utime(retained, (old, old))
        monkeypatch.setattr(media_module, '_last_sweep', {}, raising=False)
        release_admission_media(storage_handle(tmp_path), row.admission_id)
        assert not os.path.exists(retained) and list(_media_root().iterdir()) == []
    finally:
        db.close()
