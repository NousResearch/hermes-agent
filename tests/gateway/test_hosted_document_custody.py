"""Hosted non-image documents are retained inputs like any other: the native GC's holder scan
keeps the bytes a live hosted row still needs, and the row's own terminal release collects them."""
import json

from gateway.session_ingress_media import capture_native_media, release_admission_media
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _document(tmp_path, name, data):
    source = tmp_path / 'src' / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    return capture_native_media([source])[0]


def test_queued_hosted_document_survives_native_cancel_and_is_released_at_its_own_terminal(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.session_hosted_attachments import attested_submission_payload
    store = storage_handle(tmp_path)
    seed_session(store, 's', 'test')
    epoch = store.runtime.begin_epoch(instance_id='current')
    data = b'%PDF-1.4 quarterly report\n'
    native_ref = _document(tmp_path, 'report.pdf', data)
    native, _created = store.admissions.admit(epoch=epoch, principal_id='bot', session_id='s', request_id='native-1',
                                              payload={'text': 'see file', 'native_text_v1': {'media': [native_ref]}})
    # The hosted row commits exactly what the hosted transport commits for the same bytes and name.
    manifest = [{'attachment_id': 'att_' + 'a' * 32, 'event_id': 'evt_1', 'kind': 'file', 'name': 'report.pdf',
                 'mime': 'application/pdf', 'size': len(data)}]
    payload = attested_submission_payload('summarize', manifest, [native_ref['sha256']])
    assert native_ref['path'] in payload['text'] and 'attachments_v1' not in payload
    hosted, _created = store.admissions.admit(epoch=epoch, principal_id='owner', session_id='s',
                                              request_id='hosted:' + json.dumps([{'room_id': 'r'}, 1]), payload=payload)
    store.admissions.cancel(epoch=epoch, admission_id=native.admission_id)
    release_admission_media(store, native.admission_id)
    from pathlib import Path
    retained = Path(native_ref['path'])
    assert retained.read_bytes() == data, 'a queued hosted row still needs this document'
    store.admissions.cancel(epoch=epoch, admission_id=hosted.admission_id)
    release_admission_media(store, hosted.admission_id)
    assert not retained.exists(), 'the hosted row was the last holder; its terminal release collects the bytes'
