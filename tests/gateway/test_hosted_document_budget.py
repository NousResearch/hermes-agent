"""Canonical shared documents obey the existing aggregate ingress limit."""
import pytest

from gateway.platforms import base
from gateway.session_hosted_attachments import submission_payload
from gateway.session_ingress_media import _media_root
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.hosted_document_custody_helpers import local_documents


@pytest.mark.parametrize('transferred', [False, True], ids=['local-store', 'named-owner-snapshot'])
def test_document_batch_rejects_total_before_any_capture(tmp_path, monkeypatch, transferred):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    rpc, bound = local_documents(tmp_path, transferred=transferred)
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 3072)
    assert not _media_root().exists()
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        submission_payload(rpc, 'read', bound)
    assert not _media_root().exists()
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 4096)
    payload = submission_payload(rpc, 'read', bound)
    assert payload['text'].count('[Shared attachment] file:') == 2


@pytest.mark.parametrize('limit', [0, -1])
def test_disabled_document_batch_limit_preserves_existing_behavior(tmp_path, monkeypatch, limit):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: limit)
    rpc, bound = local_documents(tmp_path, transferred=False)
    assert submission_payload(rpc, 'read', bound)['text'].count('[Shared attachment] file:') == 2


@pytest.mark.parametrize('transferred', [False, True], ids=['local-store', 'named-owner-snapshot'])
def test_retained_document_still_obeys_single_file_limit(tmp_path, monkeypatch, transferred):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    rpc, bound = local_documents(tmp_path, transferred=transferred)
    if transferred:
        rpc.hosted_attachment_data = rpc.hosted_attachment_data[:1]
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 1024)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        submission_payload(rpc, 'read', bound[:1])
    assert not _media_root().exists()
