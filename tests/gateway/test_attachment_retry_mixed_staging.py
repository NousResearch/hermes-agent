"""A missing staging file must not hide a changed sibling on an exact retry."""
import pytest

from gateway.session_ingress_media import admit_attachments
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize('missing_index', [0, 1])
def test_missing_staging_sibling_does_not_skip_remaining_image_hashes(missing_index):
    from gateway.platforms.base import get_image_cache_dir
    staging = get_image_cache_dir()
    paths = [staging / 'first.png', staging / 'second.png']
    for index, path in enumerate(paths):
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + str(index).encode())
    attachments = [{'path': str(path), 'mime': 'image/png'} for path in paths]
    committed = admit_attachments(attachments)
    row = {'status': 'queued', 'payload': committed}
    paths[missing_index].unlink()
    remaining = paths[1 - missing_index]
    remaining.write_bytes(b'\x89PNG\r\n\x1a\nchanged')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit_attachments(attachments, admitted=lambda: row)

    # Restore the surviving file: a partially pruned exact retry remains valid.
    remaining.write_bytes(b'\x89PNG\r\n\x1a\n' + str(1 - missing_index).encode())
    assert admit_attachments(attachments, admitted=lambda: row) == committed


def test_direct_retry_preserves_storage_unavailable_when_both_media_copies_are_gone():
    """A direct exact retry keeps the owner-wide storage refusal classification."""
    from pathlib import Path

    from gateway.platforms.base import get_image_cache_dir

    staged = get_image_cache_dir() / 'direct-storage-preservation.png'
    staged.write_bytes(bytes.fromhex(
        '89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02'
        '0000000b4944415478da63fcff1f0003030200efa2a75b0000000049454e44ae426082'))
    attachments = [{'path': str(staged), 'mime': 'image/png'}]
    committed = admit_attachments(attachments)
    row = {'status': 'queued', 'payload': committed}
    for reference in committed['attachments_v1']['media']:
        Path(reference['path']).unlink()
    staged.unlink()

    # Deliberately omit restore_committed: this is the shared direct/non-Bot path.
    with pytest.raises(RuntimeStoreError, match='storage_unavailable') as raised:
        admit_attachments(attachments, admitted=lambda: row)
    assert raised.value.reason == 'storage_unavailable'
