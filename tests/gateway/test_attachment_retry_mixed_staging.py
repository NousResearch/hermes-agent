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
