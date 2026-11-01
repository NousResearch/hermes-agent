"""Shared publication and physical alias custody without executing a turn."""
import os
from pathlib import Path

import pytest

from gateway import session_ingress_media as media
from hermes_state_runtime import RuntimeStoreError, admit_session_input, claim_session_input
from hermes_state_runtime import get_session_admission, settle_session_input
from tests.gateway.test_native_media_budget import _authority, _staged, _submit


@pytest.mark.parametrize('reuse', ['captured', 'admitted'])
def test_rejected_batch_keeps_an_alias_another_capture_reused(tmp_path, monkeypatch, reuse):
    from gateway.platforms import base
    db, owner = _authority(tmp_path, monkeypatch)
    with db:
        monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 3072)
        paths = [Path(_staged(tmp_path, name, 2048)['path']) for name in ('shared.png', 'excess.png')]
        capture_one = media._capture_file
        reused = []
        inside = False

        def other_capture(path, *args):
            nonlocal inside
            capture_one(path, *args)
            if path != paths[0] or inside:
                return
            inside = True
            reused.extend(media.capture_native_media(paths[:1]))
            if reuse == 'admitted':
                admit_session_input(db, epoch=owner.epoch, principal_id='other', session_id='s',
                    request_id='other', payload={'text': 'other', 'native_text_v1': {'media': reused}})

        monkeypatch.setattr(media, '_capture_file', other_capture)
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            media.capture_native_media(paths)
        assert Path(media.restore_native_media(reused)[0]).stat().st_size == 2048
        assert not list(media._media_root().glob('.capture-*'))


@pytest.mark.asyncio
@pytest.mark.parametrize('alias', ['same', 'case', 'hardlink', 'distinct', 'missing-holder'])
async def test_physical_holders_and_uncertain_stat_deny_collection(tmp_path, monkeypatch, alias):
    db, owner = _authority(tmp_path, monkeypatch)
    with db:
        first = _staged(tmp_path, 'shared.png', 2048)
        name = {'same': 'shared.png', 'case': 'SHARED.png', 'hardlink': 'linked.png',
            'distinct': 'other.png', 'missing-holder': 'missing.png'}[alias]
        second = _staged(tmp_path, name, 2048)
        receipt = await _submit(owner, 'first', [first])
        first_ref = get_session_admission(db, admission_id=receipt.admission_id)['payload']['attachments_v1']['media'][0]
        if alias == 'hardlink':
            os.link(first_ref['path'], Path(first_ref['path']).with_name(name))
        other = await _submit(owner, 'other', [second])
        other_ref = get_session_admission(db, admission_id=other.admission_id)['payload']['attachments_v1']['media'][0]
        same_file = os.path.samefile(first_ref['path'], other_ref['path'])
        if alias == 'missing-holder':
            Path(other_ref['path']).unlink()
        started = claim_session_input(db, epoch=owner.epoch, session_id='s')
        settle_session_input(db, epoch=owner.epoch, admission_id=receipt.admission_id,
            generation=started['generation'], outcome='completed')
        removed = media.release_admission_media(db, receipt.admission_id)
        assert removed == (0 if same_file or alias == 'missing-holder' else 1)
        if alias == 'missing-holder':
            assert Path(first_ref['path']).exists()
        else:
            assert Path(media.restore_native_media([other_ref])[0]).stat().st_size == 2048
