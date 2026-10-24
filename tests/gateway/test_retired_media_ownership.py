"""Receipt retries preserve transcript boundaries, successor ownership and media identity."""
from pathlib import Path

import pytest

from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.asyncio
async def test_attachment_retry_survives_lost_staging_but_rejects_changed_payload(tmp_path, monkeypatch):
    from gateway.platforms.base import cache_image_from_bytes
    from tests.gateway.test_prompt_attachments import _authority, _ONE_PX_PNG
    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    staged = Path(cache_image_from_bytes(_ONE_PX_PNG, '.png'))
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
    request = Submission('retry', SessionRef('p', 's'),
        {'text': 'look', 'attachments': [{'path': str(staged), 'mime': 'image/png'}]}, 'queue')
    receipt = await authority.submit(actor, request)
    staged.unlink()
    assert await authority.submit(actor, request) == receipt
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await authority.submit(actor, Submission('retry', request.ref, {**request.payload, 'text': 'changed'}, 'queue'))
    staged.write_bytes(_ONE_PX_PNG + b'changed')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await authority.submit(actor, request)
