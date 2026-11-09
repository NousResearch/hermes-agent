"""The authority's session storage ports are resolved once, at initialisation.

Admission operations run on the ports the authority was brought up with; nothing on the
operation path re-reads the profile's storage config.
"""
import pytest

import agent.session_storage
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
async def test_admission_operations_after_init_never_read_the_storage_config(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    retained = authority.admission()

    def refuse(home):
        raise AssertionError('the storage config was re-read after the authority was initialised')

    monkeypatch.setattr(agent.session_storage, '_storage_name', refuse)
    assert authority.admission() is retained
    with db:
        receipt = await _submit(authority, 'work')
        assert receipt.status == 'queued'
        row, _first = await authority._claim_next(REF, authority.sessions['s'])
        assert row['request_id'] == 'work'
        settled = authority.admission().get(receipt.admission_id)
        assert settled is not None and settled['status'] == 'started' and settled['request_id'] == 'work'
