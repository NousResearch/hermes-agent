"""Published v2 is a fixed legacy format, never the target of new preparation."""
import pytest

from gateway.hosted_room_input_custody import _backing_root
from gateway.hosted_room_input_preparation import prepare_hosted_input
from gateway.session_hosted_attachments import submission_payload
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, v3_path


@pytest.mark.asyncio
@pytest.mark.parametrize('named', [False, True])
async def test_new_documents_never_grow_the_published_v2_namespace(tmp_path, monkeypatch, named):
    home = tmp_path / 'profiles' / 'member' if named else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, owner = owned(home, monkeypatch)
    try:
        rpc, bound = rpc_files(home, owner, named=named)
        with pytest.raises(RuntimeStoreError, match='input_preparation_required'):
            submission_payload(rpc, 'read', [bound[0][0]])
        assert not _backing_root(db.db_path).exists()
        prepared = prepare_hosted_input(rpc, request_id='hosted:future', prompt='read', attachments=[bound[0][0]])
        assert v3_path(prepared).read_bytes() == bound[0][1]
        assert not _backing_root(db.db_path).exists()
    finally:
        close(db, home)
