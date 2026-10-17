"""A persistent MCP reload opt-out changes later consent and never reports a failed save as success."""

import pytest

from tests.gateway.test_mcp_reload_new_server import _home, _reloadable
from tests.gateway.test_session_resume_config_refresh import _call, _connect, _owner_authority


@pytest.mark.asyncio
async def test_always_stops_later_bare_reloads_asking_for_confirmation(monkeypatch):
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner = _connect(authority)
    try:
        assert (await _call(owner, 'reload.mcp'))['status'] == 'confirm_required'
        assert (await _call(owner, 'reload.mcp', always=True))['status'] == 'reloaded'
        assert (await _call(owner, 'reload.mcp'))['status'] == 'reloaded'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_always_reports_an_unsaved_opt_out_as_failure(monkeypatch):
    from hermes_state_runtime import RuntimeStoreError

    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner = _connect(authority)

    def failed_write(*_args):
        raise RuntimeStoreError('storage_unavailable')

    monkeypatch.setattr('gateway.session_tui_config._write_profile_config', failed_write)
    try:
        reply = await owner.dispatch({'id': 1, 'method': 'reload.mcp', 'params': {'always': True}})
        assert 'result' not in reply, reply
        assert reply['error']['message'] == 'storage_unavailable', reply
    finally:
        await owner.close()
        store.close_all_db_handles()
