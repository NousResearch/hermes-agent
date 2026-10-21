"""The admission-start editor MCP re-check is best-effort, bounded and runs only for an authorized head.

The drain re-checks an editor session's held MCP servers before it claims the FIFO head
(``refreeze_editor_mcp``). Nothing that check raises may pause or kill the drain, every held
server is re-listed under one shared bound, and a head the drain refuses never reaches it.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit
from tests.waiting import async_wait_until


def _local_authority(tmp_path, monkeypatch):
    from gateway import session_local_recovery
    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.LOCAL)
    monkeypatch.setattr(session_local_recovery, 'restore_local_session', lambda authority, sid: None)
    monkeypatch.setattr(session_local_recovery, 'reopen_local_session', lambda authority, ref: None)
    return db, authority


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeStoreError('session_busy'), OSError('editor server crashed')],
                         ids=['runtime-store-error', 'other-exception'])
async def test_a_failing_editor_mcp_probe_does_not_stop_the_admission(tmp_path, monkeypatch, error):
    import hermes_state_local
    from gateway import session_finite, session_local_mcp, session_policy
    db, authority = _local_authority(tmp_path, monkeypatch)
    # An editor session whose borrowed servers the probe re-checks; the probe itself fails.
    data = {'scope': 'editor-session:probe', 'schemas': {}, 'names': {'0': 'editor_probe_0'}}
    authority._local_editor_mcp = {data['scope']: [{'name': 'probe'}]}
    monkeypatch.setattr(hermes_state_local, 'local_receipt', lambda db, sid: {'policy': 'frozen'})
    monkeypatch.setattr(session_policy, 'restore_policy',
                        lambda saved: SimpleNamespace(editor_mcp_json=json.dumps(data), cwd=str(tmp_path)))

    def failing_probe(*args, **kwargs):
        raise error
    monkeypatch.setattr(session_local_mcp, '_borrowed', failing_probe)
    executed = []

    async def execute(authority, ref, row):
        executed.append(row['request_id'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'work')
        await async_wait_until(lambda: executed, timeout=10, what='the head to run past the failed probe')
    assert executed == ['work']


@pytest.mark.asyncio
async def test_a_refused_head_triggers_no_editor_mcp_probe(tmp_path, monkeypatch):
    from gateway import session_local_mcp
    db, authority = _local_authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    probed = []

    async def refreeze(authority, ref, live):
        probed.append(ref.session_id)
    monkeypatch.setattr(session_local_mcp, 'refreeze_editor_mcp', refreeze)
    live = authority.sessions['s']
    with db:
        await _submit(authority, 'refused')
        # The submitter no longer owns the session's source: check_local_input refuses the head.
        live.source = SimpleNamespace(platform=Platform.LOCAL, user_id='someone-else')
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await authority._claim_next(REF, live)
        assert probed == []
        live.source = SimpleNamespace(platform=Platform.LOCAL, user_id='human')
        row, _ = await authority._claim_next(REF, live)
    assert row and probed == ['s'], 'an authorized head is still re-checked'


@pytest.fixture
def mcp_loop():
    from tools import mcp_tool_loop
    mcp_tool_loop._ensure_mcp_loop()
    yield
    mcp_tool_loop._stop_mcp_loop()


def test_held_servers_are_relisted_together_under_one_bound(monkeypatch, mcp_loop):
    """Two held servers that never answer cost one bound, not one each: each list only gets past
    its rendezvous while the other's is in flight too."""
    from gateway import session_local_mcp
    from tools import mcp_tool
    from tools.mcp_tool_scope import _server_key
    bound = 2
    monkeypatch.setattr(session_local_mcp, '_RELIST_TIMEOUT', bound)
    scope, names = 'editor-session:slow', {0: 'editor_a', 1: 'editor_b'}
    started, met = [asyncio.Event(), asyncio.Event()], []

    def server(i):
        async def list_tools():
            started[i].set()
            await started[1 - i].wait()
            met.append(i)
            await asyncio.Event().wait()  # never answers
        return SimpleNamespace(name=names[i], _rpc_lock=asyncio.Lock(), _tools=[],
                               session=SimpleNamespace(list_tools=list_tools))
    for i, name in names.items():
        monkeypatch.setitem(mcp_tool._servers, _server_key(name, scope), server(i))
    began = time.monotonic()
    assert session_local_mcp._live_tools_changed(scope, names) is False  # unanswered counts as unchanged
    elapsed = time.monotonic() - began
    assert sorted(met) == [0, 1], 'the re-lists ran one after another'
    assert elapsed < 2 * bound, elapsed
