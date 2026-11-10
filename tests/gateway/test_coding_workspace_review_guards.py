"""Supplemental regressions from the independent canonical workspace review."""
import json

import pytest

from tests.gateway.test_session_coding_workspace import create_params, rpc, workspace as _workspace

workspace = _workspace


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['{not json', 'null', '[]', '{}',
                                json.dumps({'policy': {'config_json': '{bad'}}),
                                json.dumps({'policy': {'config_json': '[]'}}),
                                json.dumps({'policy': {'config_json': json.dumps({
                                    'canonical_coding_workspace_v1': ['invalid']})}})])
async def test_unrelated_unreadable_receipt_cannot_block_workspace_create(workspace, raw):
    from hermes_state_local import POLICY_PREFIX
    _home, _source, owner, connection, binding = workspace
    owner.db.set_meta(POLICY_PREFIX + 'junk', raw)
    try:
        response = await rpc(connection, 'session.create', **create_params(binding))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        pytest.fail(f'Unrelated malformed receipt blocked workspace creation: {exc!r}')
    assert 'result' in response, response
    assert owner.db.get_meta(POLICY_PREFIX + 'junk') == raw
    assert response['result']['info']['coding_workspace']['requestId'] == binding['requestId']


@pytest.mark.asyncio
@pytest.mark.parametrize('word', ['', '@file:'])
async def test_read_only_principal_cannot_complete_an_uncreated_draft(workspace, tmp_path, word):
    from gateway.session_controls import AuthorityConnection
    _home, _source, owner, connection, _binding = workspace
    outside = tmp_path / 'unrelated-host-folder'
    outside.mkdir()
    (outside / 'private-name.txt').write_text('fixture only')
    reader = AuthorityConnection(owner, object(), {**connection._identity, 'capabilities': ['session:read']})
    try:
        reply = await rpc(reader, 'complete.path', word=word, cwd=str(outside), cwd_explicit=True)
        assert reply.get('error', {}).get('message') == 'permission_denied', reply
    finally:
        await reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('word', ['', '@file:'])
async def test_existing_session_completion_keeps_read_only_contract(workspace, word):
    from gateway.session_controls import AuthorityConnection
    _home, _source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    reader = AuthorityConnection(owner, object(), {**connection._identity, 'capabilities': ['session:read']})
    try:
        reply = await rpc(reader, 'complete.path', session_id=created['result']['session_id'], word=word)
        assert 'result' in reply, reply
        if word:
            assert '@file:sample.py' in [item['text'] for item in reply['result']['items']]
        else:
            assert reply['result'] == {'items': [], 'sourceCwd': binding['cwd']}
    finally:
        await reader.close()
