"""Every create refusal names the check that fired.

``session.create`` refused with a bare ``invalid_params`` cost three reports in one morning
(#138670, #138748, #134889): the gateway logged nothing and the client printed the code alone, so
neither the reporter nor the maintainer could tell which of the ~20 checks had rejected the launch.
These tests pin the naming at the policy, at the wire frame, in the gateway log, and in the CLI
copy that a user actually reads.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_policy import build_policy, rebind_launch_key
from hermes_state_runtime import RuntimeStoreError

_CFG = {'platform_toolsets': {'cli': ['terminal']}}


def _refusal(params, cfg=None, **kwargs):
    """The ``RuntimeStoreError`` a real ``build_policy`` raises for *params*."""
    with pytest.raises(RuntimeStoreError) as caught:
        build_policy(params, cfg if cfg is not None else dict(_CFG), **kwargs)
    return caught.value


@pytest.mark.parametrize('extra, field', [
    ({'bogus': 1}, 'params.bogus'),                    # an unknown create field
    ({'source': 'nope'}, 'source'),                     # not a known surface
    ({'source': 'cli', 'safe_mode': 'yes'}, 'safe_mode'),
    ({'source': 'cli', 'ignore_user_config': 1}, 'ignore_user_config'),
    ({'source': 'cli', 'api_key': '   '}, 'api_key'),
    ({'source': 'cli', 'model': '  '}, 'model'),
    ({'source': 'cli', 'cwd': 'relative/place'}, 'cwd'),
    ({'source': 'cli', 'row_cwd': 'relative/place'}, 'row_cwd'),
    ({'source': 'cli', 'provider': ''}, 'provider'),
    ({'source': 'cli', 'base_url': 'ftp://example.invalid'}, 'base_url'),
    ({'source': 'cli', 'yolo': 'yes'}, 'yolo'),
    ({'source': 'cli', 'verbose': 'yes'}, 'verbose'),
    ({'source': 'cli', 'service_tier': 'instant'}, 'service_tier'),
    ({'source': 'cli', 'run_budget': -1}, 'run_budget'),
    ({'source': 'cli', 'max_turns': 'soon'}, 'max_turns'),
    ({'source': 'cli', 'reasoning': 'galaxy'}, 'reasoning'),
    ({'source': 'cli', 'toolsets': 'terminal'}, 'toolsets'),                   # not a list
    ({'source': 'cli', 'toolsets': ['terminal', 'gone_plugin']}, 'toolsets'),  # unknown name
    ({'source': 'cli', 'toolsets': ['desktop_ui']}, 'toolsets'),               # Desktop-only
    ({'source': 'cli', 'skills': []}, 'skills'),                               # empty list
    ({'source': 'cli', 'skills': ['porting'], 'safe_mode': True}, 'skills'),   # bypass launch
    ({'source': 'cli', 'editor': {}}, 'editor'),                               # ACP-only launch key
    ({'source': 'acp', 'editor': {'bogus': 1}}, 'editor.bogus'),
    ({'source': 'acp', 'editor': {'edit_approval_policy': 'always'}}, 'editor.edit_approval_policy'),
    ({'source': 'acp', 'editor': {'mcp_servers': 'terminal'}}, 'editor.mcp_servers'),
    ({'source': 'acp', 'editor': {'mcp_servers': [{'type': 'stdio'}]}}, 'editor.mcp_servers'),
    ({'source': 'cli', 'toolsets': [None]}, 'toolsets'),   # a literal None is the bad element
    ({'source': 'cli', 'toolsets': [False]}, 'toolsets'),
])
def test_every_create_check_names_the_field_it_read(tmp_path, extra, field):
    refusal = _refusal({'cwd': str(tmp_path), **extra})
    assert refusal.reason == 'invalid_params'
    assert refusal.field == field, refusal.detail
    assert refusal.detail, 'a refusal names why, not only which field'


def test_the_unknown_field_is_named_with_its_own_key(tmp_path):
    """A closed key set reports the key that broke it, not the set."""
    refusal = _refusal({'cwd': str(tmp_path), 'toolsets_extra': 1})
    assert refusal.field == 'params.toolsets_extra'


def test_an_invalid_toolset_name_reports_that_name(tmp_path):
    refusal = _refusal({'cwd': str(tmp_path), 'toolsets': ['terminal', 'gone_plugin']})
    assert 'gone_plugin' in refusal.detail


def test_a_rebound_launch_key_names_api_key():
    """``session.resume`` re-supplying a launch key refuses before it touches the authority."""
    with pytest.raises(RuntimeStoreError) as caught:
        rebind_launch_key(object(), object(), '   ')
    assert (caught.value.field, caught.value.reason) == ('api_key', 'invalid_params')


def test_a_clean_launch_still_builds(tmp_path):
    """The naming must not turn a valid launch into a refusal."""
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'fixture'}, dict(_CFG))
    assert policy.cwd == str(tmp_path)


def test_the_reason_code_is_unchanged_on_the_wire():
    """Callers compare ``str(exc)`` to reason codes; naming may not move it."""
    exc = RuntimeStoreError('invalid_params', field='cwd', detail='nope')
    assert str(exc) == 'invalid_params' and exc.reason == 'invalid_params'


def test_no_bare_invalid_params_raise_on_the_create_path():
    """A new create check must name its field and why: this guard fails instead of the next report.

    Five surfaces carry the create path's checks: the policy builder, the staged local session, the
    editor policy, the MCP server list and the ``_create`` handler. Two of them are scanned at function
    scope, because their modules refuse with the bare code elsewhere — ``session_local_mcp.py`` on the
    mutate path (``session.mutate editor_mcp``) and ``session_controls.py`` in the other handlers. Those
    are a separate, off-path pass."""
    scanned = [('gateway/session_policy.py', None), ('gateway/session_local_editor.py', None),
               ('gateway/session_local.py', None), ('gateway/session_local_mcp.py', 'validate_servers'),
               ('gateway/session_controls.py', '_create')]
    root = Path(__file__).resolve().parents[2]
    bare, thin = [], []
    for module, function in scanned:
        tree = ast.parse((root / module).read_text(encoding='utf-8'))
        nodes = tree.body
        if function is not None:
            # A method lives inside a class body, so walk for it rather than reading tree.body.
            nodes = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function]
            assert nodes, f'{function} not found in {module}'
        for top in nodes:
            for node in ast.walk(top):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                name = getattr(called, 'id', None) or getattr(called, 'attr', None)
                if name == '_refuse' and not (len(node.args) >= 2 or node.keywords):
                    thin.append(f'{module}:{node.lineno}')  # a field with no reason is half a diagnosis
                first = node.args[0] if node.args else None
                if (name == 'RuntimeStoreError' and isinstance(first, ast.Constant)
                        and first.value == 'invalid_params' and not node.keywords):
                    bare.append(f'{module}:{node.lineno}')
    assert bare == [], f'unnamed invalid_params refusals on the create path: {bare}'
    assert thin == [], f'_refuse() calls without a detail: {thin}'


def test_the_profile_in_the_log_line_is_a_name_not_a_home_path(tmp_path, caplog):
    """A root profile is keyed by its home path; the log line carries its name, never the path."""
    from gateway.session_controls import _log_refusal, _profile_label

    assert _profile_label('test') == 'test'
    assert _profile_label(None) == 'default'
    assert _profile_label(str(tmp_path)) == 'default'
    with caplog.at_level('WARNING', logger='gateway.session_controls'):
        _log_refusal(str(tmp_path), 'session.create',
                     RuntimeStoreError('invalid_params', field='cwd', detail='not a directory'))
    line = caplog.records[-1].getMessage()
    assert 'profile=default' in line and str(tmp_path) not in line, line


@pytest.mark.asyncio
async def test_the_frame_carries_the_field_and_the_gateway_logs_it(tmp_path, caplog):
    """What a report can quote: the JSON-RPC ``error.data`` and one gateway log line."""
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    detail = "unknown or invalid toolset name: 'gone_plugin'"
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test', instance_id='test',
                                     db=db, epoch=epoch)

        class Connection(AuthorityConnection):
            def handlers(self):
                async def probe(ref, params):
                    raise RuntimeStoreError('invalid_params', field='toolsets', detail=detail)
                return {'session.policy_probe': probe}

        connection = Connection(authority, object(), {'user_id': 'human'})
        with caplog.at_level('WARNING', logger='gateway.session_controls'):
            reply = await connection.dispatch({'id': 7, 'method': 'session.policy_probe', 'params': {}})
        assert reply['error']['message'] == 'invalid_params'
        assert reply['error']['data'] == {'reason': 'invalid_params', 'field': 'toolsets', 'detail': detail}
        logged = [record.getMessage() for record in caplog.records]
        assert any('field=toolsets' in line and 'session.policy_probe' in line and 'profile=test' in line
                   for line in logged), logged
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_bare_refusal_does_not_warn(tmp_path, caplog):
    """A retrying client must not be able to flood the log: only a named refusal is a warning."""
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test', instance_id='test',
                                     db=db, epoch=epoch)

        class Connection(AuthorityConnection):
            def handlers(self):
                async def probe(ref, params):
                    raise RuntimeStoreError('permission_denied')
                return {'session.unnamed_probe': probe}

        connection = Connection(authority, object(), {'user_id': 'human'})
        with caplog.at_level('DEBUG', logger='gateway.session_controls'):
            reply = await connection.dispatch({'id': 8, 'method': 'session.unnamed_probe', 'params': {}})
        assert reply['error']['data'] == {'reason': 'permission_denied'}
        levels = {record.levelname for record in caplog.records}
        assert 'WARNING' not in levels, [r.getMessage() for r in caplog.records]
        assert 'DEBUG' in levels
    finally:
        db.close()


def test_the_cli_copy_names_the_field_a_user_reads():
    """The panel line said ``invalid_params`` alone; it now carries the check (#138748)."""
    from hermes_cli.cli_chat_error_copy import agent_init_failure_message
    from hermes_cli.gateway_client import GatewayRPCError

    exc = GatewayRPCError('invalid_params', 4001, {
        'reason': 'invalid_params', 'field': 'toolsets',
        'detail': "unknown or invalid toolset name: 'gone_plugin'"})
    message = agent_init_failure_message(exc)
    assert 'invalid_params' in message and 'toolsets' in message


def test_a_refusal_without_a_named_field_reads_exactly_as_before():
    """Refusals raised elsewhere keep their copy: no empty parentheses on the panel."""
    from hermes_cli.cli_chat_error_copy import agent_init_failure_message
    from hermes_cli.gateway_client import GatewayRPCError

    message = agent_init_failure_message(GatewayRPCError('storage_unavailable', 5001,
                                                         {'reason': 'storage_unavailable'}))
    assert message.count('storage_unavailable') == 1 and '(' not in message


def test_a_bypass_launch_without_a_model_names_the_model():
    """`--ignore-user-config` has no profile to inherit a model from: the refusal says so."""
    from gateway.session_local import _bypass_policy

    with pytest.raises(RuntimeStoreError) as caught:
        _bypass_policy({'source': 'cli'}, private_secrets=None)
    assert (caught.value.field, caught.value.reason) == ('model', 'invalid_params')
