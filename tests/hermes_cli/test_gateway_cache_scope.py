"""Classic and thin CLI paths forward hashed cache routing, or refuse an older gateway."""
import argparse
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_client import GatewayClientError


@pytest.mark.parametrize('supported', [True, False])
def test_classic_store_forwards_scope_or_refuses_before_creation(tmp_path, monkeypatch, supported):
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from tests.hermes_cli.test_gateway_session_store import FakeClient

    class Peer(FakeClient):
        async def rpc(self, method, *, _timeout=None, **params):
            if method == 'runtime.describe':
                return {'session_create': {'parameters': ['cache_scope'] if supported else []}}
            return await super().rpc(method, _timeout=_timeout, **params)

    peer = Peer()

    @asynccontextmanager
    async def connect():
        yield peer

    monkeypatch.chdir(tmp_path)
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    try:
        store = GatewaySessionStore(conn)
        if supported:
            store.create({'cache_scope': 'cli_' + 'a' * 32})
            params = next(params for method, params in peer.calls if method == 'session.create')
            assert params['cache_scope'] == 'cli_' + 'a' * 32
        else:
            with pytest.raises(GatewayClientError, match='update the gateway'):
                store.create({'cache_scope': 'cli_' + 'a' * 32})
            assert not peer.calls
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('supported', [True, False])
async def test_thin_client_uses_actual_workspace_and_checks_contract(tmp_path, monkeypatch, supported):
    from hermes_cli import gateway_chat
    from hermes_cli.cli_cache_scope import cli_cache_scope
    from hermes_cli.gateway_chat_view import GatewayChatView
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            if method == 'runtime.describe':
                return {'session_create': {'sources': ['cli', 'oneshot'],
                        'parameters': ['request_id', 'source', 'cwd'] + (['cache_scope'] if supported else [])}}
            calls.append((method, params))
            return {'stored_session_id': 'physical', 'messages': []}

    @asynccontextmanager
    async def connect():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        assert query == 'draft' and oneshot
        return 0

    monkeypatch.setattr(gateway_chat, 'connect_gateway', connect)
    monkeypatch.setattr(GatewayChatView, 'run', rendered)
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / 'actual-workspace'
    workspace.mkdir()
    args = argparse.Namespace(query='draft', quiet=True, cache_scope='local-label', in_dir=str(workspace))
    if supported:
        assert await gateway_chat.run_gateway_chat(args) == 0
        [(method, params)] = calls
        assert method == 'session.create' and params['source'] == 'oneshot'
        assert params['cache_scope'] == cli_cache_scope('local-label', workspace)
        assert 'local-label' not in str(params)
    else:
        with pytest.raises(GatewayClientError, match='cache_scope'):
            await gateway_chat.run_gateway_chat(args)
        assert not calls


@pytest.mark.parametrize('flags', [['--resume', 'old'], ['--continue'], ['--format', 'stream-json', '--resume', 'old']])
def test_thin_launch_rejects_scope_on_resume_before_connection(flags):
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.gateway_chat import validate_options
    args = build_top_level_parser()[0].parse_args(['chat', '-Q', '-q', 'draft', '--cache-scope', 'local-label', *flags])
    with pytest.raises(GatewayClientError, match='--resume/--continue'):
        validate_options(args)


def test_stream_json_validation_failure_still_finishes_protocol(capsys):
    import json
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.gateway_chat import launch_from_args
    args = build_top_level_parser()[0].parse_args([
        'chat', '--format', 'stream-json', '--resume', 'old', '-q', 'draft', '--cache-scope', 'local-label'])
    assert launch_from_args(args) == 2
    captured = capsys.readouterr()
    frames = [json.loads(line) for line in captured.out.splitlines()]
    assert frames[-1]['type'] == 'result'
    assert '--resume/--continue' in captured.err
