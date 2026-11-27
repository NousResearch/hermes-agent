"""Private Paperclip launch context: never a durable client environment."""
import asyncio
import subprocess
import sys

import pytest


def context(tag='a', **extra):
    return {'PAPERCLIP_API_URL': 'http://127.0.0.1:3100', 'PAPERCLIP_API_KEY': 'fixture-key-' + tag,
            'PAPERCLIP_AGENT_ID': tag, 'PAPERCLIP_COMPANY_ID': 'company', 'PAPERCLIP_RUN_ID': 'run-' + tag, **extra}


def test_bound_child_builders_and_no_daemon_inheritance(monkeypatch, tmp_path):
    from agent.paperclip_context import paperclip_scope, child_environ
    from tools.environments.local import _make_run_env
    monkeypatch.setattr('pm.environments._payload_manifest', lambda root: None)
    monkeypatch.setattr('tools.environments.local._user_tools_first', lambda p: p)
    monkeypatch.setattr('tools.environments.local._prepend_hermes_bin_dir', lambda p: p)
    from tools.code_execution_env import _build_child_env
    monkeypatch.setenv('PAPERCLIP_API_KEY', 'daemon-token')
    monkeypatch.setenv('PAPERCLIP_AGENT_JWT_SECRET', 'server-secret')
    for values in (context(), {}, context('b')):
        with paperclip_scope(values):
            envs = [child_environ(), _make_run_env({}), _build_child_env(
                rpc_endpoint='sock', rpc_token='fixture', tmpdir=str(tmp_path), child_python=sys.executable)]
            for env in envs:
                assert {k: v for k, v in env.items() if k.startswith('PAPERCLIP_')} == values
                result = subprocess.run([sys.executable, '-c',
                    'import os; assert os.environ.get("PAPERCLIP_AGENT_ID", "") == ' + repr(values.get('PAPERCLIP_AGENT_ID', ''))],
                    env=env, capture_output=True, check=False)
                assert result.returncode == 0
    assert not any(k.startswith('PAPERCLIP_') for k in child_environ())


def test_opaque_run_key_is_redacted_from_real_child_output_even_after_turn(monkeypatch, tmp_path):
    from agent.paperclip_context import paperclip_scope
    from agent.redact import clear_vault_redaction_values, redact_sensitive_text, redact_terminal_output
    from tools.code_execution_env import _build_child_env
    from tools.environments.local import _make_run_env
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'redaction-home'))
    monkeypatch.setattr('agent.redact._redact_enabled', lambda: False)
    monkeypatch.setattr('pm.environments._payload_manifest', lambda root: None)
    monkeypatch.setattr('tools.environments.local._user_tools_first', lambda p: p)
    monkeypatch.setattr('tools.environments.local._prepend_hermes_bin_dir', lambda p: p)
    opaque = 'opaque-paperclip-private-value-without-a-known-token-prefix'
    try:
        with paperclip_scope(context(PAPERCLIP_API_KEY=opaque)):
            envs = [_make_run_env({}), _build_child_env(rpc_endpoint='unused', rpc_token='fixture',
                    tmpdir=str(tmp_path), child_python=sys.executable)]
            for env in envs:
                child = subprocess.run([sys.executable, '-c', 'import os;print(os.environ["PAPERCLIP_API_KEY"])'],
                                       env=env, capture_output=True, text=True, check=True)
                assert child.stdout.strip() == opaque
                assert opaque not in redact_sensitive_text(child.stdout)
                assert opaque not in redact_terminal_output(child.stdout, 'printenv PAPERCLIP_API_KEY')
        # Background/log readers may drain after the authorizing turn has ended.
        assert opaque not in redact_sensitive_text(opaque)
    finally:
        clear_vault_redaction_values()


def test_collect_exact_allowlist_and_client_env_cannot_smuggle():
    from agent.paperclip_context import collect_context
    from agent.session_client_env import collect_client_env
    values = context(PAPERCLIP_AGENT_JWT_SECRET='server', PAPERCLIP_DB_PASSWORD='db')
    assert collect_context(values) == context()
    assert collect_client_env(values, {'terminal': {'client_env_passthrough': list(values)}}) == {}


@pytest.mark.asyncio
async def test_concurrent_scopes_are_isolated():
    from agent.paperclip_context import paperclip_scope, child_environ
    async def read(values):
        with paperclip_scope(values):
            await asyncio.sleep(0)
            return await asyncio.to_thread(lambda: {k: v for k, v in child_environ().items() if k.startswith('PAPERCLIP_')})
    assert await asyncio.gather(*(read(v) for v in [context(), context('b'), {}])) == [context(), context('b'), {}]


def test_validate_context_rejects_unknown_names_and_oversize_and_partial():
    from agent.paperclip_context import validate_context
    with pytest.raises(ValueError):
        validate_context({'PAPERCLIP_AGENT_JWT_SECRET': 'x'})  # not an allowlisted name
    with pytest.raises(ValueError):
        validate_context({'PAPERCLIP_API_KEY': 'x' * (64 * 1024 + 1)})  # oversize
    with pytest.raises(ValueError):
        validate_context({'PAPERCLIP_API_KEY': 'only-one-of-the-required-names'})  # partial, non-empty
    assert validate_context({}) == {}  # an empty context is a legitimate "no run bound" value
