"""Shared fixtures for tests/acp_adapter.

Keeps the ACP server tests offline: ``HermesACPAgent._build_model_state``
calls ``hermes_cli.inventory.build_models_payload``, which (without this
fixture) performs live network fetches — models.dev registry, GitHub model
catalog, Copilot token exchange, Anthropic model list — adding ~3s of real
SSL/socket time to every test that creates or loads a session (~147s total
for test_server.py alone).

Tests that assert model-state behavior re-patch these same attributes with
``unittest.mock.patch`` / ``monkeypatch``; inner patches win, so this
default is transparent to them.

``daemon`` (an ordinary isolated gateway process; ``model_peer`` comes from
``tests/conftest.py``) and ``server_spec`` (a stdio MCP peer) are shared by the
real-wire ACP test files, which receive them by name.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.fixture(autouse=True)
def _offline_model_inventory(monkeypatch):
    """Stub the shared model inventory so ACP tests never hit the network."""
    from hermes_cli import inventory

    class _StubPickerContext:
        def with_overrides(self, **_kwargs):
            return self

    monkeypatch.setattr(inventory, "load_picker_context", lambda: _StubPickerContext())
    monkeypatch.setattr(
        inventory,
        "build_models_payload",
        lambda *_args, **_kwargs: {"providers": []},
    )


@pytest.fixture
def daemon(tmp_path, model_peer, request):
    """An ordinary isolated ``gateway.run`` process: ``(home, descriptor, env, root)``."""
    from tests.gateway.test_normal_runtime_boot import control

    home = tmp_path / "state"
    home.mkdir(mode=0o700)
    user = tmp_path / "user"
    user.mkdir()
    root = Path(__file__).resolve().parents[2]
    model_url = f"http://127.0.0.1:{model_peer.server_port}/v1"
    config = {
        "gateway": {"multiplex_profiles": False},
        "approvals": {"mode": "manual", "timeout": 60},
        "model": {"provider": "custom", "default": "local-wire-stub", "base_url": model_url},
        "auxiliary": {"title_generation": {"enabled": False}},
    }
    for key, value in (request.param.items() if isinstance(getattr(request, 'param', None), dict) else ()):
        config.setdefault(key, {}).update(value)
    (home / "config.yaml").write_text(json.dumps(config))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED="1",
               OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=model_url,
               HERMES_ACP_SKIP_CONFIGURED_MCP="1")
    command = [sys.executable, "-m", "gateway.run"]
    if getattr(request, 'param', None) == 'proposed-acp-descriptor':
        # API-owner handoff only: execution/create/policy stay unmodified. This
        # fixture explicitly distinguishes the proposed advert from shipped HEAD.
        command = [sys.executable, '-c', '''
from gateway.session_controls import AuthorityConnection
original = AuthorityConnection.describe
async def describe(self, ref, params):
    result = await original(self, ref, params)
    result['session_create']['sources'].append('acp')
    result['capabilities'].append('acp-editor-policy-v1')
    return result
AuthorityConnection.describe = describe
import runpy
runpy.run_module('gateway.run', run_name='__main__')
''']
    log_path = tmp_path / "gateway.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            descriptor = {}
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, "identify")
                    if descriptor.get("state") == "ready":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            assert descriptor.get("state") == "ready", log_path.read_text()
            yield home, descriptor, env, root
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.fixture
def server_spec(tmp_path):
    """A stdio MCP server spec whose tool echoes a borrowed credential."""
    script = tmp_path / 'owned_mcp.py'
    script.write_text('''import json, os, sys
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'owned-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':'owned echo','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call': result={'content':[{'type':'text','text':os.environ['BORROWED_CREDENTIAL']}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    return {'name': 'same-editor-name', 'command': sys.executable, 'args': [str(script)],
            'env': [{'name': 'BORROWED_CREDENTIAL', 'value': 'PRIVATE_A'}]}
