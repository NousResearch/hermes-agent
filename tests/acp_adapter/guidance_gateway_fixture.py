"""ACP consumer fixture: ordinary gateway owner/AIAgent, loopback model only.

Fault injection is limited to a lost ACP admission response and model barriers.
Neither the execution owner nor its conversation loop is replaced.
"""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import acp
from acp.exceptions import RequestError
from acp_adapter.gateway_server import GatewayACPAgent
from agent.active_turn_guidance import CAPABILITY
from tests.gateway.fixtures.guidance_model import GuidanceModel
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon


class FixtureGateway(GatewayACPAgent):
    def __init__(self, peer, owner_pid, trace, case):
        super().__init__()
        self.peer, self.owner_pid, self.trace, self.case = peer, owner_pid, trace, case
        self.steers = 0

    async def ext_method(self, method, params):
        if method == 'ai.nous/steer':
            self.steers += 1
        result = await super().ext_method(method, params)
        if method == 'ai.nous/steer' and result.get('disposition') == 'accepted_active':
            self.peer.answer = os.environ.get('GUIDANCE_MARKER') or params['text']
            self.peer.release.set()
            if self.case == 'lost_reply':
                raise RequestError.internal_error('controlled lost ACP admission response')
        if method == 'ai.nous/guidanceStatus' and result.get('state') == 'incorporated':
            self.peer.observed.set()
        return result

    async def prompt(self, prompt, session_id, **kwargs):
        if self._conn:
            await self._conn.session_update(session_id, acp.update_agent_thought_text('fixture ready'))
        result = await super().prompt(prompt, session_id, **kwargs)
        snapshot = (result.field_meta or {}).get(CAPABILITY, {})
        with sqlite3.connect(self._home / 'state.db') as db:
            runs = db.execute('SELECT count(*) FROM session_admissions').fetchone()[0]
        trace = {'kind': 'gateway', 'pid': os.getpid(), 'ownerPid': self.owner_pid,
                 'runs': runs, 'modelCalls': len(self.peer.requests), 'steers': self.steers,
                 'turn': snapshot.get('turnId'), 'queued': [],
                 'messages': self.peer.requests[-1].get('messages', []),
                 'messageIds': [r['messageId'] for r in snapshot.get('receipts', [])]}
        self.trace.write_text(json.dumps(trace), encoding='utf-8')
        return result


def main():
    home, trace = Path(os.environ['HERMES_HOME']), Path(os.environ['GUIDANCE_TRACE'])
    case = os.environ.get('GUIDANCE_CASE', '')
    marker = os.environ.get('GUIDANCE_MARKER', '')
    env = child_env()
    user = home / 'user'
    user.mkdir(parents=True, exist_ok=True)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), GuidanceModel)
    peer.requests = []
    peer.blocked, peer.release, peer.observed = threading.Event(), threading.Event(), threading.Event()
    peer.case, peer.hold_final = case, case not in {'unconsumed', 'final_receipt'}
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'controlled-gateway', 'base_url': url},
        'platform_toolsets': {'acp': ['terminal']},
        'auxiliary': {'title_generation': {'enabled': False}}, 'model_catalog': {'enabled': False},
        'models_dev': {'url': url.removesuffix('/v1') + '/models-dev-offline'},
    }))
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1',
               HERMES_ACP_SKIP_CONFIGURED_MCP='1', GUIDANCE_MARKER=marker)
    os.environ.clear()
    os.environ.update(env)
    try:
        with daemon(ROOT, home, env, barrier=False) as (process, _):
            asyncio.run(acp.run_agent(FixtureGateway(peer, process.pid, trace, case), use_unstable_protocol=True))
    finally:
        peer.release.set()
        peer.observed.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
