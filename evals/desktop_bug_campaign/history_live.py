"""Real serve/HTTP/WS history witness. No provider calls; seeded SQLite data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--port', type=int, default=18146)
args = parser.parse_args()
repo, tag = str(Path(args.repo).resolve()), 'run'
sys.path.insert(0, repo)
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=False)
home = Path(tempfile.mkdtemp(prefix='history-'+tag+'-', dir=out))
os.environ['HOME'] = str(home)
os.environ['HERMES_HOME'] = str(home)
from hermes_state import SessionDB
from websockets.sync.client import connect

items = [{'type': 'message', 'role': 'assistant', 'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': 'HISTORY_SIDECAR_REPLY'}]}]
with SessionDB(db_path=home/'state.db') as db:
    db.create_session('history-witness', source='desktop')
    db.append_message('history-witness', 'user', 'HISTORY_USER_PROMPT')
    db.append_message('history-witness', 'assistant', '', codex_message_items=items, reasoning='THINKING_BEFORE_TOOL', tool_calls=[{'id':'call-1','type':'function','function':{'name':'terminal','arguments':'{}'}}])
    db.append_message('history-witness', 'tool', 'TOOL_OUTPUT', tool_call_id='call-1')
    db.create_session('history-control', source='desktop')
    db.append_message('history-control', 'user', 'CONTROL_USER')
    db.append_message('history-control', 'assistant', 'CONTROL_REPLY')
env = {k: v for k, v in os.environ.items() if k in ['PATH', 'HOME', 'LANG', 'USER', 'VIRTUAL_ENV']}
env.update(HOME=str(home), HERMES_HOME=str(home), HERMES_DASHBOARD_SESSION_TOKEN='history-fixture-token', PYTHONPATH=repo, HERMES_NONINTERACTIVE='1')
cmd=[sys.executable, '-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', str(args.port), '--isolated']
log=open(out/(tag+'-serve.log'),'w',encoding='utf-8')
p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
frames=[]
try:
    for _ in range(120):
        try:
            req=urllib.request.Request(f'http://127.0.0.1:{args.port}/api/status',headers={'X-Hermes-Token':'history-fixture-token'})
            status=json.load(urllib.request.urlopen(req,timeout=2));break
        except Exception:
            if p.poll() is not None: raise RuntimeError('serve exited')
            time.sleep(.5)
    else: raise RuntimeError('serve never healthy')
    def rpc(ws,method,params):
        rid=len(frames)+1
        request={'jsonrpc':'2.0','id':rid,'method':method,'params':params}
        frames.append({'sent':request});ws.send(json.dumps(request))
        while True:
            reply=json.loads(ws.recv(timeout=30));frames.append({'received':reply})
            if reply.get('id')==rid:
                if 'error' in reply: raise RuntimeError(reply)
                return reply['result']
    results={}
    for sid in ['history-witness','history-control']:
        with connect(f'ws://127.0.0.1:{args.port}/api/ws?token=history-fixture-token') as ws:
            first=rpc(ws,'session.resume',{'session_id':sid,'lazy':True})
            rid=first['session_id']
            history=rpc(ws,'session.history',{'session_id':rid})
        with connect(f'ws://127.0.0.1:{args.port}/api/ws?token=history-fixture-token') as ws:
            resumed=rpc(ws,'session.resume',{'session_id':sid,'lazy':True})
            activated=rpc(ws,'session.activate',{'session_id':resumed['session_id']})
            after=rpc(ws,'session.history',{'session_id':resumed['session_id']})
        results[sid]={'first':first,'history':history,'resumed':resumed,'activated':activated,'after':after}
    receipt={'tag':tag,'sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,stdin=subprocess.DEVNULL,text=True).strip(),'command':cmd,'home':str(home),'results':results}
    (out/(tag+'-rpc.json')).write_text(json.dumps(receipt,indent=2))
    (out/(tag+'-frames.json')).write_text(json.dumps(frames,indent=2))
    for sid, responses in results.items():
        for method, value in responses.items():
            messages = value['messages']
            assert len(messages) == (3 if sid == 'history-witness' else 2), (sid, method)
            if sid == 'history-witness':
                assert 'HISTORY_SIDECAR_REPLY' in json.dumps(messages[1]), method
            else:
                assert 'CONTROL_REPLY' in json.dumps(messages[1]), method
    print('10 real RPC history controls passed; receipts saved:', out)
finally:
    p.terminate()
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    log.close()
