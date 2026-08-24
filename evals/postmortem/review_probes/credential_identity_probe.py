"""#103526: explicit account-A key must not be replaced by the singleton's account-B key (real SDK -> loopback)

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import os, tempfile, sys, json, time, base64, threading, importlib.util
from pathlib import Path
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
ROOT = Path(sys.argv[1]); MODE = sys.argv[2]
sys.path.insert(0, str(ROOT))
home = Path(tempfile.mkdtemp(prefix='pr103526-probe-'))
os.environ.update(HOME=str(home), HERMES_HOME=str(home/'hermes'), XDG_CONFIG_HOME=str(home/'config'), CODEX_HOME=str(home/'codex'))
for k in list(os.environ):
    if any(x in k for x in ('API_KEY','TOKEN','NOUS_','SECRET')): os.environ.pop(k, None)
(home/'hermes').mkdir()
(home/'hermes'/'config.yaml').write_text('nous:\n  keepalive_interval_seconds: 0\nmemory:\n  memory_enabled: false\n', encoding='utf-8')
def guard(event,args):
    if event == 'socket.connect' and isinstance(args[1], tuple) and args[1][0] not in ('127.0.0.1','::1'):
        raise RuntimeError('External network forbidden by review probe')
sys.addaudithook(guard)
def jwt(sub, ttl):
    def part(v): return base64.urlsafe_b64encode(json.dumps(v).encode()).decode().rstrip('=')
    return part({'alg':'none'})+'.'+part({'sub':sub,'scope':'inference:invoke','exp':int(time.time()+ttl)})+'.sig'
def claims(token):
    return json.loads(base64.urlsafe_b64decode(token.split('.')[1]+'==='))
records=[]
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw_body=self.rfile.read(int(self.headers.get('Content-Length',0)))
        if self.path == '/api/oauth/token':
            records.append({'path':self.path,'refresh':True})
            raw=json.dumps({'access_token':refresh_reply,'refresh_token':'fixture-rotated','expires_in':3600,'token_type':'Bearer','scope':'inference:invoke'}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        body=json.loads(raw_body or '{}')
        bearer=self.headers.get('Authorization','').removeprefix('Bearer ')
        records.append({'path':self.path,'sub':claims(bearer).get('sub') if bearer else None})
        if bearer and claims(bearer)['exp'] < time.time():
            raw=json.dumps({'error':{'message':'expired bearer','type':'authentication_error'}}).encode();self.send_response(401);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        data={'id':'local-probe','object':'chat.completion','created':int(time.time()),'model':'hermes-test','choices':[{'index':0,'message':{'role':'assistant','content':'local-only'},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        raw=json.dumps(data).encode(); self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def log_message(self,*a): pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler); threading.Thread(target=server.serve_forever,daemon=True).start()
url=f'http://127.0.0.1:{server.server_port}/v1'
# Runtime override preserves loopback routing, without relaxing URL validation.
os.environ['NOUS_INFERENCE_BASE_URL']=url
os.environ['HERMES_SHARED_AUTH_DIR']=str(home/'shared')
from run_agent import AIAgent
from agent.turn_iteration_prep import prepare_iteration
import agent.client_lifecycle as lifecycle
import hermes_cli.auth as auth
print(json.dumps({'mode':MODE,'module':lifecycle.__file__,'has_new':hasattr(AIAgent,'_adopt_nous_key_before_expiry'),'home':str(home)}),flush=True)
if MODE=='main':
    spec=importlib.util.spec_from_file_location('main_prep',Path(__file__).with_name('main-turn_iteration_prep.py')); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);prepare_iteration=mod.prepare_iteration

def store(token):
    exp=claims(token)['exp']; state={'portal_base_url':'https://portal.nousresearch.com','inference_base_url':'https://inference-api.nousresearch.com/v1','client_id':'hermes-cli','token_type':'Bearer','scope':'inference:invoke','access_token':token,'refresh_token':'fixture-refresh-never-send','expires_at':datetime.fromtimestamp(exp,timezone.utc).isoformat(),'expires_in':3600,'agent_key':token,'agent_key_expires_at':datetime.fromtimestamp(exp,timezone.utc).isoformat()}
    (home/'hermes'/'auth.json').write_text(json.dumps({'version':1,'active_provider':'nous','providers':{'nous':state}}), encoding='utf-8')
results=[]
for case, own_sub, store_sub, ttl in [('same-account','account-A','account-A',30),('explicit-account','account-A','account-B',30),('far-from-expiry','account-A','account-B',3000)]:
    own=jwt(own_sub,ttl);fresh=jwt(store_sub,3600);store(fresh)
    agent=AIAgent(api_key=own,base_url=url,provider='nous',model='hermes-test',quiet_mode=True,skip_context_files=True,skip_memory=True,enabled_toolsets=[])
    messages=[{'role':'user','content':'local fixture'}]; before=json.dumps(messages)
    prepare_iteration(agent,messages=messages,api_call_count=0)
    client=agent._create_request_openai_client(reason='review_probe')
    reply=client.chat.completions.create(model='hermes-test',messages=messages)
    result={'case':case,'before_sub':own_sub,'after_sub':claims(agent.api_key)['sub'],'adopted_fresh':agent.api_key==fresh,'messages_unchanged':json.dumps(messages)==before,'wire':records[-1],'reply':reply.choices[0].message.content}
    results.append(result); print(json.dumps(result),flush=True)
    agent._close_request_openai_client(client,reason='probe_done')
    agent.client.close()
# Contended peer-adoption: real auth-store locking and SDK wire, no resolver mocks.
from concurrent.futures import ThreadPoolExecutor
fresh=jwt('account-A',3600); store(fresh)
expired=jwt('account-A',-30)
barrier=threading.Barrier(12)
def worker(_):
    a=AIAgent(api_key=expired,base_url=url,provider='nous',model='hermes-test',quiet_mode=True,skip_context_files=True,skip_memory=True,enabled_toolsets=[])
    barrier.wait(timeout=30)
    messages=[{'role':'user','content':'concurrent local fixture'}]
    prepare_iteration(a,messages=messages,api_call_count=0)
    c=a._create_request_openai_client(reason='concurrent_review_probe')
    try:
        c.chat.completions.create(model='hermes-test',messages=messages)
        status=200
    except Exception as e:
        status=getattr(e,'status_code',type(e).__name__)
    finally:
        a._close_request_openai_client(c,reason='probe_done'); a.client.close()
    return status
with ThreadPoolExecutor(max_workers=12) as executor:
    statuses=list(executor.map(worker,range(12)))
print(json.dumps({'concurrent_statuses':statuses,'successes':statuses.count(200),'401s':statuses.count(401)}),flush=True)
# No fresh peer key: twelve agents contend for one real local refresh POST.
refresh_reply=jwt('account-A',3600)
if MODE != 'main':
    import shutil
    shutil.rmtree(home/'shared',ignore_errors=True)
    expired=jwt('account-A',30); store(expired)
    state_file=home/'hermes'/'auth.json'
    state=json.loads(state_file.read_text(encoding='utf-8')); state['providers']['nous']['portal_base_url']=url.removesuffix('/v1');state_file.write_text(json.dumps(state), encoding='utf-8')
    barrier=threading.Barrier(12);records.clear()
    with ThreadPoolExecutor(max_workers=12) as executor:
        mint_statuses=list(executor.map(worker,range(12)))
    posts=sum(r.get('refresh',False) for r in records)
    print(json.dumps({'mint_statuses':mint_statuses,'refresh_posts':posts}),flush=True)
    assert posts == 1 and mint_statuses == [200]*12
server.shutdown()
(Path(os.environ.get('PROBE_OUT', tempfile.gettempdir()))/('cred-probe-'+MODE+'.json')).write_text(json.dumps({'cases':results,'concurrent_statuses':statuses},indent=2), encoding='utf-8')
assert all(r['messages_unchanged'] for r in results)
# Regression contract (fixed head): the explicit account-A key must stay A; same-account adoption must still happen.
assert results[1]['after_sub']=='account-A', 'explicit-account key was replaced by the singleton account (the round-1 defect)'
assert results[0]['after_sub']=='account-A' and results[0]['adopted_fresh'], 'same-account fresh key must still be adopted'
