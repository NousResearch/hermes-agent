"""#103486: nested delegate deadline through actual dispatch

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import os, sys, tempfile, pathlib, json, time, threading, socket, subprocess
from types import SimpleNamespace as NS
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root)); os.chdir(root)
for k in list(os.environ):
    if k.startswith('HERMES_') or k.endswith(('_API_KEY', '_TOKEN')):
        os.environ.pop(k, None)
home = tempfile.TemporaryDirectory(prefix='deadline-probe-')
os.environ['HERMES_HOME'] = home.name
os.environ['HERMES_DISABLE_TELEMETRY'] = '1'
pathlib.Path(home.name, 'config.yaml').write_text('timeouts:\n  tools:\n    sequential_call: 0.3\ndelegation:\n  max_summary_chars: 24000\n', encoding='utf-8')
# Any accidental provider, metadata, or telemetry request fails closed.
def no_connect(*args, **kwargs):
    raise RuntimeError('OFFLINE PROBE: network forbidden')
socket.socket.connect = no_connect
socket.create_connection = no_connect
import agent.tool_executor as te
import agent.turn_usage as tu
import tools.delegate_tool_results as dr
import agent.usage_pricing as up
import agent.codex_runtime as cr
from run_agent import AIAgent
for mod in (te, tu, dr, up, cr):
    assert pathlib.Path(mod.__file__).is_relative_to(root), mod.__file__
print(json.dumps({'sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True, encoding='utf-8', errors='replace').strip(), 'modules': {m.__name__:m.__file__ for m in (te,tu,dr,up,cr)}, 'home':home.name}), flush=True)
a = AIAgent(api_key='offline-fixture', base_url='http://127.0.0.1:9/v1', provider='openai-compat', model='offline-test', enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False)
a._delegate_depth = 1; a._delegate_role = 'orchestrator'
assert te._resolve_sequential_tool_timeout() == 0.3
rows=[]
for name in ('delegate_task','terminal','execute_code'):
    entered=threading.Event(); finished=threading.Event()
    def work(args):
        entered.set()
        time.sleep(0.65)
        finished.set()
        return json.dumps({'marker':'LEAF_DONE_MARKER'})
    start=time.monotonic()
    result=te._run_sequential_tool_execution_middleware(a,function_name=name,function_args={},effective_task_id='offline',tool_call_id='probe-'+name,execute=work)
    elapsed=time.monotonic()-start
    assert entered.is_set(), 'tool body never entered'
    row={'case':name,'elapsed':round(elapsed,4),'result_type':type(result.result).__name__,'result':str(result.result),'completed_on_return':finished.is_set()}
    assert finished.wait(2), 'fixture did not finish'
    rows.append(row)
print(json.dumps({'tool_boundary':rows}),flush=True)
# Warm real middleware, then interrupt a non-cooperative delegated operation.
a._interrupt_requested=False
entered=threading.Event(); release=threading.Event()
def blocked(args):
    entered.set(); release.wait(8); return 'late'
def interrupt():
    assert entered.wait(2)
    time.sleep(0.1)
    a.interrupt('offline cancellation probe')
t = threading.Thread(target=interrupt); t.start()
start=time.monotonic()
r=te._run_sequential_tool_execution_middleware(a,function_name='delegate_task',function_args={},effective_task_id='offline',tool_call_id='probe-interrupt',execute=blocked)
elapsed=time.monotonic()-start
release.set();t.join()
print(json.dumps({'interrupt':{'elapsed':round(elapsed,4),'result_type':type(r.result).__name__,'result':str(r.result)}}),flush=True)
# Feed provider-shaped data to the production writer, not fabricated _last_turn_usage.
# Price lookup is unrelated and potentially networked; only pricing is stubbed.
tu.estimate_usage_cost=lambda *args,**kwargs: NS(amount_usd=None,status='unknown',source='offline')
up.estimate_usage_cost=tu.estimate_usage_cost
class Compressor:
    context_length=200000
    max_tokens=8000
    threshold_tokens=190000
    def update_from_response(self, usage):
        self.last_prompt_tokens=usage.get('prompt_tokens',-1)
def parent(provider,mode='chat_completions',client=None):
    b=NS(context_compressor=Compressor(),provider=provider,api_mode=mode,model='offline',base_url='',client=client,session_id='offline-usage',_session_db=None,quiet_mode=True,verbose_logging=False)
    for key in ('api_calls','prompt_tokens','completion_tokens','total_tokens','input_tokens','output_tokens','cache_read_tokens','cache_write_tokens','reasoning_tokens','estimated_cost_usd'):
        setattr(b,'session_'+key,0)
    return b
def record(b,raw):
    tu.record_response_usage(b,NS(usage=raw),messages=[{'role':'user','content':'fixture'}],api_call_count=1,api_duration=0,compression_attempts=0,max_compression_attempts=3)
    return {'prompt':b._last_turn_usage['prompt_tokens'],'input':b._last_turn_usage['input_tokens'],'cache_read':b._last_turn_usage['cache_read_tokens'],'cache_write':b._last_turn_usage['cache_write_tokens'],'budget':dr._parent_summary_char_budget(b,1)}
chat={'prompt_tokens':30000,'completion_tokens':100,'prompt_tokens_details':{'cached_tokens':20000,'cache_write_tokens':5000}}
anth={'input_tokens':5000,'output_tokens':100,'cache_read_input_tokens':20000,'cache_creation_input_tokens':5000}
responses={'input_tokens':30000,'output_tokens':100,'input_tokens_details':{'cached_tokens':20000,'cache_write_tokens':5000}}
cases=[('openai', 'chat_completions',chat),('nous','chat_completions',chat),('openrouter','chat_completions',chat),('anthropic','anthropic_messages',anth),('minimax','anthropic_messages',anth),('minimax-cn','anthropic_messages',anth),('bedrock','chat_completions',{'prompt_tokens':30000,'completion_tokens':100,'cache_read_input_tokens':20000,'cache_creation_input_tokens':5000}),('google','chat_completions',chat),('deepseek','chat_completions',{'prompt_tokens':30000,'completion_tokens':100,'prompt_cache_hit_tokens':20000}),('moonshot','chat_completions',{'prompt_tokens':30000,'completion_tokens':100,'cached_tokens':20000}),('openai-codex','codex_responses',responses),('openai-compat','chat_completions',{'input_tokens':30000,'output_tokens':100})]
usage_rows=[]
for provider,mode,raw in cases:
    b=parent(provider,mode)
    fresh=record(b,raw)
    assert fresh['prompt']==30000
    b.session_prompt_tokens=25000000
    long_budget=dr._parent_summary_char_budget(b,1)
    usage_rows.append({'provider':provider,'mode':mode,**fresh,'long_lived_budget':long_budget})
# Raw native adapter response -> production conversion -> writer -> budget.
from agent.bedrock_adapter import normalize_converse_response
from agent.gemini_native_adapter import _usage_from_metadata
native=[]
for provider, raw in [('bedrock',normalize_converse_response({'usage':{'inputTokens':5000,'cacheReadInputTokens':20000,'cacheWriteInputTokens':5000,'outputTokens':100}}).usage),('google',_usage_from_metadata({'promptTokenCount':30000,'cachedContentTokenCount':25000,'candidatesTokenCount':100,'totalTokenCount':30100}))]:
    b=parent(provider); values=record(b,raw); assert values['prompt']==30000
    native.append({'provider':provider,**values})
print(json.dumps({'provider_writers':usage_rows,'native_adapters':native}),flush=True)
# Exercise the actual MoA accounting deposit/consume path without model calls.
from agent.moa_loop import MoAClient
moa_client = MoAClient('offline')
moa_client.chat.completions._fold_pending_accounting(up.CanonicalUsage(input_tokens=240000), None)
b=parent('moa',client=moa_client)
moa=record(b,chat)
results=[{'task_index':0,'summary':'X'*5000}]
dr._apply_summary_budget(results,b)
print(json.dumps({'moa':{**moa,'actual_aggregator_prompt':30000,'anchor_prompt':b._usage_anchor.prompt_tokens if hasattr(b._usage_anchor,'prompt_tokens') else repr(b._usage_anchor),'truncated':results[0].get('summary_truncated',False)}}),flush=True)
b=parent('openai-codex','codex_app_server')
b._last_turn_usage=None
codex=cr._record_codex_app_server_usage(b,NS(token_usage_last={'inputTokens':170000,'cachedInputTokens':20000,'outputTokens':100},model_context_window=200000))
print(json.dumps({'codex_app_server':{'returned_prompt':codex['prompt_tokens'],'last_turn_usage':b._last_turn_usage,'compressor_prompt':b.context_compressor.last_prompt_tokens,'budget':dr._parent_summary_char_budget(b,1)}}),flush=True)
# Missing usage and full-context defaults, including a current turn with no usage.
b=parent('openai-compat'); record(b,{'prompt_tokens':190000,'completion_tokens':100})
print(json.dumps({'near_full_budget':dr._parent_summary_char_budget(b,1),'near_full_batch_budget':dr._parent_summary_char_budget(b,5)}),flush=True)
# A successful provider response without usage keeps no current usage after turn reset.
b._last_turn_usage=None
record_outcome=tu.record_response_usage(b,NS(usage=None),messages=[],api_call_count=1,api_duration=0,compression_attempts=0,max_compression_attempts=3)
missing=[{'task_index':0,'summary':'X'*20000}]
dr._apply_summary_budget(missing,b)
print(json.dumps({'missing_usage_next_turn':{'session_prompt':b.session_prompt_tokens,'compressor_prompt':b.context_compressor.last_prompt_tokens,'last_turn_usage':b._last_turn_usage,'budget':dr._parent_summary_char_budget(b,1),'summary_length':len(missing[0]['summary']),'truncated':missing[0].get('summary_truncated',False)}}),flush=True)
# Full sequential executor -> resolver -> real middleware -> result commit.
# Replace only the model/delegation body; no subagents or model requests.
a._interrupt_requested=False
full_entered=threading.Event()
def full_body(args):
    full_entered.set(); time.sleep(0.65)
    return json.dumps({'marker':'FULL_BOUNDARY_DONE'})
a._dispatch_delegate_task=full_body
calls=NS(tool_calls=[NS(id='full-boundary',type='function',function=NS(name='delegate_task',arguments='{}'))])
messages=[]
start=time.monotonic()
te.execute_tool_calls_sequential(a,calls,messages,'offline-full')
assert full_entered.is_set()
print(json.dumps({'full_sequential_boundary':{'elapsed':round(time.monotonic()-start,4),'messages':messages}}),flush=True)
print('PROBE_COMPLETE',flush=True)
