"""#103513: child cap through repeated compression and persistence

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import os, sys, tempfile, json, socket, copy
from pathlib import Path
from types import SimpleNamespace
root, tag = sys.argv[1:3]
sys.path.insert(0, root)
os.chdir(root)
for k in list(os.environ):
    if any(s in k for s in ('API_KEY','TOKEN','SECRET')) or k.startswith('HERMES_'):
        os.environ.pop(k, None)
home = tempfile.mkdtemp(prefix='cap-review-')
os.environ['HERMES_HOME'] = home
os.environ['HERMES_DISABLE_REDACTION'] = 'true'
import yaml
cfg = {'model': {'default': 'anthropic/claude-fable-5.1', 'provider':'openai-compat', 'base_url':'http://127.0.0.1:1/v1', 'context_length':1000000}, 'compression':{'threshold':0.85}, 'delegation': {}}
if len(sys.argv)>3:
    cfg['delegation']['compression_threshold_tokens'] = json.loads(sys.argv[3])
Path(home,'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
def blocked(*a, **kw):
    raise RuntimeError('Network disabled in unpaid cap probe')
socket.socket.connect = blocked
socket.create_connection = blocked
from run_agent import AIAgent
import tools.delegate_tool as dt
import agent.context_compressor as mod
from agent.model_metadata import estimate_messages_tokens_rough
from hermes_state import SessionDB
from unittest.mock import patch
print('IDENTITY',json.dumps({'tag':tag,'tree':root,'delegate':dt.__file__,'compressor':mod.__file__,'cap_present':hasattr(dt,'_apply_child_compression_cap'),'home':home}),flush=True)
db=SessionDB(Path(home,'state.db'))
parent=AIAgent(api_key='test-key',base_url='http://127.0.0.1:1/v1',provider='openai-compat',model='anthropic/claude-fable-5.1',enabled_toolsets=[],quiet_mode=True,skip_context_files=True,skip_memory=True,save_trajectories=False,session_db=db)
child=dt._build_child_agent(task_index=0,goal='Review compression cap; continue the task.',context=None,toolsets=[],model=None,max_iterations=10,task_count=1,parent_agent=parent)
cc=child.context_compressor
print('SPAWN',json.dumps({'parent':parent.context_compressor.threshold_tokens,'child':cc.threshold_tokens,'cap':cc.threshold_tokens_cap,'tail':cc.tail_token_budget,'enabled':child.compression_enabled}),flush=True)
if len(sys.argv)>3:
    child.close(); parent.close(); db.close(); sys.exit(0)
# Real trigger and real compressor transitions, no hand-written compression behavior.
base=[{'role':'user','content':'Continue reviewing this project and preserve the current goal.'}]
for i in range(48):
    base.append({'role':'assistant' if i%2==0 else 'user','content':f'fixture {i} '+('A documented implementation detail with evidence and constraints. '*360)})
base.append({'role':'user','content':'Continue the review and report findings.'})
summary_calls=[]
def local_summary(**kw):
    assert kw['task']=='compression'
    summary_calls.append(len(kw['messages'][0]['content']))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='## Current task\nContinue reviewing the project. Preserve evidence and report findings.\n## Decisions\nNo changes have been applied.\n## Next steps\nInspect remaining details.',reasoning=None,reasoning_content=None),finish_reason='stop')],usage=None)
messages=base
for cycle in range(4):
    tokens=estimate_messages_tokens_rough(messages)
    cc.update_from_response({'prompt_tokens': tokens,'completion_tokens':0})
    should=cc.should_compress(tokens)
    before=len(messages)
    if should:
        with patch.object(mod,'call_llm',side_effect=local_summary):
            messages=cc.compress(messages,current_tokens=tokens)
    after=estimate_messages_tokens_rough(messages)
    print('CYCLE',json.dumps({'cycle':cycle,'before_tokens':tokens,'trigger':cc.threshold_tokens,'should':should,'before_messages':before,'after_tokens':after,'after_messages':len(messages),'compressions':cc.compression_count,'summary_calls':len(summary_calls),'blocked':cc.should_compress_info(tokens),'cap':cc.threshold_tokens_cap}),flush=True)
    cc.update_from_response({'prompt_tokens':after,'completion_tokens':0})
    if cycle < 3:
        messages=messages+copy.deepcopy(base[1:])
print('SUMMARY_INPUT_CHARS',summary_calls,flush=True)
# Model switching must keep the cap, including ratio-lower small windows.
for window in [128000,1000000]:
    cc.update_model(child.model,window,provider=child.provider,base_url=child.base_url,api_mode=child.api_mode)
    print('MODEL_SWITCH',window,cc.threshold_tokens,cc.threshold_tokens_cap,flush=True)
if hasattr(dt,'_apply_child_compression_cap'):
    for raw in ['200k','default',True,False,0,None,200000.9,float('inf')]:
        temp=SimpleNamespace(context_compressor=mod.ContextCompressor(model=child.model,threshold_percent=.85,config_context_length=1000000,quiet_mode=True))
        try:
            dt._apply_child_compression_cap(temp,{'compression_threshold_tokens':raw})
            print('CONFIG',repr(raw),temp.context_compressor.threshold_tokens,temp.context_compressor.threshold_tokens_cap,flush=True)
        except Exception as exc:
            print('CONFIG_EXCEPTION',repr(raw),type(exc).__name__,str(exc),flush=True)
# Accounting policy: the cap changes the threshold, not what route-aware pressure counts.
from agent.turn_context import _preflight_request_tokens, _agent_stale_thinking_on_wire
history=[{'role':'user','content':'review'}]
for i in range(24):
    history += [{'role':'assistant','content':'observed','reasoning_content':'reasoning detail '*4000},{'role':'user','content':'continue'}]
for provider, model in [('openai-compat','test-model'),('deepseek','deepseek-chat')]:
    route=SimpleNamespace(provider=provider,model=model,base_url='http://127.0.0.1:1/v1',api_mode='chat_completions',tools=[])
    pressure=_preflight_request_tokens(route,history,'')
    print('ACCOUNTING',provider,_agent_stale_thinking_on_wire(route),pressure,cc.should_compress(pressure),flush=True)
# Already-cached legacy tail is an explicit compatibility edge.
if hasattr(dt,'_apply_child_compression_cap'):
    legacy=SimpleNamespace(context_compressor=mod.ContextCompressor(model=child.model,threshold_percent=.85,config_context_length=1000000,tail_mode='legacy',quiet_mode=True))
    old_tail=legacy.context_compressor.tail_token_budget
    dt._apply_child_compression_cap(legacy,{})
    print('LEGACY_RESOLVED',old_tail,legacy.context_compressor.tail_token_budget,legacy.context_compressor.threshold_tokens,flush=True)
child.close();parent.close();db.close()
print('DONE',tag,flush=True)
