"""#103476: preflight estimate vs Anthropic wire size with retained thinking

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import os,sys,tempfile,copy,json,types,subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1] if len(sys.argv)>1 else os.getcwd());os.environ['HERMES_HOME']=tempfile.mkdtemp(prefix='cache-boundary-')  # usage: <repo_root>
from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.context_compressor import ContextCompressor
from agent.turn_context import _preflight_request_tokens
from agent.model_metadata import estimate_messages_tokens_rough
model='anthropic/claude-fable-5.1';url='https://inference-api.nousresearch.com/v1'
cc=ContextCompressor(model=model,provider='nous',base_url=url,api_mode='anthropic_messages',config_context_length=1000000,threshold_tokens_cap=200000)
rows=[{'role':'user','content':'Investigate repository.'}]
for i in range(48):
 text='reasoning text ' * 1800
 rows.extend([{'role':'assistant','content':'tool','reasoning':text,'reasoning_details':[{'type':'thinking','thinking':text,'signature':f'signature-{i}'}],'tool_calls':[{'id':f't{i}','type':'function','function':{'name':'terminal','arguments':'{}'}}]},{'role':'tool','tool_call_id':f't{i}','content':f'Result {i}'}])
a=types.SimpleNamespace(api_mode='anthropic_messages',provider='nous',model=model,base_url=url,tools=[],_usage_anchor=None)
pre=_preflight_request_tokens(a,rows,'');wire=convert_messages_to_anthropic(rows,base_url=url,model=model)[1];wire_est=estimate_messages_tokens_rough(wire)
print(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True, encoding='utf-8', errors='replace').strip(),'preflight':pre,'wire_estimate':wire_est,'preflight_should_compress':cc.should_compress(pre),'wire_estimate_should_compress':cc.should_compress(wire_est)}))
# Local signature-prefix checker validates the documented contract only; it is not the provider.
def prefixes(messages):
 before=[]; result={}
 for msg in messages:
  nonthinking=[]
  for b in msg['content'] if isinstance(msg['content'],list) else [{'type':'text','text':msg['content']}]:
   if b.get('type')=='thinking':result[b['signature']]=json.dumps(before,sort_keys=True)
   else:nonthinking.append(b)
  before.append({'role':msg['role'],'content':nonthinking})
 return result
before=prefixes(wire)
cc._generate_summary=lambda *a,**k:'Task: investigate repository. Continue reviewing remaining files.'
out=cc.compress(copy.deepcopy(rows),current_tokens=wire_est,force=True)
after=prefixes(convert_messages_to_anthropic(out,base_url=url,model=model)[1])
print('retained',list(after),'prefix_changed',[k for k,v in after.items() if before.get(k)!=v])
print('retained with unchanged system; actual agent rebuilds system too, so this is lower bound')
