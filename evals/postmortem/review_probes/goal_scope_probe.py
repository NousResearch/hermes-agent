"""#103496/#103534: judge process scoping and delegation WAIT lifecycle

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import os,sys,tempfile,json,subprocess,time,queue,asyncio,contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,os.getcwd())
home=tempfile.TemporaryDirectory(prefix='goals-probe-')
os.environ['HERMES_HOME']=home.name
os.environ['HERMES_TEST_MODE']='1'
from hermes_cli import goals
from tools import process_registry as pr,async_delegation as ad
from hermes_cli.cli_loops_mixin import CLILoopsMixin
from gateway.run_goals import GatewayGoalsMixin
from tui_gateway import server as pt
out={'module':goals.__file__,'sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True, encoding='utf-8', errors='replace').strip()}
proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],stdin=subprocess.DEVNULL)
try:
 reg=pr.process_registry
 now=time.time()
 records={'proc_own':pr.ProcessSession(id='proc_own',command='own sleeper',task_id='default',owner_task_id='root',pid=proc.pid,started_at=now), 'proc_child':pr.ProcessSession(id='proc_child',command='child sleeper',task_id='default',owner_task_id='sa-child',pid=proc.pid,started_at=now)}
 deleg={'d':{'status':'running','parent_session_id':'root','session_key':'','origin_ui_session_id':''}}
 with patch.object(reg,'_running',records),patch.object(reg,'_finished',{}),patch.object(ad,'_records',deleg):
  out['registry']=reg.list_sessions()
  out['all']=[r['session_id'] for r in goals.gather_background_processes()]
  if hasattr(goals,'count_active_delegations'):out['active']=goals.count_active_delegations('root')
  captured=[]
  def judge(*a,**kw):
   captured.append({'processes':[x['session_id'] for x in kw.get('background_processes',[]) or []], 'active':kw.get('active_delegations',0)})
   return ('continue','test',False,None,False)
  class CLI(CLILoopsMixin):
   def _get_goal_manager(self): return self.mgr
  c=CLI();c.session_id='root';c.agent=SimpleNamespace(session_id='root');c._pending_input=queue.Queue();c.conversation_history=[{'role':'assistant','content':'Waiting on workers'}]
  class GW(GatewayGoalsMixin):
   async def _post_turn_manager(self,*a):return self.mgr
   async def _run_in_executor_with_context(self,fn):return fn()
  g=GW()
  with patch.object(goals,'judge_goal',side_effect=judge):
   for label in ['cli','gateway','tui']:
    m=goals.GoalManager(session_id='root');m.set('finish');c.mgr=g.mgr=m
    if label=='cli':c._maybe_continue_goal_after_turn()
    elif label=='gateway':asyncio.run(g._post_turn_goal_continuation(session_entry=SimpleNamespace(session_id='root'),source=None,final_response='Waiting on workers'))
    else:
     with patch.object(pt,'_active_goal_manager',return_value=m),patch.object(pt,'_plan_goal_compression_recovery',return_value=(None,None)),patch.object(pt,'_emit'):
      pt._goal_followup_after_turn('tab',{'session_key':'root'}, {'final_response':'Waiting on workers'},'complete','Waiting on workers')
    out[label]=captured[-1]
  m=goals.GoalManager(session_id='expiry');m.set('finish');m.wait_on(proc.pid);m.state.waiting_since=now-1801;m._save()
  out['old_pid_waiting']=m.is_waiting()
  m.wait_on_session('proc_own');m.state.waiting_since=now-1801;m._save();out['old_session_waiting']=m.is_waiting()
  m.wait_for_seconds(3600);m.state.waiting_since=now-1801;m._save();out['long_timer_still_waiting']=m.is_waiting()
  m.stop_waiting();m.wait_for_seconds(1200,reason='delegates');deleg['d']['status']='completed'
  with patch.object(goals,'judge_goal',side_effect=judge):
   before=len(captured)
   kw={'active_delegations':0} if hasattr(goals,'count_active_delegations') else {}
   out['after_delegation_complete']=m.evaluate_after_turn('Workers finished; next step is integration',**kw)
   out['completion_judge_calls']=len(captured)-before
  out['persisted_wait_until']=goals.GoalManager(session_id='expiry').state.waiting_until
  # Drive the actual new judge branch, then deliver a worker-result turn.
  c._pending_input=queue.Queue(); c.mgr=goals.GoalManager(session_id='root');c.mgr.set('finish integration')
  deleg['d']['status']='running'; calls=[]
  def fake_aux(*args,**kwargs):
   text=str(kwargs.get('messages') or args);calls.append(text)
   content='{"verdict":"wait","wait_for_seconds":1200,"reason":"workers"}' if 'Active delegations:' in text else '{"verdict":"continue","reason":"integrate next"}'
   return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
  with patch('agent.auxiliary_client.call_llm',side_effect=fake_aux):
   c._maybe_continue_goal_after_turn();out['lifecycle_initial_pending']=c._pending_input.qsize()
   while not c._pending_input.empty():c._pending_input.get_nowait()
   deleg['d']['status']='completed';c.conversation_history=[{'role':'assistant','content':'Workers finished; I still need to integrate their branches.'}]
   c._maybe_continue_goal_after_turn()
   out['lifecycle_completion_pending']=c._pending_input.qsize();out['lifecycle_aux_calls']=len(calls);out['lifecycle_still_waiting']=c.mgr.is_waiting()
finally:
 proc.terminate();proc.wait(timeout=10)
Path(sys.argv[1]).write_text(json.dumps(out,indent=2), encoding='utf-8')
print(json.dumps(out,indent=2))
