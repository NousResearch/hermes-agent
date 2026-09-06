"""#103549: interim notice must not claim/ack/dedup the batch's final result

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import os, sys, tempfile, socket, json, time, threading, queue, asyncio
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
root=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(root))
for key in list(os.environ):
    if key.startswith('HERMES_') or any(s in key for s in ('API_KEY','TOKEN','SECRET')):
        os.environ.pop(key,None)
os.environ['HERMES_HOME']=tempfile.mkdtemp(prefix='notice-review-')
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden'))
import tools.async_delegation as ad
import tools.delegate_tool_dispatch as dd
from tools.process_registry import process_registry as reg
from tools.process_registry_notifications import format_process_notification
from gateway.run_notifications import GatewayNotificationsMixin
assert Path(ad.__file__).is_relative_to(root)
assert Path(dd.__file__).is_relative_to(root)
print('SOURCE',ad.__file__,dd.__file__,flush=True)
class Sink(GatewayNotificationsMixin):
    def __init__(self):
        self._completion_delivery_lock=threading.Lock()
        self._completion_deliveries_inflight=set()
        self._completion_deliveries_delivered=OrderedDict()
        self._completion_delivery_retention=100
        self.received=[]
    async def _classify_completion_target(self,sid): return 'deliver'
    async def _inject_watch_notification(self,text,evt):
        self.received.append(('notice' if evt.get('task_failure_notice') else 'final',evt.get('results')))
        return True

def batch_run(did, gates):
    tasks=[{'goal':f'worker task {i}'} for i in range(3)]
    children=[(i,t,SimpleNamespace()) for i,t in enumerate(tasks)]
    b=dd._Batch(tasks,children,SimpleNamespace(quiet_mode=True),{'model':'offline'},None,'leaf',3,did,[],[], '', '',None,None,time.monotonic())
    def child(i,t,c):
        assert gates[i].wait(15)
        return {'task_index':i,'status':'error' if i<2 else 'completed','error':'offline failure' if i<2 else None,'summary':None if i<2 else 'FINISHED_SUCCESS','duration_seconds':0.1}
    b.run_child=child
    results=[]
    kwargs={'honor_parent_interrupt':False}
    if 'detached' in __import__('inspect').signature(dd._run_children_parallel).parameters: kwargs['detached']=True
    dd._run_children_parallel(b,results,**kwargs)
    return {'results':results,'total_duration_seconds':1}

def start(did):
    gates=[threading.Event() for _ in range(3)]
    h=ad.dispatch_async_delegation_batch(goals=['worker task 0','worker task 1','worker task 2'],context=None,toolsets=None,role='leaf',model='offline',session_key='owned',parent_session_id='parent',runner=lambda:batch_run(did,gates),delegation_id=did)
    assert h['status']=='dispatched'
    return gates

def get(timeout=2): return reg.completion_queue.get(timeout=timeout)
def deliver(sink,e): return asyncio.run(sink._deliver_completion_notification(format_process_notification(e),e))
results={}
if not hasattr(ad,'push_task_failure_notice'):
    g=start('base-control');g[0].set()
    try:get(.2);raise AssertionError('unexpected early event')
    except queue.Empty:pass
    g[1].set();g[2].set();e=get();sink=Sink();results['main_control']={'no_early_notice':True,'final_delivered':deliver(sink,e),'received':sink.received}
else:
    g=start('gateway-before-final');sink=Sink();g[0].set();first=get();assert first['task_failure_notice']
    results['gateway']={'first_notice_accepted':deliver(sink,first),'after_notice':ad.get_durable_delegation('gateway-before-final')}
    g[1].set();second=get();results['gateway']['second_notice_accepted']=deliver(sink,second)
    g[2].set();final=get();assert not final.get('task_failure_notice')
    results['gateway']['final_accepted']=deliver(sink,final)
    results['gateway']['received']=sink.received
    results['gateway']['after_final']=ad.get_durable_delegation('gateway-before-final')
    print('GATEWAY_RECEIVED', [k for k,_ in sink.received])
    assert [kind for kind,_ in sink.received]==['notice','notice','final'], 'fixed head must deliver both notices AND the final'
    # Parent busy: the queued first notice gets accepted only after final persisted.
    g=start('busy-parent');g[0].set();notice=get();g[1].set();second=get();g[2].set();final=get()
    claim=ad.claim_event_delivery(notice,'tui-poller');assert claim=='', 'an interim notice must be NON-durable (empty claim token)'
    ad.complete_event_delivery(notice,claim)
    final_claim=ad.claim_event_delivery(final,'tui-poller')
    results['busy_parent']={'notice_claimed':bool(claim),'final_claim':final_claim,'row':ad.get_durable_delegation('busy-parent')}
    assert final_claim, 'the final result must still be claimable after a notice was consumed'
    # Different TUI UI-dedup keys do not fix shared durable claim identity.
    from tui_gateway.session_notifications import _notification_event_dedup_key
    results['busy_parent']['ui_keys_differ']=_notification_event_dedup_key(notice)!=_notification_event_dedup_key(final)
print(json.dumps(results,indent=2),flush=True)
print('PROBE_COMPLETE',flush=True)
