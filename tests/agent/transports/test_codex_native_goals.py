"""Native Codex owns continuation; Hermes consumes events and mirrors lifecycle."""
import pytest

from agent.transports.codex_app_server_goals import run_native_goal
from agent.transports.codex_app_server_session import CodexAppServerSession
from hermes_cli.codex_goals import CodexGoalManager
from hermes_cli.goals import load_goal
from tests.agent.transports.test_codex_app_server_session import FakeClient


class NativeClient(FakeClient):
    def __init__(self, stages=3, final_status='complete'):
        super().__init__()
        self.goal=None
        self.stages=stages
        self.final_status=final_status
        self.delivered=0

    def request(self, method, params=None, timeout=30):
        params=params or {}
        if method.startswith('thread/goal/'):
            self.requests.append((method,params))
            if method=='thread/goal/set':
                self.goal={**(self.goal or {}), 'threadId':params['threadId'],
                    'objective':params.get('objective',(self.goal or {}).get('objective','')),
                    'status':params.get('status','active'),
                    'tokensUsed':self.delivered*100,'tokenBudget':params.get('tokenBudget',(self.goal or {}).get('tokenBudget'))}
            if method=='thread/goal/clear':self.goal=None
            return {'goal':dict(self.goal) if self.goal else None}
        result=super().request(method,params,timeout)
        if method=='turn/start':
            for n in range(1,self.stages+1):
                tid='turn-fake-001' if n==1 else 'native-'+str(n)
                self._notifications.extend([
                    {'method':'turn/started','params':{'threadId':'thread-fake-001','turn':{'id':tid}}},
                    {'method':'item/completed','params':{'threadId':'thread-fake-001','turnId':tid,'item':{'type':'commandExecution','id':'tool-'+str(n),'command':'printf stage','status':'completed','aggregatedOutput':'stage','exitCode':0}}},
                    {'method':'item/completed','params':{'threadId':'thread-fake-001','turnId':tid,'item':{'type':'agentMessage','id':'msg-'+str(n),'text':'STAGE-'+str(n),'phase':'final_answer'}}},
                    {'method':'turn/completed','params':{'threadId':'thread-fake-001','turn':{'id':tid,'status':'completed'}}},
                ])
        return result

    def take_notification(self, timeout=0):
        result=super().take_notification(timeout)
        if result and result['method']=='turn/completed':
            self.delivered+=1
            if self.delivered==self.stages:self.goal['status']=self.final_status
        return result


def native_run(client, mgr, **extra):
    session=CodexAppServerSession(client_factory=lambda **kw:client)
    result=run_native_goal(session,'start',session_id=mgr.session_id,state=mgr.state,
        turn_timeout=0,idle_timeout=2,**extra)
    return result,session


def test_three_native_turns_need_only_one_turn_start():
    mgr=CodexGoalManager('native-3',token_budget=90000);mgr.set('three stages')
    client=NativeClient()
    result,_=native_run(client,mgr)
    assert not result.interrupted and result.error is None
    assert result.final_text=='STAGE-3' and result.native_turns==3
    assert [m for m,_ in client.requests].count('turn/start')==1
    state=load_goal(mgr.session_id)
    assert state.status=='done' and state.turns_used==3 and state.native_goal['status']=='complete'


def test_native_continuation_is_not_capped_at_twenty_turns():
    mgr=CodexGoalManager('native-25',token_budget=90000);mgr.set('25 stages')
    result,_=native_run(NativeClient(stages=25),mgr)
    assert result.native_turns==25 and result.error is None
    assert load_goal(mgr.session_id).status=='done'


@pytest.mark.parametrize('status',['blocked','budgetLimited','usageLimited','paused'])
def test_native_non_completion_states_are_never_success(status):
    mgr=CodexGoalManager('native-'+status);mgr.set('work')
    result,_=native_run(NativeClient(stages=1,final_status=status),mgr)
    assert result.interrupted and result.should_retire and result.error
    assert load_goal(mgr.session_id).status=='paused'


def test_user_pause_between_turns_is_not_overwritten():
    mgr=CodexGoalManager('native-pause');mgr.set('three stages')
    client=NativeClient()
    def pause_after_first(turn,continuing):
        mgr.pause()
    result,_=native_run(client,mgr,on_turn=pause_after_first)
    assert result.interrupted and result.error
    assert load_goal(mgr.session_id).status=='paused'
    assert client.delivered==1


def test_replaced_goal_is_not_completed_by_old_events():
    mgr=CodexGoalManager('native-replace');mgr.set('old')
    old_id=mgr.state.goal_id
    def replace_after_first(turn,continuing):
        mgr.set('new')
    result,_=native_run(NativeClient(),mgr,on_turn=replace_after_first)
    state=load_goal(mgr.session_id)
    assert result.interrupted and state.goal=='new' and state.status=='active' and state.goal_id!=old_id


def test_native_token_usage_is_not_reset_on_resume():
    mgr=CodexGoalManager('native-resume',token_budget=90000);mgr.set('work')
    mgr.state.native_goal={'threadId':'same','status':'paused','tokensUsed':4000}
    mgr.state.turns_used=5;mgr._save();mgr.pause();mgr.resume()
    assert mgr.state.native_goal['tokensUsed']==4000 and mgr.state.turns_used==5
