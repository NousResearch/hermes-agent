"""A native Goal records each compaction boundary without another compact or kickoff."""
from types import SimpleNamespace

from agent.codex_runtime_goals import run_app_server_work
from agent.transports.codex_app_server_session import CodexAppServerSession
from hermes_cli.codex_goals import CodexGoalManager
from hermes_cli.goals import load_goal
from tests.agent.transports.test_codex_native_goals import NativeClient


class CompactClient(NativeClient):
    def request(self, method, params=None, timeout=30):
        result=super().request(method,params,timeout)
        if method=='turn/start':
            for i,note in enumerate(self._notifications):
                if note['method']=='turn/started':
                    self._notifications.insert(i+1, {'method':'item/completed','params':{
                        'threadId':'thread-fake-001','turnId':'turn-fake-001',
                        'item':{'id':'compact-1','type':'contextCompaction'}}})
                    break
        return result


def test_compaction_bookkeeping_precedes_usage_and_is_not_repeated(tmp_path, monkeypatch):
    (tmp_path/'config.yaml').write_text('goals:\n  runtime: codex\nagent:\n  codex_turn_timeout: 0\n  codex_idle_timeout: 2\n')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    mgr=CodexGoalManager('compact-goal');mgr.set('three stages')
    client=CompactClient();session=CodexAppServerSession(client_factory=lambda **kw:client)
    events=[]
    monkeypatch.setattr('agent.codex_runtime._persist_projected_messages',lambda *a:True)
    monkeypatch.setattr('agent.codex_runtime._store_codex_thread_id',lambda *a:None)
    monkeypatch.setattr('agent.codex_runtime._record_codex_app_server_compaction',
                        lambda agent,turn: events.append(('compact',turn.compacted)))
    def usage(*a,**kw):
        events.append(('usage',None));return {}
    monkeypatch.setattr('agent.codex_runtime._record_codex_app_server_usage',usage)
    result=run_app_server_work(SimpleNamespace(session_id=mgr.session_id,_codex_session=session),'start',messages=[])
    assert result.error is None and load_goal(mgr.session_id).status=='done'
    assert events==[('compact',True),('usage',None),('compact',False),('usage',None)]
    assert result.compacted is False
    assert [m for m,p in client.requests].count('turn/start')==1
    assert not any(m=='thread/compact/start' for m,p in client.requests)
