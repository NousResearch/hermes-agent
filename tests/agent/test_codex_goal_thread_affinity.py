"""Resuming a paused native Goal must not reuse a live ordinary-chat thread."""
from types import SimpleNamespace

import pytest

from agent import codex_runtime
from agent.transports import codex_app_server_session as wire
from hermes_cli.codex_goals import CodexGoalManager
from hermes_cli.goals import load_goal
from hermes_state import SessionDB
from tests.agent.test_codex_app_server_thread_resume import _WireClient


def setup(tmp_path,monkeypatch,*,cached='ordinary',started=True,recorded=True,active=True):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setattr(wire,'CodexAppServerClient',_WireClient)
    _WireClient.instances=[];_WireClient.dead=set();_WireClient.counter=0
    db=SessionDB(tmp_path/'state.db');db.create_session(session_id='affinity',source='feishu',model='test')
    db.patch_session_model_config('affinity',{'codex_thread_id':'ordinary'})
    mgr=CodexGoalManager('affinity',token_budget=5000000);mgr.set('work')
    mgr.state.native_goal={'threadId':'budget-thread','objective':'work','status':'blocked',
                          'tokensUsed':4835727,'tokenBudget':5000000,'createdAt':123}
    mgr._save();mgr.pause()
    agent=SimpleNamespace(session_id='affinity',_session_db=db,_cached_system_prompt='stable',
                          ephemeral_system_prompt=None,session_cwd=str(tmp_path),_codex_session=None,
                          _emit_diagnostic_status=lambda *a:None)
    old=None
    if cached:
        old=wire.CodexAppServerSession(resume_thread_id=cached)
        if started:old.ensure_started()
        agent._codex_session=old
        if recorded:agent._codex_session_prompt='stable';agent._codex_session_model_provider=None
    if active:mgr.resume()
    return agent,db,old


@pytest.mark.parametrize('cached,started,recorded', [('ordinary',True,True),('ordinary',True,False),('ordinary',False,True),(None,False,False)])
def test_goal_resume_prefers_its_ledger_over_chat_cache_and_db(tmp_path,monkeypatch,cached,started,recorded):
    agent,db,old=setup(tmp_path,monkeypatch,cached=cached,started=started,recorded=recorded)
    try:
        codex_runtime._ensure_codex_session(agent,[])
        assert codex_runtime._start_codex_thread(agent)=='budget-thread'
        if old:assert old._closed and agent._codex_session is not old
        assert _WireClient.instances[-1].requests[-1][0]=='thread/resume'
        state=load_goal('affinity');assert state.token_budget==5000000 and state.native_goal['tokensUsed']==4835727
    finally:db.close()


@pytest.mark.parametrize('started',[True,False])
def test_matching_goal_thread_is_reused(tmp_path,monkeypatch,started):
    agent,db,old=setup(tmp_path,monkeypatch,cached='budget-thread',started=started)
    try:
        codex_runtime._ensure_codex_session(agent,[])
        assert agent._codex_session is old and not old._closed
    finally:db.close()


def test_paused_goal_does_not_hijack_ordinary_chat(tmp_path,monkeypatch):
    agent,db,old=setup(tmp_path,monkeypatch,active=False)
    try:
        codex_runtime._ensure_codex_session(agent,[])
        assert agent._codex_session is old
    finally:db.close()


@pytest.mark.parametrize('failure',['missing','timeout'])
def test_unresumable_active_goal_never_falls_back_to_fresh_thread(tmp_path,monkeypatch,failure):
    agent,db,_=setup(tmp_path,monkeypatch,cached=None)
    codex_runtime._ensure_codex_session(agent,[])
    if failure=='missing':_WireClient.dead={'budget-thread'}
    else:
        original=_WireClient.request
        def request(client,method,params=None,timeout=30):
            if method=='thread/resume':
                client.requests.append((method,params));raise TimeoutError('resume timed out')
            return original(client,method,params,timeout)
        monkeypatch.setattr(_WireClient,'request',request)
    try:
        with pytest.raises((wire.CodexThreadResumeError,TimeoutError)):
            codex_runtime._start_codex_thread(agent)
        assert not any(m=='thread/start' for m,p in _WireClient.instances[-1].requests)
        assert db.get_session_model_config_value('affinity','codex_thread_id')=='ordinary'
        state=load_goal('affinity');assert state.status=='paused' and state.native_goal['tokensUsed']==4835727
    finally:db.close()
