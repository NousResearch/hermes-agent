import json
import subprocess
from pathlib import Path
import pytest
from x_knowledge import repository_knowledge, memory_knowledge, safe_excerpt
from x_voice_gate import voice_gate_issues


def test_actual_generated_proposal_is_not_fabricated_experience():
    assert not voice_gate_issues("I'd give it one recurring job you already hate doing.")
    assert voice_gate_issues("I quietly doubled revenue last week.")


def test_internal_commit_is_not_deployment_proof(tmp_path):
    repo=tmp_path/'private-fork'; repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-qm','Add agent memory retrieval'],check=True)
    result=repository_knowledge('agent memory retrieval',[tmp_path])
    assert len(result)==1
    assert 'not proof' in result[0]['status']
    assert result[0]['disclosure'].startswith('internal_only')
    assert len(result[0]['provenance']['commit'])==40
    assert not repository_knowledge('football match results',[tmp_path])


def test_sensitive_lines_removed_and_memory_is_only_a_hint(tmp_path):
    assert 'password' not in safe_excerpt('agent memory\npassword=do-not-copy')
    file=tmp_path/'memory.json'
    file.write_text(json.dumps([{'memory_id':'fixture','text':'agent memory retrieval'}]))
    rows=memory_knowledge('agent memory retrieval',file)
    assert rows[0]['status'].startswith('unverified memory')


def test_x_route_uses_owning_runtime_not_other_profile(monkeypatch):
    import x_generate as x
    calls=[]
    config={'model':{'provider':'fixture','default':'fixture-model'}}
    monkeypatch.setattr(x,'load_config',lambda:config)
    monkeypatch.setattr(x,'_llm_configs',lambda **kw: calls.append(kw) or ['resolved-fixture'])
    monkeypatch.setattr(x,'_call_llm_chain',lambda s,u,**kw: calls.append(kw) or 'fixture output')
    assert x.call_x_model('system','user')=='fixture output'
    assert calls[0]=={'config':config}
    assert calls[1]['configs']==['resolved-fixture']


def test_explicit_config_supported_without_global_loader(monkeypatch):
    import llm_generate as llm
    monkeypatch.setattr(llm,'_load_hermes_config',lambda:pytest.fail('unrelated profile loaded'))
    monkeypatch.setattr(llm,'_resolve_runtime',lambda **kw:{'base_url':'http://example.invalid/v1','api_mode':'chat_completions'})
    routes=llm._llm_configs(config={'model':{'provider':'custom','default':'fixture'}})
    assert routes[0]['model']=='fixture'


def test_virtualised_feed_progress_uses_ids_not_dom_count(monkeypatch):
    import x_ingest as x
    from test_x_ingest_repair import row
    class Feed:
        index=0
        def goto(self,*a,**kw): pass
        def get_by_role(self,*a,**kw): return self
        def get_attribute(self,*a): return 'true'
        def wait_for_selector(self,*a,**kw): pass
        def locator(self,*a): return self
        def count(self): return 1
        def nth(self,i): return row(1 + self.index)
        def evaluate(self,*a): self.index=min(1,self.index+1)
    monkeypatch.setattr(x,'_extract_article',lambda article:article)
    monkeypatch.setattr(x,'tweet_age',lambda sid:1)
    monkeypatch.setattr(x.time,'sleep',lambda _:None)
    assert len(x._scrape_feed(Feed(),limit=2))==2


def test_collection_refreshes_living_own_references(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/content_engine'))
    import x_quote_scout as scout
    import x_reference_refresh
    called=[]
    monkeypatch.delenv('X_REFRESH_OWN_REFERENCES',raising=False)
    monkeypatch.setattr(x_reference_refresh,'refresh_references',lambda:called.append(True) or {'posts':1})
    monkeypatch.setattr(scout.x_ingest,'ingest',lambda **kw:[])
    monkeypatch.setattr(scout,'CANDIDATES_JSON',tmp_path/'candidates.json')
    assert scout._collect()==[]
    assert called==[True]
