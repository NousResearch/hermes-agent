import pytest


def test_remote_sources_include_private_and_public_without_disclosure_permission(monkeypatch,tmp_path):
    monkeypatch.delenv('X_GITHUB_KNOWLEDGE',raising=False)
    from x_github_knowledge import remote_knowledge
    def api(endpoint, *, paginate=False):
        if endpoint.startswith('user/repos'):
            return [[{'full_name':'fixture/public','private':False,'description':'local inference routing'},
                     {'full_name':'fixture/private','private':True,'description':'local inference routing'}]]
        if '/commits?' in endpoint:
            return [{'sha':'a'*40,'commit':{'message':'Improve local inference routing','committer':{'date':'2026-09-01T00:00:00Z'}}}]
        raise AssertionError(endpoint)
    result=remote_knowledge('local inference routing',tmp_path/'cache',api=api)
    assert result['coverage']['private']==1 and result['coverage']['public']==1
    assert len(result['references'])==2
    assert all(r['disclosure']=='internal_only' and r['deployment_status']=='unknown' for r in result['references'])
    assert all(r['commit']=='a'*40 for r in result['references'])


def test_remote_source_failure_is_reported_not_invented(monkeypatch,tmp_path):
    monkeypatch.delenv('X_GITHUB_KNOWLEDGE',raising=False)
    from x_github_knowledge import remote_knowledge
    def api(*args,**kwargs):raise RuntimeError('fixture unavailable')
    result=remote_knowledge('local inference',tmp_path/'cache',api=api)
    assert result['references']==[]
    assert result['coverage']['status']=='unavailable'
