"""Synthetic fixtures only; never evidence of live performance."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'content_engine'))


def test_empty_observations_do_not_invent_growth():
    from x_analytics import growth_report
    report = growth_report({'account':'owner','posts':[], 'followers':[]})
    assert report['unique_posts'] == 0
    assert report['follower_change'] is None
    assert all(e['status']=='insufficient_comparable_metrics' for e in report['experiments'])


def test_real_runtime_report(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts/content_engine'))
    import x_runtime_bundle
    from cron.scheduler_script import _run_job_script
    home = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(home))
    x_runtime_bundle.build_bundle(home / 'scripts')
    data = home / 'data/x-analytics/observations.json'
    data.parent.mkdir(parents=True)
    data.write_text(json.dumps({'account':'owner', 'posts':[], 'followers':[], 'capabilities':{'collection':{'status':'fixture-empty'}}}))
    ok, output = _run_job_script('x_mix_report.py', workdir=str(tmp_path))
    assert ok, output
    report = Path(next(line[6:] for line in output.splitlines() if line.startswith('MEDIA:')))
    assert 'Measured growth' in report.read_text()
    assert 'fixture-empty' in report.read_text()


import copy
import csv
import subprocess
import pytest


def fixture_row(id='1', action='original', likes=3, hours=24):
    from datetime import datetime, timedelta, timezone
    created = datetime(2026,1,1,tzinfo=timezone.utc)
    return {'id':id,'url':f'https://x.com/owner/status/{id}','author':'owner',
            'created_at':created.isoformat(),'observed_at':(created+timedelta(hours=hours)).isoformat(),
            'text':'SYNTHETIC TEST FIXTURE <script>not live</script>','topic':'testing','action':action,
            'metrics':{'likes':likes,'replies':1,'reposts':0,'views':100},
            'provenance':{'kind':'fixture','source':'explicit synthetic fixture','sha256':'0'*64}}


def dataset(*rows):
    return {'account':'owner','posts':list(rows),'followers':[],'capabilities':{'collection':{'status':'synthetic_fixture'}}}


def test_dedupe_conflict_and_null():
    from x_analytics import validate, growth_report
    row=fixture_row()
    assert len(validate(dataset(row,row))['posts'])==1
    altered=copy.deepcopy(row); altered['metrics']['likes']=4
    with pytest.raises(ValueError,match='conflicting'):
        validate(dataset(row,altered))
    row['metrics']['views']=None
    report=growth_report(dataset(row))
    assert all(g['median_visible_interactions_per_view'] is None for g in report['cohorts'])
    assert report['follower_change'] is None


@pytest.mark.parametrize('field,value', [('author','foreign'),('id','2'),('url','https://x.com/foreign/status/1'),('observed_at','2026-01-02'),('created_at','2027-01-01T00:00:00Z')])
def test_invalid_identity_time(field,value):
    from x_analytics import validate
    row=fixture_row(); row[field]=value
    with pytest.raises(ValueError): validate(dataset(row))


@pytest.mark.parametrize('value',[-1,True,1.5,'1K'])
def test_invalid_counts(value):
    from x_analytics import validate
    row=fixture_row(); row['metrics']['likes']=value
    with pytest.raises(ValueError): validate(dataset(row))


def test_age_matching_experiments_and_denominators():
    from x_analytics import growth_report
    report=growth_report(dataset(fixture_row('1'),fixture_row('2','reply',9),fixture_row('3','quote',100,hours=160)))
    action=next(e for e in report['experiments'] if e['dimension']=='action')
    assert action['compare']==['reply','original']
    assert action['small_sample'] is True
    assert action['age_hours']==[22,26]
    assert next(g for g in report['cohorts'] if g['dimension']=='action' and g['label']=='reply')['median_visible_interactions_per_view']==0.1


def test_csv_real_cli_e2e(tmp_path):
    from x_analytics import DEFINITIONS, import_csv
    row=fixture_row()
    flat={k:v for k,v in row.items() if k not in ('metrics','provenance')}; flat.update(row['metrics'])
    source=tmp_path/'synthetic.csv'
    with source.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat)); writer.writeheader(); writer.writerow(flat)
    defs=tmp_path/'definitions.json'; defs.write_text(json.dumps(DEFINITIONS))
    output=tmp_path/'report'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/content_engine/x_analytics_report.py'),'--input',str(source),'--format','csv','--account','owner','--definitions',str(defs),'--output-dir',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((output/'growth-report.json').read_text())
    assert report['unique_posts']==1
    assert json.loads((output/'voice-reference.json').read_text())['approval_status']=='unapproved'
    defs.write_text('{}')
    with pytest.raises(ValueError,match='definitions'): import_csv(source,'owner',defs)


def test_voice_fixture_exclusion_and_html_escape():
    from x_analytics import voice_export,render_growth
    assert voice_export(dataset(fixture_row()))['posts']==[]
    assert '<script>' not in render_growth({'untrusted':'<script>'})


def test_follower_dedupe_and_net_change():
    from x_analytics import growth_report
    data=dataset()
    for time,count in [('2026-01-01T00:00:00Z',10),('2026-01-02T00:00:00Z',9)]:
        data['followers'].append({'account':'owner','observed_at':time,'count':count,'provenance':fixture_row()['provenance']})
    data['followers'].append(data['followers'][0])
    report=growth_report(data)
    assert report['follower_change']==-1
    assert report['follower_observations']==2


def test_historical_posts_do_not_assume_original_or_approval(tmp_path):
    from x_analytics import import_historical,voice_export
    row=fixture_row()
    path=tmp_path/'explicit-synthetic-history.json'
    path.write_text(json.dumps({'posts':[{'id':row['id'],'url':row['url'],'timestamp':row['created_at'],'text':row['text']}]}))
    data=import_historical(path,'owner',row['observed_at'])
    assert data['posts'][0]['action']=='unknown'
    assert all(v is None for v in data['posts'][0]['metrics'].values())
    assert voice_export(data)['approval_status']=='unapproved'


def test_merge_preserves_baseline_and_new_observations():
    from x_analytics import merge_observations
    first=dataset(fixture_row())
    second=dataset(fixture_row(),fixture_row(hours=48))
    merged=merge_observations(first,second)
    assert len(merged['posts'])==2
    assert len(first['posts'])==1


def test_fresh_baseline_same_age_only():
    from datetime import datetime, timezone, timedelta
    from x_analytics import growth_report
    now=datetime(2026,1,20,tzinfo=timezone.utc)
    recent=fixture_row('2',likes=9)
    recent['created_at']=(now-timedelta(days=3)).isoformat()
    recent['observed_at']=(now-timedelta(days=2)).isoformat()
    report=growth_report(dataset(fixture_row(),recent),now=now)
    comparison=report['fresh_baseline_comparisons'][0]
    assert comparison['recent_median']==0.1
    assert comparison['baseline_median']==0.04
    assert comparison['small_sample'] is True
    recent['observed_at']=(now-timedelta(days=1)).isoformat()
    assert growth_report(dataset(fixture_row(),recent),now=now)['fresh_baseline_comparisons']==[]
