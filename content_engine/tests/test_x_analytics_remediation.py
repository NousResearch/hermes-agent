"""Synthetic defect witnesses; not live growth evidence."""
import pytest
from test_x_analytics import fixture_row, dataset
from x_analytics import growth_report, validate


def test_early_unequal_exposures_cannot_propose_timing():
    a, b = fixture_row('1', hours=.01), fixture_row('2', hours=5.99)
    b['created_at'] = '2026-01-01T04:00:00+00:00'
    b['observed_at'] = '2026-01-01T09:59:24+00:00'
    report = growth_report(dataset(a, b))
    assert report['cohorts'] == []
    assert all(e['status'] == 'insufficient_comparable_metrics' for e in report['experiments'])


def test_metric_missingness_pooled_distribution_and_sources():
    a, b, c = fixture_row('1'), fixture_row('2', likes=9), fixture_row('3')
    b['metrics']['views'] = 200
    c['metrics']['views'] = None
    group = next(g for g in growth_report(dataset(a,b,c))['cohorts'] if g['dimension']=='action')
    assert group['metric_summary']['views']['eligible'] == 2
    assert group['metric_summary']['views']['missing'] == 1
    assert group['pooled_visible_interactions_per_view'] == pytest.approx(14/300)
    assert group['ratio_distribution']['min'] == .04
    assert group['ratio_distribution']['max'] == .05
    assert group['ratio_distribution']['stdev'] > 0
    assert {p['url'] for p in group['evidence']} == {a['url'],b['url'],c['url']}
    assert all(p['age_hours']==24 for p in group['evidence'])


def test_manual_nonnull_metrics_require_explicit_contract():
    row = fixture_row(); row['provenance']['kind']='manual'
    with pytest.raises(ValueError, match='metric contract'):
        validate(dataset(row))


def test_followers_include_exact_period_and_baselines():
    data=dataset()
    for t,n in [('2026-01-01T00:00:00Z',100),('2026-01-03T00:00:00Z',110)]:
        data['followers'].append({'account':'owner','observed_at':t,'count':n,'provenance':fixture_row()['provenance']})
    report=growth_report(data)
    assert report['follower_period']['start']['count']==100
    assert report['follower_period']['end']['observed_at']=='2026-01-03T00:00:00Z'
    assert report['follower_period']['elapsed_hours']==48


def test_research_is_not_observed_efficacy_and_experiments_cite_points():
    report=growth_report(dataset(fixture_row('1'),fixture_row('2','reply',9)))
    assert report['research_evidence']['ranking_source_commit']=='6bb4594253cdfa9ea19983a54a401d5ce8f8275d'
    assert report['official_engagement_rate']['value'] is None
    experiment=next(e for e in report['experiments'] if e['dimension']=='action')
    assert len(experiment['observed_evidence'])==2
    assert all(e['evidence'][0]['observation_id'] for e in experiment['observed_evidence'])
    assert experiment['research_sources']
    assert experiment['status']=='approval_only_hypothesis'


def test_topic_comparison_does_not_pool_different_actions():
    a,b=fixture_row('1'),fixture_row('2','reply',9)
    b['topic']='other'
    report=growth_report(dataset(a,b))
    assert next(e for e in report['experiments'] if e['dimension']=='topic')['status']=='insufficient_comparable_metrics'
