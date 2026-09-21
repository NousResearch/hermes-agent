"""Offline measurement evidence; research policy is not live-account proof."""
from collections import Counter
from statistics import median, stdev

METRICS = ('likes', 'replies', 'reposts', 'views')
AGE_TARGETS = (24, 72)
AGE_TOLERANCE = 2
RESEARCH = {
    'status': 'research_policy_not_live_capability_proof',
    'source_artifacts':{'remii-policy.json':'af3e95075deacaede0c9013f57e495767951c183de12603b14ed9b96fae91f8d','remii-research.md':'27f396d3ec847f4bcad1e915e9b721366dfd2270ba35861bda5c5004ec0d6730'},
    'researched_at_utc': '2026-09-12',
    'ranking_source_commit': '6bb4594253cdfa9ea19983a54a401d5ce8f8275d',
    'sources': [
        'https://raw.githubusercontent.com/xai-org/x-algorithm/6bb4594253cdfa9ea19983a54a401d5ce8f8275d/README.md',
        'https://raw.githubusercontent.com/xai-org/x-algorithm/6bb4594253cdfa9ea19983a54a401d5ce8f8275d/home-mixer/scorers/ranking_scorer.rs',
        'https://help.x.com/en/using-x/view-counts',
        'https://business.x.com/en/help/campaign-measurement-and-analytics/tweet-activity-dashboard',
        'https://help.x.com/en/rules-and-policies/authenticity'],
    'interpretation': 'Published weights multiply predicted viewer actions, not raw engagement counts; no universal timing, reply multiplier, quota or growth guarantee.',
    'historical_source': 'https://github.com/twitter/the-algorithm (2023; not current production recipe)',
    'local_policy': '24h and 72h +/-2h, ten eligible posts per arm as exploratory guardrail, not platform thresholds or statistical power guarantees. Replicate before strategy changes.',
    'approval_required': True, 'automatic_publishing_allowed': False,
}


def distribution(values):
    return {'n': len(values), 'min': min(values) if values else None,
            'max': max(values) if values else None, 'median': median(values) if values else None,
            'stdev': stdev(values) if len(values)>1 else None,
            'stdev_missing_reason': None if len(values)>1 else 'At least two eligible posts required.'}


def summarize(rows, point):
    eligible = [r for r in rows if all(r['metrics'][k] is not None for k in METRICS) and r['metrics']['views']>0]
    ratios = [sum(r['metrics'][k] for k in METRICS[:3])/r['metrics']['views'] for r in eligible]
    metrics = {}
    for key in METRICS:
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        metrics[key] = {'eligible':len(values), 'missing':len(rows)-len(values),
                        'missing_reasons':dict(Counter(r['metric_details'][key]['reason'] for r in rows if r['metrics'][key] is None)),
                        'distribution':distribution(values)}
    return {'n_posts':len(rows), 'metric_summary':metrics, 'n_complete_denominators':len(eligible),
            'median_visible_interactions_per_view':median(ratios) if ratios else None,
            'pooled_visible_interactions_per_view':sum(sum(r['metrics'][k] for k in METRICS[:3]) for r in eligible)/sum(r['metrics']['views'] for r in eligible) if eligible else None,
            'ratio_distribution':distribution(ratios), 'zero_denominators':sum(r['metrics']['views']==0 for r in rows),
            'ratio_missing_reason':None if eligible else 'No complete exact positive-view denominator set.',
            'small_sample':len(eligible)<10, 'evidence':[point(r) for r in rows],
            'confidence_interval':None, 'uncertainty':'Intervals omitted: bounded observational sample, correlated posts and unknown selection; no justified independent sampling model.'}
