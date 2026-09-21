"""Local, read-only observations and approval-only growth experiments.

Views are displayed post views, NOT unique people or account impressions.
No missing count is interpreted as zero. Import hashes prove bytes, not truth.
"""
from __future__ import annotations
import csv
import hashlib
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from x_analytics_evidence import AGE_TARGETS, AGE_TOLERANCE, RESEARCH, summarize

METRICS = ('likes', 'replies', 'reposts', 'views')
DEFINITIONS = {'likes':'displayed cumulative likes', 'replies':'displayed cumulative replies',
               'reposts':'displayed cumulative reposts', 'views':'displayed cumulative post views; not unique viewers'}


def utc(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError('UTC timestamp required')
    return dt


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(data):
    """Validate and dedupe observation identity; conflicting same-time data fails closed."""
    account = data['account']
    if not re.fullmatch(r'[A-Za-z0-9_]{1,15}', account):
        raise ValueError('literal account required')
    seen = {}
    if data.get('schema_version', 1) not in (1, 2):
        raise ValueError('unsupported observation schema version')
    out: dict = dict(data, schema_version=2, posts=[], followers=[])
    for kind in ('posts', 'followers'):
        for original in data.get(kind, []):
            row = dict(original)
            observed = utc(row['observed_at'])
            if observed > datetime.now(timezone.utc):
                raise ValueError('future observation')
            prov = row['provenance']
            if prov.get('kind') not in ('browser','local_export','csv','manual','fixture') or not prov.get('source') or not re.fullmatch('[a-f0-9]{64}', prov.get('sha256','')):
                raise ValueError('provenance required')
            if kind == 'posts':
                m = re.fullmatch(r'https://(?:x\.com|twitter\.com)/([A-Za-z0-9_]{1,15})/status/([0-9]{1,20})', row['url'])
                if not m or m[1].lower() != account.lower() or row['author'].lower() != account.lower() or m[2] != row['id']:
                    raise ValueError('own canonical post identity required')
                if utc(row['created_at']) > observed:
                    raise ValueError('post newer than observation')
                if row.get('action') not in (None,'original','reply','quote','unknown'):
                    raise ValueError('unknown action')
                row['metrics'] = {k:row.get('metrics',{}).get(k) for k in METRICS}
                contract = row.get('metric_contract')
                if contract is None and prov['kind'] in ('manual', 'local_export') and any(v is not None for v in row['metrics'].values()):
                    raise ValueError('explicit metric contract required for imported/manual measurements')
                if contract is not None and (contract.get('definitions') != DEFINITIONS or contract.get('precision') != 'exact' or contract.get('scope') not in ('organic','promoted','unknown')):
                    raise ValueError('invalid metric contract: definitions, exact precision and scope required')
                if prov['kind'] == 'csv' and prov.get('definitions') != DEFINITIONS:
                    raise ValueError('CSV metric contract definitions required')
                row['metric_contract'] = contract or {'definitions':DEFINITIONS, 'precision':'exact', 'scope':'unknown'}
                raw = prov.get('observation', {}).get('metrics', {})
                row['metric_details'] = {k:{
                    'definition':DEFINITIONS[k], 'literal_label':k, 'raw_display':raw.get(k),
                    'precision':'exact' if v is not None else 'unknown',
                    'scope':row['metric_contract']['scope'], 'surface':prov['source'],
                    'method':prov['kind'], 'status':'available' if v is not None else 'not_observed',
                    'reason':None if v is not None else ('Unparsed or rounded display; exact count withheld.' if raw.get(k) else 'No exact counter observed in source.')}
                    for k,v in row['metrics'].items()}
                values = row['metrics'].values()
                identity = row['id']
            else:
                if row['account'].lower() != account.lower():
                    raise ValueError('foreign follower observation')
                values = [row['count']]
                identity = account.lower()
            if any(v is not None and (type(v) is not int or v < 0) for v in values):
                raise ValueError('counts must be nonnegative integers or null')
            key = (kind, identity, observed.isoformat())
            # Provenance can differ for the same fact imported through another file.
            facts = {k:v for k,v in row.items() if k not in ('provenance','metric_details')}
            if key in seen:
                if seen[key] != facts:
                    raise ValueError('conflicting duplicate observation')
                continue
            seen[key] = facts
            out[kind].append(row)
    return out


def import_csv(path, account, definitions_path):
    """Explicit schema CSV/manual fallback; sidecar must define every denominator."""
    path = Path(path)
    definitions = json.loads(Path(definitions_path).read_text())
    if definitions != DEFINITIONS:
        raise ValueError('metric definitions must exactly match displayed cumulative counters; no impressions substitution')
    raw = path.read_bytes()
    rows = []
    for line, row in enumerate(csv.DictReader(raw.decode('utf-8-sig').splitlines()), 2):
        metrics = {}
        for key in METRICS:
            value = row.get(key, '')
            if value and not re.fullmatch(r'[0-9]+', value):
                raise ValueError(f'line {line}: exact integer or blank required')
            metrics[key] = int(value) if value else None
        rows.append({k:row[k] for k in ('id','url','author','created_at','observed_at','text')} | {
            'action':row.get('action') or None,'topic':row.get('topic') or None,'metrics':metrics,
            'provenance':{'kind':'csv','source':str(path.resolve()),'sha256':digest(raw),'line':line,
                          'definitions':definitions}})
    return validate({'account':account,'posts':rows,'followers':[], 'capabilities':{
        'collection':{'status':'imported','reason':'Owner-provided export; not independently verified browser metrics.'}}})


def import_historical(path, account, observed_at):
    """Existing own public export: authorship by canonical URL, NOT human approval."""
    path = Path(path)
    raw = path.read_bytes()
    source = json.loads(raw)
    rows = []
    for container, action in (('posts','unknown'),('replies','reply')):
        for row in source.get(container, []):
            rows.append({'id':str(row['id']),'url':row['url'],'author':account,
                         'created_at':row['timestamp'],'observed_at':observed_at,'text':row['text'],
                         'action':action,'topic':None,'metrics':dict.fromkeys(METRICS),
                         'provenance':{'kind':'local_export','source':str(path.resolve()),'sha256':digest(raw),
                                       'container':container}})
    return validate({'account':account,'posts':rows,'followers':[], 'capabilities':{
        'collection':{'status':'historical_text_only','reason':'No metric observations in this real export; import time is not metric collection time.'}}})


def merge_observations(baseline, incoming):
    """Append observations, retaining history; validate the combined identity space."""
    baseline, incoming = validate(baseline), validate(incoming)
    if baseline['account'].lower() != incoming['account'].lower():
        raise ValueError('baseline account differs')
    return validate({'account':incoming['account'],
                     'posts':baseline['posts']+incoming['posts'],
                     'followers':baseline['followers']+incoming['followers'],
                     'capabilities':incoming.get('capabilities',{}),
                     'prior_capabilities':baseline.get('capabilities',{})})


def voice_export(data):
    data = validate(data)
    own = {}
    for row in data['posts']:
        if row['provenance']['kind'] != 'fixture' and row.get('text'):
            own[row['id']] = {k:row[k] for k in ('id','url','author','created_at','text','provenance')}
    return {'account':data['account'],'approval_status':'unapproved','purpose':'Historical own voice reference; human approval is separate.', 'posts':list(own.values())}


def growth_report(data, now=None):
    data = validate(data)
    now = now or datetime.now(timezone.utc)
    def point(row):
        return {'observation_id':data['account']+':'+row['id']+':'+row['observed_at'],
                **{k:row[k] for k in ('id','url','created_at','observed_at','metrics','metric_details')},
                'provenance':{k:v for k,v in row['provenance'].items() if k != 'observation'},
                'age_hours':(utc(row['observed_at'])-utc(row['created_at'])).total_seconds()/3600}
    # Select nearest observed snapshot to a declared target, never extrapolate.
    cohorts = defaultdict(dict)
    for row in data['posts']:
        hours = (utc(row['observed_at']) - utc(row['created_at'])).total_seconds()/3600
        if hours > 168 or row['provenance']['kind'] == 'local_export':
            continue
        bucket = next((target for target in AGE_TARGETS if abs(hours-target)<=AGE_TOLERANCE), None)
        if bucket is None:
            continue
        previous = cohorts[bucket].get(row['id'])
        if previous is None or abs(hours-bucket) < abs((utc(previous['observed_at'])-utc(previous['created_at'])).total_seconds()/3600-bucket):
            cohorts[bucket][row['id']] = row
    groups = []
    for age, posts in sorted(cohorts.items()):
        for dimension in ('topic','action','timing_utc'):
            split = defaultdict(list)
            for row in posts.values():
                label = str(utc(row['created_at']).hour // 4 * 4).zfill(2)+'–'+str(utc(row['created_at']).hour // 4 * 4+4).zfill(2) if dimension == 'timing_utc' else row.get(dimension) or 'unknown'
                controls = tuple((key, row.get(key) or 'unknown') for key in ('action','topic') if key != dimension) + (('scope', row['metric_contract']['scope']),)
                split[(label, controls)].append(row)
            for (label, controls), rows in sorted(split.items()):
                groups.append({'age_hours':[age-AGE_TOLERANCE,age+AGE_TOLERANCE], 'target_age_hours':age, 'controls':dict(controls), 'dimension':dimension,'label':label, **summarize(rows, point)})
    # Fresh period comparison uses post creation cohorts, holding observation age fixed.
    periods = defaultdict(lambda: {'recent_7d':[], 'previous_21d':[]})
    metric_times = [utc(r['observed_at']) for r in data['posts'] if any(r['metrics'][k] is not None for k in METRICS)]
    for age, posts in cohorts.items():
        for row in posts.values():
            days = (now-utc(row['created_at'])).total_seconds()/86400
            period = 'recent_7d' if 0 <= days < 7 else 'previous_21d' if 7 <= days < 28 else None
            if period:
                key = (age,row.get('action') or 'unknown',row.get('topic') or 'unknown',row['metric_contract']['scope'])
                periods[key][period].append(row)
    comparisons = []
    for (age,action,topic,scope), values in sorted(periods.items()):
        if all(summarize(v,point)['n_complete_denominators'] for v in values.values()):
            comparisons.append({'age_hours':[age-AGE_TOLERANCE,age+AGE_TOLERANCE],'action':action,'topic':topic,'scope':scope,
                                'recent_n':len(values['recent_7d']),'baseline_n':len(values['previous_21d']),
                                'recent_median':summarize(values['recent_7d'],point)['median_visible_interactions_per_view'], 'baseline_median':summarize(values['previous_21d'],point)['median_visible_interactions_per_view'],
                                'observed_evidence':{k:summarize(v,point) for k,v in values.items()},
                                'small_sample':any(summarize(v,point)['small_sample'] for v in values.values())})
    followers = sorted((r for r in data['followers'] if r['count'] is not None),key=lambda r:utc(r['observed_at']))
    delta = followers[-1]['count']-followers[0]['count'] if len(followers)>1 else None
    experiments = []
    for dimension in ('topic','action','timing_utc'):
        usable = [g for g in groups if g['dimension']==dimension and g['label']!='unknown' and g['median_visible_interactions_per_view'] is not None]
        candidates = defaultdict(list)
        for g in usable:
            candidates[(tuple(g['age_hours']),tuple(sorted(g['controls'].items())))].append(g)
        comparable = [v for v in candidates.values() if len(v)>1]
        if comparable:
            pair = sorted(max(comparable,key=lambda v:sum(g['n_complete_denominators'] for g in v)),key=lambda g:g['median_visible_interactions_per_view'],reverse=True)
            experiments.append({'dimension':dimension,'status':'approval_only_hypothesis','compare':[pair[0]['label'],pair[-1]['label']], 'age_hours':pair[0]['age_hours'],
                                'observed_evidence':[pair[0],pair[-1]], 'research_sources':RESEARCH['sources'],
                                'hypothesis':'Repeat the observational contrast; direction is not a winner or causal result.',
                                'instruction':'Alternate matched posts; hold topic/action/time other than this dimension constant. Reobserve at the same age; no winner or ranking claim.',
                                'small_sample':any(g['small_sample'] for g in pair)})
        else:
            experiments.append({'dimension':dimension,'status':'insufficient_comparable_metrics','instruction':'Collect matched-age observations before recommending a winner. Test original/reply/quote only with human approval; no engagement bait.'})
    latest = {}
    for row in sorted(data['posts'],key=lambda r:utc(r['observed_at'])):
        latest[row['id']] = row
    coverage = summarize(list(latest.values()),point)
    return {'title':'Measured growth','schema_version':2,'generated_at':now.isoformat(),'account':data['account'],'capabilities':data.get('capabilities',{}),
            'research_evidence':RESEARCH,
            'comparison_policy':{'targets_hours':list(AGE_TARGETS),'tolerance_hours':AGE_TOLERANCE,'snapshot_selection':'closest actual observation to target, one per post/target; no cumulative summation'},
            'official_engagement_rate':{'value':None,'reason':'Reported engagements and matching impressions with the same scope/window are unavailable; public counters omit private interactions.'},
            'coverage':{'scope':'supplied observations only; not an account census',
                        'pagination':data.get('coverage','Not provided; completeness unknown.'),
                        'oldest_created_at':min((r['created_at'] for r in data['posts']),default=None),
                        'newest_created_at':max((r['created_at'] for r in data['posts']),default=None),
                        'oldest_observed_at':min((r['observed_at'] for r in data['posts']),default=None),
                        'metric_summary_latest_snapshot_per_post':coverage['metric_summary'],
                        'observation_evidence':[point(r) for r in data['posts']]},
            'follower_period':{'start':followers[0] if followers else None,'end':followers[-1] if followers else None,
                               'elapsed_hours':(utc(followers[-1]['observed_at'])-utc(followers[0]['observed_at'])).total_seconds()/3600 if len(followers)>1 else None,
                               'missing_snapshot_count':len(data['followers'])-len(followers),
                               'reason':None if len(followers)>1 else 'At least two exact follower snapshots required; no attribution to posts.'},
            'unique_posts':len({r['id'] for r in data['posts']}),'observations':len(data['posts']),
            'latest_observed_at':max((r['observed_at'] for r in data['posts']), default=None),
            'metric_definitions':DEFINITIONS,'ratio_definition':'(likes + replies + reposts) / displayed post views; not X engagement rate, unique reach, or causal effect',
            'fresh_baseline_comparisons':comparisons,
            'baseline_status':'matched' if comparisons else 'No matched recent-seven-day versus prior-21-day metric observations; no growth trend asserted.',
            'latest_metric_observed_at':max(metric_times).isoformat() if metric_times else None,
            'stale_metrics':not metric_times or (now-max(metric_times)).total_seconds()>7*86400,
            'cohorts':groups,'experiments':experiments,'follower_change':delta,'follower_observations':len(followers),
            'limits':['Only observed 24h +/-2h and 72h +/-2h snapshots; no early, lifetime or interpolated comparison. Unknown parent exposure, media/link, distribution and timing remain confounders.','Small samples (<10 complete posts per group) are exploratory; even larger observational samples do not prove causality.',
                      'Follower change is account-level net change, never attributable to an individual post.','Unknown metrics remain null; zero denominators excluded.','Imports prove source bytes, not human approval or independent metric authenticity.']}


def render_growth(report):
    return '<section><h2>Measured growth</h2><pre>'+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+'</pre></section>'


def load_growth_section(path):
    path = Path(path)
    if not path.exists():
        return '<section><h2>Measured growth</h2><p>No observations installed. Use the read-only collector or validated owner CSV import; staging counts are not growth.</p></section>'
    try:
        return render_growth(growth_report(json.loads(path.read_text())))
    except (ValueError,KeyError,TypeError) as exc:
        return '<section><h2>Measured growth</h2><p>Invalid observations; no recommendations: '+html.escape(str(exc))+'</p></section>'
