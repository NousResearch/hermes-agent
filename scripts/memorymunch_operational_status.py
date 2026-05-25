#!/usr/bin/env python3
"""Operational proof checker for Hermes MemoryMunch + Graphify live status.
Read-only except optional JSON report output path. Does not read secrets.
"""
from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path


def load_env_names_only(env_path: Path) -> None:
    """Load non-secret MemoryMunch gate env values for read-only status checks."""
    if not env_path.exists():
        return
    allowed = {
        'HERMES_MEMORYMUNCH_ENABLE',
        'HERMES_MEMORYMUNCH_LIVE_WRITE_ENABLE',
        'HERMES_MEMORYMUNCH_AUTO_CAPTURE_ENABLE',
        'HERMES_MEMORYMUNCH_SCOPE_ENTITY',
        'HERMES_MEMORYMUNCH_DOMAIN',
    }
    for line in env_path.read_text(errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key in allowed and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")

HERMES_HOME = Path(os.environ.get('HERMES_HOME', '/home/alcoo/.hermes'))
MM_SESS = HERMES_HOME/'memorymunch'/'sessions'
PLUGIN = HERMES_HOME/'plugins'/'memorymunch'/'__init__.py'
VENDORED_PLUGIN = HERMES_HOME/'hermes-agent'/'contrib'/'plugins'/'memorymunch'/'__init__.py'
TESTS = HERMES_HOME/'hermes-agent'/'tests'/'run_agent'/'test_memorymunch_curator_relevance.py'
DOC = Path('/home/alcoo/tmp/memorymunch-live-repair-living-doc-20260523.md')
CC = Path('/mnt/c/Users/paulcooke1976')
GRAPH_OUT = CC/'claude-config'/'graphify-out'
CC_SETTINGS = CC/'.claude'/'settings.json'
OC_SETTINGS = CC/'claude-config'/'variants'/'oc'/'settings.json'
GRAPH_DIR = CC/'.claude'/'graphify'

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ''


def plugin_parity(runtime_plugin: Path, vendored_plugin: Path):
    runtime_sha = sha(runtime_plugin)
    vendored_sha = sha(vendored_plugin)
    ok = bool(runtime_sha and vendored_sha and runtime_sha == vendored_sha)
    return {
        'ok': ok,
        'runtime_path': str(runtime_plugin),
        'vendored_path': str(vendored_plugin),
        'runtime_sha256': runtime_sha,
        'vendored_sha256': vendored_sha,
        'gap': '' if ok else 'runtime_plugin_drift',
    }


def detect_live_briefing_contradictions(text: str):
    text = text or ''
    lower = text.lower()
    gaps = []
    if lower.count('<memorymunch-briefing') > 1:
        gaps.append('duplicate_memorymunch_briefing')
    technical_terms = {'memorymunch', 'hermes', 'openclaw', 'plugin', 'audit', 'production', 'curator', 'janitor'}
    unrelated_terms = {'kipbo mortgage', 'fulfilled eschatology', 'theology', 'preterist'}
    if any(term in lower for term in technical_terms) and any(term in lower for term in unrelated_terms):
        gaps.append('unrelated_activation_atom_in_technical_query')
    return {'ok': not gaps, 'gaps': gaps}


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_strings(item)


def rows_text(rows):
    return '\n'.join(s for row in rows for s in iter_strings(row))


def latest_turn_briefing_state(rows):
    last_completed = -1
    for idx, row in enumerate(rows):
        if isinstance(row, dict) and row.get('event') == 'turn_completed':
            last_completed = idx
    if last_completed < 0:
        return {'ok': False, 'gaps': ['latest_turn_missing']}
    last_start = -1
    for idx, row in enumerate(rows[:last_completed + 1]):
        if isinstance(row, dict) and row.get('event') == 'turn_started':
            last_start = idx
    window = rows[last_start:last_completed + 1] if last_start >= 0 else rows[:last_completed + 1]
    inspectable_events = {'turn_started', 'turn_completed', 'session_attached', 'session_opened'}
    inspectable_rows = [
        row for row in window
        if isinstance(row, dict) and row.get('event') in inspectable_events
    ]
    return detect_live_briefing_contradictions(rows_text(inspectable_rows))

def latest_session():
    files = sorted(MM_SESS.glob('*.jsonl'), key=lambda p:p.stat().st_mtime, reverse=True) if MM_SESS.exists() else []
    return files[0] if files else None

def latest_turn_session():
    files = sorted(MM_SESS.glob('*.jsonl'), key=lambda p:p.stat().st_mtime, reverse=True) if MM_SESS.exists() else []
    for p in files:
        rows = read_rows(p)
        if any(r.get('event') == 'turn_completed' for r in rows):
            return p
    return files[0] if files else None

def recent_rows(limit: int = 25):
    rows=[]
    files = sorted(MM_SESS.glob('*.jsonl'), key=lambda p:p.stat().st_mtime, reverse=True)[:limit] if MM_SESS.exists() else []
    for p in files:
        for row in read_rows(p):
            row.setdefault('_ledger_path', str(p))
            rows.append(row)
    return rows

def read_rows(p: Path):
    rows=[]
    if not p or not p.exists(): return rows
    for line in p.read_text(errors='replace').splitlines():
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except Exception as e: rows.append({'event':'parse_error','error':str(e)})
    return rows

def settings_has_gate(p: Path):
    if not p.exists(): return False
    txt=p.read_text(errors='ignore')
    return 'graphify-gate.py' in txt and 'graphify-marker-writer.py' in txt and 'graphify-write-tracker.py' in txt


def latest_capture_ok(rows):
    """Evaluate latest live-capture lane state from one turn ledger.

    Older ledgers can lack live_capture_attempted telemetry while still having
    live_capture_completed rows. Treat that as recovered/firing when no later
    failure/skipped event exists, but surface a telemetry warning.
    """
    events = [
        r for r in rows
        if r.get('event') in {'live_capture_attempted','live_capture_completed','live_capture_failed','live_capture_skipped'}
    ]
    live = [r for r in rows if r.get('event') == 'live_capture_completed']
    attempted = [r for r in rows if r.get('event') == 'live_capture_attempted']
    failed = [r for r in rows if r.get('event') == 'live_capture_failed']
    if not events:
        return {'ok': False, 'state': 'missing', 'telemetry_warning': ''}
    latest = events[-1]
    latest_event = latest.get('event')
    if latest_event == 'live_capture_completed':
        state = 'completed' if attempted else 'completed_without_attempted'
        return {
            'ok': True,
            'state': state,
            'telemetry_warning': '' if attempted else 'missing_live_capture_attempted',
            'latest_event': latest_event,
        }
    if latest_event == 'live_capture_failed':
        return {'ok': False, 'state': 'failed', 'telemetry_warning': '', 'latest_event': latest_event}
    if latest_event == 'live_capture_skipped':
        return {'ok': False, 'state': 'skipped', 'telemetry_warning': '', 'latest_event': latest_event}
    return {'ok': bool(live and attempted and not failed), 'state': 'attempted_only', 'telemetry_warning': '', 'latest_event': latest_event}


def main():
    load_env_names_only(HERMES_HOME/'.env')
    now=time.time()
    attach_p=latest_session()
    p=latest_turn_session()
    rows=read_rows(p) if p else []
    attach_rows=read_rows(attach_p) if attach_p else []
    all_recent_rows=recent_rows()
    live=[r for r in rows if r.get('event')=='live_capture_completed']
    attempted=[r for r in rows if r.get('event')=='live_capture_attempted']
    failed=[r for r in rows if r.get('event')=='live_capture_failed']
    latest_capture_events=[r for r in rows if r.get('event') in {'live_capture_attempted','live_capture_completed','live_capture_failed','live_capture_skipped'}]
    capture_state=latest_capture_ok(rows)
    latest_capture_ok_bool=bool(capture_state.get('ok'))
    plugin_parity_state = plugin_parity(PLUGIN, VENDORED_PLUGIN)
    completed=[r for r in rows if r.get('event')=='turn_completed']
    briefing_state = latest_turn_briefing_state(rows)
    comp=[r for r in all_recent_rows if r.get('event') in {'session_attached','compaction_checkpoint'} or r.get('reason')=='compression']
    gj=GRAPH_OUT/'graph.json'; gr=GRAPH_OUT/'GRAPH_REPORT.md'; gh=GRAPH_OUT/'graph.html'
    verdict=GRAPH_DIR/'verdict.jsonl'
    verdict_tail=verdict.read_text(errors='ignore').splitlines()[-5:] if verdict.exists() else []
    env={k:os.environ.get(k,'') for k in [
        'HERMES_MEMORYMUNCH_ENABLE','HERMES_MEMORYMUNCH_LIVE_WRITE_ENABLE','HERMES_MEMORYMUNCH_AUTO_CAPTURE_ENABLE','HERMES_MEMORYMUNCH_SCOPE_ENTITY','HERMES_MEMORYMUNCH_DOMAIN']}
    checks={
        'memorymunch_env_enabled': env.get('HERMES_MEMORYMUNCH_ENABLE','').lower() in {'1','true','yes'},
        'live_write_env_enabled': env.get('HERMES_MEMORYMUNCH_LIVE_WRITE_ENABLE','').lower() in {'1','true','yes'},
        'auto_capture_env_enabled': env.get('HERMES_MEMORYMUNCH_AUTO_CAPTURE_ENABLE','').lower() in {'1','true','yes'},
        'plugin_exists': PLUGIN.exists(),
        'runtime_plugin_matches_vendored': plugin_parity_state['ok'],
        'live_briefing_clean': briefing_state['ok'],
        'latest_ledger_exists': bool(p and p.exists()),
        'visible_turn_ledgering': len(completed)>0,
        'live_capture_firing': latest_capture_ok_bool,
        'compaction_lineage_present': len(comp)>0,
        'compression_core_injects_provider_context': 'memory_compression_context' in Path('/home/alcoo/.hermes/hermes-agent/agent/conversation_compression.py').read_text(errors='ignore'),
        'compression_core_injects_exact_user_query': 'Exact pre-compression user message' in Path('/home/alcoo/.hermes/hermes-agent/agent/conversation_compression.py').read_text(errors='ignore'),
        'graphify_data_fresh': gj.exists() and gr.exists() and (now-gj.stat().st_mtime)<24*3600 and (now-gr.stat().st_mtime)<24*3600,
        'graphify_runtime_hooks_wired': settings_has_gate(CC_SETTINGS),
        'graphify_variant_hooks_wired': settings_has_gate(OC_SETTINGS),
        'graphify_verdict_log_exists': verdict.exists(),
        'living_doc_exists': DOC.exists(),
    }
    out={
        'ts': int(now),
        'checks': checks,
        'pass_count': sum(1 for v in checks.values() if v),
        'total_count': len(checks),
        'confidence_percent': round(100*sum(1 for v in checks.values() if v)/len(checks),1),
        'env': env,
        'latest_ledger': str(attach_p) if attach_p else '',
        'latest_turn_ledger': str(p) if p else '',
        'latest_ledger_counts': {'rows':len(rows),'turn_completed':len(completed),'live_capture_attempted':len(attempted),'live_capture_completed':len(live),'live_capture_failed':len(failed),'compression_events':len(comp)},
        'latest_live_exchange_id': live[-1].get('exchange_id','') if live else '',
        'latest_capture_state': capture_state,
        'plugin_parity': plugin_parity_state,
        'live_briefing_state': briefing_state,
        'plugin_sha256': sha(PLUGIN),
        'vendored_plugin_sha256': sha(VENDORED_PLUGIN),
        'test_sha256': sha(TESTS),
        'graphify': {
            'graph_json': str(gj), 'graph_report': str(gr),
            'graph_age_seconds': int(now-gj.stat().st_mtime) if gj.exists() else None,
            'report_age_seconds': int(now-gr.stat().st_mtime) if gr.exists() else None,
            'html_age_note': 'viz may be partial/stale; data graph/report are gate source',
            'verdict_tail': verdict_tail,
        },
        'gaps': [k for k,v in checks.items() if not v],
    }
    print(json.dumps(out, indent=2))
    return 0 if all(checks.values()) else 1
if __name__ == '__main__':
    raise SystemExit(main())

