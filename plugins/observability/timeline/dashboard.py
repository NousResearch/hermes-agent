"""Self-contained HTML dashboard generation for the local timeline plugin."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable


def dashboard_data(
    *,
    list_runs: Callable[..., list[dict[str, Any]]],
    list_thread_runs: Callable[..., list[dict[str, Any]]],
    get_run: Callable[[str], tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    iso: Callable[..., str],
    limit: int = 50,
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    thread_id: str = "",
) -> dict[str, Any]:
    """Return runs with events in a shape suited for the local dashboard."""
    if chat_id or thread_id:
        runs = list_thread_runs(platform=platform, chat_id=chat_id, thread_id=thread_id, limit=limit)
    else:
        runs = list_runs(limit=limit, platform=platform, source=source)
    out_runs: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for run in runs:
        full_run, events = get_run(str(run["run_id"]))
        if not full_run:
            continue
        clean_events: list[dict[str, Any]] = []
        for event in events:
            event = dict(event)
            try:
                payload = json.loads(event.get("payload_json") or "{}")
            except Exception:
                payload = {"raw": event.get("payload_json")}
            event["payload"] = payload
            event.pop("payload_json", None)
            clean_events.append(event)
            et = str(event.get("event_type") or "unknown")
            totals[et] = totals.get(et, 0) + 1
        full_run = dict(full_run)
        full_run["events"] = clean_events
        out_runs.append(full_run)
    return {
        "generated_at": iso(),
        "filters": {
            "limit": limit,
            "platform": platform,
            "source": source,
            "chat_id": chat_id,
            "thread_id": thread_id,
        },
        "runs": out_runs,
        "totals": totals,
    }


def render_dashboard_html(data: dict[str, Any], *, api_url: str = "", poll_ms: int = 2000) -> str:
    data_json = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    live_api_url = json.dumps(api_url, ensure_ascii=False)
    live_poll_ms = max(500, int(poll_ms or 2000))
    generated = html.escape(str(data.get("generated_at") or ""))
    run_count = len(data.get("runs") or [])
    event_count = sum(len(r.get("events") or []) for r in data.get("runs") or [])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hermes Timeline</title>
  <style>
    :root {{
      --bg: #0d1110; --panel: #151b19; --panel-2: #1d2522; --ink: #edf7ef;
      --muted: #8da097; --line: #2a3631; --green: #87f7a5; --amber: #ffd36e;
      --red: #ff7c7c; --blue: #88c8ff; --violet: #c7a5ff;
      --mono: "SFMono-Regular", "JetBrains Mono", "Menlo", monospace;
      --sans: "Avenir Next", "SF Pro Display", ui-sans-serif, system-ui;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; color: var(--ink); font-family: var(--sans);
      background: radial-gradient(circle at 15% -10%, rgba(135,247,165,.20), transparent 30rem), radial-gradient(circle at 100% 10%, rgba(136,200,255,.14), transparent 24rem), linear-gradient(135deg, #0a0d0c, var(--bg)); }}
    body::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .22; background-image: linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px); background-size: 28px 28px; mask-image: linear-gradient(to bottom, black, transparent 70%); }}
    header {{ padding: 32px clamp(20px, 4vw, 56px) 18px; position: sticky; top: 0; z-index: 3; backdrop-filter: blur(18px); background: linear-gradient(to bottom, rgba(13,17,16,.92), rgba(13,17,16,.62)); border-bottom: 1px solid rgba(135,247,165,.12); }}
    .brand {{ display: flex; align-items: center; justify-content: space-between; gap: 20px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 56px); letter-spacing: -.05em; line-height: .95; }}
    .pulse {{ width: 16px; height: 16px; border-radius: 50%; background: var(--green); box-shadow: 0 0 28px var(--green); animation: pulse 1.6s infinite; }}
    @keyframes pulse {{ 50% {{ transform: scale(.72); opacity: .45; }} }}
    .sub {{ color: var(--muted); margin-top: 10px; font-family: var(--mono); font-size: 12px; }}
    .toolbar {{ display: grid; grid-template-columns: 1.5fr repeat(3, minmax(120px, .7fr)); gap: 10px; margin-top: 22px; }}
    .filterchips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .filterchips button {{ border:1px solid var(--line); background:rgba(21,27,25,.86); color:var(--muted); border-radius:999px; padding:8px 11px; font:11px var(--mono); cursor:pointer; }}
    .filterchips button:hover {{ color:var(--ink); border-color:rgba(135,247,165,.45); }}
    input, select {{ width: 100%; border: 1px solid var(--line); color: var(--ink); background: rgba(21,27,25,.78); border-radius: 14px; padding: 12px 13px; outline: none; font: 13px var(--mono); }}
    input:focus, select:focus {{ border-color: var(--green); box-shadow: 0 0 0 3px rgba(135,247,165,.12); }}
    main {{ padding: 24px clamp(20px, 4vw, 56px) 60px; display: grid; grid-template-columns: minmax(300px, 420px) 1fr; gap: 18px; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 14px; }}
    .stat, .run, .detail {{ border: 1px solid var(--line); background: rgba(21,27,25,.82); border-radius: 22px; box-shadow: 0 22px 60px rgba(0,0,0,.25); }}
    .stat {{ padding: 15px; }} .stat b {{ display: block; font: 28px var(--mono); color: var(--green); }} .stat span {{ color: var(--muted); font-size: 12px; }}
    .runlist {{ display: flex; flex-direction: column; gap: 12px; max-height: calc(100vh - 230px); overflow: auto; padding-right: 4px; }}
    .thread-group {{ border: 1px solid rgba(42,54,49,.64); border-radius: 24px; padding: 10px; background: rgba(8,12,11,.22); }}
    .thread-head {{ display:flex; justify-content:space-between; gap:10px; align-items:start; padding: 4px 4px 10px; color: var(--muted); font: 11px var(--mono); }}
    .thread-title {{ color: var(--green); word-break: break-all; }}
    .thread-sub {{ margin-top:4px; color: var(--muted); }}
    .thread-runs {{ display:flex; flex-direction:column; gap:10px; }}
    .run {{ cursor: pointer; padding: 14px; transition: transform .14s ease, border-color .14s ease; }} .run:hover, .run.active {{ transform: translateY(-1px); border-color: rgba(135,247,165,.55); }}
    .run-top {{ display:flex; justify-content:space-between; gap:10px; align-items:start; }}
    .rid {{ font: 12px var(--mono); color: var(--ink); word-break: break-all; }}
    .badge {{ font: 11px var(--mono); border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; color: var(--muted); white-space: nowrap; }}
    .badge.completed, .badge.ok {{ color: var(--green); border-color: rgba(135,247,165,.45); }} .badge.running {{ color: var(--amber); border-color: rgba(255,211,110,.45); }} .badge.error {{ color: var(--red); border-color: rgba(255,124,124,.45); }}
    .meta {{ margin-top: 10px; color: var(--muted); font: 11px var(--mono); display:flex; flex-wrap:wrap; gap:8px; }}
    .detail {{ min-height: 520px; padding: 20px; overflow: hidden; }}
    .detail h2 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: -.03em; }}
    .overview {{ display:grid; grid-template-columns: repeat(4, minmax(110px, 1fr)); gap:10px; margin: 16px 0 6px; }}
    .mini {{ border:1px solid var(--line); border-radius:16px; padding:12px; background:rgba(0,0,0,.18); }}
    .mini b {{ display:block; font: 22px var(--mono); color: var(--green); }} .mini span {{ color:var(--muted); font-size:11px; }}
    .mini.danger b {{ color: var(--red); }} .mini.warn b {{ color: var(--amber); }} .mini.blue b {{ color: var(--blue); }}
    .quick {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
    .chip {{ border:1px solid var(--line); background:rgba(29,37,34,.8); color:var(--muted); border-radius:999px; padding:7px 10px; font:11px var(--mono); }}
    .lane {{ margin-top:18px; border:1px solid rgba(42,54,49,.76); border-radius:22px; overflow:hidden; background:rgba(8,12,11,.28); }}
    .lane summary {{ list-style:none; cursor:pointer; }} .lane summary::-webkit-details-marker {{ display:none; }}
    .lane-title {{ display:flex; justify-content:space-between; align-items:center; padding:12px 15px; border-bottom:1px solid var(--line); font:12px var(--mono); letter-spacing:.02em; text-transform:uppercase; color:var(--muted); background:linear-gradient(90deg, rgba(135,247,165,.08), transparent); }}
    .lane:not([open]) .lane-title {{ border-bottom:0; }}
    .lane-body {{ padding: 12px; display:flex; flex-direction:column; gap:12px; }}
    .lane-empty {{ color:var(--muted); font:12px var(--mono); padding:10px; }}
    .timeline {{ margin-top: 20px; position: relative; display:flex; flex-direction:column; gap: 12px; }}
    .event {{ position: relative; display:grid; grid-template-columns: 182px 1fr; gap: 20px; align-items:start; }}
    .event::before {{ content:""; position:absolute; left: 170px; top: 21px; width: 10px; height: 10px; border-radius: 50%; background: var(--blue); box-shadow: 0 0 18px rgba(136,200,255,.65); }}
    .etime {{ color: var(--muted); font: 11px var(--mono); padding-top: 1px; padding-right: 18px; text-align:right; line-height:1.35; }}
    .etime .abs {{ display:block; color:#d5e4db; }}
    .etime .rel {{ display:block; color:var(--muted); }}
    .ecard {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 18px; padding: 13px 14px; }}
    .ename {{ display:flex; justify-content:space-between; gap:10px; font: 12px var(--mono); }}
    .etype {{ color: var(--blue); }} .gateway-message, .gateway-delivery {{ color: var(--green); }} .llm-request, .llm-response {{ color: var(--violet); }} .tool-call, .tool-result {{ color: var(--amber); }}
    .summary {{ margin-top: 8px; color: #d5e4db; white-space: pre-wrap; font-size: 13px; line-height: 1.45; }}
    details {{ margin-top: 8px; }} summary {{ color: var(--muted); cursor:pointer; font: 11px var(--mono); }} pre {{ white-space: pre-wrap; word-break: break-word; color: #c9d9d0; background: rgba(0,0,0,.22); border-radius: 12px; padding: 10px; max-height: 320px; overflow:auto; }}
    .empty {{ color: var(--muted); padding: 40px; text-align:center; }}
    @media (max-width: 920px) {{ main {{ grid-template-columns: 1fr; }} .toolbar {{ grid-template-columns: 1fr; }} .runlist {{ max-height: none; }} }}
  </style>
</head>
<body>
<header>
  <div class="brand"><div><h1>Hermes Timeline</h1><div class="sub" id="subline">local SQLite · generated {generated} · {run_count} runs · {event_count} events</div></div><div class="pulse" aria-hidden="true"></div></div>
  <div class="toolbar"><input id="q" placeholder="Filter runs/events…" /><select id="eventType"><option value="">All event types</option></select><select id="status"><option value="">All statuses</option></select><input id="source" placeholder="Source contains…" /></div>
  <div class="filterchips"><button type="button" data-filter="delivery">Delivery only</button><button type="button" data-filter="errors">Errors</button><button type="button" data-filter="running">Running turns</button><button type="button" data-filter="clear">Clear filters</button></div>
</header>
<main><section><div class="stats"><div class="stat"><b id="runCount">0</b><span>visible runs</span></div><div class="stat"><b id="eventCount">0</b><span>visible events</span></div><div class="stat"><b id="deliveryCount">0</b><span>deliveries</span></div><div class="stat"><b id="errorCount">0</b><span>errors</span></div></div><div id="runs" class="runlist"></div></section><section id="detail" class="detail"><div class="empty">Select a run to inspect its event waterfall.</div></section></main>
<script type="application/json" id="timeline-data">{data_json}</script>
<script>
const initialData = JSON.parse(document.getElementById('timeline-data').textContent);
let data = initialData;
const liveApiUrl = {live_api_url};
const livePollMs = {live_poll_ms};
let selected = null;
const laneOpenState = new Map();
const $ = id => document.getElementById(id);
const fmt = ts => ts ? new Date(ts * 1000).toLocaleString() : '';
const fmtEvent = ts => ts ? new Date(ts * 1000).toLocaleString('ko-KR', {{year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false}}) : '';
const ms = v => v == null ? '' : (v < 1000 ? `${{v}}ms` : `${{(v/1000).toFixed(2)}}s`);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const allEvents = data.runs.flatMap(r => r.events || []);
[...new Set(allEvents.map(e => e.event_type).filter(Boolean))].sort().forEach(t => $('eventType').insertAdjacentHTML('beforeend', `<option>${{esc(t)}}</option>`));
[...new Set([...data.runs.map(r => r.status), ...allEvents.map(e => e.status)].filter(Boolean))].sort().forEach(t => $('status').insertAdjacentHTML('beforeend', `<option>${{esc(t)}}</option>`));
function matches(run) {{
  const q = $('q').value.toLowerCase(); const et = $('eventType').value; const st = $('status').value; const src = $('source').value.toLowerCase();
  const events = run.events || [];
  const blob = JSON.stringify(run).toLowerCase();
  return (!q || blob.includes(q)) && (!src || String(run.source || '').toLowerCase().includes(src)) && (!et || events.some(e => e.event_type === et)) && (!st || run.status === st || events.some(e => e.status === st));
}}
function visibleRuns() {{ return data.runs.filter(matches); }}
function isError(e) {{ return ['error','blocked','failed'].includes(String(e.status || '').toLowerCase()) || String(e.event_type || '').includes('error'); }}
function laneOf(e) {{
  const t = String(e.event_type || '');
  if (t.startsWith('gateway')) return 'Gateway';
  if (t.startsWith('llm')) return 'LLM';
  if (t.startsWith('tool')) return 'Tools';
  if (t === 'approval') return 'Approvals';
  if (t === 'session' || t === 'turn') return 'Agent';
  return 'Other';
}}
function runMetrics(run) {{
  const events = run.events || [];
  return {{
    total: events.length,
    deliveries: events.filter(e => e.event_type === 'gateway.delivery').length,
    errors: events.filter(isError).length,
    tools: events.filter(e => String(e.event_type || '').startsWith('tool')).length,
    llm: events.filter(e => String(e.event_type || '').startsWith('llm')).length,
  }};
}}
function eventCard(e, start) {{
  const eventTs = e.ts || start;
  const relative = `+${{(eventTs - start).toFixed(3)}}s`;
  return `<div class="event"><div class="etime"><span class="abs">${{esc(fmtEvent(eventTs))}}</span><span class="rel">${{relative}}</span></div><div class="ecard"><div class="ename"><span class="etype ${{esc(String(e.event_type||'').replaceAll('.','-'))}}">${{esc(e.event_type)}} / ${{esc(e.name || '')}}</span><span class="badge ${{esc(e.status || '')}}">${{esc(e.status || '')}}</span></div>${{e.summary ? `<div class="summary">${{esc(e.summary)}}</div>` : ''}}<details><summary>payload</summary><pre>${{esc(JSON.stringify(e.payload || {{}}, null, 2))}}</pre></details></div></div>`;
}}
function conversationOf(run) {{
  const source = String(run.source || '');
  let platform = run.platform || '-';
  let chat = '';
  let thread = '';
  const agentSlack = source.match(/agent:[^:]+:slack:([^:]+):([^:]+):(.+)$/);
  const directSlack = source.match(/^slack:([^:]+):(.+)$/);
  if (agentSlack) {{ platform = 'slack'; chat = agentSlack[2]; thread = agentSlack[3]; }}
  else if (directSlack) {{ platform = 'slack'; chat = directSlack[1]; thread = directSlack[2]; }}
  const key = platform === 'slack' && chat && thread ? `slack:${{chat}}:${{thread}}` : (source || `${{platform}}:unknown`);
  const label = platform === 'slack' && chat && thread ? `Slack thread ${{thread}}` : (source || 'Unknown conversation');
  const sub = platform === 'slack' && chat && thread ? `channel ${{chat}}` : platform;
  return {{key, label, sub, chat, thread, platform}};
}}
function runCard(r) {{
  const m = runMetrics(r);
  return `<article class="run ${{r.run_id===selected?'active':''}}" data-run="${{esc(r.run_id)}}"><div class="run-top"><div class="rid">${{esc(r.run_id)}}</div><span class="badge ${{esc(r.status)}}">${{esc(r.status)}}</span></div><div class="meta"><span>${{esc(r.platform || '-')}}</span><span>${{esc(r.model || '-')}}</span><span>${{ms(r.duration_ms) || 'live'}}</span><span>${{esc(fmt(r.started_at))}}</span></div><div class="quick"><span class="chip">${{m.total}} events</span><span class="chip">${{m.deliveries}} deliveries</span><span class="chip">${{m.tools}} tool</span><span class="chip">${{m.errors}} errors</span></div><div class="meta">session=${{esc(r.session_id || '-')}}</div></article>`;
}}
function renderThreadGroup(group) {{
  const runs = group.runs;
  const events = runs.flatMap(r => r.events || []);
  const sessions = new Set(runs.map(r => r.session_id).filter(Boolean));
  const latest = runs[0]?.started_at ? fmt(runs[0].started_at) : '';
  return `<div class="thread-group" data-thread="${{esc(group.key)}}"><div class="thread-head"><div><div class="thread-title">${{esc(group.label)}}</div><div class="thread-sub">${{esc(group.sub)}} · ${{sessions.size}} sessions · latest ${{esc(latest)}}</div></div><span class="badge">${{runs.length}} runs · ${{events.length}} events</span></div><div class="thread-runs">${{runs.map(runCard).join('')}}</div></div>`;
}}
function renderRuns() {{
  const rows = visibleRuns();
  const visibleEvents = rows.flatMap(r => r.events || []);
  const totalEvents = data.runs.reduce((n,r)=>n+(r.events||[]).length,0);
  $('subline').textContent = `local SQLite · generated ${{data.generated_at || ''}} · ${{data.runs.length}} runs · ${{totalEvents}} events`;
  $('runCount').textContent = rows.length;
  $('eventCount').textContent = visibleEvents.length;
  $('deliveryCount').textContent = visibleEvents.filter(e => e.event_type === 'gateway.delivery').length;
  $('errorCount').textContent = visibleEvents.filter(isError).length;
  if (!selected && rows[0]) selected = rows[0].run_id;
  const groups = [];
  const byKey = new Map();
  rows.forEach(r => {{
    const convo = conversationOf(r);
    if (!byKey.has(convo.key)) {{ byKey.set(convo.key, {{...convo, runs: []}}); groups.push(byKey.get(convo.key)); }}
    byKey.get(convo.key).runs.push(r);
  }});
  $('runs').innerHTML = groups.map(renderThreadGroup).join('') || '<div class="empty">No matching runs.</div>';
  document.querySelectorAll('.run[data-run]').forEach(el => el.onclick = () => {{ selected = el.dataset.run; render(); }});
}}
function laneStateKey(runId, lane) {{ return `${{runId}}::${{lane}}`; }}
function laneDefaultOpen(lane, count) {{ return ['Gateway', 'Agent'].includes(lane) || count <= 6; }}
function updateLaneAction(details) {{
  const action = details.querySelector('[data-lane-action]');
  if (action) action.textContent = `${{details.dataset.count || '0'}} events · click to ${{details.open ? 'collapse' : 'expand'}}`;
}}
function renderDetail() {{
  const run = data.runs.find(r => r.run_id === selected) || visibleRuns()[0];
  if (!run) {{ $('detail').innerHTML = '<div class="empty">No run selected.</div>'; return; }}
  selected = run.run_id; const start = run.started_at || 0; const events = run.events || []; const m = runMetrics(run);
  const lanes = ['Gateway','Agent','LLM','Tools','Approvals','Other'];
  const laneHtml = lanes.map(lane => {{
    const laneEvents = events.filter(e => laneOf(e) === lane);
    if (!laneEvents.length) return '';
    const stateKey = laneStateKey(run.run_id, lane);
    const storedOpen = laneOpenState.get(stateKey);
    const isOpen = storedOpen === undefined ? laneDefaultOpen(lane, laneEvents.length) : storedOpen;
    return `<details class="lane" data-lane="${{esc(lane)}}" data-count="${{laneEvents.length}}" ${{isOpen ? 'open' : ''}}><summary class="lane-title"><span>${{lane}}</span><span data-lane-action>${{laneEvents.length}} events · click to ${{isOpen ? 'collapse' : 'expand'}}</span></summary><div class="lane-body">${{laneEvents.map(e => eventCard(e, start)).join('')}}</div></details>`;
  }}).join('');
  const errorChips = events.filter(isError).slice(0, 4).map(e => `<span class="chip">${{esc(e.event_type)}} · ${{esc(e.name || e.status || 'error')}}</span>`).join('') || '<span class="chip">No errors detected</span>';
  $('detail').innerHTML = `<h2>${{esc(run.run_id)}}</h2><div class="meta"><span>status=${{esc(run.status)}}</span><span>platform=${{esc(run.platform || '-')}}</span><span>model=${{esc(run.model || '-')}}</span><span>duration=${{ms(run.duration_ms) || 'live'}}</span></div><div class="meta">${{esc(run.source || '')}}</div><div class="overview"><div class="mini blue"><b>${{m.total}}</b><span>events</span></div><div class="mini"><b>${{m.deliveries}}</b><span>deliveries</span></div><div class="mini warn"><b>${{m.tools}}</b><span>tool events</span></div><div class="mini danger"><b>${{m.errors}}</b><span>errors</span></div></div><div class="quick">${{errorChips}}</div>${{laneHtml || '<div class="empty">No events recorded for this run.</div>'}}`;
  document.querySelectorAll('#detail details.lane').forEach(details => {{
    updateLaneAction(details);
    details.addEventListener('toggle', () => {{
      laneOpenState.set(laneStateKey(run.run_id, details.dataset.lane || ''), details.open);
      updateLaneAction(details);
    }});
  }});
}}
function render() {{ renderRuns(); renderDetail(); }}
['q','eventType','status','source'].forEach(id => $(id).addEventListener('input', () => {{ selected=null; render(); }}));
document.querySelectorAll('[data-filter]').forEach(btn => btn.addEventListener('click', () => {{
  const kind = btn.dataset.filter;
  $('q').value = ''; $('eventType').value = ''; $('status').value = '';
  if (kind === 'delivery') $('eventType').value = 'gateway.delivery';
  if (kind === 'errors') $('status').value = 'error';
  if (kind === 'running') $('status').value = 'running';
  if (kind === 'clear') $('source').value = '';
  selected = null; render();
}}));
async function refreshLiveData() {{
  if (!liveApiUrl) return;
  try {{
    const res = await fetch(liveApiUrl, {{cache: 'no-store'}});
    if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
    data = await res.json();
    render();
  }} catch (err) {{
    console.warn('timeline refresh failed', err);
  }}
}}
if (liveApiUrl) setInterval(refreshLiveData, livePollMs);
render();
</script>
</body>
</html>"""


def write_dashboard(
    path: str | Path,
    *,
    list_runs: Callable[..., list[dict[str, Any]]],
    list_thread_runs: Callable[..., list[dict[str, Any]]],
    get_run: Callable[[str], tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    iso: Callable[..., str],
    limit: int = 50,
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    thread_id: str = "",
) -> Path:
    """Write a self-contained local HTML dashboard and return its path."""
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dashboard_data(
        list_runs=list_runs,
        list_thread_runs=list_thread_runs,
        get_run=get_run,
        iso=iso,
        limit=limit,
        platform=platform,
        source=source,
        chat_id=chat_id,
        thread_id=thread_id,
    )
    out.write_text(render_dashboard_html(data), encoding="utf-8")
    return out
