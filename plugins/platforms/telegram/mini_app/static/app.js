const tg = window.Telegram?.WebApp;

function requestImmersiveLaunch(){
  if(!tg) return;
  try{
    tg.ready?.();
    tg.expand?.();
    if(tg.isVersionAtLeast?.('8.0') && !tg.isFullscreen) tg.requestFullscreen?.();
  }catch(error){
    console.warn('Telegram fullscreen request failed; keeping expanded mode.', error);
    tg.expand?.();
  }
}
requestImmersiveLaunch();

let initData = tg?.initData || '';
let sessionExchange = null;
const app = document.getElementById('app');
const nav = document.getElementById('tabs');
const topbar = document.getElementById('topbar');
const q = document.getElementById('q');

function tgOk(v){ return !!tg?.isVersionAtLeast?.(v); }
function haptic(kind='selection'){ try{ kind==='error' ? tg?.HapticFeedback?.notificationOccurred?.('error') : tg?.HapticFeedback?.selectionChanged?.(); }catch{} }
function applySafeArea(){
  const safe = tg?.safeAreaInset || {};
  const content = tg?.contentSafeAreaInset || {};
  const top = Math.max(Number(safe.top)||0, Number(content.top)||0);
  const bottom = Math.max(Number(safe.bottom)||0, Number(content.bottom)||0);
  // Fullscreen Telegram keeps its Close/menu controls below the physical notch.
  // The SDK safe-area values do not always include this floating control row.
  const fullscreenChrome = tg?.isFullscreen ? 52 : 0;
  document.documentElement.style.setProperty('--safe-top', (top+fullscreenChrome)+'px');
  document.documentElement.style.setProperty('--safe-bottom', bottom+'px');
}
function applyTelegramChrome(){
  if(tgOk('6.1')){ tg?.setHeaderColor?.('#041c1c'); tg?.setBackgroundColor?.('#041c1c'); }
  if(tgOk('7.10')) tg?.setBottomBarColor?.('#041c1c');
  applySafeArea();
}

const I = {
  home:'<svg viewBox="0 0 24 24"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/></svg>',
  cmd:'<svg viewBox="0 0 24 24"><path d="M5 7h14"/><path d="M8 12h8"/><path d="M5 17h14"/></svg>',
  sessions:'<svg viewBox="0 0 24 24"><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h10"/></svg>',
  tools:'<svg viewBox="0 0 24 24"><path d="m14.7 6.3 3-3 3 3-3 3"/><path d="M17.7 9.3 9 18l-4 1 1-4 8.7-8.7"/></svg>',
  search:'<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  x:'<svg viewBox="0 0 24 24"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
  check:'<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
  info:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/></svg>',
  copy:'<svg viewBox="0 0 24 24"><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1"/></svg>',
  bolt:'<svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 10-13h-7z"/></svg>',
  user:'<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>',
  bot:'<svg viewBox="0 0 24 24"><rect x="5" y="7" width="14" height="12" rx="3"/><path d="M12 7V3"/><path d="M9 13h.01"/><path d="M15 13h.01"/></svg>',
  clock:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  stats:'<svg viewBox="0 0 24 24"><path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 16v-5"/><path d="M12 16V8"/><path d="M16 16v-3"/></svg>',
  branch:'<svg viewBox="0 0 24 24"><path d="M6 3v6a6 6 0 0 0 6 6h6"/><circle cx="6" cy="3" r="2"/><circle cx="18" cy="15" r="2"/><circle cx="6" cy="21" r="2"/><path d="M6 9v12"/></svg>',
  download:'<svg viewBox="0 0 24 24"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>',
  plus:'<svg viewBox="0 0 24 24"><path d="M12 5v14"/><path d="M5 12h14"/></svg>',
  filter:'<svg viewBox="0 0 24 24"><path d="M4 6h16"/><path d="M7 12h10"/><path d="M10 18h4"/></svg>',
  shield:'<svg viewBox="0 0 24 24"><path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6z"/></svg>'
};
document.getElementById('searchIcon').innerHTML = I.search;
document.getElementById('searchClose').innerHTML = I.x;

const tabs = [['Home','home'],['Swarm','bot'],['Memory','branch'],['Sessions','sessions'],['Tools','tools']];
let active = 'Home';
let state = { swarmView:'board', toolsView:'toolsets', sessionLimit:12, memoryOpen:{}, savedAt:0 };
let liveTimer = 0, liveRefreshing = false;
try{
  // Older builds cached memory, session previews, and usage results. Purge that
  // snapshot before loading the intentionally tiny UI-preference allowlist.
  localStorage.removeItem('hermesMiniSnapshot');
  localStorage.removeItem('hermesMiniUpdateCheckedAt');
  const preferences=JSON.parse(localStorage.getItem('hermesMiniPreferences')||'{}')||{};
  if(['board','agents'].includes(preferences.swarmView)) state.swarmView=preferences.swarmView;
  if(['toolsets','skills'].includes(preferences.toolsView)) state.toolsView=preferences.toolsView;
  if(Number.isInteger(preferences.sessionLimit)) state.sessionLimit=Math.max(12,Math.min(60,preferences.sessionLimit));
}catch{}

const cache = new Map();
const searchableTabs = ['Swarm','Memory','Sessions','Tools'];
function esc(s){ return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function enc(s){ return encodeURIComponent(String(s??'')); }
function dec(s){ try{return decodeURIComponent(String(s??''));}catch{return String(s??'');} }
function fmt(n){ n=Number(n||0); return Math.abs(n)>=1e6?(n/1e6).toFixed(1)+'M':Math.abs(n)>=1e3?(n/1e3).toFixed(1)+'K':String(n); }
function shortText(s,n=170){ s=String(s||'').replace(/\s+/g,' ').trim(); return s.length>n?s.slice(0,n-1).trim()+'…':s; }
function when(ts){ if(!ts)return ''; const ms=typeof ts==='string'?Date.parse(ts):(ts>2e10?ts:ts*1000); const d=Date.now()-ms; if(!Number.isFinite(ms))return ''; if(d<6e4)return 'now'; if(d<36e5)return Math.floor(d/6e4)+'m ago'; if(d<864e5)return Math.floor(d/36e5)+'h ago'; return new Date(ms).toLocaleDateString(); }
function clsStatus(s){ return ['connected','running','ready','done','on','healthy'].includes(String(s).toLowerCase())?'good':['blocked','warning','queued'].includes(String(s).toLowerCase())?'warn':['error','failed','stopped','off','unknown','unavailable'].includes(String(s).toLowerCase())?'bad':''; }

const READ_ONLY_ENDPOINTS = [
  /^\/api\/me$/,
  /^\/api\/status$/,
  /^\/api\/live-usage\?include_accounts=false$/,
  /^\/api\/swarm\/board$/,
  /^\/api\/memory$/,
  /^\/api\/sessions\?limit=30$/,
  /^\/api\/tools\/toolsets$/,
  /^\/api\/skills$/
];
async function api(path){
  if(!READ_ONLY_ENDPOINTS.some(pattern=>pattern.test(path))) throw new Error('Blocked non-read-only Mini App endpoint');
  let r = await fetch(path,{method:'GET',credentials:'same-origin'});
  if(r.status===401 && initData){
    if(!sessionExchange){
      const exchangeData=initData;
      initData='';
      sessionExchange=fetch('/api/auth/session',{
        method:'POST',
        credentials:'same-origin',
        headers:{'X-Telegram-Init-Data':exchangeData}
      }).then(async response=>{
        const body=await response.json().catch(()=>({error:response.statusText}));
        if(!response.ok) throw new Error(body.error||body.detail||'Telegram session exchange failed');
        return body;
      });
    }
    await sessionExchange;
    r = await fetch(path,{method:'GET',credentials:'same-origin'});
  }
  const j = await r.json().catch(()=>({error:r.statusText}));
  if(!r.ok) throw new Error(j.error||j.detail||JSON.stringify(j));
  return j;
}
async function cached(path,ttl=30000){ const c=cache.get(path), now=Date.now(); if(c&&now-c.t<ttl)return c.v; const v=await api(path); cache.set(path,{t:now,v}); return v; }
function savePreferences(){
  try{ localStorage.setItem('hermesMiniPreferences', JSON.stringify({toolsView:state.toolsView,sessionLimit:state.sessionLimit,swarmView:state.swarmView})); }catch(e){ console.warn(e); }
}
function toast(m,kind='success'){ try{ tg?.HapticFeedback?.notificationOccurred?.(kind); }catch{} tg?.showPopup?tg.showPopup({message:m,buttons:[{type:'ok'}]}):alert(m); }
async function copyText(t){ try{ await navigator.clipboard.writeText(t); toast('Copied'); }catch{ prompt('Copy this:',t); } }

function icon(name,cls=''){ return `<div class="icon ${cls}">${I[name]||I.info}</div>`; }
function badge(text,kind=''){ return `<span class="badge ${kind}">${esc(text)}</span>`; }
function row(ic,title,desc='',right='',cls='',extra='',attrs=''){
  return `<div class="row" ${attrs}>${icon(ic,cls)}<div class="content"><div class="title">${title}</div>${desc?`<div class="desc clamp">${desc}</div>`:''}${extra}</div>${right?`<div class="right">${right}</div>`:''}</div>`;
}
function section(title,body,sub=''){ return `<section class="section"><div class="section-title">${esc(title)}${sub?`<span>${esc(sub)}</span>`:''}</div><div class="list">${body||emptyState('Nothing here','No matching items yet.','info')}</div></section>`; }
function stat(num,lab,kind=''){ return num==='···'?`<div class="stat loading"><div class="population-shim short"></div><div class="lab">${esc(lab)}</div></div>`:`<div class="stat ${kind}"><div class="num">${esc(num)}</div><div class="lab">${esc(lab)}</div></div>`; }
function emptyState(title,body,ic='info',action=''){
  return `<div class="empty-state">${icon(ic,'blue')}<h3>${esc(title)}</h3><p>${esc(body)}</p>${action?`<div class="actions center">${action}</div>`:''}</div>`;
}
function pillNav(items,current,attr){ return `<div class="pill-nav">${items.map(([id,label])=>`<button class="${id===current?'active':''}" ${attr}="${esc(id)}">${esc(label)}</button>`).join('')}</div>`; }
function actionCard(ic,title,desc,action,label='Open'){
  return `<button class="action-card" data-action="${esc(action)}">${icon(ic,'blue')}<span><b>${esc(title)}</b><small>${esc(desc)}</small></span><em>${esc(label)}</em></button>`;
}
function miniProgress(label,value,kind=''){
  value=Number(value); value=Number.isFinite(value)?Math.max(0,Math.min(100,value)):0;
  const pct = Math.round(value);
  return `<div class="mini-bar"><div><b>${esc(label)}</b><span>${pct}%</span></div><div class="bar" role="progressbar" aria-label="${esc(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span class="${kind} w${pct}"></span></div></div>`;
}
function renderTabs(){
  nav.innerHTML = tabs.map(([t,i])=>`<button class="${t===active?'active':''}" data-tab="${esc(t)}">${I[i]}<span>${t}</span></button>`).join('');
  if(tgOk('6.1')) (active==='Home'&&!topbar.classList.contains('searching')) ? tg?.BackButton?.hide?.() : tg?.BackButton?.show?.();
}
function setSearch(show,ph){ topbar.classList.toggle('searching',!!show); q.placeholder=ph||'Search'; if(!show){ q.value=''; topbar.classList.remove('has-query'); } }
function closeSearch(){ q.value=''; topbar.classList.remove('active','has-query'); if(!searchableTabs.includes(active)) topbar.classList.remove('searching'); renderCurrent(); renderTabs(); }
function renderSkeleton(t){
  const rows=Array.from({length:t==='Home'?2:5},()=>'<div class="row skel-row"><div class="skel icon"></div><div class="content"><div class="skel line w2"></div><div class="skel line small"></div></div></div>').join('');
  if(t!=='Home'){ app.innerHTML=section(t,rows); return; }
  app.innerHTML=`<div class="app-head"><div class="brand-lockup"><img class="hermes-logo" src="/static/assets/hermes-logo.webp" alt="Hermes"><div class="brand-copy"><h1>Hermes</h1><p>Telegram control surface · connecting</p></div></div></div><div class="focus-panel skel skel-hero"></div><div class="quick-dock">${[1,2,3].map(()=>'<div class="action-card skel"></div>').join('')}</div><div class="metric-ribbon">${[1,2,3,4].map(()=>'<div class="stat"><div class="skel line"></div><div class="skel line small"></div></div>').join('')}</div>`;
}
function hasData(t){ return t==='Home'?state.me&&state.status&&state.live:t==='Swarm'?state.swarm:t==='Memory'?state.memory:t==='Sessions'?state.sessions:t==='Tools'?state.tools&&state.skills:true; }

async function load(t,{force=false}={}){
  if(t==='Home'&&(force||!state.homeCheckedAt||Date.now()-state.homeCheckedAt>15000)){
    // Paint an authenticated shell first. Slower read-only calls hydrate in place.
    state.me=await api('/api/me');
    state.homeCheckedAt=Date.now();
    if(active==='Home') renderHome();
    savePreferences();
    const hydrate = (key,path) => api(path).then(value=>{
      state[key]=value;
      savePreferences();
      if(active==='Home') renderHome();
      return value;
    });
    Promise.allSettled([
      hydrate('status','/api/status'),
      hydrate('live','/api/live-usage?include_accounts=false'),
    ]).then(()=>{
      state.liveUpdatedAt=Date.now();
      loadHomeRest();
    });
  } else if(t==='Swarm'&&(force||!state.swarm)) state.swarm=await api('/api/swarm/board');
  else if(t==='Memory'&&(force||!state.memory)) state.memory=await api('/api/memory');
  else if(t==='Sessions'&&(force||!state.sessions)) state.sessions=await api('/api/sessions?limit=30');
  else if(t==='Tools'&&(force||!state.tools||!state.skills)) [state.tools,state.skills]=await Promise.all([cached('/api/tools/toolsets',60000),cached('/api/skills',60000)]);
}
async function go(t,{force=false}={}){
  haptic();
  const changed=t!==active; if(changed){ q.value=''; topbar.classList.remove('has-query'); }
  active=t; renderTabs();
  setSearch(searchableTabs.includes(t), t==='Swarm'?'Search agents, states, tasks':t==='Memory'?'Search memories':t==='Sessions'?'Search sessions':'Search tools and skills');
  if(liveTimer){ clearInterval(liveTimer); liveTimer=0; }
  hasData(t)&&!force ? renderCurrent() : renderSkeleton(t);
  try{
    await load(t,{force}); savePreferences(); if(active===t) renderCurrent();
    if(t==='Home'){
      liveTimer=setInterval(refreshLiveUsage,3000);
    } else if(t==='Swarm') liveTimer=setInterval(refreshSwarm,15000);
  }catch(e){
    if(active===t){
      const msg=String(e.message||'');
      app.innerHTML = msg.includes('initData') ? `<div class="welcome-error">${icon('shield','red')}<h2>Open from Telegram</h2><p>Telegram did not provide Mini App auth data. Use the Hermes button/menu in your authorized bot chat, not a plain browser link.</p></div>` : `<div class="err">${esc(msg)}</div>`;
    }
  }
}
async function loadHomeRest(){ try{ [state.swarm,state.memory]=await Promise.all([api('/api/swarm/board'),api('/api/memory')]); if(active==='Home')renderHome(); }catch(e){ console.warn(e); } }
async function refreshLiveUsage(){
  if(active!=='Home'||document.hidden||liveRefreshing)return;
  liveRefreshing=true;
  try{
    const core=await api('/api/live-usage?include_accounts=false');
    state.live=core;
    state.liveUpdatedAt=Date.now();
    renderHome();
  }catch(e){ console.warn(e); }
  finally{ liveRefreshing=false; }
}
async function refreshSwarm(){ if(active!=='Swarm'||document.hidden)return; try{ state.swarm=await api('/api/swarm/board'); renderSwarm(); }catch(e){ console.warn(e); } }
function renderCurrent(){ ({Home:renderHome,Swarm:renderSwarm,Memory:renderMemory,Sessions:renderSessions,Tools:renderTools}[active]||renderHome)(); }

function platformRows(p){
  const items=Object.entries(p||{});
  return items.map(([name,x])=>row(x.state==='connected'?'check':'x',esc(name),esc(x.error_message||x.state||'unknown'),badge(x.state||'unknown',clsStatus(x.state)),x.state==='connected'?'green':'red')).join('') || emptyState('No platforms','Gateway has no connected platform details yet.','bot');
}
function renderHome(){
  const live=state.live||{}, sw=state.swarm||{}, mem=state.memory||{};
  const ctx=live.context||{};
  const modelLine=live.model||'';
  const usagePct=Math.round(ctx.percent||0), promptTokens=Number(ctx.prompt_tokens||0);
  const focusTitle=!state.live?'Loading current session':usagePct>84?'Context near limit':'Current session';
  const nextActions=[
    actionCard('bot','Swarm',sw.summary?`${sw.summary.tasks||0} active tasks`:'Mission board','tab:Swarm'),
    actionCard('branch','Memory',mem.count!=null?`${mem.count} saved facts`:'Saved context','tab:Memory'),
    actionCard('clock','Resume','Recent sessions','tab:Sessions')
  ].join('');
  app.innerHTML=`<div class="app-head"><div class="brand-lockup"><img class="hermes-logo" src="/static/assets/hermes-logo.webp" alt="Hermes"><div class="brand-copy"><h1>Hermes</h1><p>Telegram control surface${modelLine?' · '+esc(modelLine):''}</p></div></div></div>`+
    `<div class="focus-panel"><div class="focus-top"><div><h2>${esc(focusTitle)}</h2><p>${esc(ctx.context_length?`${fmt(promptTokens)} of ${fmt(ctx.context_length)} prompt tokens`:'Live usage is loading in place')}</p></div><div class="focus-number">${state.live?usagePct+'%':'···'}<small>context used</small></div></div>${miniProgress('Current context',usagePct,usagePct>85?'danger':'')}<div class="live-readout"><i></i><span>${state.liveUpdatedAt?'Live · refreshed '+when(state.liveUpdatedAt):'Connecting live feed'} · every 3s</span></div></div>`+
    `<div class="quick-dock">${nextActions}</div>`+
    `<div class="metric-ribbon">${stat(state.live?fmt(live.total_tokens||0):'···','Processed')}${stat(state.live?fmt(live.api_calls||0):'···','Calls')}${stat(state.swarm?fmt(sw.summary?.running_tasks||0):'···','Running')}${stat(state.swarm?fmt(sw.summary?.blocked_tasks||0):'···','Blocked',sw.summary?.blocked_tasks?'danger':'')}</div>`;
}

function swarmStatusClass(s){ return s==='running'?'green':s==='blocked'?'amber':s==='error'?'red':s==='queued'||s==='ready'?'blue':''; }
function taskCard(t){ const meta=[t.assignee?'@'+t.assignee:'unassigned',t.priority?('p'+t.priority):''].filter(Boolean).join(' · '); return `<button class="kanban-card" data-copy="${enc(`hermes kanban show ${t.id||''}`)}"><div class="task-title">${esc(t.title||t.id||'Untitled task')}</div>${t.body?`<div class="desc clamp">${esc(shortText(t.body,110))}</div>`:''}<div class="task-meta">${esc(meta)}</div></button>`; }
function renderSwarm(){
  const sw=state.swarm||{}, query=q.value.toLowerCase(), view=state.swarmView==='agents'?'agents':'board', sum=sw.summary||{};
  const agents=(sw.agents||[]).filter(a=>!query||((a.name||'')+' '+(a.model||'')+' '+(a.status||'')).toLowerCase().includes(query));
  const cols=(sw.columns||[]).map(c=>({...c,tasks:(c.tasks||[]).filter(t=>!query||((t.title||'')+' '+(t.body||'')+' '+(t.assignee||'')+' '+(t.status||'')).toLowerCase().includes(query))}));
  const warning=(sw.warning||sw.diagnostic)?`<div class="notice">${icon('info','amber')}<div><b>${esc(sw.source==='kanban-db'?'Direct Kanban board':'Swarm degraded')}</b><span>${esc(sw.warning||sw.diagnostic)}</span></div></div>`:'';
  const boardCols=cols.filter(c=>(c.tasks||[]).length||['triage','ready','running','blocked','review'].includes(String(c.status||c.name).toLowerCase()));
  const board=boardCols.map(c=>`<section class="kanban-lane"><header><b>${esc(c.name||c.status)}</b><span>${(c.tasks||[]).length}</span></header><div class="kanban-stack">${(c.tasks||[]).slice(0,12).map(taskCard).join('')||`<div class="lane-empty">No cards</div>`}</div></section>`).join('');
  const agentRows=agents.map(a=>`<div class="agent-row ${swarmStatusClass(a.status)}"><span class="agent-signal"></span><div><b>${esc(a.name)}</b><small>${esc(a.model||a.gateway||'profile')}</small></div><div class="agent-load"><strong>${a.assigned_count||0}</strong><span>assigned</span></div>${badge(a.status||'idle',clsStatus(a.status))}</div>`).join('')||emptyState('No agent profiles','Profiles appear after `hermes profile list` returns data.','bot');
  app.innerHTML=`<div class="swarm-head"><div><span>Orchestration · ${esc(sw.source||'live')}</span><h1>Swarm</h1></div><button class="btn primary" data-action="refresh-swarm">Refresh board</button></div>`+
    `<div class="swarm-console"><div><b>${fmt(sum.tasks||0)}</b><span>tasks</span></div><div><b>${fmt(sum.running_tasks||0)}</b><span>running</span></div><div class="${sum.blocked_tasks?'danger':''}"><b>${fmt(sum.blocked_tasks||0)}</b><span>blocked</span></div><div><b>${fmt(sum.agents||0)}</b><span>agents</span></div></div>`+
    warning+pillNav([['board','Board'],['agents','Profiles']],view,'data-swarm-view')+
    (view==='agents'?`<div class="agent-list">${agentRows}</div>`:`<div class="kanban-board">${board}</div>`)+
    `<div class="swarm-foot"><span>Source: ${esc(sw.source||'live')}</span><span>Auto-refresh 15s</span></div>`;
}

function renderMemoryEntry(x){ return `<div class="memory-entry"><div class="memory-text">${esc(x.text||'')}</div><div class="memory-meta"><span>${esc(x.file||x.target||'memory')}</span><span>${esc(when(x.updated_at))}</span><button class="btn" data-copy="${enc(x.text||'')}">Copy</button></div></div>`; }
function renderMemory(){
  const query=q.value.toLowerCase(), m=state.memory||{}, open=state.memoryOpen||{};
  const files=(m.files||[]).map(f=>{ const entries=(f.entries||[]).filter(x=>!query||((x.text||'')+' '+(f.name||'')+' '+(f.label||'')).toLowerCase().includes(query)); return {...f,entries,match:!query||entries.length||((f.name||'')+' '+(f.label||'')+' '+(f.raw||'')).toLowerCase().includes(query)}; }).filter(f=>f.match);
  const head=`<div class="insight-strip">${stat(m.count??0,'Memories')}${stat((m.files||[]).length||0,'Files')}</div>`;
  app.innerHTML=head+(files.map(f=>{ const isOpen=query?true:open[f.name]===true; const body=isOpen?`<div class="memory-body">${(f.entries||[]).map(renderMemoryEntry).join('')||emptyState('No matching entries','This memory file has no matches.','search')}</div>`:''; return `<section class="section memory-file"><button class="memory-head" data-memory-file="${esc(f.name)}">${icon(f.target==='user'?'user':'bot','blue')}<div class="content"><div class="title">${esc(f.name||f.label)}</div><div class="desc">${esc(f.label||'')} · ${(f.entries||[]).length}/${f.count||0} entries · ${esc(when(f.updated_at))}</div></div><div class="chev">${isOpen?'−':'+'}</div></button>${body}</section>`; }).join('')||emptyState('No memories','Nothing matched. Memory is quiet. Suspiciously quiet.','branch'));
}

function renderSessions(){
  const query=q.value.toLowerCase(), all=state.sessions?.sessions||[], filtered=all.filter(s=>((s.title||'')+' '+(s.preview||'')+' '+s.id).toLowerCase().includes(query));
  const limit=query?filtered.length:Math.max(12,Number(state.sessionLimit||12));
  const arr=filtered.slice(0,limit), more=filtered.length-arr.length;
  const moreAction=more?`<div class="load-more"><button class="btn" data-action="more-sessions">Show ${Math.min(12,more)} more</button><span>${arr.length} of ${filtered.length}</span></div>`:'';
  app.innerHTML=`<div class="insight-strip">${stat(all.length,'Recent')}${stat(filtered.length,'Matches')}</div>`+section('Recent sessions',arr.map(s=>row('clock',esc(s.title||s.preview||s.id),`${s.source||'local'} · ${s.message_count||0} msgs · ${when(s.last_active||s.started_at)}`,'','',`<div class="desc clamp">${esc(s.preview||'')}</div><div class="actions"><button class="btn primary" data-copy="${enc('/resume '+(s.title||s.id))}">Copy resume</button><button class="btn" data-copy="${enc(s.id)}">Copy ID</button></div>`)).join('')||emptyState('No sessions','Recent sessions will appear here after you chat.','clock'))+moreAction;
}
function renderTools(){
  const query=q.value.toLowerCase(), allTools=state.tools||[], allSkills=state.skills||[], view=state.toolsView||'toolsets';
  const tools=allTools.filter(x=>((x.name||'')+' '+(x.label||'')+' '+(x.description||'')).toLowerCase().includes(query));
  const skills=allSkills.filter(x=>((x.name||'')+' '+(x.description||'')+' '+(x.category||'')).toLowerCase().includes(query));
  const switcher=pillNav([['toolsets',`Toolsets (${allTools.length})`],['skills',`Skills (${allSkills.length})`]],view,'data-tools-view');
  const summary=`<div class="insight-strip">${stat(allTools.length,'Toolsets')}${stat(allSkills.length,'Skills')}</div>`;
  const availability=(x,on,off)=>typeof x.enabled==='boolean'?(x.enabled?{icon:'check',label:on,badge:'good',row:'green'}:{icon:'x',label:off,badge:'bad',row:'red'}):{icon:'info',label:'Catalog',badge:'',row:'blue'};
  const toolRows=tools.map(x=>{const a=availability(x,'Enabled','Off');return row(a.icon,esc(x.label||x.name),esc(x.description||''),badge(a.label,a.badge),a.row);}).join('')||emptyState('No tools found','Search by toolset name or description.','tools');
  const skillRows=skills.map(x=>{const a=availability(x,'On','Off');return row(a.icon,esc(x.name),esc(x.description||x.category||''),badge(a.label,a.badge),a.row);}).join('')||emptyState('No skills found','Try “frontend”, “memory”, or “home”.','branch');
  app.innerHTML=summary+switcher+(view==='skills'?section('Skills',skillRows):section('Toolsets',toolRows));
}

q.addEventListener('focus',()=>{ topbar.classList.add('active'); renderTabs(); });
q.addEventListener('blur',()=>setTimeout(()=>{ topbar.classList.remove('active'); renderTabs(); },120));
q.addEventListener('input',()=>{ topbar.classList.toggle('has-query',!!q.value); renderCurrent(); });
document.getElementById('searchClose').addEventListener('click',closeSearch);
nav.addEventListener('click',e=>{ const b=e.target.closest('button[data-tab]'); if(b) go(b.dataset.tab); });
app.addEventListener('click',async e=>{
  const b=e.target.closest('button'); if(!b)return;
  if(b.dataset.copy!==undefined){ copyText(dec(b.dataset.copy)); return; }
  if(b.dataset.memoryFile!==undefined){ state.memoryOpen=state.memoryOpen||{}; state.memoryOpen[b.dataset.memoryFile]=!Boolean(state.memoryOpen[b.dataset.memoryFile]); renderMemory(); return; }
  if(b.dataset.swarmView){ state.swarmView=b.dataset.swarmView; renderSwarm(); savePreferences(); return; }
  if(b.dataset.toolsView){ state.toolsView=b.dataset.toolsView; renderTools(); savePreferences(); return; }
  if(b.dataset.action==='more-sessions'){ state.sessionLimit=Math.min((state.sessions?.sessions||[]).length,Number(state.sessionLimit||12)+12); renderSessions(); savePreferences(); return; }
  if(b.dataset.action?.startsWith('tab:')){ go(b.dataset.action.slice(4)); return; }
  if(b.dataset.action==='refresh-swarm'){ await refreshSwarm(); toast('Swarm refreshed'); return; }
});
applyTelegramChrome();
tg?.onEvent?.('themeChanged',applyTelegramChrome);
tg?.onEvent?.('safeAreaChanged',applySafeArea);
tg?.onEvent?.('contentSafeAreaChanged',applySafeArea);
tg?.onEvent?.('viewportChanged',applySafeArea);
tg?.onEvent?.('fullscreenChanged',applySafeArea);
tg?.onEvent?.('fullscreenFailed',error=>{
  console.warn('Telegram refused fullscreen mode.',error);
  tg?.expand?.();
  applySafeArea();
});
tg?.onEvent?.('activated',()=>{ if(active==='Home'){refreshLiveUsage();if(!liveTimer)liveTimer=setInterval(refreshLiveUsage,3000);} else if(active==='Swarm'){refreshSwarm();if(!liveTimer)liveTimer=setInterval(refreshSwarm,15000);} });
tg?.onEvent?.('deactivated',()=>{ if(liveTimer){ clearInterval(liveTimer); liveTimer=0; } });
document.addEventListener('visibilitychange',()=>{ if(!document.hidden&&active==='Home')refreshLiveUsage(); });
if(tgOk('6.1')) tg?.BackButton?.onClick?.(()=>topbar.classList.contains('searching')?closeSearch():(active==='Home'?tg.close?.():go('Home')));
renderTabs();
hasData('Home')?renderHome():renderSkeleton('Home');
requestAnimationFrame(()=>tg?.ready?.());
go('Home').then(()=>{ const warm=()=>{['Swarm','Memory'].forEach((t,i)=>setTimeout(()=>load(t).catch(console.warn),900+i*350));}; window.requestIdleCallback?requestIdleCallback(warm,{timeout:2500}):setTimeout(warm,1800); });
