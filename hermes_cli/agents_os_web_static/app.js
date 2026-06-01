const params = new URLSearchParams(location.search);
const state = { dashboard: null, view: params.get('view') || 'home', selectedTaskId: params.get('task'), taskDetail: null, bootstrappedDetail: false };
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function api(path, opts={}){
  const res = await fetch(path, {headers:{'Content-Type':'application/json'}, ...opts});
  const data = await res.json();
  if(!data.ok) throw new Error(data.error?.message || 'API error');
  return data.data;
}
function badge(value){ const v=String(value); const cls=(v==='true'||v==='ok'||v==='0'||v==='completed'||v==='succeeded'||v==='available')?'ok':(v==='false'||v==='not_verified'?'warn':'warn'); return `<span class="badge ${cls}">${escapeHtml(v)}</span>`; }
function escapeHtml(x){return String(x??'').replace(/[&<>"]/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));}
function item(title, meta='', actions=''){return `<div class="item"><div class="title">${title}</div>${meta?`<div class="meta">${meta}</div>`:''}<div>${actions}</div></div>`}

async function refresh(){
  state.dashboard = await api('/api/dashboard');
  state.sessions = await api('/api/sessions');
  state.skills = await api('/api/skills');
  state.cron = await api('/api/cron');
  $('#healthPill').innerHTML = state.dashboard.status.status === 'ok' ? 'HEALTH OK' : 'ATTENTION';
  $('#healthPill').className = 'health ' + (state.dashboard.status.status === 'ok' ? 'ok' : 'warn');
  $('#clock').textContent = new Date().toLocaleTimeString();
  renderAll();
  if(!state.bootstrappedDetail && state.selectedTaskId){ state.bootstrappedDetail=true; await showTaskDetail(state.selectedTaskId); }
  else if(!state.bootstrappedDetail && state.view !== 'home'){ state.bootstrappedDetail=true; switchView(state.view); }
}

function renderAll(){ renderHome(); renderTasks(); renderApprovals(); renderRuns(); renderSessions(); renderSkills(); renderCron(); renderAgents(); renderArtifacts(); renderWorkflows(); renderSafety(); }
function renderHome(){
  const q=state.dashboard.queue_summary;
  const cards=[['Action required',q.action_required],['Open tasks',q.open_tasks],['Running',q.running_tasks],['Blocked',q.blocked_tasks],['Approvals',q.pending_approvals],['Failed runs',q.failed_executions],['Review',q.review_tasks],['Completed',q.completed_tasks]];
  $('#metricCards').innerHTML=cards.map(([k,v])=>`<div class="card"><label>${k}</label><strong>${v}</strong></div>`).join('');
  $('#nextBestAction').innerHTML=`<div class="item"><div class="title">${escapeHtml(state.dashboard.next_best_action.label)}</div><div class="meta">kind=${escapeHtml(state.dashboard.next_best_action.kind)}</div></div>`;
  $('#doniStatus').innerHTML=`<p>Status: ${badge(state.dashboard.status.status)}</p><p class="path">Home: ${escapeHtml(state.dashboard.status.agents_os_home)}</p><p class="path">DB: ${escapeHtml(state.dashboard.status.state_db)}</p>`;
  $('#recentEvents').innerHTML=state.dashboard.events.slice(0,8).map(e=>item(escapeHtml(e.event_type), `${escapeHtml(e.created_at)} task=${escapeHtml(e.task_id||'-')}`)).join('') || '<p class="muted">No events.</p>';
  $('#latestArtifacts').innerHTML=state.dashboard.artifacts.slice(0,6).map(a=>item(escapeHtml(a.title), `${escapeHtml(a.kind)} · ${escapeHtml(a.path)}`, `<button onclick="showArtifact('${a.id}')">Open artifact</button>`)).join('') || '<p class="muted">No artifacts.</p>';
}
function renderTasks(){
  const statuses=['pending','ready','in_progress','review','blocked','needs_approval','completed'];
  $('#taskBoard').innerHTML=statuses.map(s=>`<div class="kanban-col"><h4>${s}</h4>${state.dashboard.tasks.filter(t=>t.status===s).map(t=>`<div class="task ${state.selectedTaskId===t.id?'selected':''}" onclick="showTaskDetail('${t.id}')"><strong>${escapeHtml(t.title)}</strong><div class="meta">${escapeHtml(t.id)} · ${escapeHtml(t.workflow||'-')} · priority=${escapeHtml(t.priority)}</div><div class="task-actions"><button onclick="event.stopPropagation();showTaskDetail('${t.id}')">Details</button><button onclick="event.stopPropagation();routeTask('${t.id}')">Route</button><button onclick="event.stopPropagation();executeTask('${t.id}')">Execute</button><button onclick="event.stopPropagation();closeTask('${t.id}')">Close</button></div></div>`).join('')||'<p class="muted">Empty</p>'}</div>`).join('');
  renderTaskDetail();
}
function actionButton(action){
  const id=escapeHtml(action.id); const label=escapeHtml(action.label); const disabled=action.allowed?'':'disabled';
  if(action.id==='route') return `<button ${disabled} onclick="routeTask('${state.selectedTaskId}')">${label}</button>`;
  if(action.id==='execute') return `<button ${disabled} onclick="executeTask('${state.selectedTaskId}')">${label}</button>`;
  if(action.id==='close') return `<button ${disabled} onclick="closeTask('${state.selectedTaskId}')">${label}</button>`;
  if(action.id==='open_artifact') return `<button ${disabled} onclick="openFirstTaskArtifact()">${label}</button>`;
  if(action.id==='refresh') return `<button onclick="showTaskDetail('${state.selectedTaskId}')">${label}</button>`;
  return `<button ${disabled}>${label}</button>`;
}
function miniList(rows, empty, render){return rows&&rows.length?rows.map(render).join(''):`<p class="muted">${escapeHtml(empty)}</p>`;}
function renderTaskDetail(){
  const box=$('#taskDetail'); if(!box) return;
  const d=state.taskDetail;
  if(!d){ box.innerHTML='<div class="empty-detail">Select a task to inspect dependencies, handoff, evidence and safe actions.</div>'; return; }
  const t=d.task;
  const dep=d.dependency_status||{};
  box.innerHTML=`
    <div class="detail-head"><div><p class="eyebrow">Task detail</p><h3>${escapeHtml(t.title)}</h3><div class="meta">${escapeHtml(t.id)} · ${escapeHtml(t.workflow||'-')} · agent=${escapeHtml(t.route||'unrouted')} · priority=${escapeHtml(t.priority)}</div></div>${badge(t.status)}</div>
    <div class="detail-actions">${(d.safe_next_actions||[]).map(actionButton).join('')}</div>
    <section class="detail-section ${dep.state==='blocked'||dep.state==='waiting'?'blocked':''}"><h4>Dependency / handoff</h4><p>${badge(dep.state||'unknown')} <span class="meta">reason=${escapeHtml(dep.reason||'-')} waiting=${escapeHtml(dep.waiting_for||'-')}</span></p>${d.handoff_preview?`<div class="handoff"><strong>Parent handoff</strong><p>${escapeHtml(d.handoff_preview.preview||'')}</p><div class="meta">parent=${escapeHtml(d.handoff_preview.parent_id||'-')} artifact=${escapeHtml(d.handoff_preview.artifact_id||'-')}</div></div>`:''}</section>
    <div class="detail-grid">
      <section class="detail-section"><h4>Parent</h4>${d.parent?item(escapeHtml(d.parent.title), `${escapeHtml(d.parent.id)} · ${escapeHtml(d.parent.status)}`, `<button onclick="showTaskDetail('${d.parent.id}')">Open parent</button>`):'<p class="muted">No parent.</p>'}</section>
      <section class="detail-section"><h4>Children</h4>${miniList(d.children,'No child tasks.', c=>item(escapeHtml(c.title), `${escapeHtml(c.id)} · ${escapeHtml(c.status)}`, `<button onclick="showTaskDetail('${c.id}')">Open child</button>`))}</section>
      <section class="detail-section"><h4>Approvals</h4>${miniList(d.approvals,'No approvals.', a=>item(`${escapeHtml(a.title)} ${badge(a.status)}`, `risk=${escapeHtml(a.risk)} · ${escapeHtml(a.payload||'')}`))}</section>
      <section class="detail-section"><h4>Runs</h4>${miniList(d.runs,'No runs.', r=>item(`${escapeHtml(r.id)} ${badge(r.status)}`, `kind=${escapeHtml(r.kind||'-')} · ${escapeHtml(r.created_at||'')}`, `<button onclick="showRun('${r.id}')">Open run</button>`))}</section>
      <section class="detail-section"><h4>Artifacts</h4>${miniList(d.artifacts,'No artifacts.', a=>item(escapeHtml(a.title), `${escapeHtml(a.kind)} · ${escapeHtml(a.id)}`, `<button onclick="showArtifact('${a.id}')">Open artifact</button>`))}</section>
      <section class="detail-section"><h4>Evidence / close</h4>${d.close_evidence?`<p>${escapeHtml(d.close_evidence.preview)}</p><div class="meta">source=${escapeHtml(d.close_evidence.source)}</div>`:'<p class="muted">No close evidence yet.</p>'}</section>
    </div>
    <section class="detail-section"><h4>Event timeline</h4>${miniList(d.events,'No events.', e=>item(escapeHtml(e.event_type), `${escapeHtml(e.created_at)} · run=${escapeHtml(e.run_id||'-')}<br><span class="path">${escapeHtml(e.payload||'')}</span>`))}</section>`;
}
async function showTaskDetail(id){state.selectedTaskId=id;state.taskDetail=await api(`/api/tasks/${id}`);switchView('tasks');renderTasks();}
async function openFirstTaskArtifact(){if(state.taskDetail?.artifacts?.length){await showArtifact(state.taskDetail.artifacts[0].id);}}

function renderApprovals(){
  $('#approvalList').innerHTML=state.dashboard.approvals.map(a=>item(`${escapeHtml(a.title)} ${badge(a.status)}`, `risk=${escapeHtml(a.risk)} · task=${escapeHtml(a.task_id||'-')}<br><span class="path">${escapeHtml(a.payload||'')}</span>`, `<button onclick="approve('${a.id}')">Approve</button><button onclick="deny('${a.id}')">Deny</button>`)).join('') || '<p class="muted">No approvals.</p>';
}
function renderRuns(){
  $('#runsList').innerHTML='<h3>Runs</h3>'+state.dashboard.runs.map(r=>item(`${escapeHtml(r.id)} ${badge(r.status)}`, `kind=${escapeHtml(r.kind)} · workflow=${escapeHtml(r.workflow)} · task=${escapeHtml(r.task_id||'-')}`, `<button onclick="showRun('${r.id}')">Details</button>`)).join('');
  $('#eventsList').innerHTML='<h3>Events</h3>'+state.dashboard.events.map(e=>item(escapeHtml(e.event_type), `${escapeHtml(e.created_at)} · task=${escapeHtml(e.task_id||'-')}`)).join('');
}
function renderSessions(){
  $('#sessionsList').innerHTML=(state.sessions||[]).map(s=>item(escapeHtml(s.title||s.id), `${escapeHtml(s.source)} · model=${escapeHtml(s.model||'-')} · messages=${escapeHtml(s.message_count||0)} · tools=${escapeHtml(s.tool_call_count||0)}<br><span class="path">${escapeHtml(s.last_message_preview||'')}</span>`)).join('') || '<p class="muted">No sessions found.</p>';
}
function renderSkills(){
  $('#skillsList').innerHTML=(state.skills||[]).map(s=>`<div class="card"><label>${escapeHtml(s.category)}</label><strong>${escapeHtml(s.name)}</strong><p>${escapeHtml(s.description||'No description')}</p><p class="path">${escapeHtml(s.path)}</p><p>${badge('read-only')}</p></div>`).join('') || '<p class="muted">No skills found.</p>';
}
function renderCron(){
  $('#cronList').innerHTML=(state.cron||[]).map(j=>item(`${escapeHtml(j.name||j.id)} ${badge(j.enabled && !j.paused ? 'enabled':'paused')}`, `schedule=${escapeHtml(j.schedule||'-')} · last=${escapeHtml(j.last_run_at||'-')} · next=${escapeHtml(j.next_run_at||'-')} · deliver=${escapeHtml(j.deliver||'-')} · no_agent=${escapeHtml(j.no_agent)}`)).join('') || '<p class="muted">No cron jobs found.</p>';
}
async function showRun(id){const d=await api(`/api/runs/${id}`);switchView('runs');$('#runDetail').textContent=JSON.stringify(d,null,2);}
function renderAgents(){
  $('#agentsList').innerHTML=state.dashboard.agents.map(a=>`<div class="card"><label>${escapeHtml(a.kind)}</label><strong>${escapeHtml(a.name||a.id)}</strong><p>${badge(a.status)}</p><p class="path">${escapeHtml(a.home||'runtime registered')}</p><p class="muted">${escapeHtml(a.memory_boundary||'Doni Agents OS registry')}</p><p>${escapeHtml(a.capabilities||'[]')}</p></div>`).join('');
}
function renderArtifacts(){
  $('#artifactList').innerHTML=state.dashboard.artifacts.map(a=>item(escapeHtml(a.title), `${escapeHtml(a.kind)} · ${escapeHtml(a.id)}<br><span class="path">${escapeHtml(a.path)}</span>`, `<button onclick="showArtifact('${a.id}')">Preview</button>`)).join('') || '<p class="muted">No artifacts.</p>';
}
function renderWorkflows(){
  $('#workflowGrid').innerHTML=state.dashboard.workflows.map(w=>`<div class="card"><label>${escapeHtml(w.kind)}</label><strong>${escapeHtml(w.id)}</strong><p>${escapeHtml(w.template)}</p><p>Approval: ${badge(Boolean(w.requires_approval))}</p><button onclick="runWorkflow('${w.id}')">Create from workflow</button></div>`).join('');
}
function renderSafety(){
  const s=state.dashboard.safety;
  $('#safetyPanel').innerHTML=[['Doctor',s.doctor.ok],['Mirror',s.mirror_validate.status],['Credential scan',s.credential_scan.status],['Network side effects',s.network_side_effects],['Runtime config changed',s.runtime_config_changed],['Gateway restart',s.gateway_restart],['Profile isolation',s.profile_home_isolation],['Doni/Marija/ERO separation',s.doni_marija_ero_separation]].map(([k,v])=>`<div class="panel"><h3>${k}</h3>${badge(v)}</div>`).join('');
}
function switchView(v){state.view=v;$$('.view').forEach(x=>x.classList.remove('active'));$('#'+v).classList.add('active');$$('nav button').forEach(b=>b.classList.toggle('active',b.dataset.view===v));$('#pageTitle').textContent=v[0].toUpperCase()+v.slice(1);}
function modal(title, fields, onSubmit){$('#modalTitle').textContent=title;$('#modalBody').innerHTML=fields;const d=$('#modal');d.showModal();$('#modalSubmit').onclick=(ev)=>{ev.preventDefault();const data=Object.fromEntries(Array.from(d.querySelectorAll('input,textarea,select')).map(i=>[i.name,i.value]));d.close();onSubmit(data);};}
async function createTask(){modal('Create task', '<label>Title<input name="title" required></label><label>Workflow<select name="workflow"><option>code-task</option><option>research-brief</option><option>qa-report</option><option>youtube-intake</option><option>external-action-draft</option></select></label><label>Notes<textarea name="notes"></textarea></label>', async d=>{await api('/api/tasks',{method:'POST',body:JSON.stringify(d)}); await refresh();});}
async function routeTask(id){await api(`/api/tasks/${id}/route`,{method:'POST',body:'{}'});await refresh();}
async function executeTask(id){await api(`/api/tasks/${id}/execute`,{method:'POST',body:'{}'}).catch(e=>alert(e.message));await refresh();}
async function closeTask(id){modal('Close task with evidence','<label>Evidence<textarea name="evidence" required></textarea></label>',async d=>{await api(`/api/tasks/${id}/close`,{method:'POST',body:JSON.stringify(d)}).catch(e=>alert(e.message));await refresh();});}
async function approve(id){await api(`/api/approvals/${id}/approve`,{method:'POST',body:'{}'});await refresh();}
async function deny(id){await api(`/api/approvals/${id}/deny`,{method:'POST',body:JSON.stringify({notes:'Denied from Mission Control'})});await refresh();}
async function runWorkflow(id){modal(`Run workflow: ${id}`,'<label>Title<input name="title"></label><label>Input<textarea name="input" required></textarea></label>',async d=>{await api(`/api/workflows/${id}/run`,{method:'POST',body:JSON.stringify(d)}).catch(e=>alert(e.message));await refresh();});}
async function showArtifact(id){const p=await api(`/api/artifacts/${id}`);switchView('artifacts');if(p.preview_type==='image' && p.raw_url){$('#artifactPreview').outerHTML='<div id="artifactPreview" class="preview panel"><img src="'+p.raw_url+'" style="max-width:100%;border-radius:12px" alt="artifact image"><p class="path">'+escapeHtml(p.path)+'</p></div>';}else{const old=$('#artifactPreview'); if(old.tagName.toLowerCase()!=='pre'){old.outerHTML='<pre id="artifactPreview" class="preview panel"></pre>';} $('#artifactPreview').textContent = p.content;}}
$$('nav button').forEach(b=>b.onclick=()=>switchView(b.dataset.view));$('#refreshBtn').onclick=refresh;document.body.addEventListener('click',e=>{if(e.target.dataset.action==='create-task')createTask();});
refresh().catch(e=>{document.body.innerHTML='<pre class="preview">Mission Control failed to load: '+escapeHtml(e.message)+'</pre>';});
window.createTask=createTask;window.routeTask=routeTask;window.executeTask=executeTask;window.closeTask=closeTask;window.approve=approve;window.deny=deny;window.runWorkflow=runWorkflow;window.showArtifact=showArtifact;window.showRun=showRun;window.showTaskDetail=showTaskDetail;window.openFirstTaskArtifact=openFirstTaskArtifact;