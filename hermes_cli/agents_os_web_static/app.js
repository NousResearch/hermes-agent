const state = { dashboard: null, view: 'home' };
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
  $('#healthPill').innerHTML = state.dashboard.status.status === 'ok' ? 'HEALTH OK' : 'ATTENTION';
  $('#healthPill').className = 'health ' + (state.dashboard.status.status === 'ok' ? 'ok' : 'warn');
  $('#clock').textContent = new Date().toLocaleTimeString();
  renderAll();
}
function renderAll(){ renderHome(); renderTasks(); renderApprovals(); renderRuns(); renderAgents(); renderArtifacts(); renderWorkflows(); renderSafety(); }
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
  const statuses=['pending','ready','in_progress','review','blocked','completed'];
  $('#taskBoard').innerHTML=statuses.map(s=>`<div class="kanban-col"><h4>${s}</h4>${state.dashboard.tasks.filter(t=>t.status===s).map(t=>`<div class="task"><strong>${escapeHtml(t.title)}</strong><div class="meta">${escapeHtml(t.id)} · ${escapeHtml(t.workflow||'-')}</div><button onclick="routeTask('${t.id}')">Route</button><button onclick="executeTask('${t.id}')">Execute</button><button onclick="closeTask('${t.id}')">Close</button></div>`).join('')||'<p class="muted">Empty</p>'}</div>`).join('');
}
function renderApprovals(){
  $('#approvalList').innerHTML=state.dashboard.approvals.map(a=>item(`${escapeHtml(a.title)} ${badge(a.status)}`, `risk=${escapeHtml(a.risk)} · task=${escapeHtml(a.task_id||'-')}<br><span class="path">${escapeHtml(a.payload||'')}</span>`, `<button onclick="approve('${a.id}')">Approve</button><button onclick="deny('${a.id}')">Deny</button>`)).join('') || '<p class="muted">No approvals.</p>';
}
function renderRuns(){
  $('#runsList').innerHTML='<h3>Runs</h3>'+state.dashboard.runs.map(r=>item(`${escapeHtml(r.id)} ${badge(r.status)}`, `kind=${escapeHtml(r.kind)} · workflow=${escapeHtml(r.workflow)} · task=${escapeHtml(r.task_id||'-')}`)).join('');
  $('#eventsList').innerHTML='<h3>Events</h3>'+state.dashboard.events.map(e=>item(escapeHtml(e.event_type), `${escapeHtml(e.created_at)} · task=${escapeHtml(e.task_id||'-')}`)).join('');
}
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
window.createTask=createTask;window.routeTask=routeTask;window.executeTask=executeTask;window.closeTask=closeTask;window.approve=approve;window.deny=deny;window.runWorkflow=runWorkflow;window.showArtifact=showArtifact;