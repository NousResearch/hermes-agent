const WS_URL = `ws://${location.host}/ws`;
const API = (path) => fetch(path).then(r => r.json());

let ws, inputHistory = [], autoScroll = true;

// ── WebSocket ──────────────────────────────────────────────────────────────
function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => setWsStatus(true);
  ws.onclose = () => { setWsStatus(false); setTimeout(connect, 3000); };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
}

function setWsStatus(ok) {
  document.getElementById('wsDot').className = 'ws-dot' + (ok ? '' : ' disconnected');
  document.getElementById('wsLabel').textContent = ok ? 'LIVE' : 'RECONNECTING';
}

// ── Event handler ──────────────────────────────────────────────────────────
let reasoningEntry = null;

function handleEvent(ev) {
  const now = new Date().toTimeString().slice(0,8);
  if (ev.type === 'action') {
    reasoningEntry = null;
    addLog('tool', now, `Using tool: use_emulator\nButtons: [${ev.action.join(', ')}]\nContext: ${ev.context}`);
    inputHistory.push(ev.action[0] || 'A');
    renderInputHistory();
    document.getElementById('turnCount').textContent = ev.turn;
    document.getElementById('turnBadge').textContent = ev.turn;
    refreshScreenshot();
  } else if (ev.type === 'reasoning') {
    if (!reasoningEntry || !ev.streaming) {
      reasoningEntry = addLog('thinking', now, '');
    }
    const el = reasoningEntry.querySelector('.log-content');
    if (ev.streaming) {
      el.innerHTML = `<span class="thinking-tag">&lt;thinking&gt;</span><br>${
        (el.dataset.text || '') + ev.text
      }<span class="cursor"></span>`;
      el.dataset.text = (el.dataset.text || '') + ev.text;
    } else {
      el.innerHTML = `<span class="thinking-tag">&lt;thinking&gt;</span><br>${ev.text}<br><span class="thinking-tag">&lt;/thinking&gt;</span>`;
      el.dataset.text = '';
      reasoningEntry = null;
    }
    scrollLog();
  } else if (ev.type === 'key_moment') {
    addLog('status', now, `★ ${ev.description}`);
  } else if (ev.type === 'battle_end') {
    const icon = ev.result === 'win' ? '✓' : '✗';
    addLog(ev.result === 'win' ? 'success' : 'error', now, `${icon} Battle ended vs ${ev.opponent} — ${ev.result.toUpperCase()}`);
  } else if (ev.type === 'state_update') {
    updateState(ev.state);
  }
}

// ── Log ────────────────────────────────────────────────────────────────────
function addLog(type, time, text) {
  const scroll = document.getElementById('logScroll');
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = `<div class="log-meta">${time}</div>
    <div class="log-content ${type}">${text.replace(/\n/g,'<br>')}</div>`;
  scroll.appendChild(div);
  scrollLog();
  return div;
}

function scrollLog() {
  if (!autoScroll) return;
  const s = document.getElementById('logScroll');
  s.scrollTop = s.scrollHeight;
}

document.getElementById('logScroll').addEventListener('mouseenter', () => autoScroll = false);
document.getElementById('logScroll').addEventListener('mouseleave', () => autoScroll = true);

// ── Input history ──────────────────────────────────────────────────────────
function renderInputHistory() {
  const last = inputHistory.slice(-7);
  document.getElementById('inputHistory').innerHTML = last.map((b, i) =>
    `<div class="input-btn ${i === last.length-1 ? 'recent' : ''}">${b}</div>`
  ).join('');
}

// ── Screenshot ─────────────────────────────────────────────────────────────
function refreshScreenshot() {
  const img = document.getElementById('gameImg');
  if (img) img.src = `/screenshot?t=${Date.now()}`;
}

// ── State polling ──────────────────────────────────────────────────────────
function updateState(state) {
  if (!state) return;
  if (state.badges !== undefined) document.getElementById('badgeCount').textContent = '⭐'.repeat(state.badges) || '—';
  if (state.money  !== undefined) document.getElementById('money').textContent = `$${state.money}`;
  if (state.playtime_seconds !== undefined) {
    const m = Math.floor(state.playtime_seconds / 60);
    const h = Math.floor(m / 60);
    document.getElementById('playTime').textContent = h > 0 ? `${h}h ${m%60}m` : `${m}m`;
  }
  if (state.party) renderTeam(state.party);
}

async function pollState() {
  try { updateState(await API('/state')); } catch(e) {}
}

// ── Team ───────────────────────────────────────────────────────────────────
function renderTeam(party) {
  const slots = [...party, ...Array(6)].slice(0,6);
  document.getElementById('teamCards').innerHTML = slots.map(mon => {
    if (!mon) return `<div class="pokemon-card empty"><div class="empty-circle"></div><div class="empty-label">Empty</div></div>`;
    const pct = mon.fainted ? 0 : Math.round((mon.hp / mon.max_hp) * 100);
    const cls = pct <= 25 ? 'critical' : pct <= 50 ? 'low' : '';
    const types = (mon.types||[]).map(t => `<span class="type-badge type-${t}">${t}</span>`).join('');
    return `<div class="pokemon-card ${mon.active?'active':''} ${mon.fainted?'fainted':''}">
      <div class="card-top">
        <div><div class="card-name">${mon.name}</div><div class="card-level">Lv.${mon.level}</div></div>
        <div class="card-sprite">${mon.sprite||'?'}</div>
      </div>
      <div class="type-badges">${types}${mon.fainted?'<span class="type-badge type-fnt">FNT</span>':''}</div>
      <div class="hp-section">
        <div class="hp-bar-outer"><div class="hp-bar-inner ${cls}" style="width:${pct}%"></div></div>
        <div class="hp-text">${mon.fainted?0:mon.hp}/${mon.max_hp}</div>
      </div>
    </div>`;
  }).join('');
}

// ── Init ───────────────────────────────────────────────────────────────────
connect();
pollState();
setInterval(pollState, 10000);
