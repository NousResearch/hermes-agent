/**
 * Backend client + derived view-model for Mission Control.
 *
 * Everything the UI renders comes from `GET /api/plugins/samagent/state`, which
 * returns ~25 top-level blocks. The previous UI read about a third of them; the
 * selectors here deliberately cover all of the ones a run needs, so the spec,
 * plan card, ledger, verification ladder and measurements are surfaced instead
 * of fetched and discarded.
 */

const API_BASE = '/api/plugins/samagent';

export function apiGet(path) {
  return fetch(API_BASE + path).then((r) => {
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return r.json();
  });
}

export function apiPost(path, body) {
  return fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then((r) => {
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return r.json();
  });
}

/* -------------------------------------------------------------------------
   Run state model
   -------------------------------------------------------------------------
   Codex ships four states (Running / Needs input / Ready / Blocked) but has no
   first-class failed or queued state, and the Hermes research confirms the same
   gap. A pipeline that runs L0-L4 verification and can escalate to a human
   genuinely needs all six, so we model them explicitly and render a single
   consistent vocabulary everywhere.
   ---------------------------------------------------------------------- */

export const RUN_STATE = {
  queued: { id: 'queued', label: 'Queued', tone: 'idle' },
  running: { id: 'running', label: 'Running', tone: 'info' },
  needs_input: { id: 'needs_input', label: 'Needs input', tone: 'warn' },
  blocked: { id: 'blocked', label: 'Blocked', tone: 'err' },
  ready: { id: 'ready', label: 'Ready', tone: 'ok' },
  failed: { id: 'failed', label: 'Failed', tone: 'err' },
};

const TERMINAL_TODO = new Set(['done', 'complete', 'completed', 'passed']);
const FAILED_TODO = new Set(['failed', 'error', 'errored', 'blocked']);

/** Derive the run's state from the todo board + gate, not from a guess. */
export function deriveRunState(state) {
  const todos = state && state.todos && Array.isArray(state.todos.items) ? state.todos.items : [];
  if (!todos.length) return RUN_STATE.ready;

  const anyFailed = todos.some((t) => FAILED_TODO.has(String(t.status || '').toLowerCase()));
  const anyRunning = todos.some((t) => String(t.status || '').toLowerCase() === 'running');
  const allDone = todos.every((t) => TERMINAL_TODO.has(String(t.status || '').toLowerCase()));

  if (anyRunning) return anyFailed ? RUN_STATE.blocked : RUN_STATE.running;
  if (allDone) {
    const gate = state && state.pre_prod_gate;
    if (gate && gate.ready_for_production === false) return RUN_STATE.blocked;
    return RUN_STATE.ready;
  }
  if (anyFailed) return RUN_STATE.failed;
  return RUN_STATE.queued;
}

/* -------------------------------------------------------------------------
   The five pipeline stages (docs/samagent/05-final-plan.md §11)
   ---------------------------------------------------------------------- */

export const STAGES = [
  { id: 'brief', label: 'Brief', hint: 'Describe what to build' },
  { id: 'plan', label: 'Plan', hint: 'Review scope, cost, autonomy' },
  { id: 'run', label: 'Run', hint: 'Watch the pipeline work' },
  { id: 'review', label: 'Review & Ship', hint: 'Verify against the brief' },
  { id: 'memory', label: 'Memory', hint: 'What the project remembers' },
];

/** Which stage the run is on, inferred from real state. */
export function deriveStage(state, runState) {
  const todos = (state && state.todos && state.todos.items) || [];
  const gate = state && state.pre_prod_gate;
  const hasBrief = !!(state && state.spec && state.spec.goal);
  const hasRun = todos.length > 0;
  const anyRunning = todos.some((t) => String(t.status || '').toLowerCase() === 'running');

  if (hasRun && anyRunning) return 'run';
  if (hasRun && runState && runState.id === 'ready' && gate) return 'review';
  if (hasRun) return 'run';
  if (hasBrief) return 'plan';
  return 'brief';
}

/* -------------------------------------------------------------------------
   Verification ladder — the project's actual differentiator (plan §7)
   ---------------------------------------------------------------------- */

const GATE_COPY = {
  local_dev_app_ready: { level: 'L1', label: 'Dev app boots', detail: 'The generated app starts and serves.' },
  vscode_workspace_configured: { level: 'L1', label: 'Workspace configured', detail: '.code-workspace resolves for the IDE.' },
  l0_l1_syntax_contract: { level: 'L1', label: 'Syntax + contract', detail: 'Modules parse and match the frozen contract.' },
  l2_ownership_and_tdd_red_green: { level: 'L2', label: 'Ownership + red-green', detail: 'Writes stayed in owned globs; acceptance went red first.' },
  l3_security_owasp_idor_rbac: { level: 'L3', label: 'Security: IDOR + RBAC', detail: 'Authz probes from the role matrix, run against the live app.' },
  l4_live_browser_dom_smoke: { level: 'L4', label: 'Live browser walkthrough', detail: 'A browser agent walked each user story.' },
  no_secret_leaks: { level: 'L3', label: 'No secret leaks', detail: 'No credentials in the diff or the tree.' },
};

export function buildVerificationLadder(state) {
  const gate = (state && state.pre_prod_gate) || { checks: {}, blockers: [] };
  const checks = gate.checks || {};
  const order = Object.keys(GATE_COPY).filter((k) => k in checks);
  const extras = Object.keys(checks).filter((k) => !(k in GATE_COPY));

  const rows = order.map((key) => {
    const meta = GATE_COPY[key];
    return {
      key,
      level: meta.level,
      label: meta.label,
      detail: meta.detail,
      pass: checks[key] === true,
    };
  });

  for (const key of extras) {
    rows.push({
      key,
      level: '—',
      label: key.replace(/_/g, ' '),
      detail: 'Reported by the verification runner.',
      pass: checks[key] === true,
    });
  }

  const passed = rows.filter((r) => r.pass).length;
  return {
    rows,
    passed,
    total: rows.length,
    ready: gate.ready_for_production === true,
    blockers: gate.blockers || [],
    uncommitted: gate.uncommitted_changes_count || 0,
  };
}

/* -------------------------------------------------------------------------
   Run metrics — elapsed + cost. Codex surfaces neither per-run (see the
   research: no elapsed timer in the run view, no per-turn cost), which is
   exactly the gap a Mission Control surface should close.
   ---------------------------------------------------------------------- */

export function buildRunMetrics(state, runState) {
  const todos = (state && state.todos && state.todos.items) || [];
  const plan = (state && state.plan_card) || {};
  const done = todos.filter((t) => TERMINAL_TODO.has(String(t.status || '').toLowerCase())).length;

  const costRange = plan.estimated_cost_usd_range || null;
  const timeRange = plan.estimated_minutes_range || null;
  const budget = (state && state.spec && state.spec.budget) || {};

  return {
    total: todos.length || (plan.modules ? plan.modules.length : 0),
    done,
    pct: todos.length ? Math.round((done / todos.length) * 100) : 0,
    costLow: costRange ? costRange[0] : null,
    costHigh: costRange ? costRange[1] : null,
    minutesLow: timeRange ? timeRange[0] : null,
    minutesHigh: timeRange ? timeRange[1] : null,
    budgetUsd: typeof budget.max_usd === 'number' ? budget.max_usd : null,
    localShare: typeof plan.local_task_share_pct === 'number' ? plan.local_task_share_pct : null,
    running: runState ? runState.id === 'running' : false,
  };
}

/** Group todos into pipeline phases for the timeline. */
export function buildTimeline(state) {
  const todos = (state && state.todos && state.todos.items) || [];
  const phases = new Map();
  for (const t of todos) {
    const key = t.phase || 'run';
    if (!phases.has(key)) phases.set(key, []);
    phases.get(key).push(t);
  }
  return Array.from(phases.entries()).map(([phase, items]) => {
    const done = items.filter((t) => TERMINAL_TODO.has(String(t.status || '').toLowerCase())).length;
    const running = items.some((t) => String(t.status || '').toLowerCase() === 'running');
    const failed = items.some((t) => FAILED_TODO.has(String(t.status || '').toLowerCase()));
    return {
      phase,
      items,
      total: items.length,
      done,
      pct: items.length ? Math.round((done / items.length) * 100) : 0,
      tone: running ? 'info' : failed ? 'err' : done === items.length ? 'ok' : 'idle',
    };
  });
}

/* -------------------------------------------------------------------------
   Formatting helpers
   ---------------------------------------------------------------------- */

export function fmtUsd(n) {
  if (n === null || n === undefined) return '—';
  if (n < 0.01) return '<$0.01';
  return `$${n.toFixed(2)}`;
}

export function fmtRange(low, high, fmt) {
  if (low === null || low === undefined) return '—';
  if (high === null || high === undefined || low === high) return fmt(low);
  return `${fmt(low)}–${fmt(high)}`;
}

export function fmtMinutes(n) {
  if (n === null || n === undefined) return '—';
  if (n < 60) return `${Math.round(n)}m`;
  const h = Math.floor(n / 60);
  const m = Math.round(n % 60);
  return m ? `${h}h ${m}m` : `${h}h`;
}

export function fmtAgo(seconds) {
  if (!seconds && seconds !== 0) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

export function fmtClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

export function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Count added/removed lines in a unified diff payload. */
export function diffStats(diffText) {
  let add = 0;
  let del = 0;
  for (const line of String(diffText || '').split('\n')) {
    if (line.startsWith('+') && !line.startsWith('+++')) add += 1;
    else if (line.startsWith('-') && !line.startsWith('---')) del += 1;
  }
  return { add, del };
}
