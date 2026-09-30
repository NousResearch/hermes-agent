/**
 * Application shell.
 *
 * Layout is a three-zone split with a persistent stage rail, which is the one
 * structural change from the previous Codex clone:
 *
 *   rail (52)  |  sidebar (260)  |  canvas (flex)  |  inspector (toggle)
 *
 * Two deliberate departures from the clone it replaces:
 *
 *   1. No fake window chrome. The old UI drew a macOS-style title bar with
 *      minimise/maximise buttons and File/Edit/View menus inside a browser tab
 *      that cannot actually do any of those things. A browser tab should spend
 *      its vertical space on the run, and a real menu bar is a lie when the
 *      window has no window controls to drive. Keyboard shortcuts do the work
 *      those menus implied.
 *
 *   2. The stage rail is a progress spine, not decoration. Codex's own research
 *      found its single biggest IA weakness is that progress is artifact-centric
 *      and never step-centric — an operator gets no "step 3 of 7" forward
 *      visibility. SamAgent is a pipeline, so the pipeline is the spine.
 */
import { createElement as h, useState, useEffect, useCallback, useMemo } from './sdk.js';
import {
  Button, IconButton, StatusPill, Dot, Card, Empty, Tabs as _T, cx, TabBar, SplitBar,
} from './ui.js';
import { STAGES, deriveStage, deriveRunState, buildRunMetrics, fmtUsd, fmtMinutes, fmtRange, titleCase } from './data.js';

import { BriefStage } from './stages/brief.js';
import { PlanStage } from './stages/plan.js';
import { RunStage } from './stages/run.js';
import { ReviewStage } from './stages/review.js';
import { MemoryStage } from './stages/memory.js';
import { Inspector } from './inspector.js';

const STAGE_COMPONENTS = {
  brief: BriefStage,
  plan: PlanStage,
  run: RunStage,
  review: ReviewStage,
  memory: MemoryStage,
};

export function MissionControl({ state, actions, theme, onToggleTheme, busy, notice }) {
  const runState = useMemo(() => deriveRunState(state), [state]);
  const inferredStage = useMemo(() => deriveStage(state, runState), [state, runState]);
  const metrics = useMemo(() => buildRunMetrics(state, runState), [state, runState]);

  // Stage is user-navigable, but a live run keeps pulling the view back to the
  // stage it is actually on — a background event may not hijack the viewport.
  const [stage, setStage] = useState(inferredStage);
  const [stagePinned, setStagePinned] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [inspected, setInspected] = useState('plan');
  const [mode, setMode] = useState(() => {
    try {
      return localStorage.getItem('samagent.mode') || 'simple';
    } catch (e) {
      return 'simple';
    }
  });

  useEffect(() => {
    if (!stagePinned) setStage(inferredStage);
  }, [inferredStage, stagePinned]);

  useEffect(() => {
    try {
      localStorage.setItem('samagent.mode', mode);
    } catch (e) {
      /* private mode — the toggle just does not persist */
    }
  }, [mode]);

  const goStage = useCallback((id) => {
    setStage(id);
    setStagePinned(true);
  }, []);

  const Stage = STAGE_COMPONENTS[stage] || BriefStage;

  /* Keyboard: the work the fake menu bar used to imply. */
  useEffect(() => {
    const onKey = (e) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      const n = Number(e.key);
      if (n >= 1 && n <= 5) {
        e.preventDefault();
        goStage(STAGES[n - 1].id);
        return;
      }
      if (e.key === '\\') {
        e.preventDefault();
        setInspectorOpen((o) => !o);
        return;
      }
      if (e.key === 'k') {
        e.preventDefault();
        goStage('brief');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [goStage]);

  return h(
    'div',
    { className: 'mc-app' },
    h(Rail, { stage, onStage: goStage, runState, metrics, theme, onToggleTheme, mode, onMode: setMode }),
    h(Sidebar, { state, runState, stage, onStage: goStage, metrics }),
    h(
      'main',
      { className: 'mc-canvas' },
      h(
        'header',
        { className: 'mc-canvas-head' },
        h(
          'div',
          { className: 'mc-canvas-title' },
          h('h1', { className: 'mc-h1' }, (STAGES.find((s) => s.id === stage) || STAGES[0]).label),
          h(
            'p',
            { className: 'mc-canvas-sub' },
            runState.label,
            metrics.total ? ` · ${metrics.done}/${metrics.total} steps` : '',
            metrics.minutesHigh ? ` · est ${fmtMinutes(metrics.minutesHigh)}` : ''
          )
        ),
        h(
          'div',
          { className: 'mc-canvas-actions' },
          metrics.total ? h(SegmentedMeterCompact, { metrics }) : null,
          h(
            IconButton,
            { label: inspectorOpen ? 'Hide inspector (⌘\\)' : 'Show inspector (⌘\\)', active: inspectorOpen, onClick: () => setInspectorOpen((o) => !o) },
            '◧'
          )
        )
      ),
      notice ? h('div', { className: 'mc-notice', role: 'status' }, notice) : null,
      h(
        'div',
        { className: 'mc-canvas-body' },
        h(Stage, { state, actions, runState, metrics, mode, busy, stage, onStage: goStage })
      )
    ),
    inspectorOpen
      ? h(Inspector, { state, tab: inspected, onTab: setInspected, runState, metrics, mode, actions })
      : null
  );
}

/* ---------------------------------------------------------------------- */
/* Rail — identity, pipeline spine, and the two global controls.           */
/* ---------------------------------------------------------------------- */

function Rail({ stage, onStage, runState, metrics, theme, onToggleTheme, mode, onMode }) {
  return h(
    'nav',
    { className: 'mc-rail', 'aria-label': 'Pipeline stages' },
    h('div', { className: 'mc-rail-logo', title: 'SamAgent' }, 'SA'),
    h(
      'ol',
      { className: 'mc-rail-stages' },
      STAGES.map((s, i) => {
        const done = isStageDone(s.id, stage, metrics);
        const active = s.id === stage;
        return h(
          'li',
          { key: s.id },
          h(
            'button',
            {
              type: 'button',
              className: cx('mc-rail-stage', active && 'is-active', done && 'is-done'),
              title: `${s.label} — ${s.hint} (⌘${i + 1})`,
              'aria-current': active ? 'step' : undefined,
              onClick: () => onStage(s.id),
            },
            h('span', { className: 'mc-rail-index mc-mono' }, String(i + 1)),
            h('span', { className: 'mc-rail-bar' })
          )
        );
      })
    ),
    h(
      'div',
      { className: 'mc-rail-foot' },
      h(
        'button',
        {
          type: 'button',
          className: cx('mc-rail-mode', mode === 'pro' && 'is-pro'),
          title: mode === 'simple' ? 'Simple mode — plain language' : 'Pro mode — tool calls, diffs, ledger',
          onClick: () => onMode(mode === 'simple' ? 'pro' : 'simple'),
        },
        mode === 'simple' ? 'S' : 'P'
      ),
      h(
        'button',
        {
          type: 'button',
          className: 'mc-rail-theme',
          title: `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`,
          'aria-label': 'Toggle colour theme',
          onClick: onToggleTheme,
        },
        theme === 'dark' ? '☀' : '☾'
      )
    )
  );
}

function isStageDone(id, current, metrics) {
  const order = STAGES.map((s) => s.id);
  return order.indexOf(id) < order.indexOf(current) || (id === 'run' && metrics.total > 0 && metrics.pct === 100);
}

/* ---------------------------------------------------------------------- */
/* Sidebar — workspace, run board, ledger summary.                        */
/* ---------------------------------------------------------------------- */

function Sidebar({ state, runState, stage, onStage, metrics }) {
  const workspaces = (state && state.available_workspaces) || [];
  const todos = (state && state.todos && state.todos.items) || [];
  const active = workspaces.find((w) => w.active) || workspaces[0];
  const facts = ((state && state.ledger && state.ledger.active_facts) || []).length;
  const attention = todos.filter((t) => String(t.status || '').toLowerCase() === 'running' || String(t.status || '').toLowerCase() === 'failed').length;

  return h(
    'aside',
    { className: 'mc-sidebar' },
    h(
      'div',
      { className: 'mc-sidebar-head' },
      h('span', { className: 'mc-eyebrow' }, 'Workspace'),
      h('p', { className: 'mc-sidebar-project mc-truncate', title: (active && active.path) || '' }, (active && active.name) || 'no workspace')
    ),
    h(
      'div',
      { className: 'mc-sidebar-section' },
      h('span', { className: 'mc-eyebrow' }, 'Run'),
      h(StatusPill, { tone: runState.tone }, h(Dot, { tone: runState.tone, pulse: runState.id === 'running' }), runState.label),
      metrics.total
        ? h(
            'p',
            { className: 'mc-sidebar-stat mc-muted' },
            `${metrics.done}/${metrics.total} steps`,
            metrics.localShare !== null ? ` · ${Math.round(metrics.localShare)}% local` : ''
          )
        : h('p', { className: 'mc-sidebar-stat mc-muted' }, 'No run yet')
    ),
    attention
      ? h(
          'div',
          { className: 'mc-sidebar-callout' },
          h(Dot, { tone: 'warn' }),
          h('span', null, `${attention} step${attention === 1 ? '' : 's'} need attention`)
        )
      : null,
    h(
      'div',
      { className: 'mc-sidebar-section' },
      h('span', { className: 'mc-eyebrow' }, 'Project record'),
      h(
        'button',
        { type: 'button', className: cx('mc-sidebar-link', stage === 'memory' && 'is-active'), onClick: () => onStage('memory') },
        'Ledger',
        facts ? h('span', { className: 'mc-count' }, facts) : null
      ),
      h(
        'button',
        { type: 'button', className: cx('mc-sidebar-link', stage === 'review' && 'is-active'), onClick: () => onStage('review') },
        'Verification',
        state && state.pre_prod_gate && state.pre_prod_gate.ready_for_production
          ? h('span', { className: 'mc-tone-ok' }, 'ready')
          : h('span', { className: 'mc-count' }, 'gates')
      )
    ),
    h(
      'div',
      { className: 'mc-sidebar-section mc-grow' },
      h('span', { className: 'mc-eyebrow' }, 'Milestones'),
      h(
        'ol',
        { className: 'mc-mini-timeline' },
        todos.slice(0, 8).map((t) =>
          h(
            'li',
            { key: t.id, className: cx('mc-mini-item', `is-${toneForTodo(t.status)}`), title: t.detail || t.title },
            h(Dot, { tone: toneForTodo(t.status), pulse: String(t.status).toLowerCase() === 'running' }),
            h('span', { className: 'mc-truncate' }, t.title)
          )
        )
      ),
      todos.length > 8 ? h('p', { className: 'mc-faint mc-tiny' }, `+${todos.length - 8} more`) : null
    )
  );
}

function toneForTodo(status) {
  const s = String(status || '').toLowerCase();
  if (s === 'done' || s === 'complete' || s === 'completed' || s === 'passed') return 'ok';
  if (s === 'running') return 'info';
  if (s === 'failed' || s === 'error' || s === 'blocked') return 'err';
  return 'idle';
}

function SegmentedMeterCompact({ metrics }) {
  const filled = metrics.total ? Math.round((metrics.pct / 100) * 24) : 0;
  return h(
    'div',
    { className: 'mc-meter-compact', title: `${metrics.done} of ${metrics.total} steps complete` },
    Array.from({ length: 24 }, (_, i) => h('span', { key: i, className: cx('mc-meter-seg', i < filled && 'is-ok') }))
  );
}
