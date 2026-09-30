/**
 * Stage 3 — Run.
 *
 * The run view. Three things the Codex research says its run view lacks, and
 * that a pipeline surface should not:
 *
 *   - a step timeline (Codex is artifact-centric, never step-centric)
 *   - elapsed time (Codex has no elapsed timer in the run view at all)
 *   - a live cost meter (Codex has per-turn cost nowhere in the transcript)
 *
 * The ticker is Hermes's pattern: one rolling line rather than a growing wall
 * of tool calls, so a long run stays one line tall.
 */
import { createElement as h, useState, useEffect, useCallback, useMemo } from '../sdk.js';
import { Button, Card, Ticker, SegmentedMeter, SplitBar, StatusPill, Dot, KV, Disclosure, Empty, cx, useElapsed } from '../ui.js';
import { buildTimeline, fmtUsd, fmtMinutes, fmtRange, fmtClock, titleCase } from '../data.js';

export function RunStage({ state, actions, runState, metrics, mode, busy }) {
  const [note, setNote] = useState('');
  const todos = (state && state.todos && state.todos.items) || [];
  const phases = useMemo(() => buildTimeline(state), [state]);
  const tickerItems = useMemo(
    () => todos.map((t) => `${titleCase(t.phase || 'run')} — ${t.title}`),
    [todos]
  );
  const startTs = state && state.todos && state.todos.updated_at ? state.todos.updated_at / 1000 : null;
  const elapsed = useElapsed(runState.id === 'running', startTs);

  const onSteer = useCallback(() => {
    const text = note.trim();
    if (!text) return;
    actions.steer(text);
    setNote('');
  }, [note, actions]);

  if (!todos.length) {
    return h(
      'div',
      { className: 'mc-stage' },
      h(Empty, {
        title: 'No run in flight',
        hint: 'Approve a plan and the pipeline will show its steps here as they execute.',
        action: h(Button, { variant: 'primary', onClick: () => actions.goStage('plan') }, 'Go to the plan'),
      })
    );
  }

  return h(
    'div',
    { className: 'mc-stage' },

    /* --- live status header ------------------------------------------- */
    h(
      'div',
      { className: 'mc-run-head' },
      h(
        'div',
        { className: 'mc-run-headline' },
        h(StatusPill, { tone: runState.tone }, h(Dot, { tone: runState.tone, pulse: runState.id === 'running' }), runState.label),
        h('span', { className: 'mc-mono mc-muted' }, `${metrics.done}/${metrics.total}`)
      ),
      h(
        'div',
        { className: 'mc-run-metrics' },
        elapsed > 0 ? h('span', { className: 'mc-metric' }, h('span', { className: 'mc-metric-label' }, 'elapsed'), h('span', { className: 'mc-mono' }, fmtClock(elapsed))) : null,
        metrics.costHigh !== null
          ? h(
              'span',
              { className: 'mc-metric' },
              h('span', { className: 'mc-metric-label' }, 'budget'),
              h('span', { className: 'mc-mono' }, `${fmtUsd(metrics.costLow)}–${fmtUsd(metrics.costHigh)}`)
            )
          : null,
        metrics.minutesHigh ? h('span', { className: 'mc-metric' }, h('span', { className: 'mc-metric-label' }, 'est'), h('span', { className: 'mc-mono' }, fmtMinutes(metrics.minutesHigh))) : null
      )
    ),

    h(SegmentedMeter, { total: metrics.total, done: metrics.done, tone: runState.tone === 'err' ? 'err' : 'ok' }),
    metrics.localShare !== null ? h(SplitBar, { localPct: metrics.localShare }) : null,

    tickerItems.length ? h(Ticker, { items: tickerItems }) : null,

    /* --- phase timeline ------------------------------------------------ */
    h(
      'div',
      { className: 'mc-timeline' },
      phases.map((p) =>
        h(
          'section',
          { key: p.phase, className: cx('mc-tl-phase', `is-${p.tone}`) },
          h(
            'header',
            { className: 'mc-tl-phase-head' },
            h('span', { className: 'mc-tl-phase-name' }, titleCase(p.phase)),
            h('span', { className: 'mc-mono mc-faint mc-tiny' }, `${p.done}/${p.total}`)
          ),
          h(
            'ol',
            { className: 'mc-tl-items' },
            p.items.map((t) => {
              const st = String(t.status || '').toLowerCase();
              const tone = st === 'done' || st === 'passed' ? 'ok' : st === 'running' ? 'info' : st === 'failed' || st === 'blocked' ? 'err' : 'idle';
              return h(
                'li',
                { key: t.id, className: cx('mc-tl-item', `is-${tone}`) },
                h('span', { className: 'mc-tl-rail' }, h(Dot, { tone, pulse: st === 'running' })),
                h(
                  'div',
                  { className: 'mc-tl-body' },
                  h('p', { className: 'mc-tl-title' }, t.title),
                  t.detail ? h('p', { className: 'mc-tl-detail mc-muted' }, t.detail) : null,
                  t.agent ? h('span', { className: 'mc-chip mc-chip-quiet' }, t.agent) : null
                )
              );
            })
          )
        )
      )
    ),

    /* --- steer ---------------------------------------------------------- */
    h(
      Card,
      { title: 'Redirect the run' },
      h(
        'div',
        { className: 'mc-steer' },
        h('input', {
          className: 'mc-input mc-steer-input',
          placeholder: 'Type a note — the next worker will see it',
          value: note,
          disabled: runState.id !== 'running',
          onChange: (e) => setNote(e.target.value),
          onKeyDown: (e) => {
            if (e.key === 'Enter') onSteer();
          },
        }),
        h(Button, { variant: 'secondary', onClick: onSteer, disabled: !note.trim() || runState.id !== 'running' || busy }, 'Steer')
      ),
      h(
        'p',
        { className: 'mc-faint mc-tiny' },
        runState.id === 'running'
          ? 'Notes are appended to the task brief for the next step.'
          : 'Steering is available while a run is in flight.'
      )
    ),

    /* --- pro-only: raw per-step detail --------------------------------- */
    mode === 'pro'
      ? h(
          Disclosure,
          { label: 'Step detail (raw)', detail: `${todos.length} entries` },
          h(
            'div',
            { className: 'mc-pro-table' },
            todos.map((t) =>
              h(
                'div',
                { key: t.id, className: 'mc-pro-row' },
                h('span', { className: 'mc-mono mc-faint' }, t.id),
                h('span', { className: 'mc-truncate' }, t.title),
                h('span', { className: 'mc-mono mc-faint' }, t.status),
                h('span', { className: 'mc-mono mc-faint' }, t.agent || '—')
              )
            )
          )
        )
      : null
  );
}
