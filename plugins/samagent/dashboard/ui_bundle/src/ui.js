/**
 * Presentational primitives.
 *
 * Every colour here comes from a semantic token in tokens.css — never a
 * component-named variable and never a literal. That is the rule the Hermes
 * desktop plugin guide states, and it is what makes dark mode a pure override
 * rather than a second stylesheet.
 */
import { createElement as h, useState, useEffect, useRef, useCallback } from './sdk.js';
import { RUN_STATE, fmtClock, titleCase } from './data.js';

const cx = (...parts) => parts.filter(Boolean).join(' ');

/* ---------------------------------------------------------------------- */
/* Status                                                                  */
/* ---------------------------------------------------------------------- */

export function Dot({ tone = 'idle', pulse = false, title }) {
  return h('span', { className: cx('mc-dot', `mc-tone-${tone}`, pulse && 'mc-dot-pulse'), title });
}

export function StatusPill({ tone = 'idle', children, title }) {
  return h('span', { className: cx('mc-pill', `mc-tone-${tone}`), title }, children);
}

export function RunStateBadge({ runState, elapsed = null }) {
  if (!runState) return null;
  return h(
    StatusPill,
    { tone: runState.tone, title: runState.label },
    h(Dot, { tone: runState.tone, pulse: runState.id === 'running' }),
    runState.label,
    elapsed !== null && runState.runningish !== false ? h('span', { className: 'mc-pill-sep' }, '·') : null,
    elapsed !== null ? h('span', { className: 'mc-mono' }, fmtClock(elapsed)) : null
  );
}

/* ---------------------------------------------------------------------- */
/* Buttons                                                                 */
/* ---------------------------------------------------------------------- */

export function Button({ variant = 'default', size = 'md', icon = null, children, ...rest }) {
  return h(
    'button',
    { type: 'button', className: cx('mc-btn', `mc-btn-${variant}`, `mc-btn-${size}`), ...rest },
    icon ? h('span', { className: 'mc-btn-icon' }, icon) : null,
    children ? h('span', { className: 'mc-btn-label' }, children) : null
  );
}

export function IconButton({ label, active = false, badge = null, children, ...rest }) {
  return h(
    'button',
    {
      type: 'button',
      className: cx('mc-iconbtn', active && 'is-active'),
      title: label,
      'aria-label': label,
      'aria-pressed': active ? 'true' : undefined,
      ...rest,
    },
    children,
    badge ? h('span', { className: 'mc-iconbtn-badge' }, badge) : null
  );
}

/* ---------------------------------------------------------------------- */
/* Surfaces                                                                */
/* ---------------------------------------------------------------------- */

export function Card({ title = null, actions = null, children, className = '', pad = true }) {
  return h(
    'section',
    { className: cx('mc-card', className) },
    title || actions
      ? h(
          'header',
          { className: 'mc-card-head' },
          title ? h('h3', { className: 'mc-card-title' }, title) : h('span'),
          actions ? h('div', { className: 'mc-card-actions' }, actions) : null
        )
      : null,
    h('div', { className: pad ? 'mc-card-body' : 'mc-card-body mc-flush' }, children)
  );
}

export function Empty({ title, hint = null, action = null }) {
  return h(
    'div',
    { className: 'mc-empty' },
    h('p', { className: 'mc-empty-title' }, title),
    hint ? h('p', { className: 'mc-empty-hint' }, hint) : null,
    action
  );
}

export function Spinner({ size = 14 }) {
  return h('span', { className: 'mc-spin', style: { width: size, height: size } });
}

/* ---------------------------------------------------------------------- */
/* Overlays: popover + modal                                               */
/* ---------------------------------------------------------------------- */

/**
 * Popover anchored to its own trigger. Closes on Escape and on an outside
 * pointerdown — the two dismissal gestures a developer expects from a menu.
 */
export function Popover({ trigger, children, align = 'start', width = null, label = 'Open menu' }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown, true);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown, true);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return h(
    'div',
    { className: 'mc-popover-anchor', ref },
    h(
      'button',
      {
        type: 'button',
        className: cx('mc-popover-trigger', open && 'is-open'),
        'aria-haspopup': 'dialog',
        'aria-expanded': open ? 'true' : 'false',
        'aria-label': label,
        onClick: () => setOpen((o) => !o),
      },
      trigger
    ),
    open
      ? h(
          'div',
          {
            className: cx('mc-popover', `mc-popover-${align}`),
            role: 'dialog',
            style: width ? { width } : undefined,
          },
          children
        )
      : null
  );
}

export function MenuItem({ icon = null, label, hint = null, selected = false, danger = false, onClick }) {
  return h(
    'button',
    {
      type: 'button',
      className: cx('mc-menu-item', selected && 'is-selected', danger && 'is-danger'),
      role: 'menuitem',
      onClick,
    },
    icon ? h('span', { className: 'mc-menu-icon' }, icon) : null,
    h('span', { className: 'mc-menu-label mc-truncate' }, label),
    hint ? h('span', { className: 'mc-menu-hint' }, hint) : null
  );
}

export function MenuDivider() {
  return h('div', { className: 'mc-menu-divider', role: 'separator' });
}

export function Modal({ open, onClose, title, subtitle = null, footer = null, children, width = 560 }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return h(
    'div',
    { className: 'mc-modal-backdrop', onMouseDown: (e) => e.target === e.currentTarget && onClose() },
    h(
      'div',
      { className: 'mc-modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': title, style: { maxWidth: width } },
      h(
        'header',
        { className: 'mc-modal-head' },
        h('div', null, h('h2', { className: 'mc-modal-title' }, title), subtitle ? h('p', { className: 'mc-modal-sub' }, subtitle) : null)
      ),
      h('div', { className: 'mc-modal-body' }, children),
      footer ? h('footer', { className: 'mc-modal-foot' }, footer) : null
    )
  );
}

/* ---------------------------------------------------------------------- */
/* Disclosure — the overflow contract the Hermes ExpandableBlock uses:      */
/* collapse past ~7.5rem, expand to 40dvh, fade as a pure overflow cue.    */
/* ---------------------------------------------------------------------- */

export function Disclosure({ label, detail = null, defaultOpen = false, children, maxHeight = '40dvh' }) {
  const [open, setOpen] = useState(defaultOpen);
  const ref = useRef(null);
  const [overflowing, setOverflowing] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    // Measure inside the observer, never at mount: a synchronous scrollHeight
    // read on mount forces a reflow per instance.
    const ro = new ResizeObserver(() => {
      if (el) setOverflowing(el.scrollHeight > 121);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return h(
    'div',
    { className: 'mc-disclosure' },
    h(
      'button',
      {
        type: 'button',
        className: 'mc-disclosure-head',
        'aria-expanded': open ? 'true' : 'false',
        onClick: () => setOpen((o) => !o),
      },
      h('span', { className: 'mc-disclosure-caret', 'aria-hidden': 'true' }, open ? '▾' : '▸'),
      h('span', { className: 'mc-disclosure-label' }, label),
      detail ? h('span', { className: 'mc-disclosure-detail mc-truncate' }, detail) : null
    ),
    h(
      'div',
      {
        ref,
        className: cx('mc-disclosure-body', !open && overflowing && 'is-collapsed'),
        style: open ? { maxHeight } : overflowing ? { maxHeight: '7.5rem' } : undefined,
      },
      children
    ),
    overflowing && !open
      ? h('span', { className: 'mc-disclosure-fade', 'aria-hidden': 'true' })
      : null
  );
}

/* ---------------------------------------------------------------------- */
/* Ticker — one rolling line, not a wall.                                 */
/* From the Hermes tool ticker: each new row slides the previous up and out */
/* so growing activity reads as a single line ticking in place.            */
/* ---------------------------------------------------------------------- */

export function Ticker({ items, intervalMs = 1400 }) {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState('idle');

  useEffect(() => {
    if (!items.length) return undefined;
    const id = setInterval(() => {
      setPhase('out');
      setTimeout(() => {
        setIndex((i) => (i + 1) % items.length);
        setPhase('in');
      }, 180);
    }, intervalMs);
    return () => clearInterval(id);
  }, [items.length, intervalMs]);

  if (!items.length) return null;
  return h(
    'div',
    { className: 'mc-ticker', 'aria-live': 'polite' },
    h(
      'div',
      { className: cx('mc-ticker-row', `is-${phase}`) },
      h(Dot, { tone: 'info', pulse: true }),
      h('span', { className: 'mc-truncate' }, items[index % items.length])
    )
  );
}

/* ---------------------------------------------------------------------- */
/* Diff — Cursor-style: colour + a 2px gutter, no @@ / file-header noise.  */
/* ---------------------------------------------------------------------- */

export function DiffView({ text, maxLines = 400 }) {
  const lines = String(text || '').split('\n');
  const shown = lines.slice(0, maxLines);

  if (!String(text || '').trim()) {
    return h(Empty, { title: 'No diff', hint: 'The working tree matches the last commit.' });
  }

  return h(
    'div',
    { className: 'mc-diff' },
    shown.map((line, i) => {
      let tone = 'ctx';
      if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('diff ') || line.startsWith('index ')) tone = 'meta';
      else if (line.startsWith('@@')) tone = 'hunk';
      else if (line.startsWith('+')) tone = 'add';
      else if (line.startsWith('-')) tone = 'del';
      return h(
        'div',
        { key: i, className: cx('mc-diff-line', `is-${tone}`) },
        h('span', { className: 'mc-diff-gutter' }),
        h('span', { className: 'mc-diff-text' }, line.replace(/^(\+|-)/, ''))
      );
    }),
    lines.length > maxLines
      ? h('div', { className: 'mc-diff-more' }, `${lines.length - maxLines} more lines not shown`)
      : null
  );
}

/* ---------------------------------------------------------------------- */
/* Meters                                                                  */
/* ---------------------------------------------------------------------- */

/**
 * A segmented meter. Segments read as discrete work units, which is the honest
 * representation for a todo board — a smooth bar would imply a smoothness the
 * pipeline does not have.
 */
export function SegmentedMeter({ total, done, tone = 'info', max = 40 }) {
  const shown = Math.min(total, max);
  const filled = total <= max ? done : Math.round((done / total) * max);
  return h(
    'div',
    { className: 'mc-meter', role: 'progressbar', 'aria-valuenow': done, 'aria-valuemin': 0, 'aria-valuemax': total, 'aria-label': `${done} of ${total} complete` },
    Array.from({ length: shown }, (_, i) =>
      h('span', { key: i, className: cx('mc-meter-seg', i < filled && `is-${tone}`) })
    )
  );
}

/**
 * Local/cloud split. The router policy in plan §6 is a real product decision —
 * how much of a run stays on the user's own machine — so it gets its own bar
 * rather than being buried in a tooltip.
 */
export function SplitBar({ localPct }) {
  if (localPct === null || localPct === undefined) return null;
  const local = Math.max(0, Math.min(100, localPct));
  return h(
    'div',
    { className: 'mc-split', title: `${Math.round(local)}% of tasks routed to a local model` },
    h('div', { className: 'mc-split-track' }, h('div', { className: 'mc-split-local', style: { width: `${local}%` } })),
    h('span', { className: 'mc-split-legend' }, h('span', { className: 'mc-muted' }, 'local '), `${Math.round(local)}%`)
  );
}

/* ---------------------------------------------------------------------- */
/* Key/value rows — the spec, plan and ledger all render through this.     */
/* ---------------------------------------------------------------------- */

export function KV({ k, v, mono = false, tone = null }) {
  return h(
    'div',
    { className: 'mc-kv' },
    h('span', { className: 'mc-kv-k' }, k),
    h('span', { className: cx('mc-kv-v', mono && 'mc-mono', tone && `mc-tone-${tone}`) }, v)
  );
}

/* ---------------------------------------------------------------------- */
/* Tabs                                                                    */
/* ---------------------------------------------------------------------- */

export function TabBar({ tabs, active, onSelect, counts = {} }) {
  return h(
    'div',
    { className: 'mc-tabbar', role: 'tablist' },
    tabs.map((t) =>
      h(
        'button',
        {
          key: t.id,
          type: 'button',
          role: 'tab',
          'aria-selected': active === t.id ? 'true' : 'false',
          className: cx('mc-tab', active === t.id && 'is-active'),
          onClick: () => onSelect(t.id),
        },
        t.label,
        counts[t.id] !== undefined && counts[t.id] !== null
          ? h('span', { className: 'mc-tab-count mc-mono' }, String(counts[t.id]))
          : null
      )
    )
  );
}

/* ---------------------------------------------------------------------- */
/* Ticker-friendly run summary                                             */
/* ---------------------------------------------------------------------- */

export function useElapsed(running, startTs) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => setElapsed(Math.floor(Date.now() / 1000 - (startTs || Date.now() / 1000))), 1000);
    return () => clearInterval(id);
  }, [running, startTs]);
  return elapsed;
}

export { cx, titleCase, RUN_STATE };
