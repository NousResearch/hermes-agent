/**
 * Stage 5 — Memory.
 *
 * The ledger is the reason a run survives context loss, so it gets a real
 * screen: what is remembered, why, whether it is still true, and whether it may
 * leave the machine. The `private` sensitivity tag is load-bearing — facts
 * tagged private are excluded from any cloud-routed prompt — so it is a
 * first-class control here, not metadata.
 *
 * Supersession is shown rather than hidden: a changed decision closes the old
 * fact's validity window instead of deleting it, and the history of what was
 * believed and when is part of the record.
 */
import { createElement as h, useState, useCallback } from '../sdk.js';
import { Button, Card, Empty, StatusPill, Dot, cx, Disclosure } from '../ui.js';
import { fmtAgo, titleCase } from '../data.js';

export function MemoryStage({ state, actions, mode }) {
  const ledger = (state && state.ledger) || {};
  const facts = ledger.active_facts || [];
  const superseded = ledger.superseded_facts || [];
  const attempts = ledger.attempts || [];
  const [draft, setDraft] = useState('');
  const [scope, setScope] = useState('architecture');
  const [sensitivity, setSensitivity] = useState('public');

  const onAdd = useCallback(() => {
    const text = draft.trim();
    if (text.length < 2) return;
    actions.addFact({ scope, kind: 'decision', text, sensitivity });
    setDraft('');
  }, [draft, scope, sensitivity, actions]);

  if (!facts.length && !superseded.length) {
    return h(
      'div',
      { className: 'mc-stage' },
      h(Empty, {
        title: 'Nothing remembered yet',
        hint: 'Decisions, contract versions and verified outcomes are recorded here as the run happens — so a cold resume picks up where this one left off.',
      })
    );
  }

  const byScope = groupBy(facts, (f) => f.scope || 'general');

  return h(
    'div',
    { className: 'mc-stage' },

    h(
      Card,
      { title: 'Remember something' },
      h(
        'div',
        { className: 'mc-mem-form' },
        h('input', {
          className: 'mc-input',
          placeholder: 'e.g. Auth is session-cookie based, not JWT — the mobile client cannot hold a secret',
          value: draft,
          onChange: (e) => setDraft(e.target.value),
          onKeyDown: (e) => e.key === 'Enter' && onAdd(),
        }),
        h(
          'div',
          { className: 'mc-mem-form-row' },
          h(
            'select',
            { className: 'mc-select', value: scope, onChange: (e) => setScope(e.target.value), 'aria-label': 'Scope' },
            ['architecture', 'preference', 'gotcha', 'environment'].map((s) => h('option', { key: s, value: s }, titleCase(s)))
          ),
          h(
            'label',
            { className: 'mc-switch mc-switch-inline', title: 'Private facts are never sent to a cloud model' },
            h('input', { type: 'checkbox', checked: sensitivity === 'private', onChange: (e) => setSensitivity(e.target.checked ? 'private' : 'public') }),
            h('span', null, 'Private')
          ),
          h(Button, { variant: 'secondary', size: 'sm', onClick: onAdd, disabled: draft.trim().length < 2 }, 'Add')
        )
      )
    ),

    Array.from(byScope.entries()).map(([scopeName, list]) =>
      h(
        Card,
        { key: scopeName, title: titleCase(scopeName), actions: h('span', { className: 'mc-faint mc-tiny' }, `${list.length}`) },
        h(
          'ul',
          { className: 'mc-facts' },
          list.map((f) =>
            h(
              'li',
              { key: f.id, className: cx('mc-fact', f.sensitivity === 'private' && 'is-private') },
              h(
                'div',
                { className: 'mc-fact-head' },
                h('span', { className: 'mc-fact-kind' }, titleCase(f.kind || 'fact')),
                f.sensitivity === 'private' ? h('span', { className: 'mc-chip mc-chip-warn' }, 'private') : null,
                f.source_ref ? h('span', { className: 'mc-fact-source mc-faint mc-mono mc-tiny' }, f.source_ref) : null,
                h('span', { className: 'mc-fact-when mc-faint mc-tiny' }, fmtAgo(nowMinus(f.valid_from)))
              ),
              h('p', { className: 'mc-fact-text' }, f.text)
            )
          )
        )
      )
    ),

    superseded.length
      ? h(
          Disclosure,
          { label: 'Superseded', detail: `${superseded.length} closed`, defaultOpen: false },
          h(
            'ul',
            { className: 'mc-facts' },
            superseded.map((f) =>
              h(
                'li',
                { key: f.id, className: 'mc-fact is-closed' },
                h(
                  'div',
                  { className: 'mc-fact-head' },
                  h('span', { className: 'mc-fact-kind' }, titleCase(f.kind || 'fact')),
                  f.superseded_by ? h('span', { className: 'mc-faint mc-tiny mc-mono' }, `→ ${f.superseded_by}`) : null
                ),
                h('p', { className: 'mc-fact-text mc-muted' }, f.text)
              )
            )
          )
        )
      : null,

    attempts.length
      ? h(
          Disclosure,
          { label: 'What was tried and failed', detail: `${attempts.length} attempts`, defaultOpen: false },
          h(
            'ul',
            { className: 'mc-facts' },
            attempts.map((a, i) =>
              h(
                'li',
                { key: a.id || i, className: 'mc-fact' },
                h(
                  'div',
                  { className: 'mc-fact-head' },
                  a.outcome ? h(StatusPill, { tone: a.outcome === 'success' ? 'ok' : 'warn' }, h(Dot, { tone: a.outcome === 'success' ? 'ok' : 'warn' }), a.outcome) : null,
                  a.error_signature ? h('span', { className: 'mc-fact-source mc-faint mc-mono mc-tiny' }, a.error_signature) : null
                ),
                h('p', { className: 'mc-fact-text' }, a.approach || a.text || JSON.stringify(a))
              )
            )
          )
        )
      : null,

    mode === 'pro' && ledger.mirror_markdown
      ? h(
          Disclosure,
          { label: 'Ledger mirror (markdown)', detail: 'committed to git' },
          h('pre', { className: 'mc-pre' }, ledger.mirror_markdown)
        )
      : null
  );
}

function groupBy(list, key) {
  const m = new Map();
  for (const item of list) {
    const k = key(item);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(item);
  }
  return m;
}

function nowMinus(ts) {
  if (!ts) return null;
  return Date.now() / 1000 - ts;
}
