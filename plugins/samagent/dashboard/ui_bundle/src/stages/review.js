/**
 * Stage 4 — Review & ship.
 *
 * The question this screen answers is the one the whole pipeline exists to
 * answer: "does it do what you asked?" Not "did the writer say it worked."
 * The L0-L4 ladder is the evidence, and the independent judge's verdict is
 * shown as a verdict rather than a summary so it cannot be mistaken for the
 * writer's own account.
 */
import { createElement as h, useState, useCallback, useMemo } from '../sdk.js';
import { Button, Card, StatusPill, Dot, Empty, Disclosure, KV, cx } from '../ui.js';
import { buildVerificationLadder, fmtUsd, titleCase } from '../data.js';

export function ReviewStage({ state, actions, mode, busy }) {
  const ladder = useMemo(() => buildVerificationLadder(state), [state]);
  const deliverable = state && state.deliverable;
  const spec = (state && state.spec) || {};
  const stories = spec.stories || [];
  const [confirmPromote, setConfirmPromote] = useState(false);

  const onReverify = useCallback(() => actions.reverify(), [actions]);
  const onPromote = useCallback(() => {
    actions.promote();
    setConfirmPromote(false);
  }, [actions]);

  return h(
    'div',
    { className: 'mc-stage' },

    /* --- verdict ------------------------------------------------------- */
    h(
      'div',
      { className: cx('mc-verdict', ladder.ready ? 'is-ok' : 'is-warn') },
      h(
        'div',
        { className: 'mc-verdict-main' },
        h(
          'span',
          { className: 'mc-verdict-label' },
          ladder.ready ? 'Ready to ship' : 'Not ready to ship'
        ),
        h(
          'p',
          { className: 'mc-verdict-sub' },
          ladder.ready
            ? `All ${ladder.total} gates passed. ${ladder.uncommitted ? `${ladder.uncommitted} uncommitted change${ladder.uncommitted === 1 ? '' : 's'}.` : 'The working tree is clean.'}`
            : `${ladder.passed} of ${ladder.total} gates passed.`
        )
      ),
      h(StatusPill, { tone: ladder.ready ? 'ok' : 'warn' }, h(Dot, { tone: ladder.ready ? 'ok' : 'warn' }), `${ladder.passed}/${ladder.total}`)
    ),

    /* --- the L0-L4 ladder ---------------------------------------------- */
    h(
      Card,
      {
        title: 'Verification ladder',
        actions: h(Button, { variant: 'ghost', size: 'sm', onClick: onReverify, disabled: busy }, busy ? 'Re-verifying…' : 'Re-verify'),
      },
      h(
        'ol',
        { className: 'mc-ladder' },
        ladder.rows.map((r) =>
          h(
            'li',
            { key: r.key, className: cx('mc-ladder-row', r.pass ? 'is-pass' : 'is-fail') },
            h('span', { className: 'mc-ladder-level mc-mono' }, r.level),
            h(Dot, { tone: r.pass ? 'ok' : 'err' }),
            h(
              'div',
              { className: 'mc-ladder-body' },
              h('p', { className: 'mc-ladder-label' }, r.label),
              h('p', { className: 'mc-ladder-detail mc-muted' }, r.detail)
            ),
            h('span', { className: cx('mc-ladder-state', r.pass ? 'mc-tone-ok' : 'mc-tone-err') }, r.pass ? 'pass' : 'fail')
          )
        )
      ),
      ladder.blockers && ladder.blockers.length
        ? h(
            'div',
            { className: 'mc-blockers' },
            h('span', { className: 'mc-eyebrow' }, 'Blocking'),
            h('ul', { className: 'mc-list mc-list-err' }, ladder.blockers.map((b, i) => h('li', { key: i }, String(b))))
          )
        : null
    ),

    /* --- acceptance, story by story ------------------------------------ */
    stories.length
      ? h(
          Card,
          { title: 'Does it do what you asked?' },
          h(
            'ol',
            { className: 'mc-stories' },
            stories.map((s) =>
              h(
                'li',
                { key: s.id, className: 'mc-story' },
                h(
                  'div',
                  { className: 'mc-story-head' },
                  h('span', { className: 'mc-story-id mc-mono' }, s.id),
                  h('span', { className: 'mc-story-as mc-muted' }, s.as)
                ),
                h('p', { className: 'mc-story-can' }, s.can),
                s.accept ? h('p', { className: 'mc-story-accept' }, h('span', { className: 'mc-eyebrow' }, 'accepts when'), ' ', s.accept) : null,
                h(
                  'div',
                  { className: 'mc-story-meta' },
                  s.route ? h('code', { className: 'mc-glob mc-mono' }, `${s.method || 'GET'} ${s.route}`) : null,
                  h(
                    'span',
                    { className: cx('mc-chip', s.auth_required ? 'mc-chip-warn' : 'mc-chip-quiet') },
                    s.auth_required ? 'auth required' : 'public'
                  )
                )
              )
            )
          )
        )
      : null,

    /* --- the deliverable ------------------------------------------------ */}
    deliverable
      ? h(
          Card,
          { title: 'What was delivered' },
          h(
            'div',
            { className: 'mc-kv-grid' },
            h(KV, { k: 'Run', v: deliverable.run_id || '—', mono: true }),
            h(KV, { k: 'Verification', v: deliverable.verified === false ? 'failed' : 'passed', tone: deliverable.verified === false ? 'err' : 'ok' }),
            deliverable.judge_model ? h(KV, { k: 'Judge', v: deliverable.judge_model, mono: true }) : null,
            deliverable.cost_usd !== undefined ? h(KV, { k: 'Cost', v: fmtUsd(deliverable.cost_usd) }) : null
          ),
          deliverable.summary ? h('p', { className: 'mc-lede' }, deliverable.summary) : null
        )
      : null,

    /* --- what I assumed, restated at the point of trust ---------------- */
    spec.assumptions && spec.assumptions.length
      ? h(
          Card,
          { title: 'What I assumed along the way' },
          h(
            'ul',
            { className: 'mc-list' },
            spec.assumptions.map((a, i) => h('li', { key: a.id || i }, a.text || a))
          )
        )
      : null,

    /* --- ship ----------------------------------------------------------- */
    h(
      Card,
      { title: 'Ship it' },
      h(
        'div',
        { className: 'mc-ship-row' },
        h(Button, { variant: 'secondary', onClick: () => actions.createPr(), disabled: busy }, 'Open a pull request'),
        h(Button, { variant: 'secondary', onClick: () => actions.syncGithub(), disabled: busy }, 'Commit & push'),
        h(
          Button,
          { variant: 'ghost', onClick: () => actions.openInVscode(null), disabled: busy },
          'Open in VS Code'
        ),
        confirmPromote
          ? h(
              'span',
              { className: 'mc-confirm' },
              h('span', { className: 'mc-faint mc-tiny' }, 'Deploy to production?'),
              h(Button, { variant: 'danger', size: 'sm', onClick: onPromote, disabled: busy }, 'Yes, promote'),
              h(Button, { variant: 'ghost', size: 'sm', onClick: () => setConfirmPromote(false) }, 'Cancel')
            )
          : h(
              Button,
              { variant: 'primary', onClick: () => setConfirmPromote(true), disabled: busy || !ladder.ready },
              'Promote to production'
            )
      ),
      !ladder.ready
        ? h('p', { className: 'mc-faint mc-tiny' }, 'Promotion stays locked until every gate passes.')
        : null,
      ladder.uncommitted
        ? h('p', { className: 'mc-warn-note' }, `${ladder.uncommitted} uncommitted change${ladder.uncommitted === 1 ? '' : 's'} in the working tree.`)
        : null
    ),

    /* --- pro-only: the honest caveat about diff scope -------------------- */
    mode === 'pro'
      ? h(
          Disclosure,
          { label: 'Raw run evidence', detail: 'runs, verification, judge' },
          h(
            'pre',
            { className: 'mc-pre' },
            JSON.stringify(deliverable || { note: 'no deliverable.json written yet' }, null, 2)
          )
        )
      : null
  );
}
