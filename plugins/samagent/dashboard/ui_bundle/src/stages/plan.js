/**
 * Stage 2 — Plan card.
 *
 * This is the human approval gate, and it is the screen the old UI never built
 * even though the backend had every field for it. The plan card is where the
 * pipeline's bets are made visible: which modules, what they cost, how much
 * stays on the user's own machine, what the router decided and why, and what
 * the spec critic could not resolve.
 *
 * The autonomy dial and the fan-out decision are editable because they are the
 * two things a user most wants to hold, not observe.
 */
import { createElement as h, useState, useCallback } from '../sdk.js';
import { Button, Card, KV, SplitBar, cx } from '../ui.js';
import { fmtUsd, fmtMinutes, fmtRange } from '../data.js';

const AUTONOMY = [
  { id: 'plan_only', label: 'Plan only', hint: 'Stop after the plan card. I write nothing.' },
  { id: 'milestones', label: 'Check in at milestones', hint: 'Pause at each wave for your approval.' },
  { id: 'hands_off', label: 'Hands-off', hint: 'Run to the end, then show you the result.' },
];

export function PlanStage({ state, actions, busy }) {
  const plan = (state && state.plan_card) || {};
  const modules = plan.modules || [];
  const [autonomy, setAutonomy] = useState((state.spec && state.spec.autonomy) || 'milestones');
  const [localOnly, setLocalOnly] = useState((state.spec && state.spec.router_policy) === 'local_strict');

  const onBuild = useCallback(() => {
    actions.build({ autonomy, routerPolicy: localOnly ? 'local_strict' : 'default' });
  }, [actions, autonomy, localOnly]);

  if (!modules.length) {
    return h(
      'div',
      { className: 'mc-stage' },
      h('p', { className: 'mc-lede' }, 'No plan yet — describe what you want on the Brief tab and SamAgent will draft one here for your approval.')
    );
  }

  const critique = plan.critique || {};
  const assumptions = plan.assumptions || [];
  const risks = plan.risks || [];
  const routing = plan.routing_table || [];
  const fanout = plan.fanout || {};

  return h(
    'div',
    { className: 'mc-stage' },

    /* --- the ask ------------------------------------------------------ */
    h(
      'div',
      { className: 'mc-plan-hero' },
      h('h2', { className: 'mc-h2' }, 'Approve the plan'),
      h('p', { className: 'mc-lede' }, plan.goal || (state.spec && state.spec.goal) || ''),
      h(
        'div',
        { className: 'mc-estimate' },
        h(
          'div',
          { className: 'mc-est' },
          h('span', { className: 'mc-est-value' }, fmtRange(plan.estimated_cost_usd_range && plan.estimated_cost_usd_range[0], plan.estimated_cost_usd_range && plan.estimated_cost_usd_range[1], fmtUsd)),
          h('span', { className: 'mc-est-label' }, 'estimated cost')
        ),
        h(
          'div',
          { className: 'mc-est' },
          h('span', { className: 'mc-est-value' }, fmtRange(plan.estimated_minutes_range && plan.estimated_minutes_range[0], plan.estimated_minutes_range && plan.estimated_minutes_range[1], fmtMinutes)),
          h('span', { className: 'mc-est-label' }, 'estimated time')
        ),
        h(
          'div',
          { className: 'mc-est' },
          h('span', { className: 'mc-est-value' }, String(modules.length)),
          h('span', { className: 'mc-est-label' }, modules.length === 1 ? 'module' : 'modules')
        )
      )
    ),

    /* --- autonomy dial ------------------------------------------------- */
    h(
      Card,
      { title: 'How much should I do on my own?' },
      h(
        'div',
        { className: 'mc-radio-group', role: 'radiogroup' },
        AUTONOMY.map((a) =>
          h(
            'button',
            {
              key: a.id,
              type: 'button',
              role: 'radio',
              'aria-checked': autonomy === a.id ? 'true' : 'false',
              className: cx('mc-radio-card', autonomy === a.id && 'is-selected'),
              onClick: () => setAutonomy(a.id),
            },
            h('span', { className: 'mc-radio-dot' }),
            h('span', { className: 'mc-radio-text' }, h('strong', null, a.label), h('span', { className: 'mc-muted' }, a.hint))
          )
        )
      ),
      h(
        'label',
        { className: 'mc-switch' },
        h('input', { type: 'checkbox', checked: localOnly, onChange: (e) => setLocalOnly(e.target.checked) }),
        h('span', null, 'Keep everything on this machine'),
        h('span', { className: 'mc-muted' }, 'never sends your code to a cloud model — quality drops, cost drops to near zero')
      ),
      typeof plan.local_task_share_pct === 'number' ? h(SplitBar, { localPct: plan.local_task_share_pct }) : null
    ),

    /* --- fan-out reasoning -------------------------------------------- */
    h(
      Card,
      { title: 'Parallel work' },
      h(
        'p',
        { className: cx('mc-fanout-reason', fanout.allow_parallel ? 'is-ok' : 'is-idle') },
        fanout.allow_parallel
          ? `Workers run at the same time: ${fanout.reason || 'modules are independent'}`
          : `This runs on one thread: ${fanout.reason || 'the modules are not independent yet'}`
      ),
      fanout.waves && fanout.waves.length
        ? h(
            'ol',
            { className: 'mc-waves' },
            fanout.waves.map((w, i) =>
              h(
                'li',
                { key: i, className: 'mc-wave' },
                h('span', { className: 'mc-wave-n mc-mono' }, `wave ${i + 1}`),
                h('span', null, Array.isArray(w) ? w.join(', ') : String(w))
              )
            )
          )
        : null
    ),

    /* --- modules + ownership ------------------------------------------ */
    h(
      Card,
      { title: 'What gets built' },
      h(
        'div',
        { className: 'mc-module-list' },
        modules.map((m, i) =>
          h(
            'div',
            { key: m.name || i, className: 'mc-module' },
            h(
              'div',
              { className: 'mc-module-head' },
              h('span', { className: 'mc-module-name' }, m.name),
              h('span', { className: 'mc-module-time mc-mono' }, fmtMinutes(m.estimated_minutes)),
              m.sensitivity && m.sensitivity !== 'public' ? h('span', { className: 'mc-chip mc-chip-warn' }, m.sensitivity) : null
            ),
            m.description ? h('p', { className: 'mc-module-desc' }, m.description) : null,
            m.owned_globs && m.owned_globs.length
              ? h(
                  'div',
                  { className: 'mc-globs' },
                  m.owned_globs.map((g, j) => h('code', { key: j, className: 'mc-glob mc-mono' }, g))
                )
              : null,
            m.depends_on && m.depends_on.length ? h('p', { className: 'mc-faint mc-tiny' }, `needs ${m.depends_on.join(', ')}`) : null
          )
        )
      )
    ),

    /* --- spec critique ------------------------------------------------- */
    critique && critique.issues && critique.issues.length
      ? h(
          Card,
          { title: 'What I could not resolve' },
          h(
            'ul',
            { className: 'mc-list mc-list-warn' },
            critique.issues.map((iss, i) => h('li', { key: i }, typeof iss === 'string' ? iss : iss.text || JSON.stringify(iss)))
          ),
          critique.blocking_question ? h('p', { className: 'mc-blocking' }, critique.blocking_question) : null
        )
      : critique && critique.passed
      ? h('p', { className: 'mc-ok-note' }, 'The spec passed critique — no ambiguity or missing acceptance criteria found.')
      : null,

    /* --- assumptions + risks ------------------------------------------ */
    assumptions.length
      ? h(
          Card,
          { title: 'What I assumed', actions: h('span', { className: 'mc-faint mc-tiny' }, 'editable — agents change these only through a contract change request') },
          h(
            'ul',
            { className: 'mc-list' },
            assumptions.map((a, i) =>
              h('li', { key: a.id || i }, h('code', { className: 'mc-mono mc-assumption-id' }, a.id || `A${i + 1}`), ' ', a.text || a)
            )
          )
        )
      : null,

    risks.length
      ? h(
          Card,
          { title: 'What could go wrong' },
          h('ul', { className: 'mc-list' }, risks.map((r, i) => h('li', { key: i }, r)))
        )
      : null,

    /* --- routing table: why each phase got its model ------------------- */
    routing.length
      ? h(
          Card,
          { title: 'Which model handles which step' },
          h(
            'div',
            { className: 'mc-routing' },
            routing.map((r, i) =>
              h(
                'div',
                { key: i, className: cx('mc-route', r.is_local && 'is-local') },
                h('span', { className: 'mc-route-phase' }, r.label || r.phase),
                h('span', { className: 'mc-route-model' }, r.model),
                h('span', { className: 'mc-route-provider mc-muted' }, r.provider),
                r.reason ? h('span', { className: 'mc-route-reason mc-faint' }, r.reason) : null
              )
            )
          )
        )
      : null,

    h(
      'div',
      { className: 'mc-approve-bar' },
      h('span', { className: 'mc-faint mc-tiny' }, 'Nothing is written until you approve'),
      h(Button, { variant: 'primary', size: 'lg', onClick: onBuild, disabled: busy }, busy ? 'Building…' : 'Approve & build')
    )
  );
}
