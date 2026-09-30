/**
 * Stage 1 — Brief.
 *
 * One box, per the plan: "What do you want to build?" The interview never
 * blocks; every skipped answer becomes a visible assumption the user can edit
 * later, which is why the assumptions list sits directly under the composer
 * rather than being deferred to the Plan stage.
 */
import { createElement as h, useState, useCallback } from '../sdk.js';
import { Button, Card, Empty, KV, cx } from '../ui.js';
import { fmtUsd } from '../data.js';

export function BriefStage({ state, actions, busy }) {
  const [brief, setBrief] = useState('');
  const [questions, setQuestions] = useState(null);
  const [answers, setAnswers] = useState({});
  const spec = (state && state.spec) || {};
  const hasExisting = !!spec.goal;

  const onInterview = useCallback(() => {
    const text = brief.trim();
    if (text.length < 3) return;
    setQuestions({ loading: true });
    actions
      .interview(text)
      .then((res) => {
        const list = (res && (res.questions || (res.interview && res.interview.questions))) || [];
        setQuestions({ loading: false, list });
      })
      .catch((err) => setQuestions({ loading: false, error: String(err.message || err) }));
  }, [brief, actions]);

  const onPlan = useCallback(() => {
    const text = brief.trim();
    if (text.length < 3) return;
    actions.plan(text, answers);
  }, [brief, answers, actions]);

  return h(
    'div',
    { className: 'mc-stage mc-stage-brief' },
    h(
      'div',
      { className: 'mc-brief-hero' },
      h('h2', { className: 'mc-h2' }, hasExisting ? 'Start something new' : 'What should we build?'),
      h(
        'p',
        { className: 'mc-lede' },
        'SamAgent turns a description into a frozen contract, parallel work in isolated worktrees, and verification your brief can be checked against. At most five questions, and every one has a default — you can always skip ahead.'
      )
    ),

    h(
      'div',
      { className: 'mc-composer-wrap' },
      h(
        'div',
        { className: 'mc-composer' },
        h('textarea', {
          className: 'mc-composer-input',
          placeholder: 'A booking site for my yoga studio where members cannot double-book…',
          value: brief,
          rows: 3,
          onChange: (e) => setBrief(e.target.value),
          onKeyDown: (e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) onPlan();
          },
        }),
        h(
          'div',
          { className: 'mc-composer-bar' },
          h('span', { className: 'mc-faint mc-tiny' }, '⌘↵ to plan'),
          h(
            'div',
            { className: 'mc-composer-actions' },
            h(Button, { variant: 'ghost', size: 'sm', onClick: onInterview, disabled: brief.trim().length < 3 || busy }, 'Ask me 5 questions'),
            h(Button, { variant: 'primary', size: 'sm', onClick: onPlan, disabled: brief.trim().length < 3 || busy }, busy ? 'Planning…' : 'Plan it')
          )
        )
      )
    ),

    questions && questions.error ? h('p', { className: 'mc-error' }, questions.error) : null,

    questions && questions.list
      ? h(
          'div',
          { className: 'mc-questions' },
          questions.list.map((q, i) =>
            h(
              'div',
              { key: q.id || i, className: 'mc-question' },
              h(
                'div',
                { className: 'mc-question-head' },
                h('span', { className: 'mc-question-n mc-mono' }, `Q${i + 1}`),
                h('span', { className: 'mc-question-text' }, q.question || q.text || ''),
                q.recommended ? h('span', { className: 'mc-chip mc-chip-accent' }, 'Recommended') : null
              ),
              h('input', {
                className: 'mc-input',
                placeholder: q.recommended || 'Skip — use the default',
                value: answers[q.id || i] || '',
                onChange: (e) => setAnswers((a) => ({ ...a, [q.id || i]: e.target.value })),
              })
            )
          )
        )
      : null,

    hasExisting
      ? h(
          Card,
          { title: 'Current brief' },
          h('p', { className: 'mc-goal' }, spec.goal),
          h(
            'div',
            { className: 'mc-kv-grid' },
            h(KV, { k: 'Stack', v: spec.stack || '—' }),
            h(KV, { k: 'Autonomy', v: titleCaseSafe(spec.autonomy) }),
            h(KV, { k: 'Roles', v: (spec.roles || []).join(', ') || '—' }),
            h(KV, { k: 'Budget', v: spec.budget ? `${fmtUsd(spec.budget.max_usd)} / ${spec.budget.max_minutes}m` : '—' })
          ),
          spec.non_goals && spec.non_goals.length
            ? h(
                'div',
                { className: 'mc-non-goals' },
                h('span', { className: 'mc-eyebrow' }, 'Explicitly out of scope'),
                h('ul', { className: 'mc-list' }, spec.non_goals.map((g, i) => h('li', { key: i }, g)))
              )
            : null
        )
      : null
  );
}

function titleCaseSafe(s) {
  return String(s || '').replace(/_/g, ' ');
}
