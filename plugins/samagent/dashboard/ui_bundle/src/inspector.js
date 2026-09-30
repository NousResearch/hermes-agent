/**
 * Inspector — the right-hand auxiliary column.
 *
 * Codex's review pane is its best surface, and three of its ideas transfer
 * directly: an explicit user-chosen scope rather than an inferred one, an
 * honest statement of what the diff contains, and diff stats visible without
 * entering the pane. We add the fourth thing SamAgent has that Codex does not:
 * the frozen contract, which is the artefact every worker is measured against.
 */
import { createElement as h, useMemo } from '../sdk.js';
import { TabBar, Card, Empty, Disclosure, DiffView, KV, StatusPill, Dot, cx } from '../ui.js';
import { diffStats, fmtUsd, titleCase } from '../data.js';

export function Inspector({ state, tab, onTab, runState, metrics, mode }) {
  const ide = (state && state.ide) || {};
  const git = ide.git || {};
  const files = ide.files || [];
  const diffText = git.diff || git.diff_text || git.unified_diff || '';

  const stats = useMemo(() => diffStats(diffText), [diffText]);
  const gateChecks = (state && state.pre_prod_gate && state.pre_prod_gate.checks) || {};
  const gatePassed = Object.keys(gateChecks).filter((k) => gateChecks[k] === true).length;
  const gateTotal = Object.keys(gateChecks).length;
  const contracts = state && state.contracts ? state.contracts : {};
  const contractKeys = Object.keys(contracts);

  const counts = {
    diff: stats.add + stats.del,
    files: files.length,
    contract: contractKeys.length,
  };

  return h(
    'aside',
    { className: 'mc-inspector' },
    h(
      TabBar,
      {
        tabs: [
          { id: 'diff', label: 'Diff' },
          { id: 'files', label: 'Files' },
          { id: 'contract', label: 'Contract' },
          { id: 'run', label: 'Run' },
        ],
        active: tab,
        onSelect: onTab,
        counts,
      }
    ),
    h('div', { className: 'mc-inspector-body' }, h(TAB_BODIES[tab] || TAB_BODIES.diff, { state, git, files, diffText, stats, contracts, contractKeys, runState, metrics, mode, onTab }))
  );
}

const TAB_BODIES = {
  /* The diff pane. Scope is stated, not inferred — the same honesty Codex
     documents: this is the whole working tree, not just what the agent wrote. */
  diff({ diffText, stats, git }) {
    return h(
      'div',
      null,
      h(
        'div',
        { className: 'mc-scope-note' },
        h('span', { className: 'mc-eyebrow' }, 'Scope'),
        h('span', null, 'Whole working tree — your own edits are included, not just the agent’s.'),
        git.branch ? h('code', { className: 'mc-glob mc-mono' }, git.branch) : null
      ),
      h(
        'div',
        { className: 'mc-diff-stat' },
        h('span', { className: 'mc-add mc-mono' }, `+${stats.add}`),
        h('span', { className: 'mc-del mc-mono' }, `−${stats.del}`),
        git.head_commit ? h('span', { className: 'mc-faint mc-tiny mc-mono' }, git.head_commit) : null
      ),
      h(DiffView, { text: diffText })
    );
  },

  files({ files, git }) {
    if (!files.length) return h(Empty, { title: 'No files', hint: 'Nothing generated in this workspace yet.' });
    return h(
      'div',
      null,
      h(
        'ul',
        { className: 'mc-file-list' },
        files.map((f, i) => {
          const name = typeof f === 'string' ? f : f.path || f.name || f.rel_path || '';
          const size = typeof f === 'object' && f && f.size ? f.size : null;
          return h(
            'li',
            { key: name || i, className: 'mc-file-row', title: name },
            h('span', { className: 'mc-file-name mc-mono mc-truncate' }, name),
            size !== null ? h('span', { className: 'mc-faint mc-tiny mc-mono' }, `${size}B`) : null
          );
        })
      ),
      git.dirty_files && git.dirty_files.length
        ? h(
            Disclosure,
            { label: 'Uncommitted', detail: `${git.dirty_files.length} paths` },
            h(
              'ul',
              { className: 'mc-file-list' },
              git.dirty_files.map((d, i) => h('li', { key: i, className: 'mc-file-row' }, h('span', { className: 'mc-file-name mc-mono mc-truncate' }, d)))
            )
          )
        : null
    );
  },

  /* The frozen contract. This is the artefact that makes the swarm safe and
     it has no Codex equivalent, so it gets a first-class tab. */
  contract({ contracts, contractKeys }) {
    if (!contractKeys.length) return h(Empty, { title: 'No contract frozen', hint: 'Approve a plan to freeze the API, schema and ownership map.' });
    return h(
      'div',
      null,
      h('p', { className: 'mc-scope-note' }, 'Frozen before any worker starts. Workers file change requests; only the conductor edits these files.'),
      h(
        'div',
        null,
        contractKeys.map((k) =>
          h(
            Disclosure,
            { key: k, label: k, detail: `${String(contracts[k]).split('\n').length} lines`, defaultOpen: k === 'version.json' },
            h('pre', { className: 'mc-pre' }, String(contracts[k]))
          )
        )
      )
    );
  },

  run({ state, runState, metrics }) {
    const plan = (state && state.plan_card) || {};
    const spec = (state && state.spec) || {};
    const budget = spec.budget || {};
    const dev = (state && state.dev_server) || {};
    const github = (state && state.github) || {};

    return h(
      'div',
      { className: 'mc-inspect-run' },
      h(
        Card,
        { title: 'This run' },
        h(
          'div',
          { className: 'mc-kv-stack' },
          h(KV, { k: 'State', v: runState.label, tone: runState.tone }),
          h(KV, { k: 'Progress', v: `${metrics.done}/${metrics.total} (${metrics.pct}%)` }),
          metrics.costHigh !== null ? h(KV, { k: 'Cost', v: `${fmtUsd(metrics.costLow)}–${fmtUsd(metrics.costHigh)}` }) : null,
          budget.max_usd ? h(KV, { k: 'Budget cap', v: fmtUsd(budget.max_usd) }) : null,
          budget.max_minutes ? h(KV, { k: 'Time cap', v: `${budget.max_minutes}m` }) : null,
          plan.autonomy ? h(KV, { k: 'Autonomy', v: titleCase(plan.autonomy) }) : null,
          plan.fanout && plan.fanout.allow_parallel !== undefined ? h(KV, { k: 'Parallel', v: plan.fanout.allow_parallel ? 'yes' : 'no' }) : null,
          plan.local_task_share_pct !== undefined ? h(KV, { k: 'Local share', v: `${Math.round(plan.local_task_share_pct)}%` }) : null
        )
      ),
      h(
        Card,
        { title: 'Environment' },
        h(
          'div',
          { className: 'mc-kv-stack' },
          h(KV, { k: 'Workspace', v: (state.workspace || '').split(/[\\/]/).pop(), mono: true }),
          h(KV, { k: 'Branch', v: (state.ide && state.ide.git && state.ide.git.branch) || '—', mono: true }),
          h(KV, { k: 'Dev server', v: dev.running ? `up on :${dev.port}` : 'stopped', tone: dev.running ? 'ok' : 'idle' }),
          h(KV, { k: 'Remote', v: github.remote || 'not configured' }),
          h(KV, { k: 'Model', v: (state.selected_model || '—'), mono: true })
        )
      ),
      h(
        Card,
        { title: 'Measurements' },
        h(MeasurementSummary, { state })
      )
    );
  },
};

/**
 * The ablation report is the project's own falsifiability record — every
 * component has to beat its own ablation to stay switched on. It belongs in the
 * product, not just in docs/samagent.
 */
function MeasurementSummary({ state }) {
  const m = (state && state.measurements) || {};
  const rows = [];

  const fp = m.tool_footprint;
  if (fp) {
    const pick = (obj, ...keys) => (obj ? keys.map((k) => obj[k]).find((v) => v !== undefined) : null);
    const coding = pick(fp, 'coding_posture', 'coding', 'coding_tokens');
    const lean = pick(fp, 'lean8', 'lean_8', 'lean8_tokens', 'lean');
    if (coding !== null) rows.push(['Tool schema, coding posture', `${coding} tok`]);
    if (lean !== null) rows.push(['Tool schema, lean profile', `${lean} tok`]);
  }

  const spikes = m.spikes_s1_s9;
  if (spikes && typeof spikes === 'object') {
    const list = Array.isArray(spikes) ? spikes : spikes.spikes || Object.values(spikes);
    const passed = list.filter((s) => s && (s.passed === true || s.verdict === 'pass')).length;
    if (list.length) rows.push(['Phase 0 spikes', `${passed}/${list.length} passed`]);
  }

  const bench = m.sambench_v0_report;
  if (bench) {
    const arms = bench.arms || bench.results;
    if (arms && typeof arms === 'object') {
      const keys = Object.keys(arms);
      if (keys.length) rows.push(['Benchmark arms', String(keys.length)]);
    }
  }

  if (!rows.length) return h('p', { className: 'mc-faint mc-tiny' }, 'No measurement snapshots in this workspace yet.');

  return h(
    'div',
    { className: 'mc-kv-stack' },
    rows.map(([k, v], i) => h(KV, { key: i, k, v, mono: true }))
  );
}
