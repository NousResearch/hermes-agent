#!/usr/bin/env node
/**
 * Keep ONE tracking issue in step with the scheduled install/update matrix.
 *
 * Run by .github/workflows/install-e2e-red.yml after every scheduled
 * "Install & Update E2E" run. A red run (any leg failed, was cancelled, or timed
 * out) opens the issue labelled `install-e2e-red`, or rewrites the body of the
 * one already open, so the issue always names the CURRENT red legs and the run
 * that produced them. A green run closes it with a pointer to that run. Never a
 * second issue and never a comment per red run: subscribers hear about the open
 * and the close, and the body is the live state in between.
 *
 *   node tests/install/e2e-assets/red-tracker.mjs --run-id <id> [--dry-run]
 *
 * Needs GH_TOKEN (issues: write, actions: read) and GITHUB_REPOSITORY.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

export const LABEL = 'install-e2e-red';
const RED = new Set(['failure', 'cancelled', 'timed_out', 'startup_failure']);
const MAX_ROWS = 60;

/**
 * Decide what the tracker does for one completed run. Pure, so the policy is
 * testable without GitHub.
 *
 * @param {{conclusion: string | null, html_url: string, head_sha: string, created_at: string}} run
 * @param {{name: string, conclusion: string | null, html_url: string, steps?: {name: string, conclusion: string | null}[]}[]} jobs
 * @param {{number: number} | null} openIssue
 * @returns {{action: 'open' | 'update' | 'close' | 'none', title?: string, body?: string}}
 */
export function planTracker(run, jobs, openIssue) {
  const red = jobs.filter((j) => RED.has(String(j.conclusion)));
  const runRed = RED.has(String(run.conclusion)) || red.length > 0;
  if (!runRed) {
    if (run.conclusion === 'success' && openIssue) {
      return { action: 'close', body: `The scheduled matrix is green again: ${run.html_url} (\`${run.head_sha.slice(0, 10)}\`). Closing; the next red run reopens a fresh tracker.` };
    }
    return { action: 'none' };
  }
  // Group by (conclusion, step) so 34 legs cancelled in the same step read as
  // one failure class, not 34 unrelated reds.
  /** @type {Map<string, typeof red>} */
  const classes = new Map();
  for (const job of red) {
    const step = (job.steps || []).find((s) => RED.has(String(s.conclusion)))?.name || '(no step reported)';
    // "Install v2026.9.24 (installer-script)" and "Install HEAD (installer-script+desktop)"
    // are the same step of the driver; the leg name already carries ref and method.
    const phase = step.replace(/\bv\d{4}\.\d+\.\d+(?:\.\d+)?\b|\bHEAD\b|\bNEXT\b/g, '<ref>').replace(/\s*\([^)]*\)\s*$/, '');
    const key = `${job.conclusion} in step \`${phase}\``;
    if (!classes.has(key)) classes.set(key, []);
    classes.get(key)?.push(job);
  }
  const lines = [
    `The scheduled **Install & Update E2E** run ${run.html_url} (\`${run.head_sha.slice(0, 10)}\`, ${run.created_at}) finished **${run.conclusion}** with ${red.length} red leg(s).`,
    '',
    'This issue is rewritten in place by `.github/workflows/install-e2e-red.yml` after every scheduled run and closed by the first green one. Root-cause each class below; a historical-release limitation that cannot be fixed goes into `tests/install/e2e-assets/known-failures.json`.',
    '',
    '### Red legs by failure class',
  ];
  let rows = 0;
  for (const [key, members] of [...classes].sort((a, b) => b[1].length - a[1].length)) {
    lines.push('', `**${members.length} × ${key}**`, '');
    for (const job of members) {
      if (rows >= MAX_ROWS) break;
      lines.push(`- [${job.name}](${job.html_url})`);
      rows += 1;
    }
  }
  if (red.length > rows) lines.push('', `…and ${red.length - rows} more; see the run's jobs list.`);
  const title = `Install & Update E2E matrix is red (${red.length} leg${red.length === 1 ? '' : 's'})`;
  return { action: openIssue ? 'update' : 'open', title, body: lines.join('\n') };
}

/** @param {string[]} args @returns {string} */
function gh(args) {
  return execFileSync('gh', args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

async function main() {
  const { values } = parseArgs({ options: { 'run-id': { type: 'string' }, 'dry-run': { type: 'boolean', default: false } } });
  const repo = process.env.GITHUB_REPOSITORY;
  const runId = values['run-id'];
  if (!repo || !runId || !/^\d+$/.test(runId)) throw new Error('need GITHUB_REPOSITORY and a numeric --run-id');
  const run = JSON.parse(gh(['api', `repos/${repo}/actions/runs/${runId}`]));
  const jobs = gh(['api', '--paginate', `repos/${repo}/actions/runs/${runId}/jobs?per_page=100`,
    '--jq', '.jobs[] | {name, conclusion, html_url, steps: [.steps[]? | {name, conclusion}]}'])
    .split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
  let open = [];
  try {
    open = JSON.parse(gh(['api', `repos/${repo}/issues?labels=${LABEL}&state=open&per_page=5`]))
      .filter((/** @type {any} */ i) => !i.pull_request);
  } catch (error) {
    // Issue tracking is auxiliary. Forks may intentionally have Issues disabled;
    // never turn an already-completed E2E result red because the notification
    // sink is unavailable.
    console.warn(`Issue tracking unavailable; leaving the E2E result untouched: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  const openIssue = open.length ? { number: open[0].number } : null;
  const plan = planTracker(run, jobs, openIssue);
  console.log(`run ${runId}: conclusion=${run.conclusion} red-legs=${jobs.filter((j) => RED.has(String(j.conclusion))).length} open-tracker=${openIssue ? `#${openIssue.number}` : 'none'} -> ${plan.action}`);
  if (values['dry-run'] || plan.action === 'none') {
    if (plan.body) console.log(`\n--- ${plan.title || 'comment'} ---\n${plan.body}`);
    return;
  }
  if (plan.action === 'open') {
    gh(['label', 'create', LABEL, '--repo', repo, '--force', '--color', 'B60205',
      '--description', 'The scheduled install/update E2E matrix is red (managed by install-e2e-red.yml)']);
    const url = gh(['issue', 'create', '--repo', repo, '--label', LABEL, '--title', String(plan.title), '--body', String(plan.body)]);
    console.log(`opened ${url.trim()}`);
  } else if (plan.action === 'update' && openIssue) {
    gh(['api', '-X', 'PATCH', `repos/${repo}/issues/${openIssue.number}`, '-f', `title=${plan.title}`, '-f', `body=${plan.body}`]);
    console.log(`updated #${openIssue.number} in place`);
  } else if (plan.action === 'close' && openIssue) {
    gh(['issue', 'comment', String(openIssue.number), '--repo', repo, '--body', String(plan.body)]);
    gh(['issue', 'close', String(openIssue.number), '--repo', repo]);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  await main();
}
