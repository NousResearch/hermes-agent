import { expect, test } from 'vitest'
import { planTracker } from '../tests/install/e2e-assets/red-tracker.mjs'

const TESTED = '0123456789abcdef0123456789abcdef01234567'
const MAIN_TIP = 'fedcba9876543210fedcba9876543210fedcba98'
const run = (conclusion) => ({ display_title: `Canary E2E on ${TESTED}`, conclusion, html_url: 'https://example.test/runs/1', head_sha: MAIN_TIP, created_at: '2026-09-26T19:24:13Z' })
const leg = (name, conclusion, step) => ({ name, conclusion, html_url: `https://example.test/job/${name.length}`, steps: step ? [{ name: 'checkout', conclusion: 'success' }, { name: step, conclusion }] : [] })

test('one tracker: a red run opens it, the next red run rewrites it, a green run closes it', () => {
  const jobs = [
    leg('windows: installer-script -> hermes-update (v2026.9.24 -> HEAD) / e2e', 'cancelled', 'Install v2026.9.24 (installer-script)'),
    leg('windows: installer-script+desktop -> hermes-update (HEAD -> NEXT) / e2e', 'cancelled', 'Install HEAD (installer-script+desktop)'),
    leg('linux: installer-script -> hermes-update (HEAD -> NEXT) / install & update', 'success'),
    leg('linux: installer-script -> installer-script (HEAD -> NEXT) / install & update', 'skipped'),
  ]

  const opened = planTracker(run('cancelled'), jobs, null)
  expect(opened.action).toBe('open')
  expect(opened.title).toBe(`Canary E2E on ${TESTED} is red (2 jobs)`)
  // Both Windows legs hung in the same driver step: one class, not two.
  expect(opened.body).toContain('**2 × cancelled in step `Install <ref>`**')
  expect(opened.body).not.toContain('linux:')

  expect(planTracker(run('failure'), jobs, { number: 7 }).action).toBe('update')
  expect(planTracker(run('success'), [jobs[2]], { number: 7 }).action).toBe('close')
  expect(planTracker(run('success'), [jobs[2]], null).action).toBe('none')
})

test('a run replaced while pending (cancelled, no jobs) leaves the tracker alone', () => {
  expect(planTracker(run('cancelled'), [], null).action).toBe('none')
  expect(planTracker(run('cancelled'), [], { number: 7 }).action).toBe('none')
  // A run cancelled after its jobs started still reports them.
  expect(planTracker(run('cancelled'), [leg('Full CI pipeline / e2e', 'cancelled', 'Run e2e tests')], null).action).toBe('open')
})

test('the tracker names the commit under test, never the workflow_run head', () => {
  const failed = [leg('Full CI pipeline / Python tests / e2e', 'failure', 'Run e2e tests')]
  for (const plan of [planTracker(run('failure'), failed, null), planTracker(run('success'), [], { number: 7 })]) {
    expect(plan.body).toContain(TESTED)
    expect(plan.body).not.toContain(MAIN_TIP.slice(0, 10))
  }
})

test('non-gating diagnostic Linux failures never block closing or pollute the red list', () => {
  const diag = leg('Build diagnostic Linux bundle (linux-x64) / bundled linux-x64 (linux-x64)', 'failure', 'Audit bundle architecture')
  const passed = leg('Full CI pipeline / Python tests / unit', 'success')
  expect(planTracker(run('success'), [passed, diag], { number: 7 }).action).toBe('close')
  expect(planTracker(run('success'), [passed, diag], null).action).toBe('none')

  const realFail = leg('Build signed release candidates (win32-bundle) / Assemble and stage', 'failure', 'Assemble')
  const opened = planTracker(run('failure'), [diag, realFail], null)
  expect(opened.action).toBe('open')
  expect(opened.title).toContain('(1 job)')
  expect(opened.body).toContain('win32-bundle')
  expect(opened.body).not.toContain('Build diagnostic Linux bundle')
})

test('manual replay of a failed run with only diagnostic Linux failures closes the tracker', () => {
  const diag = leg('Build diagnostic Linux bundle (linux-x64) / bundled linux-x64 (linux-x64)', 'failure', 'Audit bundle architecture')
  const passed = leg('Stable release is green', 'success')
  expect(planTracker(run('failure'), [passed, diag], { number: 7 }).action).toBe('close')
  expect(planTracker(run('failure'), [passed, diag], null).action).toBe('none')
  // A failure with no red jobs at all (no diagnostic explanation) still opens/updates fail-closed.
  expect(planTracker(run('failure'), [passed], null).action).toBe('open')
  // A cancelled run is never reclassified as green.
  const diagCancelled = leg('Build diagnostic Linux bundle (linux-x64) / bundled linux-x64 (linux-x64)', 'cancelled', 'Audit')
  expect(planTracker(run('cancelled'), [passed, diagCancelled], { number: 7 }).action).toBe('update')
})
