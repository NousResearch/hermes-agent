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

test('the tracker names the commit under test, never the workflow_run head', () => {
  const failed = [leg('Full CI pipeline / Python tests / e2e', 'failure', 'Run e2e tests')]
  for (const plan of [planTracker(run('failure'), failed, null), planTracker(run('success'), [], { number: 7 })]) {
    expect(plan.body).toContain(TESTED)
    expect(plan.body).not.toContain(MAIN_TIP.slice(0, 10))
  }
})
