import { expect, test } from 'vitest'
import { planTracker } from '../tests/install/e2e-assets/red-tracker.mjs'

const run = (conclusion) => ({ conclusion, html_url: 'https://example.test/runs/1', head_sha: 'abcdef0123456789', created_at: '2026-09-26T19:24:13Z' })
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
  expect(opened.title).toBe('Install & Update E2E matrix is red (2 legs)')
  // Both Windows legs hung in the same driver step: one class, not two.
  expect(opened.body).toContain('**2 × cancelled in step `Install <ref>`**')
  expect(opened.body).not.toContain('linux:')

  expect(planTracker(run('failure'), jobs, { number: 7 }).action).toBe('update')
  expect(planTracker(run('success'), [jobs[2]], { number: 7 }).action).toBe('close')
  expect(planTracker(run('success'), [jobs[2]], null).action).toBe('none')
})
