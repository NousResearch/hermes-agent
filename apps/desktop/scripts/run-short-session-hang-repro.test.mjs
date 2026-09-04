import assert from 'node:assert/strict'
import { existsSync, writeFileSync } from 'node:fs'
import test from 'node:test'
import { join } from 'node:path'

import {
  ReproductionError,
  classify,
  pairedSoftSignal,
  resultForError,
  validateSummary,
  waitFor,
  waitForPredicate,
  waitForResponsive,
  withTemporarySandbox,
  withTimeout
} from './run-short-session-hang-repro.mjs'

const result = (outcome, latency = 10) => ({
  hardFailure: outcome !== 'not-reproduced',
  maxGapMs: latency,
  maxOperationMs: latency,
  outcome
})

test('separates harness errors from reproduction timeouts', () => {
  const harness = resultForError(new Error('fixture mismatch'))
  const reproduced = resultForError(new ReproductionError('renderer operation exceeded 5000ms'))

  assert.equal(harness.outcome, 'harness-error')
  assert.equal(harness.maxGapMs, null)
  assert.equal(harness.maxOperationMs, null)
  assert.equal(reproduced.outcome, 'reproduced')
})

test('preserves timeout semantics through the actual nested helpers', async () => {
  const immediate = await withTimeout(Promise.reject(new Error('precondition')), 30, 'outer', ReproductionError).catch(
    error => error
  )
  assert.equal(resultForError(immediate).outcome, 'harness-error')

  const stalledCdp = { eval: async () => false }
  const inner = await withTimeout(
    waitFor(stalledCdp, 'false', 10, 'inner product operation', ReproductionError),
    50,
    'outer product operation',
    ReproductionError
  ).catch(error => error)
  assert.ok(inner instanceof ReproductionError)
  assert.match(inner.message, /inner product operation/)

  const outer = await withTimeout(
    waitFor(stalledCdp, 'false', 50, 'inner product operation', ReproductionError),
    10,
    'outer product operation',
    ReproductionError
  ).catch(error => error)
  assert.ok(outer instanceof ReproductionError)
  assert.match(outer.message, /outer product operation/)

  const nearDeadline = await withTimeout(
    new Promise(resolve => setTimeout(() => resolve('responsive'), 20)),
    50,
    'responsive operation',
    ReproductionError
  )
  assert.equal(nearDeadline, 'responsive')
})

test('distinguishes a responsive false condition from a stalled renderer evaluation', { timeout: 2_000 }, async () => {
  let transientEvaluations = 0
  await waitForResponsive(
    {
      eval: async () => {
        transientEvaluations += 1

        if (transientEvaluations === 1) {
          throw new Error('execution context was destroyed')
        }

        return true
      }
    },
    'true',
    500,
    'transient condition',
    50
  )
  assert.equal(transientEvaluations, 2)

  await assert.rejects(
    waitForResponsive({ eval: async () => false }, 'false', 10, 'responsive condition', 1_000),
    error => !(error instanceof ReproductionError) && /renderer remained responsive/.test(error.message)
  )
  await assert.rejects(
    waitForResponsive({ eval: () => new Promise(() => {}) }, 'false', 50, 'stalled condition', 10),
    ReproductionError
  )
  await assert.rejects(
    waitForPredicate(() => false, 10, 'provider request'),
    /provider request/
  )
})

test('invalidates a target when warmup or measured runs have harness errors', () => {
  const passing = Array.from({ length: 5 }, () => result('not-reproduced'))

  assert.equal(classify(passing, result('harness-error')).classification, 'invalid')
  assert.equal(
    classify([result('harness-error'), ...passing.slice(1)], result('not-reproduced')).classification,
    'invalid'
  )
  assert.equal(classify(passing, result('not-reproduced')).classification, 'not-reproduced')
  assert.equal(
    classify(
      [result('reproduced'), result('reproduced'), result('reproduced'), result('reproduced'), passing[0]],
      passing[0]
    ).classification,
    'reproduced'
  )
})

test('suppresses soft-signal comparisons when a run is invalid or reproduced', () => {
  const passing = Array.from({ length: 5 }, () => result('not-reproduced', 10))

  assert.equal(pairedSoftSignal([], passing).reason, 'insufficient-runs')
  assert.equal(pairedSoftSignal(passing, []).reason, 'insufficient-runs')
  assert.equal(pairedSoftSignal(passing, passing).reason, undefined)
  assert.equal(pairedSoftSignal([result('harness-error'), ...passing.slice(1)], passing).reason, 'hard-or-invalid-run')
  assert.equal(pairedSoftSignal([result('reproduced'), ...passing.slice(1)], passing).reason, 'hard-or-invalid-run')
})

test('rejects contradictory summary semantics', () => {
  const passing = Array.from({ length: 5 }, () => result('not-reproduced'))
  const classification = classify(passing, result('not-reproduced'))
  const target = { ...classification, runs: passing, warmup: result('not-reproduced') }
  const summary = { baseline: target, candidate: target, invalid: false }

  assert.doesNotThrow(() => validateSummary(summary, 5))
  assert.throws(
    () => validateSummary({ ...summary, baseline: { ...target, classification: 'reproduced' } }, 5),
    /inconsistent baseline summary classification/
  )
  assert.throws(
    () =>
      validateSummary(
        {
          ...summary,
          baseline: { ...target, runs: [{ ...passing[0], hardFailure: true }, ...passing.slice(1)] }
        },
        5
      ),
    /inconsistent baseline run outcome and hardFailure/
  )
})

test('removes the exact temporary sandbox when the run body throws', async () => {
  let sandbox

  await assert.rejects(
    withTemporarySandbox('cleanup-test', path => {
      sandbox = path
      writeFileSync(join(path, 'diagnostic.txt'), 'temporary')
      throw new Error('teardown report failed')
    }),
    /teardown report failed/
  )

  assert.ok(sandbox)
  assert.equal(existsSync(sandbox), false)
})
