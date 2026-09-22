import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  reduceManagedRollout,
  type ManagedRolloutPlan
} from './managed-rollout-coordinator'

const SOURCE = {
  repositoryRoot: '/srv/hermes-agent',
  originUrl: 'https://github.com/NousResearch/hermes-agent.git',
  resolvedRef: 'refs/remotes/origin/main',
  targetSha: 'a'.repeat(40),
  assuranceProfile: 'managed-ssh-review-v1',
  assuranceEvidenceSha256: '0'.repeat(64),
  assuranceGeneration: 1
}

const PLAN: ManagedRolloutPlan = {
  id: '12345678-1234-4678-9234-567812345678',
  revision: 4,
  queueGeneration: 9,
  policy: 'manual',
  targets: [
    { installId: 'canary', installationFingerprint: 'install-a', connectionId: 'ssh-canary', sourceFingerprint: 'source-a', targetSha: SOURCE.targetSha, reviewedSource: SOURCE, correlationId: 'canary-correlation', wave: 0 },
    { installId: 'later', installationFingerprint: 'install-b', connectionId: 'ssh-later', sourceFingerprint: 'source-b', targetSha: SOURCE.targetSha, reviewedSource: SOURCE, correlationId: 'later-correlation', wave: 1 }
  ]
}

function authorizedState() {
  let state = reduceManagedRollout(createManagedRolloutState(PLAN), { kind: 'start' }).state
  state = reduceManagedRollout(state, { kind: 'record-intent', installId: 'canary' }).state
  return reduceManagedRollout(state, { kind: 'launch-authorized', installId: 'canary' }).state
}

function coordinator(state = authorizedState(), calls: string[] = []) {
  return createManagedRolloutCoordinator(state, {
    journal: { persistAuthorization: async () => {} },
    service: {
      issueCapability: () => ({}),
      launch: async () => {
        calls.push('launch')
      }
    },
    evidence: { sweep: async current => ({ rolloutId: current.id, revision: current.revision, queueGeneration: current.queueGeneration, processGeneration: 1, valid: false, reason: 'not-needed', admissions: [] }) },
    recovery: {
      reprobe: async authorization => ({ correlationId: authorization.correlationId, outcome: 'unverified', terminal: false }),
      recover: async authorization => ({ correlationId: authorization.correlationId, clearanceProved: true })
    }
  })
}

test('reopen after durable authorization stays unknown and does not redispatch', async () => {
  const calls: string[] = []
  const reopened = coordinator(authorizedState(), calls)

  assert.equal(reopened.reduce({ kind: 'restart' }).ok, true)
  assert.equal(reopened.reduce({ kind: 'reconcile-unknown', installId: 'canary' }).ok, true)
  assert.equal(reopened.snapshot.attempts.canary.state, 'unverified')
  assert.equal((await reopened.reprobe('canary')).ok, true)
  assert.equal(reopened.snapshot.attempts.canary.state, 'unverified')
  assert.deepEqual(calls, [])
})

test('recovery requires exact correlation and positive clearance while preserving unresolved outcome', async () => {
  let state = authorizedState()
  state = reduceManagedRollout(state, { kind: 'restart' }).state
  state = reduceManagedRollout(state, { kind: 'reconcile-unknown', installId: 'canary' }).state
  const restored = coordinator(state)

  const result = await restored.recover('canary')

  assert.equal(result.ok, true)
  assert.equal(restored.snapshot.attempts.canary.state, 'unverified')
})

test('reconciliation cannot declare an unresolved committed attempt completed', () => {
  let state = authorizedState()
  state = reduceManagedRollout(state, { kind: 'restart' }).state

  const reconciled = reduceManagedRollout(state, { kind: 'reconciled', phase: 'completed' })

  assert.equal(reconciled.ok, false)
  assert.equal(reconciled.reason, 'reconciled-terminal-not-proven')
  assert.equal(reconciled.state.phase, 'reconciling')
})

test('an archive or exclusion cannot erase an authorized attempt or reopen a launch edge', () => {
  const state = authorizedState()

  assert.equal(reduceManagedRollout(state, { kind: 'exclude', installId: 'canary' }).ok, false)
  assert.equal(reduceManagedRollout(state, { kind: 'record-intent', installId: 'canary' }).ok, false)
  assert.equal(state.attempts.canary.state, 'authorized')
})

test('inconclusive reprobes enter a bounded cooldown instead of an eternal lockout', async () => {
  let state = authorizedState()
  state = reduceManagedRollout(state, { kind: 'restart' }).state
  state = reduceManagedRollout(state, { kind: 'reconcile-unknown', installId: 'canary' }).state
  let nowMono = 1_000
  let reprobes = 0
  const reopened = createManagedRolloutCoordinator(state, {
    journal: { persistAuthorization: async () => {} },
    service: { issueCapability: () => ({}), launch: async () => {} },
    evidence: { sweep: async current => ({ rolloutId: current.id, revision: current.revision, queueGeneration: current.queueGeneration, processGeneration: 1, valid: false, reason: 'not-needed', admissions: [] }) },
    recovery: {
      reprobe: async authorization => {
        reprobes += 1
        return { correlationId: authorization.correlationId, outcome: 'unverified' as const, terminal: false }
      },
      recover: async authorization => ({ correlationId: authorization.correlationId, clearanceProved: true })
    },
    nowMono: () => nowMono
  })

  for (let index = 0; index < 5; index += 1) {
    assert.equal((await reopened.reprobe('canary')).ok, true)
  }
  assert.equal(reprobes, 5)
  assert.equal(reopened.snapshot.attempts.canary.reprobeCount, 5)
  assert.equal(reopened.snapshot.attempts.canary.reprobeCooldownUntilMono, 61_000)

  const held = await reopened.reprobe('canary')
  assert.equal(held.ok, false)
  assert.equal(held.reason, 'reprobe-cooldown')
  assert.equal(reprobes, 5)

  nowMono = 61_000
  assert.equal((await reopened.reprobe('canary')).ok, true)
  assert.equal(reprobes, 6)
  assert.equal(reopened.snapshot.attempts.canary.reprobeCount, 1)
})

test('a foreign controller cannot steal a correlation or create a second launch', async () => {
  const calls: string[] = []
  const foreign = coordinator(authorizedState(), calls)

  const result = await foreign.terminal('canary', 'foreign-correlation', 'updated')

  assert.equal(result.ok, false)
  assert.equal(foreign.snapshot.attempts.canary.correlationId, 'canary-correlation')
  assert.deepEqual(calls, [])
})
