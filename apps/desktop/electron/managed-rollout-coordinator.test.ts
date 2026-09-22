import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  reduceManagedRollout,
  type ManagedRolloutAuthorization,
  type ManagedRolloutPlan,
  type ManagedRolloutState
} from './managed-rollout-coordinator'

const PLAN: ManagedRolloutPlan = {
  id: '12345678-1234-4678-9234-567812345678',
  revision: 4,
  queueGeneration: 9,
  policy: 'auto-after-canary',
  targets: [
    { installId: 'canary', installationFingerprint: 'install-a', connectionId: 'ssh-canary', sourceFingerprint: 'source-a', targetSha: 'a'.repeat(40), reviewedSource: reviewedSource('a'.repeat(40)), correlationId: 'canary-correlation', wave: 0 },
    { installId: 'later', installationFingerprint: 'install-b', connectionId: 'ssh-later', sourceFingerprint: 'source-b', targetSha: 'a'.repeat(40), reviewedSource: reviewedSource('a'.repeat(40)), correlationId: 'later-correlation', wave: 1 }
  ]
}

function reviewedSource(targetSha: string) {
  return {
    repositoryRoot: '/srv/hermes-agent',
    originUrl: 'https://github.com/NousResearch/hermes-agent.git',
    resolvedRef: 'refs/remotes/origin/main',
    targetSha,
    assuranceProfile: 'managed-ssh-review-v1',
    assuranceEvidenceSha256: '0'.repeat(64),
    assuranceGeneration: 1
  }
}

function runningState(): ManagedRolloutState {
  const started = reduceManagedRollout(createManagedRolloutState(PLAN), { kind: 'start' })
  assert.equal(started.ok, true)
  return started.state
}

function adapters(overrides: Partial<{
  persistAuthorization: (authorization: ManagedRolloutAuthorization) => Promise<void>
  launch: (authorization: ManagedRolloutAuthorization, capability: object) => Promise<void>
}> = {}) {
  const persisted: ManagedRolloutAuthorization[] = []
  const launches: ManagedRolloutAuthorization[] = []
  return {
    persisted,
    launches,
    deps: {
      journal: {
        persistAuthorization: async authorization => {
          persisted.push(authorization)
          await overrides.persistAuthorization?.(authorization)
        }
      },
      service: {
        issueCapability: authorization => Object.freeze({ authorization }),
        launch: async (authorization, capability) => {
          assert.ok(capability)
          launches.push(authorization)
          await overrides.launch?.(authorization, capability)
        }
      },
      evidence: {
        sweep: async state => ({
          rolloutId: state.id,
          revision: state.revision,
          queueGeneration: state.queueGeneration,
          processGeneration: 1,
          valid: true,
          reason: null,
          admissions: Object.values(state.attempts).map(attempt => ({
            installId: attempt.installId,
            installationFingerprint: attempt.installationFingerprint,
            sourceFingerprint: attempt.sourceFingerprint,
            reviewedSource: attempt.reviewedSource,
            observationGeneration: 1,
            observedAt: '2026-09-22T00:00:00.000Z'
          }))
        })
      }
    }
  }
}

test('reducer admits legal edges and refuses backward launch-state edges', () => {
  let state = runningState()
  for (const action of [
    { kind: 'record-intent' as const, installId: 'canary' },
    { kind: 'launch-authorized' as const, installId: 'canary' },
    { kind: 'launch-observed' as const, installId: 'canary' }
  ]) {
    const transition = reduceManagedRollout(state, action)
    assert.equal(transition.ok, true)
    state = transition.state
  }

  assert.equal(reduceManagedRollout(state, { kind: 'record-intent', installId: 'canary' }).ok, false)
  assert.equal(reduceManagedRollout(state, { kind: 'exclude', installId: 'canary' }).ok, false)
  assert.equal(reduceManagedRollout(state, { kind: 'terminal', installId: 'canary', outcome: 'updated' }).state.phase, 'awaiting-promotion')
})

test('duplicate start returns the original admission and cannot create a second rollout', async () => {
  const fixture = adapters()
  const coordinator = createManagedRolloutCoordinator(createManagedRolloutState(PLAN), fixture.deps)
  const first = coordinator.start('request-1')
  assert.strictEqual(first, coordinator.start('request-1'))
  await first
  await assert.rejects(coordinator.start('request-2'), /rollout-already-started/)
})

test('failed authorization persistence produces no capability or service handoff', async () => {
  const fixture = adapters({ persistAuthorization: async () => Promise.reject(new Error('disk-full')) })
  const coordinator = createManagedRolloutCoordinator(runningState(), fixture.deps)

  const result = await coordinator.authorize('canary')

  assert.equal(result.ok, false)
  assert.match(result.reason || '', /authorization-persist-failed/)
  assert.equal(fixture.persisted.length, 1)
  assert.equal(fixture.launches.length, 0)
  assert.equal(coordinator.snapshot.attempts.canary.state, 'intent-recorded')
})

test('a single committed authorization consumes one scoped handoff and enforces serial launch', async () => {
  const fixture = adapters()
  const coordinator = createManagedRolloutCoordinator(runningState(), fixture.deps)

  assert.equal((await coordinator.authorize('canary')).ok, true)
  assert.equal((await coordinator.authorize('later')).ok, false)
  assert.equal(fixture.persisted.length, 1)
  assert.equal(fixture.launches.length, 1)
  assert.deepEqual(fixture.launches[0], fixture.persisted[0])
})

test('terminal observations require the exact original correlation', async () => {
  const fixture = adapters()
  const coordinator = createManagedRolloutCoordinator(runningState(), fixture.deps)
  await coordinator.authorize('canary')

  const invalid = await coordinator.terminal('canary', 'other-correlation', 'updated')

  assert.equal(invalid.ok, false)
  assert.equal(invalid.reason, 'terminal-correlation-mismatch')
  assert.equal(coordinator.snapshot.attempts.canary.state, 'authorized')
})

test('restart forces explicit continuation and auto cannot bypass the canary', () => {
  let state = runningState()
  for (const action of [
    { kind: 'record-intent' as const, installId: 'canary' },
    { kind: 'launch-authorized' as const, installId: 'canary' },
    { kind: 'terminal' as const, installId: 'canary', outcome: 'updated' },
    { kind: 'restart' as const },
    { kind: 'reconciled' as const, phase: 'awaiting-promotion' as const }
  ]) {
    const transition = reduceManagedRollout(state, action)
    assert.equal(transition.ok, true)
    state = transition.state
  }
  assert.equal(reduceManagedRollout(state, { kind: 'promote', auto: true }).reason, 'auto-promotion-not-admissible')
  assert.equal(reduceManagedRollout(state, { kind: 'promote' }).ok, true)
})
