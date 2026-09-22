import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  reduceManagedRollout,
  type ManagedRolloutAuthorization,
  type ManagedRolloutPlan
} from './managed-rollout-coordinator'

const PLAN: ManagedRolloutPlan = {
  id: '12345678-1234-4678-9234-567812345678',
  revision: 4,
  queueGeneration: 9,
  policy: 'auto-after-canary',
  targets: [
    { installId: 'canary', installationFingerprint: 'install-a', connectionId: 'ssh-canary', sourceFingerprint: 'source-a', targetSha: 'a'.repeat(40), reviewedSource: reviewedSource('a'.repeat(40)), correlationId: 'canary-correlation', wave: 0 },
    { installId: 'later-a', installationFingerprint: 'install-b', connectionId: 'ssh-later-a', sourceFingerprint: 'source-b', targetSha: 'a'.repeat(40), reviewedSource: reviewedSource('a'.repeat(40)), correlationId: 'later-a-correlation', wave: 1 },
    { installId: 'later-b', installationFingerprint: 'install-c', connectionId: 'ssh-later-b', sourceFingerprint: 'source-c', targetSha: 'a'.repeat(40), reviewedSource: reviewedSource('a'.repeat(40)), correlationId: 'later-b-correlation', wave: 1 }
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

function proof(state: ReturnType<typeof createManagedRolloutState>, overrides: Partial<{ queueGeneration: number; valid: boolean }> = {}) {
  return {
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
    })),
    ...overrides
  }
}

function running() {
  const start = reduceManagedRollout(createManagedRolloutState(PLAN), { kind: 'start' })
  assert.equal(start.ok, true)
  return start.state
}

function createFixture() {
  const launches: ManagedRolloutAuthorization[] = []
  const coordinator = createManagedRolloutCoordinator(running(), {
    journal: { persistAuthorization: async () => {} },
    service: {
      issueCapability: authorization => Object.freeze({ authorization }),
      launch: async authorization => {
        launches.push(authorization)
      }
    },
    evidence: {
      sweep: async state => proof(state)
    }
  })
  return { coordinator, launches }
}

test('pause before authorization preserves pending work and prevents transport mutation', async () => {
  const fixture = createFixture()
  await fixture.coordinator.command({ kind: 'pause' })
  const refused = await fixture.coordinator.authorize('canary')

  assert.equal(refused.ok, false)
  assert.equal(fixture.coordinator.snapshot.attempts.canary.state, 'none')
  assert.deepEqual(fixture.launches, [])
})

test('stop before authorization skips pending rows but never erases an intent record', () => {
  let state = running()
  state = reduceManagedRollout(state, { kind: 'record-intent', installId: 'canary' }).state
  const stopped = reduceManagedRollout(state, { kind: 'stop' })

  assert.equal(stopped.ok, true)
  assert.equal(stopped.state.attempts.canary.state, 'cancelled-before-launch')
  assert.equal(stopped.state.attempts['later-a'].state, 'skipped')
  assert.equal(stopped.state.phase, 'stopped')
})

test('stop admitted after authorization waits for committed work and never authorizes the next row', async () => {
  const fixture = createFixture()
  assert.equal((await fixture.coordinator.authorize('canary')).ok, true)
  assert.equal((await fixture.coordinator.command({ kind: 'stop' })).ok, true)
  assert.equal((await fixture.coordinator.authorize('later-a')).ok, false)
  assert.equal((await fixture.coordinator.terminal('canary', 'canary-correlation', 'updated')).state.phase, 'stopped')
  assert.equal(fixture.launches.length, 1)
})

test('manual promotion sweeps first and pause during a sweep invalidates its proof', async () => {
  const fixture = createFixture()
  let state = fixture.coordinator.snapshot
  for (const action of [
    { kind: 'record-intent' as const, installId: 'canary' },
    { kind: 'launch-authorized' as const, installId: 'canary' },
    { kind: 'terminal' as const, installId: 'canary', outcome: 'updated' as const }
  ]) state = reduceManagedRollout(state, action).state
  const reviewed = createManagedRolloutCoordinator(state, {
    journal: { persistAuthorization: async () => {} },
    service: { issueCapability: () => ({}), launch: async () => {} },
    evidence: { sweep: async state => proof(state) }
  })
  const promotion = reviewed.promote(false)
  await reviewed.command({ kind: 'pause' })

  assert.equal((await promotion).ok, false)
  assert.equal(reviewed.snapshot.phase, 'paused')
})

test('canary cannot be auto-promoted and safe exclusion cannot reset an attempted row', () => {
  let state = running()
  for (const action of [
    { kind: 'record-intent' as const, installId: 'canary' },
    { kind: 'launch-authorized' as const, installId: 'canary' },
    { kind: 'terminal' as const, installId: 'canary', outcome: 'updated' as const }
  ]) state = reduceManagedRollout(state, action).state

  assert.equal(reduceManagedRollout(state, { kind: 'promote', auto: true }).ok, false)
  state = reduceManagedRollout(state, { kind: 'promote' }).state
  state = reduceManagedRollout(state, { kind: 'exclude', installId: 'later-a' }).state
  assert.equal(state.attempts['later-a'].state, 'skipped')
  assert.equal(reduceManagedRollout(state, { kind: 'record-intent', installId: 'later-a' }).ok, false)
})

test('missed acknowledgement remains authorized and restart retains an unresolved fence', () => {
  let state = running()
  state = reduceManagedRollout(state, { kind: 'record-intent', installId: 'canary' }).state
  state = reduceManagedRollout(state, { kind: 'launch-authorized', installId: 'canary' }).state
  state = reduceManagedRollout(state, { kind: 'restart' }).state

  assert.equal(state.phase, 'reconciling')
  assert.equal(state.attempts.canary.state, 'authorized')
  assert.equal(state.continuationRequired, true)
})

test('a stale sweep proof cannot authorize a promotion', async () => {
  let state = running()
  for (const action of [
    { kind: 'record-intent' as const, installId: 'canary' },
    { kind: 'launch-authorized' as const, installId: 'canary' },
    { kind: 'terminal' as const, installId: 'canary', outcome: 'updated' as const }
  ]) state = reduceManagedRollout(state, action).state
  const coordinator = createManagedRolloutCoordinator(state, {
    journal: { persistAuthorization: async () => {} },
    service: { issueCapability: () => ({}), launch: async () => {} },
    evidence: { sweep: async current => proof(current, { queueGeneration: 0 }) }
  })

  const result = await coordinator.promote()

  assert.equal(result.ok, false)
  assert.equal(result.reason, 'promotion-proof-is-stale-or-invalid')
})

test('a losing controller records no handoff after the journal rejects its authorization', async () => {
  let owner: string | null = null
  const launches: string[] = []
  const make = (id: string) =>
    createManagedRolloutCoordinator(running(), {
      journal: {
        persistAuthorization: async () => {
          if (owner && owner !== id) throw new Error('foreign-update-owner')
          owner = id
        }
      },
      service: {
        issueCapability: () => Object.freeze({}),
        launch: async () => {
          launches.push(id)
        }
      },
      evidence: { sweep: async state => proof(state) }
    })
  const winner = make('winner')
  const loser = make('loser')

  assert.equal((await winner.authorize('canary')).ok, true)
  assert.equal((await loser.authorize('canary')).ok, false)
  assert.equal(loser.snapshot.attempts.canary.state, 'unverified')
  assert.deepEqual(launches, ['winner'])
})
