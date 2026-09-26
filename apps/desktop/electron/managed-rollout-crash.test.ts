import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  type ManagedRolloutPlan,
  reduceManagedRollout
} from './managed-rollout-coordinator'
import {
  ManagedConnectionUpdateGate,
  type ManagedConnectionUpdateResult,
  type ManagedSshUpdateIntent,
  type RemoteUpdateTarget
} from './managed-ssh-update'
import {
  createManagedSshUpdateService,
  type ManagedSshLaunchCapability,
  type ManagedSshUpdateScope,
  type ManagedSshUpdateSource
} from './managed-ssh-update-service'

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
    evidence: { sweep: async current => ({ rolloutId: current.id, revision: current.revision, queueGeneration: current.queueGeneration, processGeneration: 1, completedMono: 0, priorWaveClear: false, nextAdmissionInstallIds: [], valid: false, reason: 'not-needed', admissions: [] }) },
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

test.each(['failed', 'refused', 'updated', 'already-current'] as const)(
  'recovery after a %s terminal observation can clear original scope without changing outcome or dispatching again',
  async outcome => {
    const calls: string[] = []
    const state = reduceManagedRollout(authorizedState(), { kind: 'terminal', installId: 'canary', outcome }).state
    const restored = coordinator(state, calls)

    const result = await restored.recover('canary')

    assert.equal(result.ok, true)
    assert.equal(restored.snapshot.attempts.canary.state, outcome)
    assert.deepEqual(calls, [])
  }
)

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
    evidence: { sweep: async current => ({ rolloutId: current.id, revision: current.revision, queueGeneration: current.queueGeneration, processGeneration: 1, completedMono: 0, priorWaveClear: false, nextAdmissionInstallIds: [], valid: false, reason: 'not-needed', admissions: [] }) },
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

// S11.5 — Executed two-controller receipt over the shared service seam. Two
// independent controllers hold the same rollout and installation identity; the
// winner's authorization owns the real update transaction, while the loser can
// neither create a second launch, settle the winner's correlation, nor restore
// the winner's scopes while that transaction is in flight.

const TWO_CONTROLLER_INSTALL = 'e'.repeat(32)
const TWO_CONTROLLER_CORRELATION = '44444444-4444-4444-8444-444444444444'
const LOSER_CORRELATION = '55555555-5555-4555-8555-555555555555'

interface SharedScope extends ManagedSshUpdateScope {
  state?: object | null
}

interface SharedSource extends ManagedSshUpdateSource {
  label: string
}

function sharedTarget(): RemoteUpdateTarget {
  return {
    ssh: { exec: async () => '' },
    platform: 'Linux',
    hermesPath: '/srv/hermes-agent/.venv/bin/hermes',
    hermesHome: '~/.hermes'
  }
}

test('the losing controller neither owns the winner receipt nor restores the winner scopes', async () => {
  const source: SharedSource = { id: 'shared-ssh', kind: 'ssh', label: 'shared' }
  const scope: SharedScope = { key: 'scope-main', profile: 'default' }
  const gate = new ManagedConnectionUpdateGate(() => null)
  const activeUpdates = new Map<string, Promise<ManagedConnectionUpdateResult>>()
  const primaryRestoreOwners = new Map<string, { correlationId: string; profile: string; source: SharedSource }>()
  const restoredBy: string[] = []
  const verifiedInstallIds: string[] = []
  let releaseUpdate!: () => void
  const pendingUpdate = new Promise<void>(resolve => {
    releaseUpdate = resolve
  })

  const intent: ManagedSshUpdateIntent = {
    targetSha: SOURCE.targetSha,
    expectedInstallId: TWO_CONTROLLER_INSTALL,
    expectedCurrentSha: '0'.repeat(40),
    source: {
      repositoryRoot: SOURCE.repositoryRoot,
      originUrl: SOURCE.originUrl,
      resolvedRef: SOURCE.resolvedRef,
      targetSha: SOURCE.targetSha,
      assuranceProfile: SOURCE.assuranceProfile,
      assuranceEvidenceSha256: SOURCE.assuranceEvidenceSha256,
      assuranceGeneration: SOURCE.assuranceGeneration
    }
  }
  const expectedSource = {
    installId: TWO_CONTROLLER_INSTALL,
    installationFingerprint: 'f'.repeat(64),
    sourceFingerprint: '1'.repeat(64)
  }

  const service = createManagedSshUpdateService<SharedSource, SharedScope>({
    gate,
    activeUpdates,
    activeRecoveries: new Map(),
    primaryRestoreOwners,
    resolveSource: id => (id === source.id ? source : null),
    resolveInstallationId: async () => TWO_CONTROLLER_INSTALL,
    readRecoveryRecords: () => [],
    captureScopes: async () => [scope],
    openTransport: async () => ({ target: sharedTarget(), close: async () => undefined }),
    targetFromState: () => sharedTarget(),
    executeRemoteUpdate: async (_target, correlationId, context) => {
      await context.beforeLaunchDispatch()
      await pendingUpdate

      return {
        exitCode: 0,
        receipt: {
          correlationId,
          outcome: 'success',
          preSha: '0'.repeat(40),
          postSha: SOURCE.targetSha,
          startedAt: '2026-09-26T00:00:00.000Z',
          finishedAt: '2026-09-26T00:01:00.000Z'
        }
      }
    },
    verifyCoordinatorSource: async (_selected, _target, expected) => {
      verifiedInstallIds.push(expected.installId)
    },
    preflightRemote: async () => undefined,
    awaitRestoreClearance: async () => undefined,
    drainScope: async () => undefined,
    closeTransports: async () => undefined,
    restoreScope: async (_selected, _transportSource, correlationId) => {
      restoredBy.push(correlationId)
    },
    prepareRecovery: async () => undefined,
    completeRecovery: async () => undefined,
    restoreRecoveryScope: async () => undefined
  })

  const targetsFor = (correlationId: string) => [{
    installId: TWO_CONTROLLER_INSTALL,
    installationFingerprint: expectedSource.installationFingerprint,
    connectionId: source.id,
    sourceFingerprint: expectedSource.sourceFingerprint,
    targetSha: SOURCE.targetSha,
    reviewedSource: SOURCE,
    correlationId,
    wave: 0
  }]
  const winnerPlan: ManagedRolloutPlan = { ...PLAN, targets: targetsFor(TWO_CONTROLLER_CORRELATION) }
  const loserPlan: ManagedRolloutPlan = { ...PLAN, targets: targetsFor(LOSER_CORRELATION) }

  let winnerResult: Promise<ManagedConnectionUpdateResult> | null = null
  const winner = createManagedRolloutCoordinator(reduceManagedRollout(createManagedRolloutState(winnerPlan), { kind: 'start' }).state, {
    journal: { persistAuthorization: async () => undefined },
    service: {
      issueCapability: authorization => service.issueLaunchCapability(authorization.connectionId, authorization.correlationId, intent, expectedSource),
      launch: (authorization, capability) => {
        const admission = service.requestCoordinator(authorization.connectionId, {
          correlationId: authorization.correlationId,
          intent,
          launchCapability: capability as ManagedSshLaunchCapability,
          expectedSource
        })

        assert.equal(admission.admitted, true)
        winnerResult = admission.operation

        return admission.operation.then(() => undefined)
      }
    },
    evidence: {
      sweep: async current => ({
        rolloutId: current.id,
        revision: current.revision,
        queueGeneration: current.queueGeneration,
        processGeneration: 1,
        completedMono: 0,
        priorWaveClear: false,
        nextAdmissionInstallIds: [],
        valid: false,
        reason: 'not-needed',
        admissions: []
      })
    }
  })

  let loserRefusal: string | null = null
  const loser = createManagedRolloutCoordinator(reduceManagedRollout(createManagedRolloutState(loserPlan), { kind: 'start' }).state, {
    journal: { persistAuthorization: async () => undefined },
    service: {
      issueCapability: authorization => service.issueLaunchCapability(authorization.connectionId, authorization.correlationId, intent, expectedSource),
      launch: (authorization, capability) => {
        const admission = service.requestCoordinator(authorization.connectionId, {
          correlationId: authorization.correlationId,
          intent,
          launchCapability: capability as ManagedSshLaunchCapability,
          expectedSource
        })

        if (admission.admitted === false) {
          loserRefusal = admission.reason
          throw new Error(admission.reason)
        }

        return admission.operation.then(() => undefined)
      }
    },
    evidence: {
      sweep: async current => ({
        rolloutId: current.id,
        revision: current.revision,
        queueGeneration: current.queueGeneration,
        processGeneration: 1,
        completedMono: 0,
        priorWaveClear: false,
        nextAdmissionInstallIds: [],
        valid: false,
        reason: 'not-needed',
        admissions: []
      })
    }
  })

  // The winner's authorization owns the real update transaction.
  const winnerAuthorized = await winner.authorize(TWO_CONTROLLER_INSTALL)

  assert.equal(winnerAuthorized.ok, true)
  assert.equal(winner.snapshot.attempts[TWO_CONTROLLER_INSTALL].state, 'authorized')
  assert.equal(gate.owner(source.id), TWO_CONTROLLER_CORRELATION)
  assert.equal(activeUpdates.size, 1)

  // The loser cannot create a second launch for the winner's connection.
  const loserAuthorized = await loser.authorize(TWO_CONTROLLER_INSTALL)

  assert.equal(loserAuthorized.ok, false)
  assert.match(String(loserAuthorized.reason), /service-handoff-failed.*already in progress/)
  assert.equal(loserRefusal, 'A managed update is already in progress.')
  assert.equal(loser.snapshot.attempts[TWO_CONTROLLER_INSTALL].state, 'unverified')
  assert.equal(activeUpdates.size, 1)

  // The loser cannot settle the winner's receipt correlation.
  const stolen = await loser.terminal(TWO_CONTROLLER_INSTALL, TWO_CONTROLLER_CORRELATION, 'updated')

  assert.equal(stolen.ok, false)
  assert.equal(stolen.reason, 'terminal-correlation-mismatch')
  assert.equal(loser.snapshot.attempts[TWO_CONTROLLER_INSTALL].state, 'unverified')

  // The loser cannot restore the winner's scopes while the transaction owns
  // the connection.
  let loserRestoreRan = false

  await assert.rejects(
    service.restorePrimary(source, 'default', LOSER_CORRELATION, async () => {
      loserRestoreRan = true

      return 'loser-restored'
    }),
    /is paused while its managed update is in progress/
  )

  assert.equal(loserRestoreRan, false)
  assert.equal(primaryRestoreOwners.size, 0)

  releaseUpdate()
  const settled = await winnerResult!

  assert.equal(settled.ok, true)
  assert.equal(settled.outcome, 'updated')
  assert.equal(settled.receipt?.correlationId, TWO_CONTROLLER_CORRELATION)
  assert.deepEqual(restoredBy, [TWO_CONTROLLER_CORRELATION])
  assert.deepEqual(verifiedInstallIds, [TWO_CONTROLLER_INSTALL])
  assert.equal(gate.owner(source.id), null)
  assert.equal(activeUpdates.size, 0)

  const winnerSettlement = await winner.terminal(TWO_CONTROLLER_INSTALL, TWO_CONTROLLER_CORRELATION, 'updated')

  assert.equal(winnerSettlement.ok, true)
  assert.equal(winner.snapshot.attempts[TWO_CONTROLLER_INSTALL].state, 'updated')
})
