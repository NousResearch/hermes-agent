import { describe, expect, it } from 'vitest'

import type { HealthEvidence, RolloutSnapshot, ScopeEvidence, TargetAttempt } from '../src/lib/managed-rollout-contract'
import {
  canPromote,
  canContinueAfterRestart,
  isExcluded,
  needsManualPromotion,
  promotionContextDigest,
  priorWaveStillValid,
  targetHealthy
} from './managed-rollout-policy'

const SHA = 'a'.repeat(40)
const INSTALL = '1'.repeat(32)

function scope(scopeId: string, codeSha = SHA): ScopeEvidence {
  return {
    scopeId,
    profile: scopeId,
    restored: true,
    ready: true,
    codeSha,
    processIdentityVerified: true
  }
}

function health(
  observationId = 'epoch-1',
  scopes: ScopeEvidence[] = [scope('default')],
  installId = INSTALL
): HealthEvidence {
  return {
    observationId,
    observedAt: '2026-09-21T00:00:00.000Z',
    installId,
    checkoutSha: SHA,
    installReady: true,
    markerClear: true,
    receiptCorrelated: true,
    receiptSucceeded: true,
    dependencyReady: true,
    recoveryClear: true,
    scopeCapture: 'complete',
    scopes,
    reasons: []
  }
}

function attempt(overrides: Partial<TargetAttempt> = {}): TargetAttempt {
  return {
    identity: {
      connectionId: 'conn-a',
      installId: INSTALL,
      aliasConnectionIds: [],
      label: 'A',
      displayAddress: 'host-a',
      installationFingerprint: 'f'.repeat(64),
      sourceFingerprint: 's'.repeat(64),
      admittedSha: SHA
    },
    correlationId: 'corr-a',
    wave: 0,
    phase: 'updated',
    launchState: 'observed',
    requiredScopeIds: ['default'],
    skipReason: null,
    reprobes: 0,
    receipt: {
      correlationId: 'corr-a',
      installId: INSTALL,
      requestedSha: SHA,
      preSha: 'b'.repeat(40),
      postSha: SHA,
      outcome: 'success',
      startedAt: null,
      finishedAt: null,
      stopReason: null
    },
    health: health(),
    recoveryRequired: false,
    reasons: [],
    ...overrides
  }
}

function snapshot(overrides: Partial<RolloutSnapshot> = {}): RolloutSnapshot {
  return {
    schemaVersion: 1,
    id: 'rollout-1',
    revision: 3,
    createdAt: '2026-09-21T00:00:00.000Z',
    updatedAt: '2026-09-21T00:00:00.000Z',
    finishedAt: null,
    retryOf: null,
    archivedAt: null,
    target: { repositoryId: 'github.com/acme/hermes', branch: 'main', sha: SHA, protocol: 1 },
    phase: 'awaiting-promotion',
    activeWave: 0,
    concurrency: 1,
    promotionPolicy: 'manual',
    canaryApproved: true,
    continuationRequired: false,
    attempts: [
      attempt(),
      attempt({
        identity: { ...attempt().identity, connectionId: 'conn-b', installId: '2'.repeat(32), label: 'B' },
        correlationId: 'corr-b',
        wave: 1,
        phase: 'queued',
        launchState: 'none',
        requiredScopeIds: [],
        receipt: null,
        health: null
      })
    ],
    eventCount: 0,
    ...overrides
  }
}

describe('managed rollout policy', () => {
  it('accepts only fresh complete healthy evidence', () => {
    const current = attempt()

    expect(targetHealthy(current, SHA, 'epoch-1')).toBe(true)
    expect(targetHealthy(current, SHA, 'old-epoch')).toBe(false)
    expect(targetHealthy({ ...current, requiredScopeIds: null }, SHA, 'epoch-1')).toBe(false)
    expect(targetHealthy({ ...current, health: health('epoch-1', []) }, SHA, 'epoch-1')).toBe(false)
    expect(
      targetHealthy({ ...current, health: health('epoch-1', [scope('default'), scope('default')]) }, SHA, 'epoch-1')
    ).toBe(false)
  })

  it('distinguishes a known-empty scope set from missing scope evidence', () => {
    const empty = attempt({
      requiredScopeIds: [],
      health: health('epoch-1', [])
    })

    expect(targetHealthy(empty, SHA, 'epoch-1')).toBe(true)
    expect(targetHealthy({ ...empty, requiredScopeIds: null }, SHA, 'epoch-1')).toBe(false)
  })

  it('requires correlated receipt and readiness for an updated target', () => {
    const current = attempt()

    expect(targetHealthy({ ...current, receipt: null }, SHA, 'epoch-1')).toBe(false)
    expect(
      targetHealthy({ ...current, health: health('epoch-1', [scope('default', 'c'.repeat(40))]) }, SHA, 'epoch-1')
    ).toBe(false)
    expect(
      targetHealthy({ ...current, health: { ...health(), reasons: ['dependency-not-ready'] } }, SHA, 'epoch-1')
    ).toBe(false)
  })

  it('allows already-current only with complete readiness evidence', () => {
    const current = attempt({ phase: 'already-current', launchState: 'observed', receipt: null })

    expect(targetHealthy(current, SHA, 'epoch-1')).toBe(true)
    expect(targetHealthy({ ...current, health: { ...health(), dependencyReady: false } }, SHA, 'epoch-1')).toBe(false)
  })

  it('does not turn an executed failed canary into an exclusion', () => {
    expect(isExcluded(attempt({ phase: 'skipped', skipReason: 'operator-excluded', launchState: 'none' }))).toBe(true)
    expect(isExcluded(attempt({ phase: 'failed', skipReason: 'operator-excluded' }))).toBe(false)
    expect(isExcluded(attempt({ phase: 'skipped', skipReason: 'operator-excluded', launchState: 'authorized' }))).toBe(
      false
    )
    expect(isExcluded(attempt({ phase: 'skipped', skipReason: 'operator-excluded', recoveryRequired: true }))).toBe(
      false
    )
  })

  it('requires a manual canary and rejects stale or restarted promotion proofs', () => {
    const current = snapshot()
    const proof = {
      observationId: 'epoch-1',
      rolloutId: current.id,
      revision: current.revision,
      queueGeneration: 4,
      processGeneration: 2,
      evidenceGeneration: 3,
      contextDigest: promotionContextDigest(current),
      wave: current.activeWave,
      sweepStartedMono: 1_000,
      sweepFinishedMono: 2_000,
      nextAdmissionInstallIds: ['2'.repeat(32)],
      approval: 'manual' as const
    }

    expect(needsManualPromotion(current)).toBe(true)
    const context = { processGeneration: 2, evidenceGeneration: 3 }
    expect(canPromote(current, proof, 2_500, 4, context)).toBe(true)
    expect(canPromote(current, proof, 2_500, 4)).toBe(false)
    expect(canPromote(current, proof, 2_500, 4, { ...context, processGeneration: 4 })).toBe(false)
    expect(canPromote(current, proof, 2_500, 4, { ...context, evidenceGeneration: 4 })).toBe(false)
    expect(canPromote(current, { ...proof, contextDigest: 'stale' }, 2_500, 4, context)).toBe(false)
    expect(canPromote({ ...current, promotionPolicy: 'auto-if-healthy' }, proof, 2_500, 4, context)).toBe(false)
    expect(canPromote({ ...current, canaryApproved: false }, proof, 2_500, 4, context)).toBe(false)
    expect(canPromote(current, { ...proof, approval: 'automatic' }, 2_500, 4, context)).toBe(false)
    expect(canPromote({ ...current, continuationRequired: true }, proof, 2_500, 4, context)).toBe(false)
    expect(canPromote(current, { ...proof, observationId: 'stale' }, 2_500, 4, context)).toBe(false)
    expect(canPromote(current, { ...proof, revision: current.revision + 1 }, 2_500, 4, context)).toBe(false)
    expect(canPromote(current, { ...proof, queueGeneration: 5 }, 2_500, 4, context)).toBe(false)
    expect(canPromote(current, proof, 12_001, 4, context)).toBe(false)
    expect(canPromote(current, { ...proof, sweepFinishedMono: 301_001 }, 301_500, 4, context)).toBe(false)
  })

  it('requires explicit continuation after restart reconciliation', () => {
    const current = snapshot()

    expect(canContinueAfterRestart(current)).toBe(true)
    expect(canContinueAfterRestart({ ...current, continuationRequired: true })).toBe(false)
    expect(canContinueAfterRestart({ ...current, phase: 'reconciling' })).toBe(false)
  })

  it('uses retained local settlement evidence for prior waves instead of requiring a new epoch', () => {
    const prior = attempt({ health: health('old-epoch') })
    const settled = attempt({
      identity: { ...attempt().identity, connectionId: 'conn-b', installId: '2'.repeat(32), label: 'B' },
      correlationId: 'corr-b',
      wave: 1,
      receipt: { ...attempt().receipt!, installId: '2'.repeat(32), correlationId: 'corr-b' },
      health: health('epoch-2', [scope('default')], '2'.repeat(32))
    })
    const next = attempt({
      identity: { ...attempt().identity, connectionId: 'conn-c', installId: '3'.repeat(32), label: 'C' },
      correlationId: 'corr-c',
      wave: 2,
      phase: 'queued',
      launchState: 'none',
      requiredScopeIds: [],
      receipt: null,
      health: null
    })
    const current = snapshot({
      activeWave: 1,
      attempts: [prior, settled, next]
    })
    const proof = {
      observationId: 'epoch-2',
      rolloutId: current.id,
      revision: current.revision,
      queueGeneration: 4,
      processGeneration: 2,
      evidenceGeneration: 3,
      contextDigest: promotionContextDigest(current),
      wave: current.activeWave,
      sweepStartedMono: 1_000,
      sweepFinishedMono: 2_000,
      nextAdmissionInstallIds: ['3'.repeat(32)],
      approval: 'manual' as const
    }

    expect(priorWaveStillValid(prior, SHA)).toBe(true)
    expect(targetHealthy(settled, SHA, 'epoch-2')).toBe(true)
    expect(canPromote(current, proof, 2_500, 4, { processGeneration: 2, evidenceGeneration: 3 })).toBe(true)
  })

  it('accepts the manual canary approval before canaryApproved is set', () => {
    const current = snapshot({ canaryApproved: false })
    const proof = {
      observationId: 'epoch-1',
      rolloutId: current.id,
      revision: current.revision,
      queueGeneration: 4,
      processGeneration: 2,
      evidenceGeneration: 3,
      contextDigest: promotionContextDigest(current),
      wave: current.activeWave,
      sweepStartedMono: 1_000,
      sweepFinishedMono: 2_000,
      nextAdmissionInstallIds: ['2'.repeat(32)],
      approval: 'manual' as const
    }

    expect(canPromote(current, proof, 2_500, 4, { processGeneration: 2, evidenceGeneration: 3 })).toBe(true)
  })
})
