import { describe, expect, it } from 'vitest'

import {
  validateHealthEvidence,
  validateReceiptSummary,
  validateRolloutCapabilities,
  validateRolloutPlan,
  validateReviewedSourceBinding,
  validateScopeEvidence,
  validateTargetIdentity
} from './managed-rollout-contract'

const SHA = 'a'.repeat(40)
const INSTALL = '1'.repeat(32)
const FINGERPRINT = 'f'.repeat(64)

function reviewedSource() {
  return {
    repositoryRoot: 'C:/reviewed/hermes',
    originUrl: 'https://github.com/acme/hermes.git',
    resolvedRef: 'refs/remotes/origin/main',
    targetSha: SHA,
    assuranceProfile: 'managed-ssh',
    assuranceEvidenceSha256: FINGERPRINT,
    assuranceGeneration: 7
  }
}

function plan() {
  return {
    target: { repositoryId: 'github.com/acme/hermes', branch: 'main', sha: SHA, protocol: 1 as const },
    waves: [[INSTALL]],
    concurrency: 1,
    promotionPolicy: 'manual' as const,
    rows: [
      {
        installId: INSTALL,
        connectionId: 'connection-a',
        installationFingerprint: FINGERPRINT,
        sourceFingerprint: FINGERPRINT,
        admittedHead: SHA,
        requiredScopeIds: [],
        eligible: true
      }
    ],
    retryOf: null,
    exclusions: []
  }
}

function health() {
  return {
    observationId: 'epoch-1',
    observedAt: '2026-09-21T00:00:00.000Z',
    installId: INSTALL,
    checkoutSha: SHA,
    installReady: true,
    markerClear: true,
    receiptCorrelated: true,
    receiptSucceeded: true,
    dependencyReady: true,
    recoveryClear: true,
    scopeCapture: 'complete' as const,
    scopes: [],
    reasons: []
  }
}

describe('managed rollout runtime contract', () => {
  it('validates the exact Python reviewed source shape without accepting missing fields', () => {
    expect(validateReviewedSourceBinding(reviewedSource())).toEqual(reviewedSource())
    expect(() => validateReviewedSourceBinding({ ...reviewedSource(), assuranceEvidenceSha256: undefined })).toThrow()
    expect(() => validateReviewedSourceBinding({ ...reviewedSource(), trustedByRenderer: true })).toThrow('unknown field')
    expect(() => validateReviewedSourceBinding({ ...reviewedSource(), targetSha: 'a'.repeat(39) })).toThrow()
    expect(() => validateReviewedSourceBinding({ ...reviewedSource(), repositoryRoot: 'relative/repo' })).toThrow()
  })

  it('rejects missing authority fields and unknown fields', () => {
    const value = plan() as Record<string, unknown>
    delete value.target
    expect(() => validateRolloutPlan(value)).toThrow('missing authority field')

    const withUnknown = { ...plan(), rendererSaysHealthy: true }
    expect(() => validateRolloutPlan(withUnknown)).toThrow('unknown field')
  })

  it('requires every plan row to belong to exactly one wave or exclusion', () => {
    expect(() => validateRolloutPlan({ ...plan(), waves: [[]] })).toThrow('waves must be non-empty')
    expect(() => validateRolloutPlan({ ...plan(), waves: [[]], exclusions: [INSTALL] })).toThrow(
      'waves must be non-empty'
    )
    expect(() => validateRolloutPlan({ ...plan(), waves: [['other-install']] })).toThrow('unknown installation')
    expect(() => validateRolloutPlan({ ...plan(), waves: [[INSTALL]], exclusions: [INSTALL] })).toThrow(
      'still assigned'
    )
  })

  it('keeps missing scope capture distinct from a known-empty capture', () => {
    expect(validateHealthEvidence(health()).scopeCapture).toBe('complete')
    expect(validateHealthEvidence({ ...health(), scopeCapture: 'missing' as const }).scopeCapture).toBe('missing')
    expect(() =>
      validateHealthEvidence({
        ...health(),
        scopeCapture: 'missing' as const,
        scopes: [
          {
            scopeId: 'default',
            profile: 'default',
            restored: true,
            ready: true,
            codeSha: SHA,
            processIdentityVerified: true
          }
        ]
      })
    ).toThrow('missing capture')
  })

  it('bounds advertised capability instead of accepting an unavailable parallel draft', () => {
    expect(() =>
      validateRolloutCapabilities({
        protocol: 1,
        available: false,
        reason: 'capacity-unverified',
        maxConcurrency: 1,
        maxInstallations: 0
      })
    ).toThrow('zero concurrency')
    expect(() =>
      validateRolloutCapabilities({
        protocol: 1,
        available: true,
        reason: null,
        maxConcurrency: 4,
        maxInstallations: 501
      })
    ).toThrow('release bounds')
    expect(
      validateRolloutCapabilities({
        protocol: 1,
        available: true,
        reason: null,
        maxConcurrency: 1,
        maxInstallations: 1
      }).maxConcurrency
    ).toBe(1)
  })

  it('rejects malformed identity, scope, receipt, and target evidence fields', () => {
    expect(() =>
      validateTargetIdentity({
        connectionId: 'connection-a',
        installId: 'Z'.repeat(32),
        aliasConnectionIds: [],
        label: 'A',
        displayAddress: 'host-a',
        installationFingerprint: FINGERPRINT,
        sourceFingerprint: FINGERPRINT,
        admittedSha: SHA
      })
    ).toThrow('install id')
    expect(() =>
      validateScopeEvidence({
        scopeId: 'default',
        profile: '',
        restored: true,
        ready: true,
        codeSha: SHA,
        processIdentityVerified: true
      })
    ).toThrow('bounded string')
    expect(() =>
      validateReceiptSummary({
        correlationId: 'corr-a',
        installId: INSTALL,
        requestedSha: 'not-a-sha',
        preSha: null,
        postSha: SHA,
        outcome: 'success',
        startedAt: null,
        finishedAt: null,
        stopReason: null
      })
    ).toThrow('40-character SHA')
    expect(() => validateRolloutPlan({ ...plan(), target: { ...plan().target, sha: 'not-a-sha' } })).toThrow(
      '40-character SHA'
    )
  })
})
