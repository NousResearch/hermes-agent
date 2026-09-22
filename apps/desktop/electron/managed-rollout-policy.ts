import crypto from 'node:crypto'

import type { RolloutSnapshot, TargetAttempt } from '../src/lib/managed-rollout-contract'

export interface PromotionProof {
  observationId: string
  rolloutId: string
  revision: number
  queueGeneration: number
  processGeneration: number
  evidenceGeneration: number
  contextDigest: string
  wave: number
  sweepStartedMono: number
  sweepFinishedMono: number
  nextAdmissionInstallIds: string[]
  approval: 'manual' | 'automatic'
}

export const SWEEP_DEADLINE_MS = 5 * 60 * 1000
export const PROMOTION_PROOF_MAX_AGE_MS = 10 * 1000

/** Bind the sweep to the exact reviewed policy, source, installation and scopes. */
export function promotionContextDigest(snapshot: RolloutSnapshot): string {
  return crypto.createHash('sha256').update(JSON.stringify([
    snapshot.id, snapshot.revision, snapshot.activeWave, snapshot.promotionPolicy,
    snapshot.canaryApproved, snapshot.target,
    snapshot.attempts.map(attempt => [
      attempt.identity.installId, attempt.identity.installationFingerprint,
      attempt.identity.sourceFingerprint, attempt.identity.admittedSha,
      attempt.wave, attempt.requiredScopeIds === null ? null : [...attempt.requiredScopeIds].sort(),
      attempt.recoveryRequired
    ])
  ])).digest('hex')
}

function unique(values: readonly string[]): boolean {
  return new Set(values).size === values.length
}

export function isExcluded(attempt: TargetAttempt): boolean {
  return (
    attempt.phase === 'skipped' &&
    attempt.skipReason === 'operator-excluded' &&
    attempt.launchState === 'none' &&
    !attempt.recoveryRequired
  )
}

export function isSafeExclusion(attempt: TargetAttempt): boolean {
  return (
    isExcluded(attempt) ||
    (attempt.launchState === 'none' &&
      attempt.phase === 'refused' &&
      !attempt.recoveryRequired &&
      attempt.skipReason === null)
  )
}

/**
 * Health is bound to the exact observation epoch supplied by the sweep. A
 * responding endpoint, a matching SHA, or a cached receipt alone is not a
 * healthy target.
 */
export function targetHealthy(attempt: TargetAttempt, sha: string, observationId: string): boolean {
  const health = attempt.health

  if (!health || !observationId || health.observationId !== observationId) return false
  if (attempt.phase !== 'updated' && attempt.phase !== 'already-current') return false
  if (attempt.identity.admittedSha !== sha || attempt.requiredScopeIds === null || attempt.recoveryRequired)
    return false
  if (!unique(attempt.requiredScopeIds)) return false

  const observedScopes = health.scopes.map(scope => scope.scopeId)

  if (!unique(observedScopes)) return false
  if (attempt.requiredScopeIds.length !== observedScopes.length) return false
  if (!attempt.requiredScopeIds.every(scopeId => observedScopes.includes(scopeId))) return false
  if (health.installId !== attempt.identity.installId || health.checkoutSha !== sha) return false
  if (health.scopeCapture !== 'complete') return false
  if (!health.installReady || !health.markerClear || !health.recoveryClear || !health.dependencyReady) return false
  if (health.reasons.length > 0) return false
  if (
    !health.scopes.every(
      scope => scope.restored && scope.ready && scope.codeSha === sha && scope.processIdentityVerified
    )
  ) {
    return false
  }

  if (attempt.phase === 'already-current') return true

  const receipt = attempt.receipt

  return Boolean(
    receipt &&
    health.receiptCorrelated &&
    health.receiptSucceeded &&
    receipt.outcome === 'success' &&
    receipt.correlationId === attempt.correlationId &&
    receipt.installId === attempt.identity.installId &&
    receipt.requestedSha === sha &&
    receipt.postSha === sha
  )
}

function finiteMonotonic(value: number): boolean {
  return Number.isFinite(value) && value >= 0
}

/**
 * Consume only a fresh proof produced for this exact revision and queue
 * generation. The proof is intentionally a narrow internal predicate, not an
 * IPC authorization surface.
 */
export function canPromote(
  snapshot: RolloutSnapshot,
  proof: PromotionProof,
  nowMono: number,
  queueGeneration: number,
  context?: { processGeneration: number; evidenceGeneration: number }
): boolean {
  if (!context || !Number.isSafeInteger(context.processGeneration) || !Number.isSafeInteger(context.evidenceGeneration))
    return false
  if (snapshot.phase !== 'awaiting-promotion' || snapshot.continuationRequired) return false
  if (!proof.observationId || proof.rolloutId !== snapshot.id || proof.revision !== snapshot.revision) return false
  if (proof.wave !== snapshot.activeWave || proof.queueGeneration !== queueGeneration) return false
  if (proof.processGeneration !== context.processGeneration || proof.evidenceGeneration !== context.evidenceGeneration)
    return false
  if (proof.contextDigest !== promotionContextDigest(snapshot)) return false
  if (snapshot.activeWave === 0 && !snapshot.canaryApproved) return false
  if (snapshot.activeWave === 0 && snapshot.promotionPolicy === 'auto-if-healthy') return false
  if (snapshot.activeWave === 0 && proof.approval !== 'manual') return false
  if (snapshot.activeWave > 0 && snapshot.promotionPolicy === 'manual' && proof.approval !== 'manual') return false
  if (snapshot.activeWave > 0 && snapshot.promotionPolicy === 'auto-if-healthy' && proof.approval !== 'automatic')
    return false
  if (snapshot.activeWave > 0 && !snapshot.canaryApproved) return false
  if (![nowMono, proof.sweepStartedMono, proof.sweepFinishedMono].every(finiteMonotonic)) return false
  if (proof.sweepFinishedMono < proof.sweepStartedMono) return false
  if (proof.sweepFinishedMono - proof.sweepStartedMono > SWEEP_DEADLINE_MS) return false
  if (nowMono < proof.sweepFinishedMono || nowMono - proof.sweepFinishedMono > PROMOTION_PROOF_MAX_AGE_MS) return false
  if (!unique(proof.nextAdmissionInstallIds)) return false

  const completed = snapshot.attempts.filter(attempt => attempt.wave <= snapshot.activeWave && !isExcluded(attempt))
  const next = snapshot.attempts.filter(attempt => attempt.wave === snapshot.activeWave + 1 && !isExcluded(attempt))

  if (!completed.length || !next.length) return false
  if (
    !next.every(
      attempt => attempt.phase === 'queued' && attempt.launchState === 'none' && attempt.requiredScopeIds !== null
    )
  )
    return false

  const admitted = new Set(proof.nextAdmissionInstallIds)

  if (admitted.size !== proof.nextAdmissionInstallIds.length || admitted.size !== next.length) return false
  if (!next.every(attempt => admitted.has(attempt.identity.installId))) return false

  return completed.every(attempt => targetHealthy(attempt, snapshot.target.sha, proof.observationId))
}

export function needsManualPromotion(snapshot: RolloutSnapshot): boolean {
  return (
    snapshot.activeWave === 0 ||
    !snapshot.canaryApproved ||
    snapshot.promotionPolicy === 'manual' ||
    snapshot.continuationRequired
  )
}

export function canContinueAfterRestart(snapshot: RolloutSnapshot): boolean {
  return !snapshot.continuationRequired && snapshot.phase !== 'reconciling'
}

export function launchStateRank(state: TargetAttempt['launchState']): number {
  return ({ none: 0, 'intent-recorded': 1, authorized: 2, observed: 3 } as const)[state]
}

export function preservesLaunchState(
  previous: TargetAttempt['launchState'],
  next: TargetAttempt['launchState']
): boolean {
  return launchStateRank(next) >= launchStateRank(previous)
}
