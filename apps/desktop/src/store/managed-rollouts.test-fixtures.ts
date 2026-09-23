import type { RolloutCapabilities, RolloutSnapshot, TargetAttempt } from '@/lib/managed-rollout-contract'

export const availableCapability: RolloutCapabilities = {
  protocol: 1,
  available: true,
  reason: null,
  maxConcurrency: 1,
  maxInstallations: 500
}

export function rolloutSnapshot(revision: number, options: { id?: string; phase?: RolloutSnapshot['phase']; attempts?: TargetAttempt[] } = {}): RolloutSnapshot {
  return {
    schemaVersion: 1,
    id: options.id ?? '11111111-1111-4111-8111-111111111111',
    revision,
    createdAt: '2026-09-23T00:00:00.000Z',
    updatedAt: '2026-09-23T00:00:01.000Z',
    finishedAt: null,
    retryOf: null,
    archivedAt: null,
    target: {
      repositoryId: 'github.com/NousResearch/hermes-agent',
      branch: 'main',
      sha: 'a'.repeat(40),
      protocol: 1
    },
    phase: options.phase ?? 'running',
    activeWave: 0,
    concurrency: 1,
    promotionPolicy: 'manual',
    canaryApproved: false,
    continuationRequired: false,
    attempts: options.attempts ?? [],
    eventCount: 1
  }
}

export function rolloutAttempt(index: number): TargetAttempt {
  const installId = index.toString(16).padStart(32, '0')

  return {
    identity: {
      connectionId: `connection-${index}`,
      installId,
      aliasConnectionIds: [],
      label: `Installation ${index}`,
      displayAddress: `host-${index}`,
      installationFingerprint: 'b'.repeat(64),
      sourceFingerprint: 'c'.repeat(64),
      admittedSha: 'd'.repeat(40)
    },
    correlationId: `correlation-${index}`,
    wave: index,
    phase: 'queued',
    launchState: 'none',
    requiredScopeIds: [],
    skipReason: null,
    reprobes: 0,
    receipt: null,
    health: null,
    recoveryRequired: false,
    reasons: []
  }
}
