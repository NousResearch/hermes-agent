import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  createManagedRolloutProvider,
  type ManagedRolloutProviderDependencies,
  type ManagedRolloutObservation
} from './managed-rollout-provider'
import { createManagedRolloutJournal } from './managed-rollout-journal'
import {
  installationFingerprint,
  sourceFingerprint,
  type SourceFingerprintInput
} from './managed-rollout-identity'
import type { TrustedSourceReader } from './managed-rollout-assurance'
import type { TrustedInventorySnapshot } from './managed-rollout-inventory'
import type { TargetResolution } from './managed-rollout-preflight'
import type {
  HealthEvidence,
  RolloutPlan,
  RolloutTarget,
  ReviewedSourceBinding
} from '../src/lib/managed-rollout-contract'

const NOW = 1_700_000_000_000
const NOW_ISO = new Date(NOW).toISOString()
const NOW_MONO = 1_000
const INSTALL_ID = 'a'.repeat(32)
const CONNECTION_ID = '11111111-1111-4111-8111-111111111111'
const TARGET_SHA = 'b'.repeat(40)
const ADMITTED_SHA = 'c'.repeat(40)
const REPOSITORY_ID = 'github.com/nousresearch/hermes-agent'
const ORIGIN = 'https://github.com/NousResearch/hermes-agent.git'
const CODE_ROOT = '/srv/hermes-agent'
const PROFILE = 'managed-ssh-review-v1'
const ASSURANCE_CONTROL_RECEIPT = 'd'.repeat(64)

function assuranceEnvelope(): Uint8Array {
  const document = {
    schema: 1,
    profile: PROFILE,
    generation: 1,
    repositoryId: REPOSITORY_ID,
    targetSha: TARGET_SHA,
    sourceFingerprint: SOURCE_FINGERPRINT,
    observedAt: new Date(NOW - 1_000).toISOString(),
    expiresAt: new Date(NOW + 60_000).toISOString(),
    controls: [
      {
        id: 'managed-rollout-admission',
        required: true,
        result: 'pass',
        receiptSha256: ASSURANCE_CONTROL_RECEIPT
      }
    ]
  }
  return new TextEncoder().encode(JSON.stringify(document))
}

const INSTALLATION_FINGERPRINT = installationFingerprint({
  installId: INSTALL_ID,
  codeRoot: CODE_ROOT,
  repositoryId: REPOSITORY_ID
})

const SOURCE_INPUT: SourceFingerprintInput = {
  installationFingerprint: INSTALLATION_FINGERPRINT,
  connectionId: CONNECTION_ID,
  connectionConfigRevision: 1,
  verifiedHostKeyFingerprint: 'e'.repeat(64),
  remoteUser: 'hermes',
  port: 22,
  configuredProfile: 'default',
  configuredCodePath: CODE_ROOT
}

const SOURCE_FINGERPRINT = sourceFingerprint(SOURCE_INPUT)
const ASSURANCE_EVIDENCE_SHA = crypto.createHash('sha256').update(assuranceEnvelope()).digest('hex')

const TARGET: RolloutTarget = {
  repositoryId: REPOSITORY_ID,
  branch: 'main',
  sha: TARGET_SHA,
  protocol: 1
}

const REVIEWED_SOURCE: ReviewedSourceBinding = {
  repositoryRoot: CODE_ROOT,
  originUrl: ORIGIN,
  resolvedRef: 'refs/remotes/origin/main',
  targetSha: TARGET_SHA,
  assuranceProfile: PROFILE,
  assuranceEvidenceSha256: ASSURANCE_EVIDENCE_SHA,
  assuranceGeneration: 1
}

const INVENTORY: TrustedInventorySnapshot = {
  inventoryRevision: 'inventory-1',
  capturedMono: NOW_MONO - 100,
  observations: [
    {
      installId: INSTALL_ID,
      connectionId: CONNECTION_ID,
      aliasConnectionIds: [],
      codeRoot: CODE_ROOT,
      repositoryId: REPOSITORY_ID,
      headSha: ADMITTED_SHA,
      requiredScopeIds: ['scope-main'],
      source: {
        connectionId: SOURCE_INPUT.connectionId,
        connectionConfigRevision: SOURCE_INPUT.connectionConfigRevision,
        verifiedHostKeyFingerprint: SOURCE_INPUT.verifiedHostKeyFingerprint,
        remoteUser: SOURCE_INPUT.remoteUser,
        port: SOURCE_INPUT.port,
        configuredProfile: SOURCE_INPUT.configuredProfile,
        configuredCodePath: SOURCE_INPUT.configuredCodePath
      }
    }
  ]
}

const BASE_PLAN: RolloutPlan = {
  target: TARGET,
  inventoryRevision: INVENTORY.inventoryRevision,
  waves: [[INSTALL_ID]],
  concurrency: 1,
  promotionPolicy: 'auto-if-healthy',
  rows: [
    {
      installId: INSTALL_ID,
      connectionId: CONNECTION_ID,
      installationFingerprint: INSTALLATION_FINGERPRINT,
      sourceFingerprint: SOURCE_FINGERPRINT,
      admittedHead: ADMITTED_SHA,
      requiredScopeIds: ['scope-main'],
      eligible: true,
      reviewedSource: REVIEWED_SOURCE
    }
  ],
  retryOf: null,
  exclusions: []
}

const RESOLUTION: TargetResolution = {
  id: 'resolution-1',
  target: TARGET,
  fingerprint: 'f'.repeat(64),
  cachePath: '/var/cache/hermes/managed-rollout/resolution-1',
  createdAt: NOW - 1_000,
  expiresAt: NOW + 60_000
}

function sourceReader(origin = ORIGIN): TrustedSourceReader {
  return {
    nowMono: () => NOW_MONO,
    async git(args) {
      if (args[0] === 'rev-parse') return CODE_ROOT
      if (args[0] === 'remote') return origin
      if (args[0] === 'symbolic-ref') return 'main'
      if (args[0] === 'cat-file') return 'commit'
      if (args[0] === 'merge-base') return ''
      if (args[0] === 'show') return '{"protocol":1}'
      throw new Error(`unexpected git probe: ${args.join(' ')}`)
    }
  }
}

function health(): HealthEvidence {
  return {
    observationId: 'observation-1',
    observedAt: NOW_ISO,
    installId: INSTALL_ID,
    checkoutSha: TARGET_SHA,
    installReady: true,
    markerClear: true,
    receiptCorrelated: true,
    receiptSucceeded: true,
    dependencyReady: true,
    recoveryClear: true,
    scopeCapture: 'complete',
    scopes: [
      {
        scopeId: 'scope-main',
        profile: 'default',
        restored: true,
        ready: true,
        codeSha: TARGET_SHA,
        processIdentityVerified: true
      }
    ],
    reasons: []
  }
}

function makeDependencies(options: { origin?: string } = {}): {
  dependencies: ManagedRolloutProviderDependencies
  journalDirectory: string
  launchCalls: string[]
} {
  const journalDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-provider-'))
  const journal = createManagedRolloutJournal({
    directory: journalDirectory,
    clock: () => NOW_ISO,
    retentionLimit: 20
  })
  const launchCalls: string[] = []
  const fakeService = {
    issueLaunchCapability: () => ({ internal: 'not-for-ipc' }),
    request: async (connectionId: string, input: { mode: string; correlationId: string }) => {
      assert.equal(input.mode, 'coordinator')
      launchCalls.push(`${connectionId}:${input.correlationId}`)
      return {
        connectionId,
        correlationId: input.correlationId,
        ok: true,
        updateOk: true,
        restoreOk: true,
        outcome: 'updated' as const,
        exitCode: 0,
        receipt: {
          correlationId: input.correlationId,
          outcome: 'updated',
          startedAt: NOW_ISO,
          finishedAt: NOW_ISO,
          preSha: ADMITTED_SHA,
          postSha: TARGET_SHA
        },
        scopes: []
      }
    }
  }

  const evidence = {
    async sweep(state: { id: string; revision: number; queueGeneration: number; attempts: Record<string, { installId: string; installationFingerprint: string; sourceFingerprint: string; reviewedSource: ReviewedSourceBinding }> }) {
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
          observedAt: NOW_ISO
        }))
      }
    }
  }

  const dependencies: ManagedRolloutProviderDependencies = {
    inventoryReader: { capture: async () => INVENTORY },
    sourceReader: sourceReader(options.origin),
    assuranceReader: {
      readEvidence: async () => assuranceEnvelope(),
      readProfile: async () => ({ generation: 1, requiredControlIds: ['managed-rollout-admission'] })
    },
    resolveTarget: async request => {
      assert.deepEqual(request.connectionIds, [CONNECTION_ID])
      assert.equal(request.inventoryRevision, INVENTORY.inventoryRevision)
      return { plan: BASE_PLAN, resolution: RESOLUTION }
    },
    journal,
    managedSshUpdateService: fakeService as unknown as ManagedRolloutProviderDependencies['managedSshUpdateService'],
    observe: {
      observe: async ({ authorization, update }): Promise<ManagedRolloutObservation> => ({
        outcome: update.outcome === 'updated' ? 'updated' : 'unverified',
        receipt: update.receipt,
        health: health(),
        authorization
      })
    },
    evidence,
    now: () => NOW,
    nowMono: () => NOW_MONO,
    processGeneration: 1
  }
  return { dependencies, journalDirectory, launchCalls }
}

test('fails closed without trusted local rollout dependencies', async () => {
  const provider = createManagedRolloutProvider({})
  const capabilities = await provider.capabilities()

  assert.deepEqual(capabilities, {
    protocol: 1,
    available: false,
    reason: 'trusted-rollout-dependencies-unavailable',
    maxConcurrency: 0,
    maxInstallations: 0
  })
  await assert.rejects(() => provider.inventory(), /trusted-rollout-dependencies-unavailable/)
  await assert.rejects(
    () => provider.start({ token: 'token', requestId: '11111111-1111-4111-8111-111111111111' }),
    /trusted-rollout-dependencies-unavailable/
  )
})

test('runs an injected trusted rollout without exposing the launch capability', async () => {
  const { dependencies, journalDirectory, launchCalls } = makeDependencies()
  try {
    const provider = createManagedRolloutProvider(dependencies)
    assert.equal((await provider.capabilities()).available, true)

    const resolution = await provider.resolveTarget({
      connectionIds: [CONNECTION_ID],
      inventoryRevision: INVENTORY.inventoryRevision,
      retryOf: null
    }) as Record<string, unknown>
    assert.equal(resolution.resolutionId, RESOLUTION.id)
    assert.equal('cachePath' in resolution, false)

    const preflight = await provider.preflight({
      inventoryRevision: INVENTORY.inventoryRevision,
      targetResolutionId: RESOLUTION.id,
      waves: [[INSTALL_ID]],
      concurrency: 1,
      promotionPolicy: 'auto-if-healthy',
      retryOf: null
    }) as Record<string, unknown>
    assert.equal(typeof preflight.token, 'string')
    assert.equal(typeof preflight.requestId, 'string')
    assert.equal(typeof preflight.rolloutId, 'string')

    await assert.rejects(
      () => provider.start({ token: preflight.token as string, requestId: '22222222-2222-4222-8222-222222222222' }),
      /preflight-request-mismatch/
    )

    const started = await provider.start({
      token: preflight.token as string,
      requestId: preflight.requestId as string
    }) as Record<string, unknown>
    assert.equal(started.ok, true)
    assert.equal('capability' in started, false)
    assert.equal(JSON.stringify(started).includes('not-for-ipc'), false)
    assert.deepEqual(
      await provider.start({ token: preflight.token as string, requestId: preflight.requestId as string }),
      started
    )

    await provider.waitForIdle()
    assert.equal(launchCalls.length, 1)
    const snapshot = await provider.get(started.id as string) as Record<string, unknown>
    assert.equal(snapshot?.phase, 'completed')
    assert.equal((snapshot.attempts as Array<Record<string, unknown>>)[0].phase, 'updated')
    assert.equal(await provider.activeRevision(), null)

    const command = {
      id: started.id as string,
      expectedRevision: snapshot.revision as number,
      revision: snapshot.revision as number,
      requestId: '33333333-3333-4333-8333-333333333333',
      kind: 'archive' as const,
      action: 'archive' as const,
      installId: null,
      reason: 'retained for test evidence',
      promotionPolicy: null
    }
    const archived = await provider.command(command)
    assert.equal((archived as Record<string, unknown>).ok, true)
    assert.deepEqual(await provider.command(command), archived)

    const history = await provider.history({ limit: 50 }) as { items: unknown[]; nextCursor: string | null }
    assert.equal(history.items.length, 1)
    assert.equal(history.nextCursor, null)
    const events = await provider.events({ id: started.id as string, limit: 50 }) as { items: unknown[]; nextCursor: string | null }
    assert.ok(events.items.length >= 3)
    assert.equal(events.nextCursor, null)
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('rejects exact-source drift before issuing a review token', async () => {
  const { dependencies, journalDirectory } = makeDependencies({ origin: 'https://github.com/NousResearch/other.git' })
  try {
    const provider = createManagedRolloutProvider(dependencies)
    await assert.rejects(
      () => provider.resolveTarget({
        connectionIds: [CONNECTION_ID],
        inventoryRevision: INVENTORY.inventoryRevision,
        retryOf: null
      }),
      /reviewed-origin-mismatch/
    )
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})
