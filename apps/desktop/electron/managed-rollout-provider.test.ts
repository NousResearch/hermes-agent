import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import type {
  HealthEvidence,
  ReviewedSourceBinding,
  RolloutPlan,
  RolloutTarget
} from '../src/lib/managed-rollout-contract'

import type { TrustedSourceReader } from './managed-rollout-assurance'
import {
  installationFingerprint,
  sourceFingerprint,
  type SourceFingerprintInput
} from './managed-rollout-identity'
import type { TrustedInventorySnapshot } from './managed-rollout-inventory'
import { createManagedRolloutJournal } from './managed-rollout-journal'
import type { TargetResolution } from './managed-rollout-preflight'
import {
  createManagedRolloutProvider,
  type ManagedRolloutObservation,
  type ManagedRolloutProviderDependencies
} from './managed-rollout-provider'
import type { ManagedSshUpdateIntent } from './managed-ssh-update'

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
      if (args[0] === 'rev-parse') {return CODE_ROOT}

      if (args[0] === 'remote') {return origin}

      if (args[0] === 'symbolic-ref') {return 'main'}

      if (args[0] === 'cat-file') {return 'commit'}

      if (args[0] === 'merge-base') {return ''}

      if (args[0] === 'show') {return '{"protocol":1}'}
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

function makeDependencies(options: { origin?: string; plan?: RolloutPlan } = {}): {
  dependencies: ManagedRolloutProviderDependencies
  journalDirectory: string
  launchCalls: string[]
  issuedIntents: ManagedSshUpdateIntent[]
  launchIntents: ManagedSshUpdateIntent[]
} {
  const journalDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-provider-'))

  const journal = createManagedRolloutJournal({
    directory: journalDirectory,
    clock: () => NOW_ISO,
    retentionLimit: 20
  })

  const launchCalls: string[] = []
  const issuedIntents: ManagedSshUpdateIntent[] = []
  const launchIntents: ManagedSshUpdateIntent[] = []

  const fakeService = {
    issueLaunchCapability: (connectionId: string, _correlationId: string, intent: ManagedSshUpdateIntent) => {
      assert.equal(connectionId, CONNECTION_ID)
      issuedIntents.push(intent)

      return { internal: 'not-for-ipc' }
    },
    requestCoordinator: (connectionId: string, input: { correlationId: string; intent: ManagedSshUpdateIntent }) => {
      assert.equal(connectionId, CONNECTION_ID)
      launchCalls.push(`${connectionId}:${input.correlationId}`)
      launchIntents.push(input.intent)

      return { admitted: true as const, operation: Promise.resolve({
        connectionId: String(connectionId),
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
      }) }
    }
  }

  const evidence = {
    async sweep(state: { id: string; revision: number; queueGeneration: number; currentWave: number; attempts: Record<string, { installId: string; installationFingerprint: string; sourceFingerprint: string; reviewedSource: ReviewedSourceBinding; wave: number; excluded?: boolean }> }) {
      return {
        rolloutId: state.id,
        revision: state.revision,
        queueGeneration: state.queueGeneration,
        processGeneration: 1,
        completedMono: NOW_MONO,
        priorWaveClear: true,
        nextAdmissionInstallIds: Object.values(state.attempts).filter(attempt => attempt.wave === state.currentWave + 1 && !attempt.excluded).map(attempt => attempt.installId).sort(),
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

      return { plan: options.plan ?? BASE_PLAN, resolution: RESOLUTION }
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
    processGeneration: 1,
    measuredMaxInstallations: () => 1
  }

  return { dependencies, journalDirectory, launchCalls, issuedIntents, launchIntents }
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

test('fails closed when the main-process generation is absent instead of defaulting it', async () => {
  const { dependencies, journalDirectory } = makeDependencies()

  try {
    const { processGeneration: _ignored, ...withoutGeneration } = dependencies
    const provider = createManagedRolloutProvider(withoutGeneration)
    assert.equal((await provider.capabilities()).available, false)
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('reports missing real-SSH capacity even when a review manifest is not installed', async () => {
  const { dependencies, journalDirectory } = makeDependencies()

  try {
    const provider = createManagedRolloutProvider({
      ...dependencies,
      ready: () => false,
      measuredMaxInstallations: undefined
    })

    const capability = await provider.capabilities()
    assert.equal(capability.available, false)
    assert.equal(capability.reason, 'unverified-capacity')
    assert.equal(capability.maxInstallations, 0)
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('runs an injected trusted rollout without exposing the launch capability', async () => {
  const { dependencies, journalDirectory, launchCalls, issuedIntents, launchIntents } = makeDependencies()

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

    const expectedIntent = {
      targetSha: TARGET_SHA,
      expectedInstallId: INSTALL_ID,
      expectedCurrentSha: ADMITTED_SHA,
      source: REVIEWED_SOURCE
    }

    assert.deepEqual(issuedIntents, [expectedIntent])
    assert.deepEqual(launchIntents, [expectedIntent])

    const facts = dependencies.journal.read(started.id as string).facts.map(fact => fact.kind)
    assert.ok(facts.includes('authorization-committed'))
    assert.ok(facts.includes('handoff-accepted'))
    assert.ok(facts.includes('detached-intent'))
    assert.match(
      dependencies.journal.read(started.id as string).facts.find(fact => fact.kind === 'detached-intent')?.basis || '',
      /remote launch remains unverified/
    )
    assert.ok(facts.includes('terminal-receipt'))
    assert.ok(facts.includes('settlement-validated'))
    const snapshot = await provider.get(started.id as string) as Record<string, unknown>
    assert.equal(snapshot?.phase, 'completed')
    assert.equal((snapshot.attempts as Array<Record<string, unknown>>)[0].phase, 'updated')
    assert.equal(await provider.activeRevision(), null)
    assert.deepEqual(await provider.read(null), { revision: 0, snapshot: null })

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

test.each([
  { serviceOutcome: 'update-failed' as const, observedOutcome: 'failed' as const },
  { serviceOutcome: 'refused' as const, observedOutcome: 'refused' as const }
])('records a correlated $observedOutcome terminal result instead of losing it to an unverified launch', async ({ serviceOutcome, observedOutcome }) => {
  const { dependencies, journalDirectory } = makeDependencies()
  let observations = 0
  dependencies.managedSshUpdateService = {
    ...dependencies.managedSshUpdateService,
    requestCoordinator: (connectionId, options) => ({
      admitted: true,
      operation: Promise.resolve({
        connectionId: String(connectionId), correlationId: options?.correlationId || '',
        ok: false, updateOk: false, restoreOk: true, outcome: serviceOutcome,
        exitCode: serviceOutcome === 'refused' ? null : 1,
        receipt: {
          correlationId: options?.correlationId || '', outcome: observedOutcome,
          startedAt: NOW_ISO, finishedAt: NOW_ISO, preSha: ADMITTED_SHA, postSha: ADMITTED_SHA
        },
        scopes: []
      })
    })
  }
  dependencies.observe = {
    observe: async ({ authorization, update }) => {
      observations += 1
      return { authorization, outcome: observedOutcome, receipt: update.receipt, health: null }
    },
    reprobe: async authorization => ({ correlationId: authorization.correlationId, outcome: 'unverified', terminal: false }),
    recover: async authorization => ({ correlationId: authorization.correlationId, clearanceProved: true })
  }

  try {
    const provider = createManagedRolloutProvider(dependencies)
    await provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: null })
    const preflight = await provider.preflight({
      inventoryRevision: INVENTORY.inventoryRevision, targetResolutionId: RESOLUTION.id,
      waves: [[INSTALL_ID]], concurrency: 1, promotionPolicy: 'auto-if-healthy', retryOf: null
    }) as { token: string; requestId: string }
    const started = await provider.start(preflight) as { id: string }
    await provider.waitForIdle()

    const record = dependencies.journal.read(started.id)
    assert.equal(observations, 1)
    assert.equal(record.snapshot.phase, 'attention-required')
    assert.equal((record.snapshot.attempts[0] as any).phase, observedOutcome)
    assert.equal(record.unresolved.length, 1)
    assert.equal(record.facts.some(item => item.kind === 'settlement-validated'), false)

    const recovered = await provider.command({
      id: started.id, expectedRevision: record.snapshot.revision,
      requestId: crypto.randomUUID(), action: 'recover', kind: 'recover',
      installId: INSTALL_ID, reason: null, promotionPolicy: null
    } as any) as Record<string, unknown>
    assert.equal(recovered.ok, true)
    const settled = dependencies.journal.read(started.id)
    assert.equal((settled.snapshot.attempts[0] as any).phase, observedOutcome)
    assert.equal(settled.unresolved.length, 0)
    assert.ok(settled.facts.some(item => item.kind === 'settlement-validated'))
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('keeps a missing terminal receipt unverified after a failed managed update', async () => {
  const { dependencies, journalDirectory } = makeDependencies()
  dependencies.managedSshUpdateService = {
    ...dependencies.managedSshUpdateService,
    requestCoordinator: (connectionId, options) => ({
      admitted: true,
      operation: Promise.resolve({
        connectionId: String(connectionId), correlationId: options?.correlationId || '',
        ok: false, updateOk: false, restoreOk: true, outcome: 'update-failed',
        exitCode: 1, receipt: null, scopes: []
      })
    })
  }
  dependencies.observe = {
    observe: async ({ authorization }) => ({ authorization, outcome: 'failed', receipt: null, health: null })
  }

  try {
    const provider = createManagedRolloutProvider(dependencies)
    await provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: null })
    const preflight = await provider.preflight({
      inventoryRevision: INVENTORY.inventoryRevision, targetResolutionId: RESOLUTION.id,
      waves: [[INSTALL_ID]], concurrency: 1, promotionPolicy: 'auto-if-healthy', retryOf: null
    }) as { token: string; requestId: string }
    const started = await provider.start(preflight) as { id: string }
    await provider.waitForIdle()

    const record = dependencies.journal.read(started.id)
    assert.equal((record.snapshot.attempts[0] as any).phase, 'unverified')
    assert.equal(record.unresolved.length, 1)
    assert.equal(record.facts.some(item => item.kind === 'settlement-validated'), false)
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('hydrates an authorized running journal record as recoverable unknown without redispatch', async () => {
  const { dependencies, journalDirectory, launchCalls } = makeDependencies()
  const rolloutId = '33333333-3333-4333-8333-333333333333'
  const correlationId = '44444444-4444-4444-8444-444444444444'
  let clearanceMode: 'missing' | 'foreign' | 'proved' = 'missing'

  try {
    dependencies.journal.create({
      schemaVersion: 1,
      id: rolloutId,
      revision: 0,
      createdAt: NOW_ISO,
      updatedAt: NOW_ISO,
      finishedAt: null,
      retryOf: null,
      archivedAt: null,
      target: TARGET,
      phase: 'running',
      activeWave: 0,
      concurrency: 1,
      promotionPolicy: 'auto-if-healthy',
      canaryApproved: false,
      continuationRequired: false,
      attempts: [{
        identity: {
          connectionId: CONNECTION_ID,
          installId: INSTALL_ID,
          aliasConnectionIds: [],
          label: INSTALL_ID,
          displayAddress: CONNECTION_ID,
          installationFingerprint: INSTALLATION_FINGERPRINT,
          sourceFingerprint: SOURCE_FINGERPRINT,
          admittedSha: ADMITTED_SHA
        },
        correlationId,
        wave: 0,
        phase: 'updating',
        launchState: 'authorized',
        requiredScopeIds: ['scope-main'],
        skipReason: null,
        reprobes: 0,
        receipt: null,
        health: null,
        recoveryRequired: false,
        reasons: []
      }],
      eventCount: 0
    } as any, {
      metadata: {
        schema: 1,
        queueGeneration: 0,
        currentWave: 0,
        phase: 'running',
        policy: 'auto-after-canary',
        canaryApproved: false,
        continuationRequired: false,
        stopRequested: false,
        plan: BASE_PLAN
      },
      events: [],
      unresolved: [{
        key: `managed-rollout:${rolloutId}:${INSTALL_ID}:${correlationId}`,
        rolloutId, installId: INSTALL_ID, correlationId,
        reason: 'remote-launch-settlement-required', recordedAt: NOW_ISO
      }]
    })

    const freshJournal = createManagedRolloutJournal({ directory: journalDirectory, clock: () => NOW_ISO })

    const freshDependencies: ManagedRolloutProviderDependencies = {
      ...dependencies,
      journal: freshJournal,
      observe: {
        ...dependencies.observe,
        reprobe: async authorization => ({
          correlationId: authorization.correlationId,
          outcome: 'updated' as const,
          terminal: true,
          recoveryRecordClear: true,
          receipt: {
            correlationId: authorization.correlationId, outcome: 'updated',
            startedAt: NOW_ISO, finishedAt: NOW_ISO, preSha: ADMITTED_SHA, postSha: TARGET_SHA
          },
          health: health()
        }),
        recover: async authorization => ({
          correlationId: clearanceMode === 'foreign' ? 'foreign-correlation' : authorization.correlationId,
          clearanceProved: clearanceMode !== 'missing'
        })
      }
    }

    const provider = createManagedRolloutProvider(freshDependencies)
    const before = await provider.get(rolloutId) as Record<string, unknown>

    assert.equal(before.phase, 'attention-required')
    assert.equal((before.attempts as Array<Record<string, unknown>>)[0].phase, 'unverified')
    assert.equal((before.attempts as Array<Record<string, unknown>>)[0].recoveryRequired, true)
    assert.equal(freshJournal.read(rolloutId).unresolved.length, 1)
    assert.equal(launchCalls.length, 0)

    let revision = before.revision as number

    for (const [mode, requestId] of [
      ['missing', '77777777-7777-4777-8777-777777777777'],
      ['foreign', '88888888-8888-4888-8888-888888888888']
    ] as const) {
      clearanceMode = mode
      const refused = await provider.command({
        id: rolloutId, expectedRevision: revision, requestId,
        action: 'recover', kind: 'recover', installId: INSTALL_ID,
        reason: null, promotionPolicy: null
      } as any) as Record<string, unknown>
      assert.equal(refused.ok, false)
      assert.equal(freshJournal.read(rolloutId).unresolved.length, 1)
      assert.equal(freshJournal.read(rolloutId).facts.some(item => item.kind === 'settlement-validated'), false)
      revision = refused.revision as number
    }

    clearanceMode = 'proved'
    const recovered = await provider.command({
      id: rolloutId, expectedRevision: revision,
      requestId: '99999999-9999-4999-8999-999999999999',
      action: 'recover', kind: 'recover', installId: INSTALL_ID,
      reason: null, promotionPolicy: null
    } as any)
    assert.equal((recovered as Record<string, unknown>).ok, true)
    assert.equal(freshJournal.read(rolloutId).unresolved.length, 0)
    assert.ok(freshJournal.read(rolloutId).facts.some(item => item.kind === 'settlement-validated'))

    const afterRecovery = await provider.get(rolloutId) as Record<string, unknown>
    assert.equal((afterRecovery.attempts as Array<Record<string, unknown>>)[0].phase, 'unverified')

    const result = await provider.command({
      id: rolloutId,
      expectedRevision: afterRecovery.revision as number,
      requestId: '55555555-5555-4555-8555-555555555555',
      action: 'reprobe',
      kind: 'reprobe',
      installId: INSTALL_ID,
      reason: null,
      promotionPolicy: null
    } as any)

    assert.equal((result as Record<string, unknown>).ok, true)
    assert.equal((await provider.get(rolloutId) as Record<string, unknown>).phase, 'completed')
    assert.equal(launchCalls.length, 0)
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('refuses retry while the prior rollout fence is unresolved and permits it after settlement evidence', async () => {
  const { dependencies, journalDirectory } = makeDependencies()

  try {
    const provider = createManagedRolloutProvider(dependencies)
    const resolution = await provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: null }) as Record<string, unknown>

    const preflight = await provider.preflight({
      inventoryRevision: INVENTORY.inventoryRevision,
      targetResolutionId: resolution.resolutionId,
      waves: [[INSTALL_ID]],
      concurrency: 1,
      promotionPolicy: 'auto-if-healthy',
      retryOf: null
    }) as Record<string, unknown>

    const started = await provider.start({ token: preflight.token as string, requestId: preflight.requestId as string }) as Record<string, unknown>
    let prior = dependencies.journal.read(started.id as string)

    for (let attempt = 0; attempt < 100 && prior.snapshot.phase !== 'completed'; attempt += 1) {
      await new Promise(resolveWait => setTimeout(resolveWait, 1))
      prior = dependencies.journal.read(started.id as string)
    }

    assert.equal(prior.snapshot.phase, 'completed')
    const correlationId = (prior.snapshot.attempts[0] as any).correlationId

    const fence = {
      key: `${prior.id}:${INSTALL_ID}`,
      rolloutId: prior.id,
      installId: INSTALL_ID,
      correlationId,
      reason: 'recovery-required',
      recordedAt: NOW_ISO
    }

    dependencies.journal.record({
      id: prior.id,
      expectedRevision: prior.snapshot.revision,
      requestId: '66666666-6666-4666-8666-666666666666',
      payload: { kind: 'test-fence' },
      snapshot: prior.snapshot as any,
      unresolved: { add: [fence] }
    })
    const retryResolution = { ...RESOLUTION, id: 'resolution-retry-1' }
    dependencies.resolveTarget = async request => ({
      plan: { ...BASE_PLAN, retryOf: request.retryOf },
      resolution: retryResolution
    })

    await assert.rejects(
      () => provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: prior.id }),
      /retry-unresolved-fence/
    )

    const fenced = dependencies.journal.read(prior.id)
    dependencies.journal.record({
      id: prior.id,
      expectedRevision: fenced.snapshot.revision,
      requestId: '77777777-7777-4777-8777-777777777777',
      payload: { kind: 'settle-test-fence' },
      snapshot: fenced.snapshot as any,
      facts: [{
        kind: 'settlement-validated',
        rolloutId: prior.id,
        correlationId,
        installId: INSTALL_ID,
        observedAt: NOW_ISO,
        basis: 'validated terminal receipt and restored scope'
      }],
      unresolved: { remove: [fence.key] }
    })

    const resolved = await provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: prior.id }) as Record<string, unknown>
    assert.equal(resolved.resolutionId, retryResolution.id)
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

test('refuses a selected route edit after preflight before creating a rollout journal', async () => {
  const { dependencies, journalDirectory, launchCalls } = makeDependencies()
  let liveInventory = INVENTORY
  dependencies.inventoryReader = { capture: async () => liveInventory }

  try {
    const provider = createManagedRolloutProvider(dependencies)
    await provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: null })

    const preflight = await provider.preflight({
      inventoryRevision: INVENTORY.inventoryRevision,
      targetResolutionId: RESOLUTION.id,
      waves: [[INSTALL_ID]],
      concurrency: 1,
      promotionPolicy: 'auto-if-healthy',
      retryOf: null
    }) as { token: string; requestId: string }

    liveInventory = {
      ...INVENTORY,
      observations: [{
        ...INVENTORY.observations[0],
        source: { ...INVENTORY.observations[0].source, verifiedHostKeyFingerprint: '9'.repeat(64) }
      }]
    }

    await assert.rejects(() => provider.start(preflight), /inventory-source-or-scope-mismatch/)
    assert.equal(dependencies.journal.history({ limit: 50 }).items.length, 0)
    assert.deepEqual(launchCalls, [])
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})

test('does not record service handoff when service refuses coordinator admission', async () => {
  const { dependencies, journalDirectory } = makeDependencies()
  dependencies.managedSshUpdateService = {
    ...dependencies.managedSshUpdateService,
    requestCoordinator: (connectionId, options) => ({
      admitted: false,
      reason: 'coordinator-source-binding-mismatch',
      operation: Promise.resolve({
        connectionId: String(connectionId),
        correlationId: options?.correlationId || '',
        ok: false,
        updateOk: false,
        restoreOk: true,
        outcome: 'refused',
        exitCode: null,
        receipt: null,
        scopes: [],
        error: 'coordinator-source-binding-mismatch'
      })
    })
  }

  try {
    const provider = createManagedRolloutProvider(dependencies)
    await provider.resolveTarget({ connectionIds: [CONNECTION_ID], inventoryRevision: INVENTORY.inventoryRevision, retryOf: null })

    const preflight = await provider.preflight({
      inventoryRevision: INVENTORY.inventoryRevision,
      targetResolutionId: RESOLUTION.id,
      waves: [[INSTALL_ID]],
      concurrency: 1,
      promotionPolicy: 'auto-if-healthy',
      retryOf: null
    }) as { token: string; requestId: string }

    const started = await provider.start(preflight) as { id: string }
    await provider.waitForIdle()

    const facts = dependencies.journal.read(started.id).facts.map(item => item.kind)
    assert.equal(facts.includes('handoff-accepted'), false)
    assert.equal(facts.includes('detached-intent'), false)
  } finally {
    fs.rmSync(journalDirectory, { recursive: true, force: true })
  }
})
