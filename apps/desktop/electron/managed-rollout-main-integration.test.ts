import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, test, vi } from 'vitest'

vi.mock('./managed-ssh-update', () => ({
  observeManagedRemoteUpdate: vi.fn()
}))

import type { ManagedRolloutAttempt, ManagedRolloutState } from './managed-rollout-coordinator'
import { buildHealthEvidence } from './managed-rollout-evidence'
import { installationFingerprint, sourceFingerprint } from './managed-rollout-identity'
import { createManagedRolloutMainIntegration, verifyManagedRolloutSelectedTarget } from './managed-rollout-main-integration'
import { observeManagedRemoteUpdate } from './managed-ssh-update'

const INSTALL_ID = 'a'.repeat(32)
const TARGET_SHA = 'b'.repeat(40)
const ROOT = '/srv/hermes-agent'
const CONNECTION_ID = '11111111-1111-4111-8111-111111111111'
const CORRELATION_ID = '22222222-2222-4222-8222-222222222222'
const REPOSITORY_ID = 'github.com/nousresearch/hermes-agent'
const NEXT_INSTALL_ID = 'd'.repeat(32)
const NEXT_CONNECTION_ID = '33333333-3333-4333-8333-333333333333'
const NEXT_CORRELATION_ID = '55555555-5555-4555-8555-555555555555'
const ROLLOUT_ID = '44444444-4444-4444-8444-444444444444'

const temporaryDirectories: string[] = []

afterEach(() => {
  vi.restoreAllMocks()

  for (const directory of temporaryDirectories.splice(0)) {fs.rmSync(directory, { recursive: true, force: true })}
})

function makeIntegration(
  headSha = TARGET_SHA,
  extra: Record<string, unknown> = {},
  additional: Array<{ source: { id: string; kind: string; label: string }; installId: string; headSha: string }> = []
) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-main-integration-'))
  temporaryDirectories.push(directory)
  const sshFor = (installId: string, observedHead: string) => {
    let inspectionHeadReads = 0

    return { exec: vi.fn(async (command: string) => {
      if (command.includes("'--show-toplevel'")) {return ROOT}

      if (command.includes("'remote' 'get-url'")) {return 'https://github.com/nousresearch/hermes-agent.git'}

      if (command.includes("'rev-parse' 'HEAD'")) {return inspectionHeadReads++ === 0 ? TARGET_SHA : observedHead}

      if (command.includes('echo "${HERMES_HOME:-$HOME/.hermes}"')) {return '~/.hermes'}

      if (command.includes('if [ -f')) {return installId}

      return observedHead
    }) }
  }
  const ssh = sshFor(INSTALL_ID, headSha)

  const target = {
    platform: 'Linux',
    hermesPath: `${ROOT}/.venv/bin/hermes`,
    hermesHome: '~/.hermes',
    ssh
  }

  const source = { id: CONNECTION_ID, kind: 'ssh', label: 'test-source' }
  const targets = new Map<string, { source: typeof source; target: typeof target }>([[source.id, { source, target }]])

  for (const item of additional) {
    targets.set(item.source.id, {
      source: item.source,
      target: { ...target, ssh: sshFor(item.installId, item.headSha) }
    })
  }

  const options = {
    nowMono: () => 1000,
    processOwner: () => true,
    listSources: () => [...targets.values()].map(item => item.source),
    getSource: connectionId => targets.get(connectionId)?.source ?? null,
    managedSshConfig: () => ({ user: 'hermes', host: 'source.example.test', port: 22 }),
    openTransport: async selected => ({ target: targets.get(selected.id)!.target, close: async () => undefined }),
    captureScopes: async () => [],
    readHostKeyFingerprint: async () => 'SHA256:host-key',
    effectiveConfigFingerprint: async () => 'config-revision',
    reviewManifestPath: path.join(directory, 'review.json'),
    assuranceRoot: path.join(directory, 'assurance'),
    journalRoot: path.join(directory, 'journal'),
    ...extra
  }
  const integration = createManagedRolloutMainIntegration(options)

  return { integration, source, target, options, ssh }
}

function attempt(
  sourceFingerprint: string,
  installation: string,
  installId = INSTALL_ID,
  connectionId = CONNECTION_ID,
  wave = 0,
  correlationId = CORRELATION_ID
): ManagedRolloutAttempt {
  return {
    installId,
    installationFingerprint: installation,
    connectionId,
    sourceFingerprint,
    targetSha: TARGET_SHA,
    reviewedSource: {
      repositoryRoot: ROOT,
      originUrl: 'https://github.com/NousResearch/hermes-agent.git',
      resolvedRef: 'refs/remotes/origin/main',
      targetSha: TARGET_SHA,
      assuranceProfile: 'profile-v1',
      assuranceEvidenceSha256: 'c'.repeat(64),
      assuranceGeneration: 1
    },
    correlationId,
    wave,
    excluded: false,
    state: wave === 0 ? 'updated' : 'none',
    reprobeCount: 0,
    reprobeCooldownUntilMono: null
  }
}

function rolloutState(rows: ManagedRolloutAttempt[], currentWave = 0): ManagedRolloutState {
  return {
    id: ROLLOUT_ID,
    revision: 1,
    queueGeneration: 1,
    phase: 'awaiting-promotion',
    policy: 'manual',
    currentWave,
    canaryApproved: false,
    continuationRequired: false,
    stopRequested: false,
    attempts: Object.fromEntries(rows.map(row => [row.installId, row]))
  }
}

function seedJournal(
  integration: ReturnType<typeof createManagedRolloutMainIntegration>,
  state: ManagedRolloutState,
  options: { priorHealthy?: boolean; fencePrior?: boolean } = {}
): void {
  integration.journal.create({
    schemaVersion: 1,
    id: state.id,
    revision: 0,
    createdAt: '2026-09-21T00:00:00.000Z',
    updatedAt: '2026-09-21T00:00:00.000Z',
    phase: state.phase,
    attempts: Object.values(state.attempts).map(row => ({
      identity: {
        installId: row.installId,
        installationFingerprint: row.installationFingerprint,
        sourceFingerprint: row.sourceFingerprint,
        admittedSha: TARGET_SHA
      },
      correlationId: row.correlationId,
      wave: row.wave,
      phase: row.state,
      requiredScopeIds: [],
      receipt: options.priorHealthy && row.wave < state.currentWave
        ? { correlationId: row.correlationId, postSha: TARGET_SHA, outcome: 'updated' }
        : null,
      health: options.priorHealthy && row.wave < state.currentWave
        ? buildHealthEvidence({
            observationId: 'prior-local-proof', observedAt: '2026-09-21T00:00:00.000Z',
            installId: row.installId, checkoutSha: TARGET_SHA,
            installReady: true, markerClear: true, receiptCorrelated: true, receiptSucceeded: true,
            dependencyReady: true, recoveryClear: true, scopes: [], reasons: []
          })
        : null,
      recoveryRequired: false
    })),
    eventCount: 0
  }, options.fencePrior ? { unresolved: [{
    key: `${state.id}:${INSTALL_ID}`,
    rolloutId: state.id,
    installId: INSTALL_ID,
    correlationId: CORRELATION_ID,
    reason: 'authorization not cleared',
    recordedAt: '2026-09-21T00:00:00.000Z'
  }] } : {})
}

async function promotionFixture(
  headSha = TARGET_SHA,
  includeThirdWave = false,
  currentWave = 0,
  localOptions: { priorHealthy?: boolean; fencePrior?: boolean } = {}
) {
  const additional = [{
    source: { id: NEXT_CONNECTION_ID, kind: 'ssh', label: 'next-source' },
    installId: NEXT_INSTALL_ID,
    headSha: TARGET_SHA
  }]

  if (includeThirdWave) {
    additional.push({
      source: { id: '66666666-6666-4666-8666-666666666666', kind: 'ssh', label: 'third-source' },
      installId: 'e'.repeat(32), headSha: TARGET_SHA
    })
  }

  const { integration, ssh } = makeIntegration(headSha, {}, additional)
  const inventory = await integration.adapters.inventoryReader.capture()
  expect(inventory).not.toBeNull()
  const byId = new Map(inventory!.observations.map((row: any) => [row.installId, row]))
  const current = attempt(
    byId.get(INSTALL_ID)!.computedSourceFingerprint,
    installationFingerprint({ installId: INSTALL_ID, codeRoot: ROOT, repositoryId: REPOSITORY_ID })
  )
  const next = attempt(
    byId.get(NEXT_INSTALL_ID)!.computedSourceFingerprint,
    installationFingerprint({ installId: NEXT_INSTALL_ID, codeRoot: ROOT, repositoryId: REPOSITORY_ID }),
    NEXT_INSTALL_ID, NEXT_CONNECTION_ID, 1, NEXT_CORRELATION_ID
  )
  const attempts = [current, next]

  if (currentWave === 1) {next.state = 'updated'}

  if (includeThirdWave) {
    const id = 'e'.repeat(32)
    attempts.push(attempt(
      byId.get(id)!.computedSourceFingerprint,
      installationFingerprint({ installId: id, codeRoot: ROOT, repositoryId: REPOSITORY_ID }),
      id, additional[1].source.id, 2, '77777777-7777-4777-8777-777777777777'
    ))
  }

  const state = rolloutState(attempts, currentWave)
  seedJournal(integration, state, localOptions)
  const currentCorrelation = currentWave === 1 ? NEXT_CORRELATION_ID : CORRELATION_ID
  vi.mocked(observeManagedRemoteUpdate).mockImplementation(async (_target, correlationId) => correlationId === currentCorrelation
    ? {
        marker: 'absent', launchIntent: 'absent',
        receipt: { correlationId: currentCorrelation, outcome: 'updated', postSha: TARGET_SHA },
        coordinatorReady: { correlationId: currentCorrelation, pid: 1 }, exitCode: 0
      } as any
    : { marker: 'absent', launchIntent: 'absent', receipt: null, coordinatorReady: null, exitCode: 0 } as any)

  return { integration, state, ssh }
}

describe('managed rollout main integration', () => {
  test('rechecks the exact selected transport target before a coordinator launch', async () => {
    const { integration, source, target, options } = makeIntegration()
    const inventory = await integration.adapters.inventoryReader.capture()
    const observed = inventory!.observations[0]
    const expectedInstallation = installationFingerprint({ installId: INSTALL_ID, codeRoot: ROOT, repositoryId: REPOSITORY_ID })
    const expected = {
      installId: INSTALL_ID,
      installationFingerprint: expectedInstallation,
      sourceFingerprint: sourceFingerprint({ ...observed.source, installationFingerprint: expectedInstallation })
    }
    const openTransport = vi.spyOn(options, 'openTransport')

    await expect(verifyManagedRolloutSelectedTarget(options, source, target, expected)).resolves.toBeUndefined()
    expect(openTransport).not.toHaveBeenCalled()

    const otherSsh = {
      exec: vi.fn(async (command: string) => command.includes('if [ -f') ? NEXT_INSTALL_ID : target.ssh.exec(command))
    }

    await expect(verifyManagedRolloutSelectedTarget(options, source, { ...target, ssh: otherSsh }, expected))
      .rejects.toThrow('managed-rollout-source-binding-changed')
    expect(openTransport).not.toHaveBeenCalled()
  })

  test('refuses journal reconstruction before a process owner is acquired', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-owner-refusal-'))
    temporaryDirectories.push(directory)
    const journalRoot = path.join(directory, 'journal')

    expect(() => makeIntegration(TARGET_SHA, { journalRoot, processOwner: () => false }))
      .toThrow('managed-rollout-owner-unavailable')
    expect(fs.existsSync(journalRoot)).toBe(false)
  })

  test('runs an independent bounded evidence sweep and verifies target HEAD', async () => {
    const { integration, state, ssh } = await promotionFixture()
    const bulkCapture = vi.spyOn(integration.adapters.inventoryReader, 'capture').mockImplementation(async () => {
      throw new Error('unbounded-inventory-capture-used-during-promotion')
    })
    const proof = await integration.evidence.sweep(state)

    expect(proof.valid).toBe(true)
    expect(proof.admissions).toHaveLength(2)
    expect(proof.admissions[0].installId).toBe(INSTALL_ID)
    expect(proof.nextAdmissionInstallIds).toEqual([NEXT_INSTALL_ID])
    expect(proof.priorWaveClear).toBe(true)
    expect(bulkCapture).not.toHaveBeenCalled()
    expect((ssh.exec.mock.calls as unknown as Array<[string, { signal?: AbortSignal }?]>).some(([, options]) => options?.signal instanceof AbortSignal)).toBe(true)
  })

  test('refuses promotion evidence when an independent Git HEAD read disagrees', async () => {
    const { integration, state } = await promotionFixture('d'.repeat(40))
    const proof = await integration.evidence.sweep(state)

    expect(proof.valid).toBe(false)
    expect(proof.reason).toBe('health-evidence-not-proven')
  })

  test('sweeps the settled wave and immediate successor without probing a later wave', async () => {
    const { integration, state } = await promotionFixture(TARGET_SHA, true)
    const proof = await integration.evidence.sweep(state)

    expect(proof.valid).toBe(true)
    expect(proof.admissions.map(admission => admission.installId)).toEqual([INSTALL_ID, NEXT_INSTALL_ID])
    expect(proof.processGeneration).toBe(integration.processGeneration)
    expect(proof.processGeneration).not.toBe(1)
  })

  test('requires prior-wave scope proof and rejects a retained local fence before remote probing', async () => {
    const missing = await promotionFixture(TARGET_SHA, true, 1)
    const missingProof = await missing.integration.evidence.sweep(missing.state)

    expect(missingProof.valid).toBe(false)
    expect(missingProof.reason).toBe('prior-wave-local-proof-missing')

    const fenced = await promotionFixture(TARGET_SHA, true, 1, { priorHealthy: true, fencePrior: true })
    const fencedProof = await fenced.integration.evidence.sweep(fenced.state)

    expect(fencedProof.valid).toBe(false)
    expect(fencedProof.reason).toBe('prior-wave-local-proof-missing')

    const clear = await promotionFixture(TARGET_SHA, true, 1, { priorHealthy: true })
    const clearProof = await clear.integration.evidence.sweep(clear.state)

    expect(clearProof.valid).toBe(true)
    expect(clearProof.nextAdmissionInstallIds).toEqual(['e'.repeat(32)])
  })

  test('rejects a next-wave target with a live update marker', async () => {
    const { integration, state } = await promotionFixture()
    vi.mocked(observeManagedRemoteUpdate).mockImplementation(async (_target, correlationId) => ({
      marker: correlationId === NEXT_CORRELATION_ID ? 'live' : 'absent',
      launchIntent: 'absent',
      receipt: correlationId === CORRELATION_ID
        ? { correlationId, outcome: 'updated', postSha: TARGET_SHA } : null,
      coordinatorReady: correlationId === CORRELATION_ID ? { correlationId, pid: 1 } : null,
      exitCode: 0
    } as any))

    const proof = await integration.evidence.sweep(state)

    expect(proof.valid).toBe(false)
    expect(proof.reason).toBe('health-evidence-not-proven')
  })

  test('rejects local journal generation changes during the sweep', async () => {
    const { integration, state } = await promotionFixture()
    const actualRead = integration.journal.read.bind(integration.journal)
    let reads = 0

    vi.spyOn(integration.journal, 'read').mockImplementation(id => {
      const record = actualRead(id)

      return ++reads === 2 ? { ...record, generation: 'changed-during-sweep' } : record
    })

    const proof = await integration.evidence.sweep(state)

    expect(proof.valid).toBe(false)
    expect(proof.reason).toBe('prior-wave-local-proof-missing')
  })

  test('wires read-only reprobe and correlated recovery through the production integration', async () => {
    const durableScope = { key: 'ssh:profile:default', kind: 'registry', profile: 'default' }
    let durableRecord: any = {
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID,
      phase: 'launching',
      scopes: [durableScope],
      source: { id: CONNECTION_ID, kind: 'ssh', label: 'original-host' }
    }
    let recoveredRecord: any = null

    const { integration } = makeIntegration(TARGET_SHA, {
      readRecoveryRecord: () => durableRecord,
      recoverManagedSsh: async (record: any) => {
        recoveredRecord = record
        durableRecord = null
      }
    })

    const authorization = {
      rolloutId: 'rollout-1',
      installId: INSTALL_ID,
      connectionId: CONNECTION_ID,
      installationFingerprint: 'e'.repeat(64),
      sourceFingerprint: 'f'.repeat(64),
      targetSha: TARGET_SHA,
      reviewedSource: {
        repositoryRoot: ROOT,
        originUrl: 'https://github.com/nousresearch/hermes-agent.git',
        resolvedRef: 'refs/remotes/origin/main',
        targetSha: TARGET_SHA,
        assuranceProfile: 'profile-v1',
        assuranceEvidenceSha256: 'c'.repeat(64),
        assuranceGeneration: 1
      },
      correlationId: CORRELATION_ID,
      queueGeneration: 1
    }

    vi.mocked(observeManagedRemoteUpdate).mockResolvedValue({
      marker: 'absent',
      launchIntent: 'absent',
      receipt: { correlationId: CORRELATION_ID, outcome: 'updated', postSha: TARGET_SHA },
      coordinatorReady: { correlationId: CORRELATION_ID, pid: 1 },
      exitCode: 0
    } as any)

    await expect((integration.observe as any).reprobe(authorization)).resolves.toMatchObject({
      correlationId: CORRELATION_ID,
      outcome: 'updated',
      terminal: true
    })
    await expect((integration.observe as any).recover(authorization)).resolves.toEqual({
      correlationId: CORRELATION_ID,
      clearanceProved: true
    })
    expect(recoveredRecord).toMatchObject({
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID,
      phase: 'launching',
      scopes: [durableScope],
      source: { label: 'original-host' }
    })
  })

  test('Recover refuses clearance without the original durable scope record', async () => {
    const recoverManagedSsh = vi.fn(async () => undefined)
    const { integration } = makeIntegration(TARGET_SHA, {
      readRecoveryRecord: () => null,
      recoverManagedSsh
    })

    vi.mocked(observeManagedRemoteUpdate).mockResolvedValue({
      marker: 'absent',
      launchIntent: 'absent',
      receipt: { correlationId: CORRELATION_ID, outcome: 'updated', postSha: TARGET_SHA }
    } as any)

    await expect((integration.observe as any).recover({
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID
    })).resolves.toEqual({ correlationId: CORRELATION_ID, clearanceProved: false })
    expect(recoverManagedSsh).not.toHaveBeenCalled()
  })

  test('Recover leaves clearance unproved while an original scope remains pending', async () => {
    const record = {
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID,
      phase: 'launching',
      scopes: [{ key: 'ssh:profile:default', kind: 'registry', profile: 'default' }],
      source: { id: CONNECTION_ID, kind: 'ssh', label: 'original-host' }
    }
    const recoverManagedSsh = vi.fn(async () => undefined)
    const { integration } = makeIntegration(TARGET_SHA, {
      readRecoveryRecord: () => record,
      recoverManagedSsh
    })

    vi.mocked(observeManagedRemoteUpdate).mockResolvedValue({
      marker: 'absent',
      launchIntent: 'absent',
      receipt: { correlationId: CORRELATION_ID, outcome: 'updated', postSha: TARGET_SHA }
    } as any)

    await expect((integration.observe as any).recover({
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID
    })).resolves.toEqual({ correlationId: CORRELATION_ID, clearanceProved: false })
    expect(recoverManagedSsh).toHaveBeenCalledWith(record)
  })

  test('Recover restores a prepared prelaunch obligation without inventing a receipt', async () => {
    let record: any = {
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID,
      phase: 'prepared',
      scopes: [{ key: 'ssh:profile:default', kind: 'registry', profile: 'default' }],
      source: { id: CONNECTION_ID, kind: 'ssh', label: 'original-host' }
    }
    const recoverManagedSsh = vi.fn(async () => {record = null})
    const { integration } = makeIntegration(TARGET_SHA, {
      readRecoveryRecord: () => record,
      recoverManagedSsh
    })

    vi.mocked(observeManagedRemoteUpdate).mockResolvedValue({
      marker: 'absent', launchIntent: 'absent', receipt: null
    } as any)

    await expect((integration.observe as any).recover({
      connectionId: CONNECTION_ID,
      correlationId: CORRELATION_ID
    })).resolves.toEqual({ correlationId: CORRELATION_ID, clearanceProved: true })
    expect(recoverManagedSsh).toHaveBeenCalledWith(expect.objectContaining({
      phase: 'prepared', scopes: [{ key: 'ssh:profile:default', kind: 'registry', profile: 'default' }]
    }))
  })
})
