/** Disposable Electron process fixture. It never opens a window or a real SSH transport. */
import fs from 'node:fs'
import path from 'node:path'

import { app } from 'electron'

import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  type ManagedRolloutAction,
  type ManagedRolloutPlan,
  reduceManagedRollout
} from '../managed-rollout-coordinator'
import { type JournalSnapshot, ManagedRolloutJournal } from '../managed-rollout-journal'
import { acquireManagedRolloutProcessOwner } from '../managed-rollout-process-owner'
import type { ManagedSshUpdateIntent, RemoteUpdateTarget } from '../managed-ssh-update'
import { createManagedSshUpdateService } from '../managed-ssh-update-service'

const ROLLOUT_ID = '11111111-1111-4111-8111-111111111111'
const FIRST_CORRELATION = '12345678-1234-4678-9234-567812345678'
const STALE_CORRELATION = '22345678-1234-4678-9234-567812345678'
const TARGET_SHA = 'abcdef0123456789abcdef0123456789abcdef01'

const INTENT: ManagedSshUpdateIntent = {
  targetSha: TARGET_SHA,
  expectedInstallId: 'a'.repeat(32),
  expectedCurrentSha: 'b'.repeat(40),
  source: {
    repositoryRoot: '/srv/disposable-fixture',
    originUrl: 'https://example.test/disposable-fixture.git',
    resolvedRef: 'refs/remotes/origin/main',
    targetSha: TARGET_SHA,
    assuranceProfile: 'fixture-v1',
    assuranceEvidenceSha256: 'a'.repeat(64),
    assuranceGeneration: 1
  }
}

const EXPECTED_SOURCE = {
  installId: 'a'.repeat(32),
  installationFingerprint: 'b'.repeat(64),
  sourceFingerprint: 'c'.repeat(64),
  expectedCurrentSha: INTENT.expectedCurrentSha
}

function required(name: string): string {
  const value = process.env[name]

  if (!value) {throw new Error(`missing-fixture-variable:${name}`)}

  return value
}

function snapshot(): JournalSnapshot {
  return {
    schemaVersion: 1,
    id: ROLLOUT_ID,
    revision: 0,
    createdAt: '2026-09-23T00:00:00.000Z',
    updatedAt: '2026-09-23T00:00:00.000Z',
    phase: 'running',
    finishedAt: null,
    archivedAt: null,
    attempts: [],
    eventCount: 0
  }
}

function target(): RemoteUpdateTarget {
  return {
    ssh: { exec: async () => '' },
    platform: 'Linux',
    hermesPath: '/fixture/hermes',
    hermesHome: '/fixture/home'
  }
}

function service(counters: { transports: number; mutations: number }) {
  return createManagedSshUpdateService({
    resolveSource: id => id === 'fixture-ssh' ? { id: 'fixture-ssh', kind: 'ssh' as const } : null,
    resolveInstallationId: async () => INTENT.expectedInstallId,
    readRecoveryRecords: () => [],
    captureScopes: async () => [],
    openTransport: async () => {
      counters.transports += 1

      return { target: target(), close: async () => {} }
    },
    targetFromState: () => target(),
    verifyCoordinatorSource: async (_source, _target, expected) => {
      if (JSON.stringify(expected) !== JSON.stringify(EXPECTED_SOURCE)) {throw new Error('fixture-source-binding-changed')}
    },
    executeRemoteUpdate: async (_target, correlationId, context) => {
      counters.mutations += 1
      await context.beforeLaunchDispatch()

      return { exitCode: 0, receipt: { correlationId, outcome: 'success' } }
    },
    preflightRemote: async () => {},
    awaitRestoreClearance: async () => {},
    drainScope: async () => {},
    closeTransports: async () => {},
    restoreScope: async () => {},
    prepareRecovery: async () => {},
    completeRecovery: async () => {},
    restoreRecoveryScope: async () => {}
  })
}

function stateAwaitingPromotion() {
  const plan: ManagedRolloutPlan = {
    id: ROLLOUT_ID,
    revision: 1,
    queueGeneration: 1,
    policy: 'manual',
    targets: [
      {
        installId: 'fixture-canary',
        installationFingerprint: 'installation-canary',
        connectionId: 'fixture-ssh',
        sourceFingerprint: 'source-canary',
        targetSha: TARGET_SHA,
        reviewedSource: INTENT.source,
        correlationId: FIRST_CORRELATION,
        wave: 0
      },
      {
        installId: 'fixture-later',
        installationFingerprint: 'installation-later',
        connectionId: 'fixture-later-ssh',
        sourceFingerprint: 'source-later',
        targetSha: TARGET_SHA,
        reviewedSource: INTENT.source,
        correlationId: STALE_CORRELATION,
        wave: 1
      }
    ]
  }

  let state = createManagedRolloutState(plan)

  const actions: ManagedRolloutAction[] = [
    { kind: 'start' },
    { kind: 'record-intent', installId: 'fixture-canary' },
    { kind: 'launch-authorized', installId: 'fixture-canary' },
    { kind: 'terminal', installId: 'fixture-canary', outcome: 'updated' }
  ]

  for (const action of actions) {
    const transition = reduceManagedRollout(state, action)

    if (!transition.ok) {throw new Error(`fixture-state-invalid:${transition.reason}`)}
    state = transition.state
  }

  if (state.phase !== 'awaiting-promotion') {throw new Error('fixture-promotion-state-unavailable')}

  return state
}

async function run(): Promise<void> {
  const mode = required('HERMES_OWNER_FIXTURE_MODE')
  const userData = path.resolve(required('HERMES_OWNER_FIXTURE_USER_DATA'))
  const resultPath = path.resolve(required('HERMES_OWNER_FIXTURE_RESULT'))
  const stalePath = path.resolve(required('HERMES_OWNER_FIXTURE_STALE_CAPABILITY'))
  const generation = Number(required('HERMES_OWNER_FIXTURE_GENERATION'))
  const journalRoot = path.join(userData, 'managed-rollouts', 'journal')

  const report = (result: Record<string, unknown>) => {
    fs.writeFileSync(resultPath, JSON.stringify({ pid: process.pid, appName: app.getName(), mode, ...result }))
  }

  fs.mkdirSync(userData, { recursive: true })
  app.setPath('userData', userData)
  const owner = acquireManagedRolloutProcessOwner(app)

  if (!owner.acquired) {
    let admissionRefused = false

    try { owner.assert() } catch { admissionRefused = true }
    report({ acquired: false, admissionRefused, journalConstructed: false })
    app.exit(0)

    return
  }

  owner.assert()
  const journal = new ManagedRolloutJournal({ directory: journalRoot, processOwner: owner.owns })
  const counters = { transports: 0, mutations: 0 }
  const updateService = service(counters)

  if (mode === 'hold') {
    journal.create(snapshot(), { metadata: { processGeneration: generation } })
    const freshCapability = updateService.issueLaunchCapability('fixture-ssh', FIRST_CORRELATION, INTENT, EXPECTED_SOURCE)
    const staleCapability = updateService.issueLaunchCapability('fixture-ssh', STALE_CORRELATION, INTENT, EXPECTED_SOURCE)

    const accepted = await updateService.requestCoordinator('fixture-ssh', {
      correlationId: FIRST_CORRELATION,
      intent: INTENT,
      expectedSource: EXPECTED_SOURCE,
      launchCapability: freshCapability
    }).operation

    if (!accepted.ok || counters.mutations !== 1) {throw new Error('fixture-owner-admission-failed')}
    fs.writeFileSync(stalePath, JSON.stringify(staleCapability))
    report({ acquired: true, journalRevision: journal.read(ROLLOUT_ID).snapshot.revision, ...counters })
    // The parent kills this disposable process to prove the OS releases ownership.
    setInterval(() => undefined, 1000)

    return
  }

  if (mode !== 'takeover') {throw new Error(`unknown-fixture-mode:${mode}`)}

  const before = journal.read(ROLLOUT_ID)
  const oldGeneration = Number(before.metadata?.processGeneration)
  const serializedCapability = JSON.parse(fs.readFileSync(stalePath, 'utf8'))

  const staleAttempt = await updateService.requestCoordinator('fixture-ssh', {
    correlationId: STALE_CORRELATION,
    intent: INTENT,
    expectedSource: EXPECTED_SOURCE,
    launchCapability: serializedCapability
  }).operation

  const state = stateAwaitingPromotion()
  const coordinator = createManagedRolloutCoordinator(state, {
    processGeneration: generation,
    journal: { persistAuthorization: async () => { throw new Error('stale-proof-mutated-journal') } },
    service: {
      issueCapability: () => { throw new Error('stale-proof-issued-capability') },
      launch: async () => { throw new Error('stale-proof-launched') }
    },
    evidence: {
      sweep: async current => ({
        rolloutId: current.id,
        revision: current.revision,
        queueGeneration: current.queueGeneration,
        processGeneration: oldGeneration,
        completedMono: Number(process.hrtime.bigint() / 1_000_000n),
        priorWaveClear: true,
        nextAdmissionInstallIds: ['fixture-later'],
        valid: true,
        reason: null,
        admissions: Object.values(current.attempts).map(attempt => ({
          installId: attempt.installId,
          installationFingerprint: attempt.installationFingerprint,
          sourceFingerprint: attempt.sourceFingerprint,
          reviewedSource: attempt.reviewedSource,
          observationGeneration: oldGeneration,
          observedAt: '2026-09-23T00:00:00.000Z'
        }))
      })
    }
  })

  const staleProof = await coordinator.promote()
  journal.record({
    id: ROLLOUT_ID,
    expectedRevision: before.snapshot.revision,
    requestId: 'new-owner-record',
    payload: { kind: 'owner-takeover' },
    snapshot: snapshot(),
    metadata: { processGeneration: generation }
  })
  report({
    acquired: true,
    priorGeneration: oldGeneration,
    processGeneration: generation,
    staleCapabilityRefused: !staleAttempt.ok && /single-use launch capability/.test(staleAttempt.error || ''),
    staleProofRefused: !staleProof.ok && staleProof.reason === 'promotion-proof-is-stale-or-invalid',
    journalRevision: journal.read(ROLLOUT_ID).snapshot.revision,
    ...counters
  })
  app.exit(0)
}

void run().catch(error => {
  const resultPath = process.env.HERMES_OWNER_FIXTURE_RESULT

  if (resultPath) {
    fs.writeFileSync(resultPath, JSON.stringify({ pid: process.pid, error: String(error) }))
  }

  app.exit(1)
})
