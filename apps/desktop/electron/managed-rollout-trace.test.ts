import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, test, vi } from 'vitest'

import {
  createManagedRolloutCoordinator,
  createManagedRolloutState,
  type ManagedRolloutAuthorization,
  type ManagedRolloutPlan,
  reduceManagedRollout
} from './managed-rollout-coordinator'
import { canonicalRepositoryId, installationFingerprint, sourceFingerprint } from './managed-rollout-identity'
import { createManagedRolloutMainIntegration, readInstallId, verifyManagedRolloutSelectedTarget } from './managed-rollout-main-integration'
import { createManagedRolloutProvider } from './managed-rollout-provider'
import {
  assertManagedUpdatePreflightClear,
  executeManagedRemoteUpdate,
  ManagedConnectionUpdateGate,
  waitForManagedRemoteClearance
} from './managed-ssh-update'
import {
  createManagedSshUpdateService,
  type ManagedSshUpdateScope,
  type ManagedSshUpdateSource
} from './managed-ssh-update-service'

const unionDirectories: string[] = []

afterEach(() => {
  for (const directory of unionDirectories.splice(0)) {fs.rmSync(directory, { recursive: true, force: true })}
})

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
    completedMono: 1_000,
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
  ]) {state = reduceManagedRollout(state, action).state}

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
  ]) {state = reduceManagedRollout(state, action).state}

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
  ]) {state = reduceManagedRollout(state, action).state}

  const coordinator = createManagedRolloutCoordinator(state, {
    journal: { persistAuthorization: async () => {} },
    service: { issueCapability: () => ({}), launch: async () => {} },
    evidence: { sweep: async current => proof(current, { queueGeneration: 0 }) }
  })

  const result = await coordinator.promote()

  assert.equal(result.ok, false)
  assert.equal(result.reason, 'promotion-proof-is-stale-or-invalid')
})

test('promotion refuses proof consumed after ten seconds or missing a next-wave admission', async () => {
  let state = running()

  for (const action of [
    { kind: 'record-intent' as const, installId: 'canary' },
    { kind: 'launch-authorized' as const, installId: 'canary' },
    { kind: 'terminal' as const, installId: 'canary', outcome: 'updated' as const }
  ]) {state = reduceManagedRollout(state, action).state}

  const make = (now: number, omitNext = false) => createManagedRolloutCoordinator(state, {
    journal: { persistAuthorization: async () => {} },
    service: { issueCapability: () => ({}), launch: async () => {} },
    nowMono: () => now,
    evidence: {
      sweep: async current => {
        const result = proof(current)

        return {
          ...result,
          completedMono: 1_000,
          nextAdmissionInstallIds: ['later-a', 'later-b'],
          priorWaveClear: true,
          admissions: omitNext ? result.admissions.filter(row => row.installId !== 'later-b') : result.admissions
        }
      }
    }
  })

  assert.equal((await make(11_001).promote()).reason, 'promotion-proof-is-stale-or-invalid')
  assert.equal((await make(2_000, true).promote()).reason, 'promotion-proof-is-stale-or-invalid')
  assert.equal((await make(2_000).promote()).ok, true)
})

test('a losing controller records no handoff after the journal rejects its authorization', async () => {
  let owner: string | null = null
  const launches: string[] = []

  const make = (id: string) =>
    createManagedRolloutCoordinator(running(), {
      journal: {
        persistAuthorization: async () => {
          if (owner && owner !== id) {throw new Error('foreign-update-owner')}
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

// S10.8 — Union of the real production service, durable journal, and evidence
// adapters. The traces above execute the coordinator against injected fakes;
// this suite composes the same critical traces inside the production object
// graph: createManagedRolloutProvider over createManagedRolloutMainIntegration
// (the real durable journal and evidence sweep) and the shared managed-SSH
// update service admission. Only the SSH edge is scripted — every ssh.exec
// response is a fixed byte string, exactly what a healthy remote would print.

const UNION_ROOT = '/srv/hermes-agent'
const UNION_ORIGIN = 'https://github.com/NousResearch/hermes-agent.git'
const UNION_REPOSITORY_ID = canonicalRepositoryId(UNION_ORIGIN)
const UNION_TARGET_SHA = 'b'.repeat(40)
const UNION_ADMITTED_SHA = 'c'.repeat(40)
const UNION_PROFILE = 'managed-ssh-review-v1'
const UNION_NOW = 1_700_000_000_000
const UNION_NOW_ISO = new Date(UNION_NOW).toISOString()
const UNION_NOW_MONO = 1_000
const UNION_CANARY_ID = 'a'.repeat(32)
const UNION_LATER_ID = 'd'.repeat(32)
const UNION_CANARY_CONNECTION = '11111111-1111-4111-8111-111111111111'
const UNION_LATER_CONNECTION = '33333333-3333-4333-8333-333333333333'

function unionDeferred() {
  let resolve!: () => void
  const promise = new Promise<void>(done => {resolve = done})

  return { promise, resolve }
}

interface UnionSource extends ManagedSshUpdateSource {
  label: string
}

interface UnionScope extends ManagedSshUpdateScope {
  state?: object | null
}

async function makeUnionHarness(options: { beforeLaunch?: () => Promise<void> } = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-union-'))
  unionDirectories.push(directory)
  const journalRoot = path.join(directory, 'journal')
  const assuranceRoot = path.join(directory, 'assurance')
  const reviewManifestPath = path.join(directory, 'review.json')

  const updatedCorrelations = new Set<string>()
  const launches: string[] = []

  const observationFor = (correlationId: string): any => updatedCorrelations.has(correlationId)
    ? {
        marker: 'absent',
        launchIntent: 'dead',
        exitCode: 0,
        receipt: {
          correlationId,
          outcome: 'success',
          requestedSha: UNION_TARGET_SHA,
          preSha: UNION_ADMITTED_SHA,
          postSha: UNION_TARGET_SHA,
          startedAt: UNION_NOW_ISO,
          finishedAt: UNION_NOW_ISO
        },
        coordinatorReady: { correlationId, pid: 4242 }
      }
    : { marker: 'absent', launchIntent: 'absent', exitCode: null, receipt: null, coordinatorReady: null }

  const sshFor = (installId: string) => {
    let head = UNION_ADMITTED_SHA

    return {
      exec: vi.fn(async (command: string) => {
        if (command.includes('MANAGED_UPDATE_STARTED')) {
          const marker = 'HERMES_UPDATE_CORRELATION_ID='
          const tail = command.slice(command.indexOf(marker) + marker.length)
          const correlation = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/.exec(tail)?.[0]
          assert.ok(correlation, 'union launch must embed a correlation')

          updatedCorrelations.add(correlation!)
          launches.push(correlation)
          head = UNION_TARGET_SHA

          return 'MANAGED_UPDATE_STARTED'
        }

        if (command.includes('python3 -c')) {
          const correlation = command.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g)!.at(-1)!

          return JSON.stringify(observationFor(correlation))
        }

        if (command.includes("'--show-toplevel'")) {return UNION_ROOT}
        if (command.includes("'remote' 'get-url'")) {return UNION_ORIGIN}
        if (command.includes("'symbolic-ref'")) {return 'main'}
        if (command.includes("'cat-file'")) {return 'commit'}
        if (command.includes("'merge-base'")) {return ''}
        if (command.includes(':hermes_cli/update_rollout_protocol.json')) {return '{"protocol":1}'}
        if (command.includes("'rev-parse' 'HEAD'")) {return head}
        if (command.includes('echo "${HERMES_HOME:-$HOME/.hermes}"')) {return '~/.hermes'}
        if (command.includes('if [ -f')) {return installId}

        throw new Error(`unexpected union ssh command: ${command}`)
      })
    }
  }

  const sources = new Map<string, { source: UnionSource; target: any }>([
    [UNION_CANARY_CONNECTION, {
      source: { id: UNION_CANARY_CONNECTION, kind: 'ssh', label: 'union-canary' },
      target: { platform: 'Linux', hermesPath: `${UNION_ROOT}/.venv/bin/hermes`, hermesHome: '~/.hermes', ssh: sshFor(UNION_CANARY_ID) }
    }],
    [UNION_LATER_CONNECTION, {
      source: { id: UNION_LATER_CONNECTION, kind: 'ssh', label: 'union-later' },
      target: { platform: 'Linux', hermesPath: `${UNION_ROOT}/.venv/bin/hermes`, hermesHome: '~/.hermes', ssh: sshFor(UNION_LATER_ID) }
    }]
  ])

  const managedSshConfig = () => ({ user: 'hermes', host: 'union.example.test', port: 22 })
  const readHostKeyFingerprint = async () => 'SHA256:union-host-key'
  const effectiveConfigFingerprint = async () => 'union-config-revision'

  const integration = createManagedRolloutMainIntegration({
    nowMono: () => UNION_NOW_MONO,
    processOwner: () => true,
    listSources: () => [...sources.values()].map(entry => entry.source),
    getSource: connectionId => sources.get(connectionId)?.source ?? null,
    managedSshConfig,
    openTransport: async selected => ({ target: sources.get(selected.id)!.target, close: async () => undefined }),
    captureScopes: async () => [],
    readHostKeyFingerprint,
    effectiveConfigFingerprint,
    readRecoveryRecord: () => null,
    recoverManagedSsh: async () => undefined,
    reviewManifestPath,
    assuranceRoot,
    journalRoot
  })

  const capture = await integration.adapters.inventoryReader.capture()
  assert.ok(capture)
  const observations = new Map<string, any>(
    capture!.observations.map((observation: any) => [observation.installId, observation] as [string, any])
  )

  fs.mkdirSync(path.join(assuranceRoot, 'profiles'), { recursive: true })
  fs.writeFileSync(
    path.join(assuranceRoot, 'profiles', `${UNION_PROFILE}.json`),
    JSON.stringify({ generation: 1, requiredControlIds: ['managed-rollout-admission'] })
  )

  const evidenceFor = (sourceFingerprintValue: string): string => {
    const envelope = {
      schema: 1,
      profile: UNION_PROFILE,
      generation: 1,
      repositoryId: UNION_REPOSITORY_ID,
      targetSha: UNION_TARGET_SHA,
      sourceFingerprint: sourceFingerprintValue,
      observedAt: new Date(UNION_NOW - 1_000).toISOString(),
      expiresAt: new Date(UNION_NOW + 3_600_000).toISOString(),
      controls: [{ id: 'managed-rollout-admission', required: true, result: 'pass', receiptSha256: 'd'.repeat(64) }]
    }
    const bytes = Buffer.from(JSON.stringify(envelope), 'utf8')
    const evidenceDirectory = path.join(assuranceRoot, 'evidence', UNION_PROFILE, UNION_TARGET_SHA)

    fs.mkdirSync(evidenceDirectory, { recursive: true })
    fs.writeFileSync(path.join(evidenceDirectory, `${sourceFingerprintValue}.json`), bytes)

    return createHash('sha256').update(bytes).digest('hex')
  }

  const rowFor = (installId: string, connectionId: string) => {
    const observation = observations.get(installId)!
    const installation = installationFingerprint({ installId, codeRoot: observation.codeRoot, repositoryId: observation.repositoryId })
    const computed = sourceFingerprint({ ...observation.source, installationFingerprint: installation })

    assert.equal(computed, observation.computedSourceFingerprint)

    return {
      installId,
      connectionId,
      installationFingerprint: installation,
      sourceFingerprint: computed,
      admittedHead: observation.headSha,
      requiredScopeIds: [...observation.requiredScopeIds],
      eligible: true,
      reviewedSource: {
        repositoryRoot: observation.codeRoot,
        originUrl: UNION_ORIGIN,
        resolvedRef: 'refs/remotes/origin/main',
        targetSha: UNION_TARGET_SHA,
        assuranceProfile: UNION_PROFILE,
        assuranceEvidenceSha256: evidenceFor(computed),
        assuranceGeneration: 1
      }
    }
  }

  const plan = {
    target: { repositoryId: UNION_REPOSITORY_ID, branch: 'main', sha: UNION_TARGET_SHA, protocol: 1 },
    inventoryRevision: capture!.inventoryRevision,
    waves: [[UNION_CANARY_ID], [UNION_LATER_ID]],
    concurrency: 1,
    promotionPolicy: 'manual',
    rows: [rowFor(UNION_CANARY_ID, UNION_CANARY_CONNECTION), rowFor(UNION_LATER_ID, UNION_LATER_CONNECTION)],
    retryOf: null,
    exclusions: []
  }

  const resolution = {
    id: 'union-resolution',
    target: plan.target,
    fingerprint: 'f'.repeat(64),
    cachePath: path.join(directory, 'resolution'),
    createdAt: UNION_NOW - 1_000,
    expiresAt: UNION_NOW + 3_600_000
  }

  fs.writeFileSync(reviewManifestPath, JSON.stringify({ schema: 1, plan, resolution }))

  const gate = new ManagedConnectionUpdateGate(() => null)

  const service = createManagedSshUpdateService<UnionSource, UnionScope>({
    gate,
    activeUpdates: new Map(),
    activeRecoveries: new Map(),
    primaryRestoreOwners: new Map(),
    resolveSource: connectionId => sources.get(connectionId)?.source ?? null,
    resolveInstallationId: async (_source, target) => {
      const id = String((await readInstallId(target)) || '').trim().toLowerCase()

      return /^[0-9a-f]{32}$/.test(id) ? id : null
    },
    readRecoveryRecords: () => [],
    captureScopes: async () => [],
    openTransport: async selected => ({ target: sources.get(selected.id)!.target, close: async () => undefined }),
    targetFromState: () => { throw new Error('union-target-from-state-unused') },
    verifyCoordinatorSource: (selected, target, expected) =>
      verifyManagedRolloutSelectedTarget({ managedSshConfig, readHostKeyFingerprint, effectiveConfigFingerprint }, selected, target, expected),
    executeRemoteUpdate: async (target, correlationId, context) => {
      if (options.beforeLaunch) {await options.beforeLaunch()}

      return executeManagedRemoteUpdate(target, correlationId, { pollMs: 1, timeoutMs: 5_000 }, context.beforeLaunchDispatch, context.intent)
    },
    preflightRemote: assertManagedUpdatePreflightClear,
    awaitRestoreClearance: (target, correlationId, clearance) =>
      waitForManagedRemoteClearance(target, correlationId, { pollMs: 1, timeoutMs: 5_000, requireTerminal: clearance.requireTerminal }),
    drainScope: async () => undefined,
    closeTransports: async () => undefined,
    restoreScope: async () => undefined,
    prepareRecovery: async () => undefined,
    completeRecovery: async () => undefined,
    restoreRecoveryScope: async () => undefined
  })

  const provider = createManagedRolloutProvider({
    ...integration.adapters,
    journal: integration.journal,
    managedSshUpdateService: {
      issueLaunchCapability: (...args: Parameters<typeof service.issueLaunchCapability>) => service.issueLaunchCapability(...args),
      requestCoordinator: (...args: Parameters<typeof service.requestCoordinator>) => service.requestCoordinator(...args)
    },
    observe: integration.observe,
    evidence: integration.evidence,
    processGeneration: integration.processGeneration,
    ready: integration.adapters.ready,
    now: () => UNION_NOW,
    nowMono: () => UNION_NOW_MONO,
    measuredMaxInstallations: () => 2
  })

  return { provider, integration, journal: integration.journal, directory, launches, plan, observations }
}

async function startUnionRollout(harness: Awaited<ReturnType<typeof makeUnionHarness>>) {
  const resolved = await harness.provider.resolveTarget({
    connectionIds: [UNION_CANARY_CONNECTION, UNION_LATER_CONNECTION],
    inventoryRevision: harness.plan.inventoryRevision,
    retryOf: null
  }) as any

  const preflight = await harness.provider.preflight({
    inventoryRevision: harness.plan.inventoryRevision,
    targetResolutionId: resolved.resolutionId,
    waves: harness.plan.waves,
    concurrency: 1,
    promotionPolicy: 'manual',
    retryOf: null
  }) as any

  assert.equal(typeof preflight.token, 'string')
  assert.deepEqual(preflight.blockers, [])

  const started = await harness.provider.start({ token: preflight.token, requestId: preflight.requestId }) as any
  assert.equal(started.ok, true)

  return started
}

describe('production adapter union', () => {
  test('the real service, durable journal, and evidence adapters drive a two-wave fleet through promotion to completion', async () => {
    const harness = await makeUnionHarness()
    const started = await startUnionRollout(harness)

    await harness.provider.waitForIdle()

    const afterCanary = await harness.provider.get(started.id) as any
    assert.equal(afterCanary.phase, 'awaiting-promotion')
    assert.equal(afterCanary.attempts[0].phase, 'updated')
    assert.equal(afterCanary.attempts[1].phase, 'queued')

    const settled = harness.journal.read(started.id)
    assert.equal(settled.unresolved.length, 0)
    assert.ok(settled.facts.some(fact => fact.kind === 'authorization-committed'))
    assert.ok(settled.facts.some(fact => fact.kind === 'settlement-validated'))

    const promoted = await harness.provider.command({
      id: started.id,
      expectedRevision: (await harness.provider.read(null) as any).revision,
      requestId: randomUUID(),
      action: 'promote',
      installId: null,
      reason: null,
      promotionPolicy: null
    } as any) as any
    assert.equal(promoted.ok, true)
    assert.equal(promoted.code, null)

    await harness.provider.waitForIdle()

    const completed = await harness.provider.get(started.id) as any
    assert.equal(completed.phase, 'completed')
    assert.equal(completed.attempts[0].phase, 'updated')
    assert.equal(completed.attempts[1].phase, 'updated')
    assert.equal(completed.attempts[0].receipt.postSha, UNION_TARGET_SHA)
    assert.equal(completed.attempts[1].receipt.postSha, UNION_TARGET_SHA)
    assert.equal(harness.launches.length, 2)

    const record = harness.journal.read(started.id)
    assert.equal(record.unresolved.length, 0)

    for (const kind of ['authorization-committed', 'handoff-accepted', 'detached-intent', 'terminal-receipt', 'settlement-validated']) {
      assert.ok(record.facts.some(fact => fact.kind === kind), `expected durable fact ${kind}`)
    }

    assert.equal(record.facts.filter(fact => fact.kind === 'settlement-validated').length, 2)
    assert.equal(await harness.provider.activeRevision(), null)
  })

  test('stop admitted after authorization through the real service waits for the committed update and never authorizes the next wave', async () => {
    const entered = unionDeferred()
    const release = unionDeferred()
    const harness = await makeUnionHarness({
      beforeLaunch: async () => {
        entered.resolve()
        await release.promise
      }
    })
    const started = await startUnionRollout(harness)

    await entered.promise

    const stop = await harness.provider.command({
      id: started.id,
      expectedRevision: (await harness.provider.read(null) as any).revision,
      requestId: randomUUID(),
      action: 'stop',
      installId: null,
      reason: null,
      promotionPolicy: null
    } as any) as any
    assert.equal(stop.ok, true)

    const paused = await harness.provider.get(started.id) as any
    assert.equal(paused.phase, 'paused')
    assert.equal(paused.attempts[0].phase, 'awaiting-receipt')
    assert.equal(paused.attempts[1].phase, 'skipped')

    release.resolve()
    await harness.provider.waitForIdle()

    const stopped = await harness.provider.get(started.id) as any
    assert.equal(stopped.phase, 'stopped')
    assert.equal(stopped.attempts[0].phase, 'updated')
    assert.equal(stopped.attempts[1].phase, 'skipped')
    assert.equal(stopped.attempts[1].skipReason, 'stop-recorded')

    assert.equal(harness.launches.length, 1)

    const record = harness.journal.read(started.id)
    assert.equal(record.facts.filter(fact => fact.kind === 'settlement-validated').length, 1)
    assert.equal(record.unresolved.length, 0)
    assert.ok(record.events.some(event => event.kind === 'stop-requested'))
  })
})
