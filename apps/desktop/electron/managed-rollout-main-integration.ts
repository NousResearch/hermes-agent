import { randomInt, randomUUID } from 'node:crypto'

import { validateHealthEvidence } from '../src/lib/managed-rollout-contract'

import type { ManagedRolloutState } from './managed-rollout-coordinator'
import { buildHealthEvidence, runEvidenceSweep, type SweepProbeContext } from './managed-rollout-evidence'
import { canonicalRepositoryId, installationFingerprint, sourceFingerprint } from './managed-rollout-identity'
import { createManagedRolloutJournal, type ManagedRolloutJournal } from './managed-rollout-journal'
import { createManagedRolloutProductionAdapters } from './managed-rollout-production-adapters'
import { observeManagedRemoteUpdate } from './managed-ssh-update'
import type { ManagedSshRecoveryRecord } from './managed-ssh-update-service'
import * as remoteLifecycle from './remote-lifecycle'
import * as windowsRemote from './windows-remote-lifecycle'

export interface ManagedRolloutMainIntegrationOptions {
  nowMono: () => number
  processOwner: () => boolean
  /** A fresh value per Electron main-process incarnation. */
  processGeneration?: number
  listSources: () => readonly any[]
  getSource: (connectionId: string) => any | null
  managedSshConfig: (source: any) => any | null
  openTransport: (source: any, context?: { signal: AbortSignal }) => Promise<{ close: () => Promise<void>; target: any }>
  captureScopes: (source: any) => Promise<readonly any[]>
  readHostKeyFingerprint: (config: any) => Promise<string>
  effectiveConfigFingerprint: (config: any) => Promise<string>
  reviewManifestPath: string
  assuranceRoot: string
  journalRoot: string
  readRecoveryRecord?: (connectionId: string, correlationId: string) => ManagedSshRecoveryRecord | null
  recoverManagedSsh?: (record: ManagedSshRecoveryRecord) => Promise<void>
}

function remoteGitCommand(repositoryRoot: string, args: readonly string[]): string {
  return `git -C ${remoteLifecycle.expandRemotePath(repositoryRoot)} ${args.map(remoteLifecycle.shq).join(' ')}`
}

function parseLastJson(raw: unknown): any {
  const lines = String(raw || '').replace(/^\uFEFF/, '').trim().split(/\r?\n/).filter(Boolean)

  return JSON.parse(lines.at(-1) || 'null')
}

function boundedTarget(target: any, options: ManagedRolloutMainIntegrationOptions, context?: SweepProbeContext): any {
  if (!context) {return target}

  return {
    ...target,
    ssh: {
      exec: (command: string, execOptions: { timeoutMs?: number; stdinData?: string } = {}) => {
        const remainingMs = Math.floor(context.deadlineMono - options.nowMono())

        if (context.signal.aborted || !Number.isFinite(remainingMs) || remainingMs <= 0) {throw new Error('probe-timeout')}

        return target.ssh.exec(command, {
          ...execOptions,
          timeoutMs: Math.min(execOptions.timeoutMs ?? remainingMs, remainingMs),
          signal: context.signal
        })
      }
    }
  }
}

function windowsInspectionCommand(runtime: any): string {
  const hermes = windowsRemote.psLiteral(runtime.hermesPath)

  return windowsRemote.powerShellCommand([
    '$ErrorActionPreference="Stop"',
    `$hermes=${hermes}`,
    '$root=(& git -C (Split-Path -Parent $hermes) rev-parse --show-toplevel).Trim()',
    'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}',
    '$origin=(& git -C $root remote get-url origin).Trim()',
    'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}',
    '$head=(& git -C $root rev-parse HEAD).Trim()',
    'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}',
    '[ordered]@{codeRoot=$root;originUrl=$origin;headSha=$head}|ConvertTo-Json -Compress'
  ].join(';'))
}

async function windowsInstallId(target: any, runtime: any): Promise<string> {
  const home = windowsRemote.psLiteral(runtime.hermesHome)

  const script = [
    '$ErrorActionPreference="Stop"',
    `$home=${home}`,
    '$root=$home',
    '$parent=Split-Path -Parent $home',
    'if((Split-Path -Leaf $parent) -ieq "profiles"){$root=Split-Path -Parent $parent}',
    '$file=Join-Path $root "install_id"',
    'if(Test-Path -LiteralPath $file -PathType Leaf){(Get-Content -LiteralPath $file -Raw).Trim()}'
  ].join(';')

  return String(await target.ssh.exec(windowsRemote.powerShellCommand(script))).trim().split(/\r?\n/).pop() || ''
}

async function readInstallId(target: any): Promise<string> {
  if (target.platform !== 'Windows') {return remoteLifecycle.readRemoteInstallId(target.ssh)}
  const runtime = await windowsRemote.probeWindowsRemote(target.ssh, target.hermesPath)

  return windowsInstallId(target, runtime)
}

async function verifyProcessIdentity(target: any, scope: any, options: ManagedRolloutMainIntegrationOptions, context?: SweepProbeContext): Promise<boolean> {
  const state = scope.state

  if (!state?.ssh || !state.pid || !state.spawnNonce || !state.hermesPath || !state.hermesHome) {return false}

  if (target.platform === 'Windows') {
    if (!state.creationTimeNs) {return false}
    const runtime = await windowsRemote.probeWindowsRemote(target.ssh, target.hermesPath)

    const result = await windowsRemote.helper(target.ssh, runtime, 'process-state', [
      String(state.pid), String(state.creationTimeNs), String(state.hermesPath), String(state.spawnNonce)
    ])

    return result?.alive === true && result?.owned === true && result?.indeterminate !== true
  }

  return remoteLifecycle.pidIsOurDashboard(
    boundedTarget({ ssh: state.ssh }, options, context).ssh,
    state.pid,
    state.spawnNonce,
    state.hermesPath,
    state.hermesHome,
    state.ownershipId,
    scope.profile
  )
}

async function inspectSource(options: ManagedRolloutMainIntegrationOptions, source: any, context?: SweepProbeContext) {
  if (source?.kind !== 'ssh') {return null}
  const config = options.managedSshConfig(source)

  if (!config) {return null}
  const transport = await options.openTransport(source, context)

  try {
    const target = boundedTarget(transport.target, options, context)
    const isWindows = target.platform === 'Windows'

    const runtime = isWindows
      ? await windowsRemote.probeWindowsRemote(target.ssh, target.hermesPath)
      : null

    const hermesPath = runtime?.hermesPath || target.hermesPath

    const remoteGit = async (args: readonly string[]) =>
      String(await target.ssh.exec(`git -C "$(dirname ${remoteLifecycle.expandRemotePath(hermesPath)})" ${args.map(remoteLifecycle.shq).join(' ')}`, { timeoutMs: 10_000 })).trim()

    const [sourceData, installId] = isWindows
      ? [parseLastJson(await target.ssh.exec(windowsInspectionCommand(runtime))), await windowsInstallId(target, runtime)]
      : [{
          codeRoot: await remoteGit(['rev-parse', '--show-toplevel']),
          originUrl: await remoteGit(['remote', 'get-url', 'origin']),
          headSha: await remoteGit(['rev-parse', 'HEAD'])
        }, await remoteLifecycle.readRemoteInstallId(target.ssh)]

    const codeRoot = String(sourceData.codeRoot || '')
    const originUrl = String(sourceData.originUrl || '')
    const headSha = String(sourceData.headSha || '')

    if (!installId || !/^[0-9a-f]{32}$/.test(String(installId)) || !/^[0-9a-f]{40}$/.test(headSha)) {return null}
    const scopes = await options.captureScopes(source)
    const config = options.managedSshConfig(source)

    if (!config) {return null}
    const connectionConfigRevision = source.effectiveConfigFingerprint || await options.effectiveConfigFingerprint(config)

    return {
      installId: String(installId),
      codeRoot,
      repositoryId: canonicalRepositoryId(originUrl),
      headSha,
      requiredScopeIds: scopes.map(scope => String(scope.key)).sort(),
      source: {
        connectionId: source.id,
        connectionConfigRevision,
        verifiedHostKeyFingerprint: await options.readHostKeyFingerprint(config),
        remoteUser: String(config.user || ''),
        port: Number(config.port || 22),
        configuredProfile: String(config.remoteProfile || 'default'),
        configuredCodePath: hermesPath
      }
    }
  } catch {
    return null
  } finally {
    await transport.close().catch(() => undefined)
  }
}

async function runGit(options: ManagedRolloutMainIntegrationOptions, connectionId: string, args: readonly string[], repositoryRoot: string) {
  const source = options.getSource(connectionId)

  if (!source) {throw new Error('reviewed-source-connection-unavailable')}
  const transport = await options.openTransport(source)

  try {
    if (transport.target.platform === 'Windows') {
      const command = windowsRemote.powerShellCommand([
        '$ErrorActionPreference="Stop"',
        `& git -C ${windowsRemote.psLiteral(repositoryRoot)} ${args.map(windowsRemote.psLiteral).join(' ')}`,
        'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}'
      ].join(';'))

      return await transport.target.ssh.exec(command, { timeoutMs: 10_000 })
    }

    return await transport.target.ssh.exec(remoteGitCommand(repositoryRoot, args), { timeoutMs: 10_000 })
  } finally {
    await transport.close().catch(() => undefined)
  }
}

function sameScopeIds(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length || new Set(left).size !== left.length || new Set(right).size !== right.length) {return false}

  const sortedRight = [...right].sort()

  return [...left].sort().every((id, index) => id === sortedRight[index])
}

function localAttemptMatches(attempt: any, persisted: any): boolean {
  return persisted?.identity?.installId === attempt.installId &&
    persisted.identity.installationFingerprint === attempt.installationFingerprint &&
    persisted.identity.sourceFingerprint === attempt.sourceFingerprint &&
    persisted.correlationId === attempt.correlationId && persisted.wave === attempt.wave &&
    Array.isArray(persisted.requiredScopeIds) &&
    persisted.requiredScopeIds.every((id: unknown) => typeof id === 'string')
}

function priorWaveLocalClear(state: ManagedRolloutState, record: ReturnType<ManagedRolloutJournal['read']>): boolean {
  const prior = Object.values(state.attempts).filter(attempt => !attempt.excluded && attempt.wave < state.currentWave)
  const persistedById = new Map<string, any>(record.snapshot.attempts.map((row: any) => [row.identity?.installId, row]))

  if (record.unresolved.length > 0) {return false}

  return prior.every(attempt => {
    const persisted = persistedById.get(attempt.installId)

    if (!localAttemptMatches(attempt, persisted) ||
        !['updated', 'already-current'].includes(attempt.state) ||
        !['updated', 'already-current'].includes(persisted.phase) ||
        persisted.recoveryRequired || persisted.receipt?.correlationId !== attempt.correlationId ||
        persisted.receipt?.postSha !== attempt.targetSha) {return false}

    let health

    try {health = validateHealthEvidence(persisted.health)}
    catch {return false}

    return health.installId === attempt.installId && health.checkoutSha === attempt.targetSha &&
      health.installReady && health.markerClear && health.receiptCorrelated && health.receiptSucceeded &&
      health.dependencyReady && health.recoveryClear && health.scopeCapture === 'complete' &&
      sameScopeIds(persisted.requiredScopeIds, health.scopes.map(scope => scope.scopeId)) &&
      health.scopes.every(scope => scope.restored && scope.ready && scope.processIdentityVerified && scope.codeSha === attempt.targetSha)
  })
}

function evidenceAdapter(
  options: ManagedRolloutMainIntegrationOptions,
  journal: ManagedRolloutJournal,
  processGeneration: number
) {
  return {
    async sweep(state: ManagedRolloutState) {
      const refused = (reason: string) => ({
        rolloutId: state.id, revision: state.revision, queueGeneration: state.queueGeneration,
        processGeneration, completedMono: options.nowMono(), priorWaveClear: false,
        nextAdmissionInstallIds: [] as string[], valid: false, reason, admissions: []
      })
      let before: ReturnType<ManagedRolloutJournal['read']>

      try {before = journal.read(state.id)}
      catch {return refused('promotion-journal-unavailable')}

      const attempts = Object.values(state.attempts).filter(attempt => !attempt.excluded)
      const nextWave = Math.min(...attempts.filter(attempt => attempt.wave > state.currentWave).map(attempt => attempt.wave))

      if (nextWave !== state.currentWave + 1) {return refused('next-wave-unavailable')}

      const sweepAttempts = attempts.filter(attempt => attempt.wave === state.currentWave || attempt.wave === nextWave)
      const byAttempt = new Map(sweepAttempts.map(attempt => [attempt.installId, attempt]))
      const localAttempts = new Map<string, any>(before.snapshot.attempts.map((row: any) => [row.identity?.installId, row]))

      if (sweepAttempts.some(attempt => !localAttemptMatches(attempt, localAttempts.get(attempt.installId)))) {
        return refused('promotion-local-state-mismatch')
      }

      if (!priorWaveLocalClear(state, before)) {return refused('prior-wave-local-proof-missing')}

      const targets = sweepAttempts.map(attempt => ({
        installId: attempt.installId,
        requiredScopeIds: localAttempts.get(attempt.installId).requiredScopeIds as string[],
        wave: attempt.wave,
        excluded: false
      }))

      const facts = new Map<string, { installationFingerprint: string; sourceFingerprint: string; headSha: string; scopeIds: string[]; markerClear: boolean }>()
      const epochId = `managed-rollout:${state.id}:${state.revision}:${state.queueGeneration}:${processGeneration}:${randomUUID()}`

      let sweep: Awaited<ReturnType<typeof runEvidenceSweep>>

      try {sweep = await runEvidenceSweep(
        targets,
        async (target, context) => {
          const attempt = byAttempt.get(target.installId)

          if (!attempt) {throw new Error('sweep-attempt-missing')}
          const source = options.getSource(attempt.connectionId)

          if (!source) {throw new Error('sweep-source-unavailable')}

          const authorization = {
            rolloutId: state.id,
            installId: attempt.installId,
            connectionId: attempt.connectionId,
            installationFingerprint: attempt.installationFingerprint,
            sourceFingerprint: attempt.sourceFingerprint,
            targetSha: attempt.targetSha,
            reviewedSource: attempt.reviewedSource,
            correlationId: attempt.correlationId,
            queueGeneration: state.queueGeneration
          }

          const observed = attempt.wave === state.currentWave
            ? await observeHealth(options, authorization, null, context.epochId, null, context)
            : null
          const inspection = await inspectSource(options, source, context)

          if (!inspection) {throw new Error('sweep-source-inspection-failed')}

          const installation = installationFingerprint(inspection)
          const fingerprint = sourceFingerprint({ ...inspection.source, installationFingerprint: installation })
          let markerClear = true

          if (!observed) {
            const transport = await options.openTransport(source, context)

            try {
              const raw: any = await observeManagedRemoteUpdate(boundedTarget(transport.target, options, context), attempt.correlationId)

              markerClear = ['absent', 'dead'].includes(raw.marker) &&
                ['absent', 'dead'].includes(raw.launchIntent) && !raw.receipt
            } finally {
              await transport.close().catch(() => undefined)
            }
          }

          facts.set(target.installId, {
            installationFingerprint: installation, sourceFingerprint: fingerprint,
            headSha: inspection.headSha, scopeIds: [...inspection.requiredScopeIds], markerClear
          })

          if (observed) {
            return { health: buildHealthEvidence({
              ...observed.health,
              checkoutSha: inspection.headSha,
              installReady: observed.health.installReady && inspection.headSha === attempt.targetSha,
              reasons: inspection.headSha === attempt.targetSha
                ? observed.health.reasons : [...observed.health.reasons, 'target-head-mismatch']
            }) }
          }

          return { health: buildHealthEvidence({
            observationId: context.epochId, observedAt: new Date().toISOString(),
            installId: inspection.installId, checkoutSha: inspection.headSha,
            installReady: false, markerClear, receiptCorrelated: false, receiptSucceeded: false,
            dependencyReady: false, recoveryClear: markerClear,
            scopes: inspection.requiredScopeIds.map(scopeId => ({
              scopeId, profile: scopeId, restored: false, ready: false,
              codeSha: inspection.headSha, processIdentityVerified: false
            })),
            reasons: ['next-wave-admission-only']
          }) }
        },
        { epochId, nowMono: options.nowMono }
      )} catch (error) {
        return refused(error instanceof Error ? error.message : 'evidence-sweep-failed')
      }

      let after: ReturnType<ManagedRolloutJournal['read']>

      try {after = journal.read(state.id)}
      catch {return refused('promotion-journal-unavailable')}

      const localStable = before.generation === after.generation &&
        before.snapshot.revision === after.snapshot.revision && priorWaveLocalClear(state, after)
      const byObservedInstall = new Map(sweep.observations.map(observation => [observation.installId, observation]))
      const checked = sweepAttempts.every(attempt => {
        const observation = byObservedInstall.get(attempt.installId)
        const fact = facts.get(attempt.installId)
        const expected = localAttempts.get(attempt.installId)

        if (!observation || !fact ||
            fact.installationFingerprint !== attempt.installationFingerprint ||
            fact.sourceFingerprint !== attempt.sourceFingerprint ||
            !sameScopeIds(fact.scopeIds, expected.requiredScopeIds) ||
            observation.health.installId !== attempt.installId) {return false}

        if (attempt.wave === nextWave) {
          return fact.markerClear && fact.headSha === expected.identity.admittedSha
        }

        const health = observation.health

        return fact.headSha === attempt.targetSha && health.checkoutSha === attempt.targetSha &&
          health.installReady && health.markerClear && health.receiptCorrelated && health.receiptSucceeded &&
          health.dependencyReady && health.recoveryClear && health.scopeCapture === 'complete' &&
          sameScopeIds(expected.requiredScopeIds, health.scopes.map(scope => scope.scopeId)) &&
          health.scopes.every(scope => scope.restored && scope.ready && scope.processIdentityVerified && scope.codeSha === attempt.targetSha)
      })
      const valid = sweep.ok && localStable && checked

      const admissions = sweep.observations.flatMap(observation => {
        const attempt = byAttempt.get(observation.installId)
        const fact = facts.get(observation.installId)

        return attempt && fact ? [{
          installId: observation.installId,
          installationFingerprint: fact.installationFingerprint,
          sourceFingerprint: fact.sourceFingerprint,
          reviewedSource: attempt.reviewedSource,
          observationGeneration: processGeneration,
          observedAt: observation.health.observedAt
        }] : []
      })

      return {
        rolloutId: state.id,
        revision: state.revision,
        queueGeneration: state.queueGeneration,
        processGeneration,
        completedMono: sweep.finishedMono,
        priorWaveClear: localStable,
        nextAdmissionInstallIds: sweep.nextAdmissionInstallIds,
        valid,
        reason: valid ? null : (sweep.errors[0]?.reason || (!localStable ? 'prior-wave-local-proof-missing' : 'health-evidence-not-proven')),
        admissions
      }
    }
  }
}

async function observeHealth(
  options: ManagedRolloutMainIntegrationOptions,
  authorization: any,
  expectedReceipt: any = null,
  observationId = authorization.correlationId,
  expectedScopes: readonly any[] | null = null,
  context?: SweepProbeContext
) {
  const source = options.getSource(authorization.connectionId)

  if (!source) {throw new Error('observed-source-connection-unavailable')}
  const transport = await options.openTransport(source, context)

  try {
    const target = boundedTarget(transport.target, options, context)
    const raw: any = await observeManagedRemoteUpdate(target, authorization.correlationId)
    const installId = await readInstallId(target)
    const scopes = await options.captureScopes(source)
    const receipt = expectedReceipt || raw.receipt

    const receiptCorrelated = Boolean(
      raw.receipt?.correlationId === authorization.correlationId &&
      (!expectedReceipt || expectedReceipt.correlationId === authorization.correlationId)
    )

    const receiptSucceeded = Boolean(
      raw.receipt && ['updated', 'already-current'].includes(raw.receipt.outcome) &&
      (!expectedReceipt || expectedReceipt.outcome === raw.receipt.outcome)
    )

    const markerClear = raw.marker === 'absent' || raw.marker === 'dead'
    const recoveryClear = markerClear && ['absent', 'dead'].includes(raw.launchIntent)
    const dependencyReady = raw.coordinatorReady?.correlationId === authorization.correlationId

    const scopeResults = []

    for (const scope of scopes) {
      const state = scope.state

      const processIdentityVerified = state?.ssh && state.pid && state.spawnNonce && state.hermesPath && state.hermesHome && state.ownershipId
        ? await verifyProcessIdentity(target, scope, options, context)
        : false

      const restored = expectedReceipt
        ? expectedScopes?.find((item: any) => item.profile === scope.profile)?.restored === true
        : true

      scopeResults.push({
        scopeId: String(scope.key),
        profile: String(scope.profile),
        restored,
        ready: restored && processIdentityVerified,
        codeSha: receipt?.postSha || raw.receipt?.postSha || null,
        processIdentityVerified
      })
    }

    const health = buildHealthEvidence({
      observationId,
      observedAt: new Date().toISOString(),
      installId: installId || null,
      checkoutSha: receipt?.postSha || raw.receipt?.postSha || null,
      installReady: receiptSucceeded && markerClear,
      markerClear,
      receiptCorrelated,
      receiptSucceeded,
      dependencyReady,
      recoveryClear,
      scopes: scopeResults,
      reasons: [
        ...(receiptCorrelated ? [] : ['receipt-correlation-missing']),
        ...(receiptSucceeded ? [] : ['receipt-success-not-proven']),
        ...(markerClear ? [] : ['update-marker-not-clear']),
        ...(dependencyReady ? [] : ['coordinator-readiness-not-proven']),
        ...(recoveryClear ? [] : ['recovery-not-clear'])
      ]
    })

    return { health, receipt }
  } finally {
    await transport.close().catch(() => undefined)
  }
}

async function observeRemote(options: ManagedRolloutMainIntegrationOptions, input: any) {
  try {
    const observed = await observeHealth(options, input.authorization, input.update.receipt, input.authorization.correlationId, input.update.scopes || [])

    const outcome: 'updated' | 'already-current' | 'failed' | 'refused' | 'unverified' = observed.receipt?.outcome === 'already-current'
      ? 'already-current'
      : input.update.ok && input.update.updateOk && input.update.restoreOk
        ? 'updated'
        : input.update.outcome === 'refused' ? 'refused' : 'failed'

    return { outcome, receipt: observed.receipt, health: observed.health, authorization: input.authorization }
  } catch {
    return { outcome: 'unverified' as const, receipt: input.update.receipt, health: null, authorization: input.authorization }
  }
}

async function reprobeRemote(options: ManagedRolloutMainIntegrationOptions, authorization: any) {
  const source = options.getSource(authorization.connectionId)

  if (!source) {throw new Error('reprobe-source-connection-unavailable')}
  const transport = await options.openTransport(source)

  try {
    const raw: any = await observeManagedRemoteUpdate(transport.target, authorization.correlationId)
    const receipt = raw.receipt
    const correlationId = typeof receipt?.correlationId === 'string' ? receipt.correlationId : ''

    const terminal = correlationId === authorization.correlationId &&
      ['updated', 'already-current', 'failed', 'refused'].includes(receipt?.outcome) &&
      ['absent', 'dead'].includes(raw.marker) && ['absent', 'dead'].includes(raw.launchIntent)

    return {
      correlationId,
      outcome: terminal ? receipt.outcome : 'unverified',
      terminal
    }
  } finally {
    await transport.close().catch(() => undefined)
  }
}

async function recoverRemote(options: ManagedRolloutMainIntegrationOptions, authorization: any) {
  if (!options.recoverManagedSsh || !options.readRecoveryRecord) {
    return { correlationId: authorization.correlationId, clearanceProved: false }
  }

  const record = options.readRecoveryRecord(authorization.connectionId, authorization.correlationId)

  if (
    !record || record.connectionId !== authorization.connectionId ||
    record.correlationId !== authorization.correlationId ||
    record.source?.id !== authorization.connectionId || record.source.kind !== 'ssh' ||
    !Array.isArray(record.scopes) || !['prepared', 'launching'].includes(record.phase)
  ) {
    return { correlationId: authorization.correlationId, clearanceProved: false }
  }

  const transport = await options.openTransport(record.source)

  try {
    const raw: any = await observeManagedRemoteUpdate(transport.target, authorization.correlationId)
    const receipt = raw.receipt
    const correlated = receipt?.correlationId === authorization.correlationId
    const clear = ['absent', 'dead'].includes(raw.marker) && ['absent', 'dead'].includes(raw.launchIntent)

    if (!correlated || !clear) {
      return { correlationId: typeof receipt?.correlationId === 'string' ? receipt.correlationId : '', clearanceProved: false }
    }

    await options.recoverManagedSsh(record)

    // The service may return after a blocked or incomplete restoration. Only
    // removal of the original durable obligation proves local clearance.
    const remaining = options.readRecoveryRecord(authorization.connectionId, authorization.correlationId)

    return { correlationId: authorization.correlationId, clearanceProved: remaining === null }
  } finally {
    await transport.close().catch(() => undefined)
  }
}

export function createManagedRolloutMainIntegration(options: ManagedRolloutMainIntegrationOptions) {
  if (!options.processOwner()) {throw new Error('managed-rollout-owner-unavailable')}
  const processGeneration = options.processGeneration ?? randomInt(1, 2_147_483_647)

  if (!Number.isSafeInteger(processGeneration) || processGeneration < 1) {
    throw new Error('managed-rollout-process-generation-invalid')
  }

  const adapters = createManagedRolloutProductionAdapters({
    nowMono: options.nowMono,
    listSources: options.listSources,
    inspectSource: source => inspectSource(options, source),
    git: (connectionId, args, repositoryRoot) => runGit(options, connectionId, args, repositoryRoot),
    reviewManifestPath: options.reviewManifestPath,
    assuranceRoot: options.assuranceRoot
  })

  const journal = createManagedRolloutJournal({
    directory: options.journalRoot,
    processOwner: options.processOwner,
    clock: () => new Date().toISOString(),
    retentionLimit: 200
  })

  return {
    adapters,
    journal,
    evidence: evidenceAdapter(options, journal, processGeneration),
    processGeneration,
    observe: {
      observe: (input: any) => observeRemote(options, input),
      reprobe: (authorization: any) => reprobeRemote(options, authorization),
      recover: (authorization: any) => recoverRemote(options, authorization)
    }
  }
}
