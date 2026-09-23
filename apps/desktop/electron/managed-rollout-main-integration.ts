import { buildHealthEvidence, runEvidenceSweep } from './managed-rollout-evidence'
import { canonicalRepositoryId } from './managed-rollout-identity'
import { createManagedRolloutJournal } from './managed-rollout-journal'
import { createManagedRolloutProductionAdapters } from './managed-rollout-production-adapters'
import { observeManagedRemoteUpdate } from './managed-ssh-update'
import * as remoteLifecycle from './remote-lifecycle'
import * as windowsRemote from './windows-remote-lifecycle'

export interface ManagedRolloutMainIntegrationOptions {
  nowMono: () => number
  listSources: () => readonly any[]
  getSource: (connectionId: string) => any | null
  managedSshConfig: (source: any) => any | null
  openTransport: (source: any) => Promise<{ close: () => Promise<void>; target: any }>
  captureScopes: (source: any) => Promise<readonly any[]>
  readHostKeyFingerprint: (config: any) => Promise<string>
  effectiveConfigFingerprint: (config: any) => Promise<string>
  reviewManifestPath: string
  assuranceRoot: string
  journalRoot: string
}

function remoteGitCommand(repositoryRoot: string, args: readonly string[]): string {
  return `git -C ${remoteLifecycle.expandRemotePath(repositoryRoot)} ${args.map(remoteLifecycle.shq).join(' ')}`
}

function parseLastJson(raw: unknown): any {
  const lines = String(raw || '').replace(/^\uFEFF/, '').trim().split(/\r?\n/).filter(Boolean)

  return JSON.parse(lines.at(-1) || 'null')
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

async function verifyProcessIdentity(target: any, scope: any): Promise<boolean> {
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
    state.ssh,
    state.pid,
    state.spawnNonce,
    state.hermesPath,
    state.hermesHome,
    state.ownershipId,
    scope.profile
  )
}

async function inspectSource(options: ManagedRolloutMainIntegrationOptions, source: any) {
  if (source?.kind !== 'ssh') {return null}
  const config = options.managedSshConfig(source)

  if (!config) {return null}
  const transport = await options.openTransport(source)

  try {
    const isWindows = transport.target.platform === 'Windows'

    const runtime = isWindows
      ? await windowsRemote.probeWindowsRemote(transport.target.ssh, transport.target.hermesPath)
      : null

    const hermesPath = runtime?.hermesPath || transport.target.hermesPath

    const remoteGit = async (args: readonly string[]) =>
      String(await transport.target.ssh.exec(`git -C "$(dirname ${remoteLifecycle.expandRemotePath(hermesPath)})" ${args.map(remoteLifecycle.shq).join(' ')}`, { timeoutMs: 10_000 })).trim()

    const [sourceData, installId] = isWindows
      ? [parseLastJson(await transport.target.ssh.exec(windowsInspectionCommand(runtime))), await windowsInstallId(transport.target, runtime)]
      : await Promise.all([
          Promise.all([
            remoteGit(['rev-parse', '--show-toplevel']),
            remoteGit(['remote', 'get-url', 'origin']),
            remoteGit(['rev-parse', 'HEAD'])
          ]).then(([codeRoot, originUrl, headSha]) => ({ codeRoot, originUrl, headSha })),
          remoteLifecycle.readRemoteInstallId(transport.target.ssh)
        ])

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

function evidenceAdapter(
  options: ManagedRolloutMainIntegrationOptions,
  inventoryReader: { capture: () => Promise<any> },
  runGitForSource: (connectionId: string, args: readonly string[], repositoryRoot: string) => Promise<unknown>
) {
  return {
    async sweep(state: any) {
      const inventory = await inventoryReader.capture()
      const rows = inventory?.observations || []
      const byInstall = new Map<string, any>(rows.map((row: any) => [row.installId, row]))
      const attempts = Object.values(state.attempts) as any[]

      const targets = attempts.map(attempt => ({
        installId: attempt.installId,
        requiredScopeIds: byInstall.get(attempt.installId)?.requiredScopeIds ?? null,
        wave: attempt.wave,
        excluded: Boolean(attempt.excluded)
      }))

      const epochId = `managed-rollout:${state.id}:${state.revision}:${state.queueGeneration}`

      const sweep = await runEvidenceSweep(
        targets,
        async (target, context) => {
          const attempt = attempts.find(candidate => candidate.installId === target.installId)

          if (!attempt) {throw new Error('sweep-attempt-missing')}

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

          const observed = await observeHealth(options, authorization, null, context.epochId)

          const head = String(await runGitForSource(
            attempt.connectionId,
            ['rev-parse', 'HEAD'],
            attempt.reviewedSource.repositoryRoot
          )).trim()

          if (head !== attempt.targetSha) {
            return {
              health: buildHealthEvidence({
                ...observed.health,
                checkoutSha: head || null,
                installReady: false,
                reasons: [...observed.health.reasons, 'target-head-mismatch']
              })
            }
          }

          return observed
        },
        { epochId, nowMono: options.nowMono }
      )

      const byObservedInstall = new Map(sweep.observations.map(observation => [observation.installId, observation]))

      const healthy = attempts.filter(attempt => !attempt.excluded).every(attempt => {
        const observation = byObservedInstall.get(attempt.installId)

        return Boolean(
          observation && observation.health.installId === attempt.installId &&
          observation.health.checkoutSha === attempt.targetSha &&
          observation.health.installReady && observation.health.markerClear &&
          observation.health.receiptCorrelated && observation.health.receiptSucceeded &&
          observation.health.dependencyReady && observation.health.recoveryClear
        )
      })

      const valid = sweep.ok && healthy

      const admissions = sweep.observations.map(observation => {
        const attempt = attempts.find(candidate => candidate.installId === observation.installId)

        return {
          installId: observation.installId,
          installationFingerprint: attempt?.installationFingerprint || '',
          sourceFingerprint: attempt?.sourceFingerprint || '',
          reviewedSource: attempt?.reviewedSource,
          observationGeneration: 1,
          observedAt: observation.health.observedAt
        }
      }).filter(admission => admission.reviewedSource)

      return {
        rolloutId: state.id,
        revision: state.revision,
        queueGeneration: state.queueGeneration,
        processGeneration: 1,
        valid,
        reason: valid ? null : (sweep.errors[0]?.reason || (healthy ? 'evidence-sweep-incomplete' : 'health-evidence-not-proven')),
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
  expectedScopes: readonly any[] | null = null
) {
  const source = options.getSource(authorization.connectionId)

  if (!source) {throw new Error('observed-source-connection-unavailable')}
  const transport = await options.openTransport(source)

  try {
    const raw: any = await observeManagedRemoteUpdate(transport.target, authorization.correlationId)
    const installId = await readInstallId(transport.target)
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

    const scopeResults = await Promise.all(scopes.map(async (scope: any) => {
      const state = scope.state

      const processIdentityVerified = state?.ssh && state.pid && state.spawnNonce && state.hermesPath && state.hermesHome && state.ownershipId
        ? await verifyProcessIdentity(transport.target, scope)
        : false

      const restored = expectedReceipt
        ? expectedScopes?.find((item: any) => item.profile === scope.profile)?.restored === true
        : true

      return {
        scopeId: String(scope.key),
        profile: String(scope.profile),
        restored,
        ready: restored && processIdentityVerified,
        codeSha: receipt?.postSha || raw.receipt?.postSha || null,
        processIdentityVerified
      }
    }))

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

export function createManagedRolloutMainIntegration(options: ManagedRolloutMainIntegrationOptions) {
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
    clock: () => new Date().toISOString(),
    retentionLimit: 200
  })

  return {
    adapters,
    journal,
    evidence: evidenceAdapter(
      options,
      adapters.inventoryReader,
      (connectionId, args, repositoryRoot) => runGit(options, connectionId, args, repositoryRoot)
    ),
    observe: { observe: (input: any) => observeRemote(options, input) }
  }
}
