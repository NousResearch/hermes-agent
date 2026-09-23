import { buildHealthEvidence } from './managed-rollout-evidence'
import { createManagedRolloutJournal } from './managed-rollout-journal'
import { createManagedRolloutProductionAdapters } from './managed-rollout-production-adapters'
import { canonicalRepositoryId, installationFingerprint } from './managed-rollout-identity'
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
  if (target.platform !== 'Windows') return remoteLifecycle.readRemoteInstallId(target.ssh)
  const runtime = await windowsRemote.probeWindowsRemote(target.ssh, target.hermesPath)
  return windowsInstallId(target, runtime)
}

async function verifyProcessIdentity(target: any, scope: any): Promise<boolean> {
  const state = scope.state
  if (!state?.ssh || !state.pid || !state.spawnNonce || !state.hermesPath || !state.hermesHome) return false
  if (target.platform === 'Windows') {
    if (!state.creationTimeNs) return false
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
  if (source?.kind !== 'ssh') return null
  const config = options.managedSshConfig(source)
  if (!config) return null
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
    if (!installId || !/^[0-9a-f]{32}$/.test(String(installId)) || !/^[0-9a-f]{40}$/.test(headSha)) return null
    const scopes = await options.captureScopes(source)
    const config = options.managedSshConfig(source)
    if (!config) return null
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
  if (!source) throw new Error('reviewed-source-connection-unavailable')
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

function evidenceAdapter(inventoryReader: { capture: () => Promise<any> }) {
  return {
    async sweep(state: any) {
      const inventory = await inventoryReader.capture()
      const rows = inventory?.observations || []
      const byInstall = new Map(rows.map((row: any) => [row.installId, row]))
      const active = Object.values(state.attempts).filter((attempt: any) => !attempt.excluded)
      const admissions = active.map((attempt: any) => {
        const row: any = byInstall.get(attempt.installId)
        const installation = row
          ? installationFingerprint({ installId: row.installId, codeRoot: row.codeRoot, repositoryId: row.repositoryId })
          : ''
        const valid = Boolean(row && row.sourceFingerprint === attempt.sourceFingerprint && installation === attempt.installationFingerprint)
        return {
          installId: attempt.installId,
          installationFingerprint: valid ? installation : '',
          sourceFingerprint: valid ? row.sourceFingerprint : '',
          reviewedSource: attempt.reviewedSource,
          observationGeneration: 1,
          observedAt: new Date().toISOString()
        }
      })
      const valid = Boolean(inventory) && admissions.every((admission: any) => admission.installationFingerprint && admission.sourceFingerprint)
      return {
        rolloutId: state.id,
        revision: state.revision,
        queueGeneration: state.queueGeneration,
        processGeneration: 1,
        valid,
        reason: valid ? null : 'inventory-evidence-mismatch',
        admissions
      }
    }
  }
}

async function observeRemote(options: ManagedRolloutMainIntegrationOptions, input: any) {
  const source = options.getSource(input.authorization.connectionId)
  if (!source) return { outcome: 'unverified' as const, receipt: input.update.receipt, health: null, authorization: input.authorization }
  const transport = await options.openTransport(source)
  try {
    const raw: any = await observeManagedRemoteUpdate(transport.target, input.authorization.correlationId)
    const installId = await readInstallId(transport.target)
    const scopes = await options.captureScopes(source)
    const receipt = input.update.receipt
    const receiptCorrelated = Boolean(receipt?.correlationId === input.authorization.correlationId && raw.receipt?.correlationId === input.authorization.correlationId)
    const receiptSucceeded = Boolean(receipt && ['updated', 'already-current'].includes(receipt.outcome) && raw.receipt?.outcome === receipt.outcome)
    const markerClear = raw.marker === 'absent' || raw.marker === 'dead'
    const recoveryClear = markerClear && ['absent', 'dead'].includes(raw.launchIntent)
    const dependencyReady = raw.coordinatorReady?.correlationId === input.authorization.correlationId
    const scopeResults = await Promise.all(scopes.map(async (scope: any) => {
      const state = scope.state
      let processIdentityVerified = false
      if (state?.ssh && state.pid && state.spawnNonce && state.hermesPath && state.hermesHome && state.ownershipId) {
        processIdentityVerified = await verifyProcessIdentity(transport.target, scope)
      }
      const restored = input.update.scopes?.find((item: any) => item.profile === scope.profile)?.restored === true
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
      observationId: input.authorization.correlationId,
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
    const outcome: 'updated' | 'already-current' | 'failed' | 'refused' | 'unverified' = receiptSucceeded && receipt?.outcome === 'already-current'
      ? 'already-current'
      : input.update.ok && input.update.updateOk && input.update.restoreOk
        ? 'updated'
        : input.update.outcome === 'refused' ? 'refused' : 'failed'
    return { outcome, receipt, health, authorization: input.authorization }
  } catch {
    return { outcome: 'unverified' as const, receipt: input.update.receipt, health: null, authorization: input.authorization }
  } finally {
    await transport.close().catch(() => undefined)
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
    evidence: evidenceAdapter(adapters.inventoryReader),
    observe: { observe: (input: any) => observeRemote(options, input) }
  }
}
