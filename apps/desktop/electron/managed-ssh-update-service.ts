/**
 * Main-process managed SSH update service.
 *
 * This is the ownership seam extracted from main.ts.  It owns admission,
 * correlation, durable fencing, and the shared update/restore transaction;
 * Electron's pool, transport, and recovery-journal details remain injected.
 */

import crypto from 'node:crypto'

import {
  ManagedConnectionUpdateGate,
  type ManagedConnectionUpdateResult,
  type ManagedSshRecoveryScope,
  type ManagedSshUpdateIntent,
  recoverManagedSshScopes,
  refusedManagedSshUpdate,
  type RemoteUpdateProof,
  type RemoteUpdateTarget,
  runManagedSshUpdate,
  validateCorrelationId,
  validateManagedSshUpdateIntent
} from './managed-ssh-update'

export interface ManagedSshUpdateSource {
  id: string
  kind: string
  [key: string]: unknown
}

export interface ManagedSshUpdateScope {
  key: string
  profile: string
  state?: unknown
  primary?: boolean
  registryScoped?: boolean
  [key: string]: unknown
}

export interface ManagedSshRecoveryRecord {
  connectionId: string
  correlationId: string
  /** Canonical remote install_id, persisted before a remote mutation. */
  installationId?: string
  phase: string
  scopes: ManagedSshRecoveryScope[]
  source: ManagedSshUpdateSource
  [key: string]: unknown
}

export interface ManagedSshUpdateTransport {
  target: RemoteUpdateTarget
  close: () => Promise<void>
}

export type ManagedSshUpdateMode = 'legacy' | 'coordinator'

/** Opaque, service-issued authority to send one reviewed remote mutation. */
export interface ManagedSshLaunchCapability {
  readonly __managedSshLaunchCapability: true
}

/** Main-owned installation and SSH route reviewed for one coordinator launch. */
export interface ManagedSshCoordinatorSourceBinding {
  installId: string
  installationFingerprint: string
  sourceFingerprint: string
}

export interface ManagedSshUpdateRequestOptions {
  correlationId?: string
  intent?: ManagedSshUpdateIntent
  mode?: ManagedSshUpdateMode
  launchCapability?: ManagedSshLaunchCapability
  expectedSource?: ManagedSshCoordinatorSourceBinding
}

/** Synchronous admission verdict; the operation continues asynchronously after acceptance. */
export type ManagedSshUpdateAdmission =
  | { admitted: true; operation: Promise<ManagedConnectionUpdateResult> }
  | { admitted: false; reason: string; operation: Promise<ManagedConnectionUpdateResult> }

export interface ManagedSshPreparationReceipt {
  kind: 'preparation'
  correlationId: string
  [key: string]: unknown
}

export interface ManagedSshPreparationResult {
  connectionId: string
  correlationId: string
  ok: boolean
  outcome: 'prepared' | 'preparation-failed' | 'refused'
  receipt: ManagedSshPreparationReceipt | null
  error?: string
}

export interface ManagedSshUpdateServiceDependencies<
  TSource extends ManagedSshUpdateSource = ManagedSshUpdateSource,
  TScope extends ManagedSshUpdateScope = ManagedSshUpdateScope,
  TRecord extends ManagedSshRecoveryRecord = ManagedSshRecoveryRecord
> {
  resolveSource: (connectionId: string) => TSource | null | undefined
  /** Read the selected remote target's install_id without mutating it. */
  resolveInstallationId: (source: TSource, target: RemoteUpdateTarget) => Promise<string | null>
  readRecoveryRecords: () => TRecord[]
  /** Reuse the Desktop-owned admission gate and operation maps across every update entry point. */
  gate?: ManagedConnectionUpdateGate
  activeUpdates?: Map<string, Promise<ManagedConnectionUpdateResult>>
  activeRecoveries?: Map<string, Promise<void>>
  primaryRestoreOwners?: Map<string, { correlationId: string; profile: string; source: TSource }>
  captureScopes: (source: TSource) => Promise<TScope[]>
  openTransport: (source: TSource) => Promise<ManagedSshUpdateTransport>
  targetFromState: (state: unknown) => RemoteUpdateTarget
  executeRemoteUpdate: (
    target: RemoteUpdateTarget,
    correlationId: string,
    context: { connectionId: string; intent?: ManagedSshUpdateIntent; beforeLaunchDispatch: () => Promise<void> }
  ) => Promise<RemoteUpdateProof>
  /** Probe the selected transport, live registry route, installation, and host key before mutation. */
  verifyCoordinatorSource?: (
    source: TSource,
    target: RemoteUpdateTarget,
    expected: ManagedSshCoordinatorSourceBinding & { expectedCurrentSha: string }
  ) => Promise<void>
  preflightRemote: (target: RemoteUpdateTarget, correlationId: string) => Promise<unknown>
  awaitRestoreClearance: (
    target: RemoteUpdateTarget,
    correlationId: string,
    options: { requireTerminal: boolean }
  ) => Promise<unknown>
  drainScope: (scope: TScope) => Promise<unknown>
  closeTransports: (scopes: TScope[], ephemeral: ManagedSshUpdateTransport | null) => Promise<unknown>
  restoreScope: (scope: TScope, source: TSource, correlationId: string) => Promise<unknown>
  prepareRecovery: (source: TSource, correlationId: string, scopes: TScope[], installationId: string) => Promise<unknown>
  completeRecovery: (source: TSource, correlationId: string) => Promise<unknown>
  restoreRecoveryScope: (record: TRecord, scope: ManagedSshRecoveryScope, correlationId: string) => Promise<unknown>
  prepareRemote?: (source: TSource, correlationId: string) => Promise<ManagedSshPreparationReceipt>
  recordPreparationReceipt?: (receipt: ManagedSshPreparationReceipt) => Promise<unknown>
  refreshEligibility?: (source: TSource, receipt: ManagedSshPreparationReceipt) => Promise<unknown>
  createCorrelationId?: () => string
  isManagedSshSource?: (source: TSource) => boolean
  logRecovery?: (message: string) => void
}

export interface ManagedSshUpdateService<TSource extends ManagedSshUpdateSource = ManagedSshUpdateSource> {
  readonly gate: ManagedConnectionUpdateGate
  readonly activeUpdates: Map<string, Promise<ManagedConnectionUpdateResult>>
  readonly activePreparations: Map<string, Promise<ManagedSshPreparationResult>>
  readonly activeRecoveries: Map<string, Promise<void>>
  readonly request: (rawId: unknown, options?: ManagedSshUpdateRequestOptions) => Promise<ManagedConnectionUpdateResult>
  readonly requestCoordinator: (rawId: unknown, options: ManagedSshUpdateRequestOptions) => ManagedSshUpdateAdmission
  readonly prepare: (
    rawId: unknown,
    options?: Pick<ManagedSshUpdateRequestOptions, 'correlationId' | 'intent'>
  ) => Promise<ManagedSshPreparationResult>
  readonly issueLaunchCapability: (
    connectionId: string,
    correlationId: string,
    intent: ManagedSshUpdateIntent,
    expectedSource: ManagedSshCoordinatorSourceBinding
  ) => ManagedSshLaunchCapability
  readonly recover: (record: ManagedSshRecoveryRecord) => Promise<void>
  readonly resumeRecoveries: () => Promise<void>
  readonly waitForOperations: () => Promise<void>
  readonly assertCanMutatePrimaryRouting: () => void
  readonly restorePrimary: <T>(
    source: TSource,
    profile: string,
    correlationId: string,
    restore: () => Promise<T>
  ) => Promise<T>
  readonly primaryRestoreOwnerForProfile: (
    profile: string
  ) => { correlationId: string; profile: string; source: TSource } | null
  readonly hasPrimaryRestoreOwners: () => boolean
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function sourceId(source: ManagedSshUpdateSource): string {
  return String(source.id || '').trim()
}

function isSshSource<TSource extends ManagedSshUpdateSource>(source: TSource): boolean {
  return source.kind === 'ssh'
}

interface LaunchCapabilityRecord {
  connectionId: string
  correlationId: string
  intent: ManagedSshUpdateIntent
  expectedSource: ManagedSshCoordinatorSourceBinding
  consumed: boolean
}

const launchCapabilityRecords = new WeakMap<object, LaunchCapabilityRecord>()

function sameIntent(left: ManagedSshUpdateIntent, right: ManagedSshUpdateIntent): boolean {
  return (
    left.targetSha === right.targetSha &&
    left.expectedInstallId === right.expectedInstallId &&
    left.expectedCurrentSha === right.expectedCurrentSha &&
    left.source.repositoryRoot === right.source.repositoryRoot &&
    left.source.originUrl === right.source.originUrl &&
    left.source.resolvedRef === right.source.resolvedRef &&
    left.source.targetSha === right.source.targetSha &&
    left.source.assuranceProfile === right.source.assuranceProfile &&
    left.source.assuranceEvidenceSha256 === right.source.assuranceEvidenceSha256 &&
    left.source.assuranceGeneration === right.source.assuranceGeneration
  )
}

function validateSourceBinding(value: ManagedSshCoordinatorSourceBinding | undefined): ManagedSshCoordinatorSourceBinding {
  if (
    !value || !/^[0-9a-f]{32}$/.test(value.installId) ||
    !/^[0-9a-f]{64}$/.test(value.installationFingerprint) ||
    !/^[0-9a-f]{64}$/.test(value.sourceFingerprint)
  ) {throw new Error('Coordinator update requires a reviewed installation and source binding.')}

  return value
}

function sameSourceBinding(left: ManagedSshCoordinatorSourceBinding, right: ManagedSshCoordinatorSourceBinding): boolean {
  return left.installId === right.installId &&
    left.installationFingerprint === right.installationFingerprint &&
    left.sourceFingerprint === right.sourceFingerprint
}

function createLaunchCapability(
  connectionId: string,
  correlationId: string,
  intent: ManagedSshUpdateIntent,
  expectedSource: ManagedSshCoordinatorSourceBinding
): ManagedSshLaunchCapability {
  const capability = Object.freeze({ __managedSshLaunchCapability: true }) as ManagedSshLaunchCapability
  launchCapabilityRecords.set(capability, { connectionId, correlationId, intent, expectedSource, consumed: false })

  return capability
}

function consumeLaunchCapability(
  capability: ManagedSshLaunchCapability | undefined,
  connectionId: string,
  correlationId: string,
  intent: ManagedSshUpdateIntent | undefined,
  expectedSource: ManagedSshCoordinatorSourceBinding | undefined
): void {
  const record = capability ? launchCapabilityRecords.get(capability) : undefined

  if (!record) {
    throw new Error('Coordinator update requires an internal single-use launch capability.')
  }

  if (record.consumed) {
    throw new Error('The managed SSH launch capability has already been consumed.')
  }

  if (!intent || !expectedSource || record.connectionId !== connectionId || record.correlationId !== correlationId ||
      !sameIntent(record.intent, intent) || !sameSourceBinding(record.expectedSource, expectedSource)) {
    throw new Error('The managed SSH launch capability was bound to a different update transaction.')
  }

  record.consumed = true
}

export function createManagedSshUpdateService<
  TSource extends ManagedSshUpdateSource = ManagedSshUpdateSource,
  TScope extends ManagedSshUpdateScope = ManagedSshUpdateScope,
  TRecord extends ManagedSshRecoveryRecord = ManagedSshRecoveryRecord
>(deps: ManagedSshUpdateServiceDependencies<TSource, TScope, TRecord>): ManagedSshUpdateService<TSource> {
  const activeUpdates = deps.activeUpdates ?? new Map<string, Promise<ManagedConnectionUpdateResult>>()
  const activePreparations = new Map<string, Promise<ManagedSshPreparationResult>>()
  const activeRecoveries = deps.activeRecoveries ?? new Map<string, Promise<void>>()
  const primaryRestoreOwners = deps.primaryRestoreOwners ?? new Map<string, { correlationId: string; profile: string; source: TSource }>()
  const installationOwners = new Map<string, { connectionId: string; correlationId: string }>()

  const validInstallationId = (value: unknown): value is string =>
    typeof value === 'string' && /^[0-9a-f]{32}$/.test(value)

  const claimInstallation = (installationId: string, connectionId: string, correlationId: string): boolean => {
    if (!validInstallationId(installationId) || installationOwners.has(installationId)) {return false}

    // Older recovery records without a canonical identity cannot safely be
    // excluded from this installation. Keep the fence until recovery clears it.
    try {
      if (deps.readRecoveryRecords().some(record =>
        !validInstallationId(record.installationId) || record.installationId === installationId
      )) {return false}
    } catch {
      return false
    }

    installationOwners.set(installationId, { connectionId, correlationId })
    return true
  }

  const releaseInstallation = (installationId: string | null, connectionId: string, correlationId: string): void => {
    if (!installationId) {return}
    const owner = installationOwners.get(installationId)
    if (owner?.connectionId === connectionId && owner.correlationId === correlationId) {
      installationOwners.delete(installationId)
    }
  }

  const gate = deps.gate ?? new ManagedConnectionUpdateGate(connectionId => {
    const record = deps.readRecoveryRecords().find(item => item.connectionId === connectionId)

    return record?.correlationId || null
  })

  const execute = async (
    source: TSource,
    correlationId: string,
    options: ManagedSshUpdateRequestOptions = {},
    installationClaim: { id: string | null }
  ): Promise<ManagedConnectionUpdateResult> => {
    const connectionId = sourceId(source)
    const intent = validateManagedSshUpdateIntent(options.intent)
    const sourceSnapshot = { ...source } as TSource
    const scopes = await deps.captureScopes(sourceSnapshot)
    let ephemeral: ManagedSshUpdateTransport | null = null
    let launchAttempted = false
    const firstState = scopes.find(scope => scope.state !== undefined && scope.state !== null)?.state

    // An active scope may still hold a socket to the prior registry route.
    // Fleet mutation must use a fresh connection from the reviewed snapshot.
    const target = options.mode !== 'coordinator' && firstState
      ? deps.targetFromState(firstState)
      : (ephemeral = await deps.openTransport(sourceSnapshot)).target

    try {
      const observedInstallId = await deps.resolveInstallationId(sourceSnapshot, target)
      if (!validInstallationId(observedInstallId)) {
        throw new Error('The selected SSH target has no valid installation identity.')
      }
      if (options.mode === 'coordinator' && observedInstallId !== options.expectedSource?.installId) {
        throw new Error('The selected SSH target no longer matches the reviewed installation.')
      }
      if (!installationClaim.id) {
        if (!claimInstallation(observedInstallId, connectionId, correlationId)) {
          throw new Error('A managed update or recovery for this installation is already in progress.')
        }
        installationClaim.id = observedInstallId
      }
    } catch (error) {
      if (ephemeral) {await ephemeral.close().catch(() => undefined)}
      throw error
    }

    return runManagedSshUpdate({
      connectionId,
      correlationId,
      scopes,
      preflightRemote: async () => {
        await deps.preflightRemote(target, correlationId)
      },
      drainScope: async scope => {
        await deps.drainScope(scope)
      },
      updateRemote: async () => {
        // Consume only at the mutation edge. A preflight or drain failure has
        // not dispatched a remote update and therefore must not burn approval.
        if (options.mode === 'coordinator') {
          await deps.verifyCoordinatorSource!(sourceSnapshot, target, {
            ...options.expectedSource!, expectedCurrentSha: intent!.expectedCurrentSha
          })
          consumeLaunchCapability(options.launchCapability, connectionId, correlationId, intent, options.expectedSource)
        }

        return deps.executeRemoteUpdate(target, correlationId, {
          connectionId,
          intent,
          beforeLaunchDispatch: async () => {
            launchAttempted = true
          }
        })
      },
      awaitRestoreClearance: async () => {
        await deps.awaitRestoreClearance(target, correlationId, { requireTerminal: launchAttempted })
      },
      closeTransports: async () => {
        await deps.closeTransports(scopes, ephemeral)
      },
      restoreScope: async scope => {
        await deps.restoreScope(scope, sourceSnapshot, correlationId)
      },
      prepareRecovery: async () => {
        await deps.prepareRecovery(sourceSnapshot, correlationId, scopes, installationClaim.id!)
      },
      completeRecovery: async () => {
        await deps.completeRecovery(sourceSnapshot, correlationId)
      },
      releaseGate: () => gate.release(connectionId, correlationId)
    })
  }

  const admitRequest = (
    rawId: unknown,
    options: ManagedSshUpdateRequestOptions = {}
  ): ManagedSshUpdateAdmission => {
    const connectionId = String(rawId || '').trim()

    const refuse = (correlationId: string, reason: string): ManagedSshUpdateAdmission => ({
      admitted: false,
      reason,
      operation: Promise.resolve(refusedManagedSshUpdate(connectionId, correlationId, reason))
    })

    const existing = activeUpdates.get(connectionId)

    if (existing) {
      return options.mode === 'coordinator'
        ? refuse(String(options.correlationId || ''), 'A managed update is already in progress.')
        : { admitted: true, operation: existing }
    }

    let correlationId: string

    try {
      correlationId = options.correlationId
        ? validateCorrelationId(options.correlationId)
        : deps.createCorrelationId?.() || crypto.randomUUID()
    } catch (error) {
      return refuse(String(options.correlationId || ''), errorMessage(error))
    }

    let intent: ManagedSshUpdateIntent | undefined

    try {
      intent = validateManagedSshUpdateIntent(options.intent)
    } catch (error) {
      return refuse(correlationId, errorMessage(error))
    }

    if (options.mode === 'coordinator' && !intent) {
      return refuse(correlationId, 'Coordinator update requires a reviewed pinned target.')
    }

    if (intent && options.mode !== 'coordinator') {
      return refuse(correlationId, 'Reviewed pinned updates require coordinator admission.')
    }

    if (options.mode === 'coordinator') {
      try {
        validateSourceBinding(options.expectedSource)
        if (intent!.expectedInstallId !== options.expectedSource!.installId) {
          throw new Error('Coordinator update intent does not match the reviewed installation.')
        }
      } catch (error) {
        return refuse(correlationId, errorMessage(error))
      }

      if (!deps.verifyCoordinatorSource) {
        return refuse(correlationId, 'Coordinator source verifier is unavailable.')
      }
    }

    const source = deps.resolveSource(connectionId)

    if (!source) {
      return refuse(correlationId, `No connection with id "${connectionId}".`)
    }

    const sourceIsManaged = deps.isManagedSshSource?.(source) ?? isSshSource(source)

    if (!sourceIsManaged) {
      return refuse(correlationId, 'Only registered Desktop-managed SSH connections can use this update lifecycle.')
    }

    if (!gate.claim(connectionId, correlationId)) {
      return refuse(correlationId, 'A managed update is already in progress.')
    }

    const installationClaim: { id: string | null } = { id: null }
    if (options.mode === 'coordinator') {
      const installationId = options.expectedSource!.installId
      if (!claimInstallation(installationId, connectionId, correlationId)) {
        gate.release(connectionId, correlationId)
        return refuse(correlationId, 'A managed update or recovery for this installation is already in progress.')
      }
      installationClaim.id = installationId
    }

    const operation = (async () => {
      try {
        return await execute(source, correlationId, { ...options, intent }, installationClaim)
      } catch (error) {
        return refusedManagedSshUpdate(connectionId, correlationId, errorMessage(error))
      } finally {
        gate.release(connectionId, correlationId)
        releaseInstallation(installationClaim.id, connectionId, correlationId)
        activeUpdates.delete(connectionId)
      }
    })()

    activeUpdates.set(connectionId, operation)

    return { admitted: true, operation }
  }

  const request = (rawId: unknown, options: ManagedSshUpdateRequestOptions = {}): Promise<ManagedConnectionUpdateResult> =>
    admitRequest(rawId, options).operation

  const requestCoordinator = (rawId: unknown, options: ManagedSshUpdateRequestOptions): ManagedSshUpdateAdmission =>
    admitRequest(rawId, { ...options, mode: 'coordinator' })

  const prepare = (
    rawId: unknown,
    options: Pick<ManagedSshUpdateRequestOptions, 'correlationId' | 'intent'> = {}
  ): Promise<ManagedSshPreparationResult> => {
    const connectionId = String(rawId || '').trim()
    const existing = activePreparations.get(connectionId)

    if (existing) {
      return existing
    }

    let correlationId: string

    try {
      correlationId = options.correlationId
        ? validateCorrelationId(options.correlationId)
        : deps.createCorrelationId?.() || crypto.randomUUID()
    } catch (error) {
      return Promise.resolve({
        connectionId,
        correlationId: String(options.correlationId || ''),
        ok: false,
        outcome: 'refused',
        receipt: null,
        error: errorMessage(error)
      })
    }

    if (options.intent) {
      return Promise.resolve({
        connectionId,
        correlationId,
        ok: false,
        outcome: 'refused',
        receipt: null,
        error: 'Managed SSH preparation must not include a pinned target; prepare before target review.'
      })
    }

    const source = deps.resolveSource(connectionId)

    if (!source) {
      return Promise.resolve({
        connectionId,
        correlationId,
        ok: false,
        outcome: 'refused',
        receipt: null,
        error: `No connection with id "${connectionId}".`
      })
    }

    const sourceIsManaged = deps.isManagedSshSource?.(source) ?? isSshSource(source)

    if (!sourceIsManaged) {
      return Promise.resolve({
        connectionId,
        correlationId,
        ok: false,
        outcome: 'refused',
        receipt: null,
        error: 'Only registered Desktop-managed SSH connections can use preparation.'
      })
    }

    if (!gate.claim(connectionId, correlationId)) {
      return Promise.resolve({
        connectionId,
        correlationId,
        ok: false,
        outcome: 'refused',
        receipt: null,
        error: 'A managed update or preparation is already in progress.'
      })
    }

    const operation = (async (): Promise<ManagedSshPreparationResult> => {
      let receipt: ManagedSshPreparationReceipt | null = null
      let transport: ManagedSshUpdateTransport | null = null
      let installationId: string | null = null
      let preparationBegan = false

      try {
        if (!deps.prepareRemote) {
          throw new Error('Managed SSH preparation is not configured for this service.')
        }

        transport = await deps.openTransport(source)
        const observedInstallId = await deps.resolveInstallationId(source, transport.target)
        if (!validInstallationId(observedInstallId)) {
          throw new Error('The selected SSH target has no valid installation identity.')
        }
        if (!claimInstallation(observedInstallId, connectionId, correlationId)) {
          throw new Error('A managed update or recovery for this installation is already in progress.')
        }
        installationId = observedInstallId
        await transport.close()
        transport = null

        preparationBegan = true
        receipt = await deps.prepareRemote(source, correlationId)

        if (receipt.kind !== 'preparation' || receipt.correlationId !== correlationId) {
          throw new Error('Managed SSH preparation receipt did not match this transaction.')
        }

        await deps.recordPreparationReceipt?.(receipt)
        // This re-resolves eligibility after the unpinned branch-tip action.
        // The caller may only freeze/review a target after this has completed.
        await deps.refreshEligibility?.(source, receipt)

        return { connectionId, correlationId, ok: true, outcome: 'prepared', receipt }
      } catch (error) {
        return {
          connectionId,
          correlationId,
          ok: false,
          outcome: preparationBegan ? 'preparation-failed' : 'refused',
          receipt,
          error: errorMessage(error)
        }
      } finally {
        if (transport) {await transport.close().catch(() => undefined)}
        gate.release(connectionId, correlationId)
        releaseInstallation(installationId, connectionId, correlationId)
        activePreparations.delete(connectionId)
      }
    })()

    activePreparations.set(connectionId, operation)

    return operation
  }

  const recover = async (record: TRecord): Promise<void> => {
    const connectionId = record.connectionId

    if (activeRecoveries.has(connectionId) || activeUpdates.has(connectionId)) {
      return
    }

    const recoveryCorrelation = record.correlationId

    if (!gate.claim(connectionId, recoveryCorrelation)) {
      return
    }

    const operation = (async () => {
      let transport: ManagedSshUpdateTransport | null = null

      try {
        transport = await deps.openTransport(record.source as TSource)

        const results = await recoverManagedSshScopes({
          scopes: record.scopes,
          awaitClearance: async () => {
            await deps.awaitRestoreClearance(transport!.target, record.correlationId, {
              requireTerminal: record.phase === 'launching'
            })
          },
          afterClearance: async () => {
            await transport!.close()
            transport = null
          },
          restoreScope: async scope => {
            await deps.restoreRecoveryScope(record, scope, recoveryCorrelation)
          },
          completeRecovery: async () => {
            await deps.completeRecovery(record.source as TSource, record.correlationId)
          }
        })

        if (results.every(result => result.status === 'fulfilled')) {
          deps.logRecovery?.(
            `[ssh-update] restored ${record.scopes.length} scope(s) from durable recovery for ${connectionId}`
          )
        } else {
          const failures = results.filter(result => result.status === 'rejected').length
          deps.logRecovery?.(
            `[ssh-update] durable recovery for ${connectionId} left ${failures} scope(s) pending; will retry next launch`
          )
        }
      } catch (error) {
        deps.logRecovery?.(`[ssh-update] durable recovery for ${connectionId} remains pending: ${errorMessage(error)}`)
      } finally {
        if (transport) {
          await transport.close().catch(() => undefined)
        }

        gate.release(connectionId, recoveryCorrelation)
        activeRecoveries.delete(connectionId)
      }
    })()

    activeRecoveries.set(connectionId, operation)
    await operation
  }

  const resumeRecoveries = async (): Promise<void> => {
    await Promise.allSettled(deps.readRecoveryRecords().map(record => recover(record)))
  }

  const waitForOperations = async (): Promise<void> => {
    for (;;) {
      const pending = [...activeUpdates.values(), ...activePreparations.values(), ...activeRecoveries.values()]

      if (pending.length === 0) {
        return
      }

      await Promise.allSettled(pending)
    }
  }

  const assertCanMutatePrimaryRouting = (): void => {
    const durableIds = deps.readRecoveryRecords().map(record => record.connectionId)

    const ids = new Set([
      ...activeUpdates.keys(),
      ...activePreparations.keys(),
      ...activeRecoveries.keys(),
      ...primaryRestoreOwners.keys(),
      ...durableIds
    ])

    if (ids.size > 0) {
      const error: any = new Error(
        `Primary connection routing cannot change while managed SSH update recovery is pending for ${[...ids].join(', ')}.`
      )

      error.code = 'managed-update-in-progress'
      throw error
    }
  }

  const restorePrimary = async <T>(
    source: TSource,
    profile: string,
    correlationId: string,
    restore: () => Promise<T>
  ): Promise<T> => {
    const connectionId = sourceId(source)
    gate.assertCanDial(connectionId, correlationId)
    const profileKey = String(profile || '').trim() || 'default'

    if (primaryRestoreOwners.size > 0 && !primaryRestoreOwners.has(connectionId)) {
      throw new Error('Another managed SSH primary restore is already in progress.')
    }

    primaryRestoreOwners.set(connectionId, { correlationId, profile: profileKey, source })

    try {
      return await restore()
    } finally {
      if (primaryRestoreOwners.get(connectionId)?.correlationId === correlationId) {
        primaryRestoreOwners.delete(connectionId)
      }
    }
  }

  return {
    gate,
    activeUpdates,
    activePreparations,
    activeRecoveries,
    request,
    requestCoordinator,
    prepare,
    issueLaunchCapability: (connectionId, correlationId, intent, expectedSource) => {
      const normalizedIntent = validateManagedSshUpdateIntent(intent)

      if (!normalizedIntent) {
        throw new Error('Coordinator launch capability requires a reviewed pinned target.')
      }

      return createLaunchCapability(
        String(connectionId || '').trim(),
        validateCorrelationId(correlationId),
        normalizedIntent,
        { ...validateSourceBinding(expectedSource) }
      )
    },
    recover,
    resumeRecoveries,
    waitForOperations,
    assertCanMutatePrimaryRouting,
    restorePrimary,
    primaryRestoreOwnerForProfile: profile => {
      const profileKey = String(profile || '').trim() || 'default'

      return [...primaryRestoreOwners.values()].find(owner => owner.profile === profileKey) || null
    },
    hasPrimaryRestoreOwners: () => primaryRestoreOwners.size > 0
  }
}
