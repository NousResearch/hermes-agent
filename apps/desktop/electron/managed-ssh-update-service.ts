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
  recoverManagedSshScopes,
  refusedManagedSshUpdate,
  runManagedSshUpdate,
  validateCorrelationId,
  validateManagedSshUpdateIntent,
  type ManagedConnectionUpdateResult,
  type ManagedSshRecoveryScope,
  type ManagedSshUpdateIntent,
  type RemoteUpdateProof,
  type RemoteUpdateTarget
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

export interface ManagedSshUpdateRequestOptions {
  correlationId?: string
  intent?: ManagedSshUpdateIntent
  mode?: ManagedSshUpdateMode
  launchCapability?: ManagedSshLaunchCapability
}

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
  readRecoveryRecords: () => TRecord[]
  captureScopes: (source: TSource) => Promise<TScope[]>
  openTransport: (source: TSource) => Promise<ManagedSshUpdateTransport>
  targetFromState: (state: unknown) => RemoteUpdateTarget
  executeRemoteUpdate: (
    target: RemoteUpdateTarget,
    correlationId: string,
    context: { connectionId: string; intent?: ManagedSshUpdateIntent; onLaunchProved: () => Promise<void> }
  ) => Promise<RemoteUpdateProof>
  preflightRemote: (target: RemoteUpdateTarget, correlationId: string) => Promise<unknown>
  awaitRestoreClearance: (
    target: RemoteUpdateTarget,
    correlationId: string,
    options: { requireTerminal: boolean }
  ) => Promise<unknown>
  drainScope: (scope: TScope) => Promise<unknown>
  closeTransports: (scopes: TScope[], ephemeral: ManagedSshUpdateTransport | null) => Promise<unknown>
  restoreScope: (scope: TScope, source: TSource, correlationId: string) => Promise<unknown>
  prepareRecovery: (source: TSource, correlationId: string, scopes: TScope[]) => Promise<unknown>
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
  readonly prepare: (
    rawId: unknown,
    options?: Pick<ManagedSshUpdateRequestOptions, 'correlationId' | 'intent'>
  ) => Promise<ManagedSshPreparationResult>
  readonly issueLaunchCapability: (
    connectionId: string,
    correlationId: string,
    intent: ManagedSshUpdateIntent
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
  consumed: boolean
}

const launchCapabilityRecords = new WeakMap<object, LaunchCapabilityRecord>()

function sameIntent(left: ManagedSshUpdateIntent, right: ManagedSshUpdateIntent): boolean {
  return (
    left.targetSha === right.targetSha &&
    left.source.repositoryRoot === right.source.repositoryRoot &&
    left.source.originUrl === right.source.originUrl &&
    left.source.resolvedRef === right.source.resolvedRef &&
    left.source.targetSha === right.source.targetSha &&
    left.source.assuranceProfile === right.source.assuranceProfile &&
    left.source.assuranceEvidenceSha256 === right.source.assuranceEvidenceSha256 &&
    left.source.assuranceGeneration === right.source.assuranceGeneration
  )
}

function createLaunchCapability(
  connectionId: string,
  correlationId: string,
  intent: ManagedSshUpdateIntent
): ManagedSshLaunchCapability {
  const capability = Object.freeze({ __managedSshLaunchCapability: true }) as ManagedSshLaunchCapability
  launchCapabilityRecords.set(capability, { connectionId, correlationId, intent, consumed: false })
  return capability
}

function consumeLaunchCapability(
  capability: ManagedSshLaunchCapability | undefined,
  connectionId: string,
  correlationId: string,
  intent: ManagedSshUpdateIntent | undefined
): void {
  const record = capability ? launchCapabilityRecords.get(capability) : undefined

  if (!record) {
    throw new Error('Coordinator update requires an internal single-use launch capability.')
  }

  if (record.consumed) {
    throw new Error('The managed SSH launch capability has already been consumed.')
  }

  if (!intent || record.connectionId !== connectionId || record.correlationId !== correlationId || !sameIntent(record.intent, intent)) {
    throw new Error('The managed SSH launch capability was bound to a different update transaction.')
  }

  record.consumed = true
}

export function createManagedSshUpdateService<
  TSource extends ManagedSshUpdateSource = ManagedSshUpdateSource,
  TScope extends ManagedSshUpdateScope = ManagedSshUpdateScope,
  TRecord extends ManagedSshRecoveryRecord = ManagedSshRecoveryRecord
>(deps: ManagedSshUpdateServiceDependencies<TSource, TScope, TRecord>): ManagedSshUpdateService<TSource> {
  const activeUpdates = new Map<string, Promise<ManagedConnectionUpdateResult>>()
  const activePreparations = new Map<string, Promise<ManagedSshPreparationResult>>()
  const activeRecoveries = new Map<string, Promise<void>>()
  const primaryRestoreOwners = new Map<string, { correlationId: string; profile: string; source: TSource }>()
  const gate = new ManagedConnectionUpdateGate(connectionId => {
    const record = deps.readRecoveryRecords().find(item => item.connectionId === connectionId)
    return record?.correlationId || null
  })

  const execute = async (
    source: TSource,
    correlationId: string,
    options: ManagedSshUpdateRequestOptions = {}
  ): Promise<ManagedConnectionUpdateResult> => {
    const connectionId = sourceId(source)
    const intent = validateManagedSshUpdateIntent(options.intent)
    const sourceSnapshot = { ...source } as TSource
    const scopes = await deps.captureScopes(sourceSnapshot)
    let ephemeral: ManagedSshUpdateTransport | null = null
    let launchAttempted = false
    const firstState = scopes.find(scope => scope.state !== undefined && scope.state !== null)?.state
    const target = firstState
      ? deps.targetFromState(firstState)
      : (ephemeral = await deps.openTransport(sourceSnapshot)).target

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
      updateRemote: () => {
        // Consume only at the mutation edge. A preflight or drain failure has
        // not dispatched a remote update and therefore must not burn approval.
        if (options.mode === 'coordinator') {
          consumeLaunchCapability(options.launchCapability, connectionId, correlationId, intent)
        }

        return deps.executeRemoteUpdate(target, correlationId, {
          connectionId,
          intent,
          onLaunchProved: async () => {
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
        await deps.prepareRecovery(sourceSnapshot, correlationId, scopes)
      },
      completeRecovery: async () => {
        await deps.completeRecovery(sourceSnapshot, correlationId)
      },
      releaseGate: () => gate.release(connectionId, correlationId)
    })
  }

  const request = (
    rawId: unknown,
    options: ManagedSshUpdateRequestOptions = {}
  ): Promise<ManagedConnectionUpdateResult> => {
    const connectionId = String(rawId || '').trim()
    const existing = activeUpdates.get(connectionId)

    if (existing) {
      return existing
    }

    let correlationId: string

    try {
      correlationId = options.correlationId
        ? validateCorrelationId(options.correlationId)
        : deps.createCorrelationId?.() || crypto.randomUUID()
    } catch (error) {
      return Promise.resolve(
        refusedManagedSshUpdate(connectionId, String(options.correlationId || ''), errorMessage(error))
      )
    }

    let intent: ManagedSshUpdateIntent | undefined

    try {
      intent = validateManagedSshUpdateIntent(options.intent)
    } catch (error) {
      return Promise.resolve(refusedManagedSshUpdate(connectionId, correlationId, errorMessage(error)))
    }

    if (options.mode === 'coordinator' && !intent) {
      return Promise.resolve(
        refusedManagedSshUpdate(connectionId, correlationId, 'Coordinator update requires a reviewed pinned target.')
      )
    }

    if (intent && options.mode !== 'coordinator') {
      return Promise.resolve(
        refusedManagedSshUpdate(connectionId, correlationId, 'Reviewed pinned updates require coordinator admission.')
      )
    }
    const source = deps.resolveSource(connectionId)

    if (!source) {
      return Promise.resolve(
        refusedManagedSshUpdate(connectionId, correlationId, `No connection with id "${connectionId}".`)
      )
    }

    const sourceIsManaged = deps.isManagedSshSource?.(source) ?? isSshSource(source)

    if (!sourceIsManaged) {
      return Promise.resolve(
        refusedManagedSshUpdate(
          connectionId,
          correlationId,
          'Only registered Desktop-managed SSH connections can use this update lifecycle.'
        )
      )
    }

    if (!gate.claim(connectionId, correlationId)) {
      return Promise.resolve(
        refusedManagedSshUpdate(connectionId, correlationId, 'A managed update is already in progress.')
      )
    }

    const operation = (async () => {
      try {
        return await execute(source, correlationId, { ...options, intent })
      } catch (error) {
        return refusedManagedSshUpdate(connectionId, correlationId, errorMessage(error))
      } finally {
        gate.release(connectionId, correlationId)
        activeUpdates.delete(connectionId)
      }
    })()

    activeUpdates.set(connectionId, operation)
    return operation
  }

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

      try {
        if (!deps.prepareRemote) {
          throw new Error('Managed SSH preparation is not configured for this service.')
        }

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
          outcome: 'preparation-failed',
          receipt,
          error: errorMessage(error)
        }
      } finally {
        gate.release(connectionId, correlationId)
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
    prepare,
    issueLaunchCapability: (connectionId, correlationId, intent) => {
      const normalizedIntent = validateManagedSshUpdateIntent(intent)

      if (!normalizedIntent) {
        throw new Error('Coordinator launch capability requires a reviewed pinned target.')
      }

      return createLaunchCapability(
        String(connectionId || '').trim(),
        validateCorrelationId(correlationId),
        normalizedIntent
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
