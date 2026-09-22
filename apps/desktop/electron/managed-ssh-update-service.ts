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
  type ManagedConnectionUpdateResult,
  type ManagedSshRecoveryScope,
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
    context: { connectionId: string; onLaunchProved: () => Promise<void> }
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
  createCorrelationId?: () => string
  isManagedSshSource?: (source: TSource) => boolean
  logRecovery?: (message: string) => void
}

export interface ManagedSshUpdateService<TSource extends ManagedSshUpdateSource = ManagedSshUpdateSource> {
  readonly gate: ManagedConnectionUpdateGate
  readonly activeUpdates: Map<string, Promise<ManagedConnectionUpdateResult>>
  readonly activeRecoveries: Map<string, Promise<void>>
  readonly request: (rawId: unknown) => Promise<ManagedConnectionUpdateResult>
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

export function createManagedSshUpdateService<
  TSource extends ManagedSshUpdateSource = ManagedSshUpdateSource,
  TScope extends ManagedSshUpdateScope = ManagedSshUpdateScope,
  TRecord extends ManagedSshRecoveryRecord = ManagedSshRecoveryRecord
>(deps: ManagedSshUpdateServiceDependencies<TSource, TScope, TRecord>): ManagedSshUpdateService<TSource> {
  const activeUpdates = new Map<string, Promise<ManagedConnectionUpdateResult>>()
  const activeRecoveries = new Map<string, Promise<void>>()
  const primaryRestoreOwners = new Map<string, { correlationId: string; profile: string; source: TSource }>()
  const gate = new ManagedConnectionUpdateGate(connectionId => {
    const record = deps.readRecoveryRecords().find(item => item.connectionId === connectionId)
    return record?.correlationId || null
  })

  const execute = async (source: TSource, correlationId: string): Promise<ManagedConnectionUpdateResult> => {
    const connectionId = sourceId(source)
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
      updateRemote: () =>
        deps.executeRemoteUpdate(target, correlationId, {
          connectionId,
          onLaunchProved: async () => {
            launchAttempted = true
          }
        }),
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

  const request = (rawId: unknown): Promise<ManagedConnectionUpdateResult> => {
    const connectionId = String(rawId || '').trim()
    const existing = activeUpdates.get(connectionId)

    if (existing) {
      return existing
    }

    const correlationId = deps.createCorrelationId?.() || crypto.randomUUID()
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
        return await execute(source, correlationId)
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
      const pending = [...activeUpdates.values(), ...activeRecoveries.values()]

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
    activeRecoveries,
    request,
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
