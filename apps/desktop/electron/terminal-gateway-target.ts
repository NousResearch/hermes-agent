// Where a terminal tab's shell runs. A tab records the gateway it was opened
// on; this resolves that record against the registry and the live SSH pool so
// the PTY spawns on THAT host no matter which gateway the window shows later.
import { backendScopeKey, type ConnectionRegistry, LOCAL_CONNECTION_ID } from './connection-registry'
import { type RegistrySshPoolEntry, registrySshPoolScopeByConnectionId } from './window-connection-route'

export interface SshTerminalTarget {
  scope: string
  ssh: unknown
}

/** `missing`: the gateway was removed. `no-shell`: a URL/token or cloud
 *  gateway, which has no shell transport; never answered with a local shell
 *  under the remote's name. `pending`: SSH gateway whose tunnel is not up yet. */
export type TerminalGatewayTarget = 'local' | 'missing' | 'no-shell' | 'pending' | SshTerminalTarget

export interface TerminalGatewayOwner {
  connectionId: string
  profile?: string
}

export function terminalGatewayTarget(
  owner: TerminalGatewayOwner,
  registry: ConnectionRegistry,
  pool: ReadonlyMap<string, RegistrySshPoolEntry | undefined>
): TerminalGatewayTarget {
  const id = owner.connectionId.trim()

  if (!id || id === LOCAL_CONNECTION_ID) {
    return 'local'
  }

  const source = registry.connections.find(connection => connection.id === id)

  if (!source) {
    return 'missing'
  }

  if (source.kind === 'local') {
    return 'local'
  }

  if (source.kind !== 'ssh') {
    return 'no-shell'
  }

  const scope = backendScopeKey(id, owner.profile)
  const state = pool.get(scope)

  if (state?.ssh) {
    return { scope, ssh: state.ssh }
  }

  // The pool's single writer publishes under its per-profile bootstrap key
  // while stamping the entry with the registry connection id (#97345).
  const pooledScope = registrySshPoolScopeByConnectionId(pool, id)
  const pooled = pooledScope === null ? undefined : pool.get(pooledScope)

  return pooledScope !== null && pooled?.ssh ? { scope: pooledScope, ssh: pooled.ssh } : 'pending'
}

/** The gateway a start payload names, or null for a tab that recorded none
 *  (an older Desktop's tab, or a host without a gateway registry). */
export function requestedTerminalGateway(payload: unknown): null | TerminalGatewayOwner {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
  const connectionId = typeof record.connectionId === 'string' ? record.connectionId.trim() : ''

  if (!connectionId) {
    return null
  }

  const profile = typeof record.profile === 'string' && record.profile.trim() ? record.profile.trim() : undefined

  return { connectionId, ...(profile ? { profile } : {}) }
}
