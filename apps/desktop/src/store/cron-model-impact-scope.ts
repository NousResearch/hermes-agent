import type { HermesConnection } from '@/global'

let generation = 0
let connectionIdentity = ''
let credentialIdentity = ''
const invalidationListeners = new Set<() => void>()

export interface CronModelImpactScopeSnapshot {
  connection: string
  generation: number
}

export function cronModelImpactScopeIsLocal(): boolean {
  return connectionIdentity.startsWith('local\u0000')
}

export function getCronModelImpactScope(): CronModelImpactScopeSnapshot {
  return { connection: connectionIdentity, generation }
}

export function beginCronModelImpactAssignment(): CronModelImpactScopeSnapshot {
  generation += 1

  return getCronModelImpactScope()
}

export function invalidateCronModelImpactScopeState(): void {
  generation += 1
  invalidationListeners.forEach(listener => listener())
}

export function onCronModelImpactScopeInvalidated(listener: () => void): () => void {
  invalidationListeners.add(listener)

  return () => invalidationListeners.delete(listener)
}

function identityForConnection(connection: HermesConnection | null): string {
  if (!connection) {
    return ''
  }

  const backendIdentity =
    connection.remoteKind === 'ssh'
      ? connection.remoteIdentity || connection.remoteHost || ''
      : connection.remoteIdentity || connection.baseUrl

  return [connection.mode ?? '', connection.remoteKind ?? '', backendIdentity, connection.profile ?? ''].join('\u0000')
}

function credentialsForConnection(connection: HermesConnection): string {
  return JSON.stringify([
    connection.authMode ?? '',
    connection.token ?? '',
    Object.entries(connection.headers ?? {}).sort(([left], [right]) => left.localeCompare(right))
  ])
}

/** Keep pending responses and action closures bound to the exact backend owner
 * that issued them. The exposed scope identity remains credential-free. */
export function syncCronModelImpactConnection(connection: HermesConnection | null): void {
  // A null descriptor is an ordinary reconnect state, not evidence that the
  // user selected another backend. Retain the last durable identity so the
  // reconnect can prove whether the backend actually changed.
  if (!connection) {
    return
  }

  const next = identityForConnection(connection)
  const nextCredentials = credentialsForConnection(connection)

  if (connectionIdentity && (connectionIdentity !== next || credentialIdentity !== nextCredentials)) {
    invalidateCronModelImpactScopeState()
  }

  connectionIdentity = next
  credentialIdentity = nextCredentials
}
