interface RemotePoolEntry {
  process?: unknown
}

interface RemoteBackendRecoveryDeps {
  pool: Map<string, RemotePoolEntry>
  cancelAndWait: (key: string, afterCancel: () => Promise<void>) => Promise<void>
  disconnectTransport: (key: string) => Promise<void>
}

export function sshRecoveryTransport<T>(state: { recovering?: boolean; ssh: T } | undefined): T | null | undefined {
  return state?.recovering ? null : state?.ssh
}

/** Registry credentials may have been replaced by a different profile's token.
 * Reuse only this scope's credential for the same SSH configuration.
 */
export function sshRecoveryToken(
  state: { fingerprint: string; token?: string } | undefined,
  fingerprint: string,
  fallback: string
): string {
  return state?.fingerprint === fingerprint && state.token ? state.token : fallback
}

/** A failed network probe cannot establish that the remote agent has stopped.
 * Retire its route under the bootstrap fence, leaving ownership and active work
 * on the host intact for the authenticated reconnect to reuse.
 */
export async function recoverRemotePoolBackend(key: string, deps: RemoteBackendRecoveryDeps): Promise<void> {
  const entry = deps.pool.get(key)

  if (!entry || entry.process) {
    return
  }

  deps.pool.delete(key)
  await deps.cancelAndWait(key, () => deps.disconnectTransport(key))
}
