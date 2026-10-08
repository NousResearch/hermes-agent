import { $apiRequestScope, type ResolvedOwner } from '@/api/client'
import { getMemoryStatus, memoryDiscoveryKey } from '@/api/system'
import { queryClient } from '@/lib/query-client'
import { installAgentPlugin } from '@/store/agent-plugins'
import { activeGateway } from '@/store/gateway'
import { normalizeProfileKey } from '@/store/profile'

/** What a memory-provider install from Settings › Memory ended with. */
export type MemoryInstallOutcome = 'discovered' | 'missing' | 'owner-changed' | 'timed-out' | { error: string }

/** True while the foreground connection + profile are the ones the installer was opened for. */
export function memoryOwnerIsForeground(owner: ResolvedOwner, scope = $apiRequestScope.get()): boolean {
  return (
    scope.connectionId === owner.connectionId &&
    normalizeProfileKey(scope.profile) === normalizeProfileKey(owner.profile)
  )
}

/**
 * Install and admit a catalog memory provider for exactly the owner Memory
 * settings opened it for, without selecting it, then re-read discovery so the
 * row shows the new provider. The gateway socket is pinned before the first
 * await: a profile or connection switch mid-install must never retarget it.
 */
export async function installMemoryProvider(opts: {
  catalogName?: string
  name: string
  owner: ResolvedOwner
  ref?: string
  repo: string
}): Promise<MemoryInstallOutcome> {
  const gateway = activeGateway()

  if (!gateway || !memoryOwnerIsForeground(opts.owner)) {
    return 'owner-changed'
  }

  const result = await installAgentPlugin(gateway.request.bind(gateway), {
    identifier: opts.repo,
    enable: true,
    catalogName: opts.catalogName,
    ref: opts.ref,
    profile: opts.owner.profile
  })

  if (result.timedOut) {
    return 'timed-out'
  }

  if (!result.ok) {
    return { error: result.error ?? '' }
  }

  try {
    const status = await getMemoryStatus(opts.owner)
    queryClient.setQueryData(memoryDiscoveryKey(opts.owner), status)

    return status.providers.some(provider => provider.name === opts.name && provider.status !== 'missing')
      ? 'discovered'
      : 'missing'
  } catch {
    return 'missing'
  }
}
