import { atom } from 'nanostores'

import type { DesktopConnectionsRegistry } from '@/global'

/** Null only for the legacy profile-only Desktop topology. Once Electron has
 * published a registry, profile names are source-local and are not owners. */
export const $connectionsRegistry = atom<DesktopConnectionsRegistry | null>(null)

let pendingRegistryRead: Promise<void> | undefined

/** Consumers needing ownership proof cannot depend on optional switcher UI
 * mounting first. Read Electron's registry, never synthesize a local entry. */
export function ensureConnectionsRegistry(): Promise<void> {
  const list = window.hermesDesktop?.connections?.list

  if ($connectionsRegistry.get() !== null || !list) {
    return Promise.resolve()
  }

  return (pendingRegistryRead ??= list()
    .then(registry => {
      // A settings/event refresh that completed meanwhile is newer authority.
      if ($connectionsRegistry.get() === null) {
        $connectionsRegistry.set(registry)
      }
    })
    .finally(() => {
      pendingRegistryRead = undefined
    }))
}

export function hasRegistryTopology(): boolean {
  // The bridge exists before its asynchronous cache load. Treat that window
  // (and a failed list IPC) as registry topology so owner routing fails closed;
  // only an older Desktop without the registry capability is truly legacy.
  return $connectionsRegistry.get() !== null || Boolean(window.hermesDesktop?.connections?.list)
}
