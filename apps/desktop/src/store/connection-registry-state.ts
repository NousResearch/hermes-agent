import { atom } from 'nanostores'

import type { DesktopConnectionsRegistry } from '@/global'

/** Null only for the legacy profile-only Desktop topology. Once Electron has
 * published a registry, profile names are source-local and are not owners. */
export const $connectionsRegistry = atom<DesktopConnectionsRegistry | null>(null)

export function hasRegistryTopology(): boolean {
  // The bridge exists before its asynchronous cache load. Treat that window
  // (and a failed list IPC) as registry topology so owner routing fails closed;
  // only an older Desktop without the registry capability is truly legacy.
  return $connectionsRegistry.get() !== null || Boolean(window.hermesDesktop?.connections?.list)
}
