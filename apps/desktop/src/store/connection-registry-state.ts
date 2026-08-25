import { atom } from 'nanostores'

import type { DesktopConnectionsRegistry } from '@/global'

/** Null only for the legacy profile-only Desktop topology. Once Electron has
 * published a registry, profile names are source-local and are not owners. */
export const $connectionsRegistry = atom<DesktopConnectionsRegistry | null>(null)

export function hasRegistryTopology(): boolean {
  return $connectionsRegistry.get() !== null
}
