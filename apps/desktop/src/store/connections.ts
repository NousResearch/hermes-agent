import { atom, computed } from 'nanostores'

import type { DesktopConnectionsRegistry } from '@/global'
import { persistStringRecord, storedStringRecord } from '@/lib/storage'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $showAllProfiles,
  ensureGatewayAgent,
  normalizeProfileKey,
  refreshActiveProfile,
  requestFreshSession
} from '@/store/profile'
import { $connection } from '@/store/session'

const LAST_PROFILE_STORAGE_KEY = 'hermes.desktop.lastProfileByConnection'

export const $connectionsRegistry = atom<DesktopConnectionsRegistry | null>(null)

// Use the resolved descriptor identity Electron publishes. The registry
// primary is only a compatibility fallback for older mains.
export const $activeConnectionId = computed(
  [$connectionsRegistry, $connection],
  (registry, connection) => connection?.connectionId ?? (connection ? (registry?.primary ?? null) : null)
)

export const $hasMultipleConnections = computed(
  $connectionsRegistry,
  registry => (registry?.connections.length ?? 0) > 1
)

const $lastProfileByConnection = atom<Record<string, string>>(storedStringRecord(LAST_PROFILE_STORAGE_KEY))
let pendingTarget: null | string = null
let switchRevision = 0

export const $pendingConnectionId = atom<null | string>(null)

$lastProfileByConnection.subscribe(value => persistStringRecord(LAST_PROFILE_STORAGE_KEY, value))

const $activeConnectionProfile = computed(
  [$activeConnectionId, $activeGatewayProfile, $connection],
  (connectionId, profile, connection) => ({
    connectionId,
    descriptorProfile: normalizeProfileKey(connection?.profile),
    profile: normalizeProfileKey(profile),
    registryScoped: connection?.registryScoped === true
  })
)

// Remember one profile per source, so switching machines is a re-home rather
// than a reset to `default`. The map is local UI preference only; Electron
// remains the authority for the connection registry and all secrets.
$activeConnectionProfile.subscribe(({ connectionId, descriptorProfile, profile, registryScoped }) => {
  // A migrated v1 per-profile remote may expose a client-side alias such as
  // "work" while the registered source's actual profile is "default". Only
  // remember a source/profile pair after Electron confirms that exact v2
  // descriptor. This also rejects the brief startup window where the profile
  // atom still carries the previous app run's alias.
  if (
    !connectionId ||
    !registryScoped ||
    descriptorProfile !== profile ||
    $lastProfileByConnection.get()[connectionId] === profile
  ) {
    return
  }

  $lastProfileByConnection.set({ ...$lastProfileByConnection.get(), [connectionId]: profile })
})

/** @internal Reset module-owned preferences and switch coordination for tests. */
export function _resetConnectionsForTests(): void {
  $lastProfileByConnection.set({})
  pendingTarget = null
  switchRevision = 0
  $pendingConnectionId.set(null)
}

export function setConnectionsRegistry(registry: DesktopConnectionsRegistry): void {
  $connectionsRegistry.set(registry)
}

/** Refresh the renderer cache from Electron's local registry. No backend is contacted. */
export async function refreshConnectionsRegistry(): Promise<DesktopConnectionsRegistry | null> {
  const bridge = window.hermesDesktop?.connections

  if (!bridge) {
    return null
  }

  const registry = await bridge.list()
  setConnectionsRegistry(registry)

  return registry
}

/**
 * Re-home Sessions to one registered source, restoring that source's last
 * profile. Only the selected source is dialed; merely rendering the switcher
 * never probes or opens remote gateways.
 */
export async function selectConnection(connectionId: string): Promise<void> {
  const registry = $connectionsRegistry.get()
  const targetConnection = registry?.connections.find(connection => connection.id === connectionId)

  if (!registry || !targetConnection) {
    return
  }

  const currentConnectionId = $activeConnectionId.get()
  const currentProfile = normalizeProfileKey($activeGatewayProfile.get())
  const targetProfile = normalizeProfileKey($lastProfileByConnection.get()[connectionId] ?? 'default')
  const targetKey = `${connectionId}::${targetProfile}`

  if (pendingTarget === targetKey) {
    return
  }

  const switching =
    pendingTarget !== null ||
    $showAllProfiles.get() ||
    currentConnectionId !== connectionId ||
    currentProfile !== targetProfile

  if (!switching) {
    return
  }

  if (pendingTarget === null && currentConnectionId === connectionId && currentProfile === targetProfile) {
    $showAllProfiles.set(false)
    $newChatProfile.set(targetProfile)
    requestFreshSession()

    return
  }

  const revision = ++switchRevision
  pendingTarget = targetKey
  $pendingConnectionId.set(connectionId)

  try {
    // Always use the explicit registry route. `local` must mean This device,
    // and a registry primary can differ from a legacy per-profile override.
    await ensureGatewayAgent(connectionId, targetProfile)

    if ($connection.get()?.connectionId !== connectionId) {
      throw new Error(`Connection "${targetConnection.label}" did not become active.`)
    }

    // A newer click owns the final refresh. Serialized gateway activation
    // already makes the latest source win; this guard also prevents an older
    // request from repainting its profile list after that newer activation.
    if (revision === switchRevision) {
      wipeSessionListsForGatewaySwitch()
      $showAllProfiles.set(false)
      $newChatProfile.set(targetProfile)
      requestFreshSession()
      await refreshActiveProfile()
    }
  } catch (error) {
    if (revision === switchRevision) {
      throw error
    }
  } finally {
    if (revision === switchRevision) {
      pendingTarget = null
      $pendingConnectionId.set(null)
    }
  }
}
