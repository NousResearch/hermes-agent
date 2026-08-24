import { atom, computed } from 'nanostores'

import type { DesktopConnectionsRegistry } from '@/global'
import { persistStringRecord, storedStringRecord } from '@/lib/storage'
import {
  beginGatewaySwitch,
  endGatewaySwitch,
  recoverActiveSourceAfterFailedGatewaySwitch
} from '@/store/gateway-switch'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $showAllProfiles,
  ensureGatewayAgent,
  normalizeProfileKey,
  openGatewayAgent,
  refreshActiveProfile,
  requestFreshSession
} from '@/store/profile'
import { $connection } from '@/store/session'

const LAST_PROFILE_STORAGE_KEY = 'hermes.desktop.lastProfileByConnection'

export const $connectionsRegistry = atom<DesktopConnectionsRegistry | null>(null)

// Use only the resolved descriptor identity Electron publishes. `primary`
// means the registry default, not necessarily the source this window is using;
// guessing it here would paint the wrong source as active for an unmatched v1
// route or while a legacy main is still resolving the descriptor.
export const $activeConnectionId = computed($connection, connection => connection?.connectionId ?? null)

export const $hasMultipleConnections = computed(
  $connectionsRegistry,
  registry => (registry?.connections.length ?? 0) > 1
)

const $lastProfileByConnection = atom<Record<string, string>>(storedStringRecord(LAST_PROFILE_STORAGE_KEY))
let pendingTarget: null | string = null
let restoreAttempted = false
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
  restoreAttempted = false
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

async function rememberConnection(connectionId: string): Promise<void> {
  const setLastUsed = window.hermesDesktop?.connections?.setLastUsed

  if (!setLastUsed) {
    return
  }

  try {
    const result = await setLastUsed(connectionId)
    setConnectionsRegistry(result.registry)
  } catch {
    // The source is already usable. A read-only/full userData directory must
    // not turn a successful backend switch into a false connection failure.
  }
}

/**
 * Load the registry once for Sessions and restore the last successfully used
 * source. Later registry refreshes stay side-effect free, so editing Settings
 * in another window never changes the active workspace.
 */
export async function initializeConnectionsRegistry(): Promise<DesktopConnectionsRegistry | null> {
  const registry = await refreshConnectionsRegistry()

  if (!registry || restoreAttempted) {
    return registry
  }

  restoreAttempted = true

  // Residual drift: a window can be live on a source the registry cannot name
  // (a v1-configured remote that reconciliation has not repaired yet, e.g. a
  // read-only userData that rejected the healed write). $activeConnectionId is
  // null there, so the preferred-id guard below would miss and "restore" the
  // registry primary over a connection that is already up and painting —
  // re-homing the user onto a different backend seconds after boot. The
  // registry has no claim on a source it does not know; leave the live one be.
  if ($connection.get() && $activeConnectionId.get() === null) {
    return registry
  }

  const lastUsed = registry.connections.some(connection => connection.id === registry.lastUsed)
    ? registry.lastUsed
    : registry.primary

  const preferredId = registry.launchMode === 'last-used' ? lastUsed : registry.primary

  if (!preferredId) {
    return registry
  }

  if ($activeConnectionId.get() === preferredId) {
    await rememberConnection(preferredId)
  } else {
    await selectConnection(preferredId)
  }

  return $connectionsRegistry.get() ?? registry
}

/**
 * Re-home Sessions to one registered source, restoring that source's last
 * profile. Only the selected source is dialed; merely rendering the switcher
 * never probes or opens remote gateways.
 *
 * Two phases, same commit contract as a Settings → Gateway apply (softSwitch):
 *  1. Dial the target WITHOUT activating it. The previous source stays fully
 *     bound and painted, so a dead target fails with nothing lost.
 *  2. beginGatewaySwitch() — barrier up, machine-context reset, session
 *     bindings wiped — then activate the already-open socket. Nothing awaits
 *     between the wipe and the publication, so no route/session effect can
 *     observe the new source while $activeSessionId still names the previous
 *     backend's runtime. Activating first and wiping after (across an IPC
 *     round-trip) is exactly how that id leaked to the new backend and came
 *     back as "session not found" (#93937).
 */
export async function selectConnection(connectionId: string): Promise<void> {
  const registry = $connectionsRegistry.get()
  const targetConnection = registry?.connections.find(connection => connection.id === connectionId)

  if (!registry || !targetConnection) {
    return
  }

  // A user-initiated source switch collapses "All profiles" browse mode: the
  // picker is a concrete-source action. The silent boot-time restore (below,
  // from initializeConnectionsRegistry) is not — it must leave the persisted
  // browse-mode preference alone so it survives restart (#93197).
  const restoreOnBoot = pendingTarget === null && $activeConnectionId.get() === null

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
    await rememberConnection(connectionId)

    return
  }

  if (pendingTarget === null && currentConnectionId === connectionId && currentProfile === targetProfile) {
    $showAllProfiles.set(false)
    $newChatProfile.set(targetProfile)
    requestFreshSession()
    await rememberConnection(connectionId)

    return
  }

  const revision = ++switchRevision
  pendingTarget = targetKey
  $pendingConnectionId.set(connectionId)

  try {
    // Phase 1 — open the target's socket; the active route is untouched.
    // Always use the explicit registry route. `local` must mean This device,
    // and a registry primary can differ from a legacy per-profile override.
    await openGatewayAgent(connectionId, targetProfile)

    // A newer click owns the switch from here on. The superseded dial never
    // activates, so the user doesn't flip through it on the way to the source
    // they picked last; its socket stays warm for that click or idles out.
    if (revision !== switchRevision) {
      return
    }

    // Phase 2 — commit: sever the previous backend's bindings FIRST, then
    // activate. Synchronous from the wipe to the publication (the socket is
    // already open), and behind the barrier until the descriptor lands.
    beginGatewaySwitch()

    try {
      await ensureGatewayAgent(connectionId, targetProfile)

      if ($connection.get()?.connectionId !== connectionId) {
        throw new Error(`Connection "${targetConnection.label}" did not become active.`)
      }
    } catch (error) {
      // The wipe already happened but the previous source is still the active
      // one, and nothing reactive re-pulls its lists (no scope moved). Repaint
      // it and land on a fresh draft there, matching what a failed Settings
      // apply leaves behind.
      recoverActiveSourceAfterFailedGatewaySwitch()
      requestFreshSession()
      throw error
    } finally {
      endGatewaySwitch()
    }

    // A newer click owns the final refresh. Serialized gateway activation
    // already makes the latest source win; this guard also prevents an older
    // request from repainting its profile list after that newer activation.
    if (revision === switchRevision) {
      await rememberConnection(connectionId)

      if (!restoreOnBoot) {
        $showAllProfiles.set(false)
      }

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
