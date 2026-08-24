import { atom, computed } from 'nanostores'

import type { DesktopConnectionsRegistry } from '@/global'
import { persistStringRecord, storedStringRecord } from '@/lib/storage'
import { isTimeoutError, withTimeout } from '@/lib/with-timeout'
import {
  beginGatewaySwitch,
  endGatewaySwitch,
  type GatewaySwitchToken,
  recoverActiveSourceAfterFailedGatewaySwitch
} from '@/store/gateway-switch'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $showAllProfiles,
  captureNewChatSource,
  ensureGatewayAgent,
  normalizeProfileKey,
  openGatewayAgent,
  refreshActiveProfile,
  requestFreshSession
} from '@/store/profile'
import { $connection } from '@/store/session'

const LAST_PROFILE_STORAGE_KEY = 'hermes.desktop.lastProfileByConnection'

// Every await of a source switch is bounded. A wedged spawn, ticket mint,
// handshake or IPC (the #93454 class) must surface as a failed click — not a
// spinner that also swallows every later click on the same source, and never
// a barrier left up or a wipe left unpainted.
const SWITCH_DIAL_TIMEOUT_MS = 20_000
const SWITCH_COMMIT_TIMEOUT_MS = 20_000
const SWITCH_REMEMBER_TIMEOUT_MS = 5_000

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
    const result = await withTimeout(
      setLastUsed(connectionId),
      SWITCH_REMEMBER_TIMEOUT_MS,
      'Timed out remembering the last-used connection'
    )

    setConnectionsRegistry(result.registry)
  } catch {
    // The source is already usable. A read-only/full userData directory (or
    // a stalled IPC) must not turn a successful backend switch into a false
    // connection failure.
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
 *  2. Commit: beginGatewaySwitch() — barrier up, machine-context reset,
 *     session bindings wiped — then activate the already-open socket. The
 *     wipe runs inside the activation's serialized section, synchronously
 *     before the publication, so no route/session effect can observe the new
 *     source while $activeSessionId still names the previous backend's
 *     runtime. Activating first and wiping after (across an IPC round-trip)
 *     is exactly how that id leaked to the new backend and came back as
 *     "session not found" (#93937).
 *
 * Overlap and stalls: clicks can supersede a switch at any point. A switch
 * superseded while queued behind another activation declines its commit —
 * no wipe, no activation — so the winner's wipe is always the one that
 * precedes the final publication, and the barrier is owned by the latest
 * switch. Every await is bounded; a commit that stalls after the wipe lowers
 * the barrier and repaints the source that is still active.
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

  const targetIsActive = () => {
    const active = $connection.get()

    return active?.connectionId === connectionId && normalizeProfileKey(active.profile) === targetProfile
  }

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
    // A connection switch is a new-chat intent on THAT source: keep the
    // registry identity with the profile so the next create names local::x /
    // <source>::x exactly, never a bare profile string.
    captureNewChatSource()
    requestFreshSession()
    await rememberConnection(connectionId)

    return
  }

  const revision = ++switchRevision
  pendingTarget = targetKey
  $pendingConnectionId.set(connectionId)
  // Set by the commit hook once THIS switch has wiped — i.e. it owns the
  // barrier and, if the commit then fails, owes the still-active source a
  // repaint. Null while queued, or if it stepped aside before its turn.
  let token = null as GatewaySwitchToken | null

  try {
    // Phase 1 — open the target's socket; the active route is untouched.
    // Always use the explicit registry route. `local` must mean This device,
    // and a registry primary can differ from a legacy per-profile override.
    await withTimeout(
      openGatewayAgent(connectionId, targetProfile),
      SWITCH_DIAL_TIMEOUT_MS,
      `Timed out connecting to "${targetConnection.label}".`
    )

    // A newer click owns the switch from here on. The superseded dial never
    // activates, so the user doesn't flip through it on the way to the source
    // they picked last; its socket stays warm for that click or idles out.
    if (revision !== switchRevision) {
      return
    }

    // Phase 2 — commit. The hook runs inside the activation's serialized
    // section, right before the socket is activated: sever the previous
    // backend's bindings, then publish, with nothing in between. A click that
    // superseded this switch while it was queued makes the hook decline —
    // neither wipe nor activation — so the user never flips through it.
    const activationController = new AbortController()
    let markActivationStarted: () => void = () => undefined

    const activationStarted = new Promise<void>(resolve => {
      markActivationStarted = resolve
    })

    try {
      try {
        const activation = ensureGatewayAgent(connectionId, targetProfile, {
          signal: activationController.signal,
          beforeActivate: () => {
            if (revision !== switchRevision) {
              return false
            }

            token = beginGatewaySwitch()
            markActivationStarted()

            return true
          }
        })

        const timedActivation = activationStarted.then(() =>
          withTimeout(
            activation,
            SWITCH_COMMIT_TIMEOUT_MS,
            `Timed out activating "${targetConnection.label}".`,
            error => {
              // withTimeout does not cancel its input. Every timed-out owner
              // loses future activation/publication rights, even when low-level
              // activation already published the target and the commit remains
              // fail-open. The shared activation signal suppresses any trailing
              // descriptor/profile publication when stale work later settles.
              activationController.abort(error)
            }
          )
        )

        // Queue time belongs to the profile-store mutex. Start the bounded
        // commit window only once beforeActivate grants this request its turn.
        await Promise.race([activation, timedActivation])
      } catch (error) {
        // The socket is activated and its descriptor published synchronously;
        // only best-effort descriptor resync trails it. A commit that timed out
        // AFTER the new source became active has landed, so keep it fail-open;
        // the timeout signal still revokes all trailing publication rights.
        if (!isTimeoutError(error) || !targetIsActive()) {
          throw error
        }
      }

      if (revision !== switchRevision) {
        return
      }

      if (!targetIsActive()) {
        throw new Error(`Connection "${targetConnection.label}" did not become active.`)
      }
    } finally {
      // Lower the barrier the moment the commit settles — before the
      // bookkeeping awaits below — but only if this switch still owns it.
      if (token !== null) {
        endGatewaySwitch(token)
      }
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
      captureNewChatSource()
      requestFreshSession()
      await refreshActiveProfile()
    }
  } catch (error) {
    if (revision === switchRevision) {
      if (token !== null) {
        // This switch wiped for a commit that never landed. The previous
        // source is still the active one, and nothing reactive re-pulls its
        // lists (no scope moved): repaint it and land on a fresh draft there,
        // matching what a failed Settings apply leaves behind.
        recoverActiveSourceAfterFailedGatewaySwitch(token)
        requestFreshSession()
      }

      throw error
    }
  } finally {
    if (revision === switchRevision) {
      pendingTarget = null
      $pendingConnectionId.set(null)
    }
  }
}
