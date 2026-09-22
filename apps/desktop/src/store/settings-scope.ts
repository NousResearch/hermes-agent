import { atom, computed, onMount } from 'nanostores'

import type { ConnectionOwner, HermesConnection } from '@/global'
import { translateNow } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'

// ── Shared settings "Applies to" scope ──────────────────────────────────────
// One selection shared by every config-backed settings page (Model, Workspace,
// Safety, Memory & Context, Voice, Tools & Keys) and the Messaging overlay, so
// picking a profile on one page carries to the next instead of resetting per
// page. `null` means "follow the app's active profile" — the default, which
// keeps single-profile users on the exact pre-existing code path (requests
// fall back to the app-wide active profile in api/client.ts `profileScoped`).
export const $settingsScopeOverride = atom<null | string>(null)

// The profile the settings pages are currently editing (a concrete key).
export const $settingsScopeProfile = computed([$settingsScopeOverride, $activeGatewayProfile], (override, active) =>
  normalizeProfileKey(override ?? active)
)

// Whether the settings pages are editing a profile OTHER than the default
// one. The scope follows the app's active profile when no override is set —
// which, after opening any Bot Mode chat, is the BOT's profile — so an edit
// can land in profiles/<bot>/config.yaml while the user believes they are
// editing their main config (#89190/#89162 class). Surfaces render this
// loudly. Until the roster has loaded (no is_default entry yet) the root
// profile's canonical key is assumed, so an unknown default fails loud, not
// quiet.
export const $settingsScopeEditsNonDefault = computed([$settingsScopeProfile, $profiles], (selected, profiles) => {
  const defaultProfile = profiles.find(profile => profile.is_default)

  return selected !== normalizeProfileKey(defaultProfile?.name)
})

// ponytail: use Electron's resolved identity; never infer a remote owner from a URL.
// Env/unmatched remotes have a descriptor but deliberately no registry identity.
const $settingsConnectionId = computed($connection, connection => {
  if (!connection) {
    return null
  }

  if (Object.hasOwn(connection, 'connectionId')) {
    return typeof connection.connectionId === 'string' && connection.connectionId.trim()
      ? connection.connectionId
      : null
  }

  return connection.mode === 'local' ? 'local' : null
})

let previousConnectionOwner: ConnectionOwner | null = null

function sameHeaders(left?: Record<string, string>, right?: Record<string, string>): boolean {
  const leftEntries = Object.entries(left ?? {})
  const rightEntries = Object.entries(right ?? {})

  return leftEntries.length === rightEntries.length && leftEntries.every(([key, value]) => right?.[key] === value)
}

function sameConnectionOwner(left: ConnectionOwner, right: ConnectionOwner): boolean {
  return (
    left.mode === right.mode &&
    left.baseUrl === right.baseUrl &&
    left.token === right.token &&
    left.authMode === right.authMode &&
    left.remoteHost === right.remoteHost &&
    left.remoteIdentity === right.remoteIdentity &&
    left.remoteKind === right.remoteKind &&
    sameHeaders(left.headers, right.headers)
  )
}

function connectionOwnerFrom(connection: HermesConnection): ConnectionOwner {
  return {
    mode: connection.mode,
    baseUrl: connection.baseUrl,
    token: connection.token,
    authMode: connection.authMode,
    remoteHost: connection.remoteHost,
    remoteIdentity: connection.remoteIdentity,
    remoteKind: connection.remoteKind,
    headers: connection.headers
  }
}

const $settingsConnectionOwner = computed($connection, connection => {
  const hasRegisteredClaim = Boolean(connection && Object.hasOwn(connection, 'connectionId'))

  const registeredId =
    hasRegisteredClaim && typeof connection?.connectionId === 'string' ? connection.connectionId.trim() : ''

  if (!connection?.mode || (hasRegisteredClaim && !registeredId)) {
    previousConnectionOwner = null

    return null
  }

  const next = connectionOwnerFrom(connection)

  if (previousConnectionOwner && sameConnectionOwner(previousConnectionOwner, next)) {
    return previousConnectionOwner
  }

  previousConnectionOwner = next

  return next
})

// The foreground process is only authoritative for its own profile. A named
// Settings selection can use a different pooled port/token on the same gateway.
const $settingsForegroundProfile = computed([$connection, $activeGatewayProfile], (connection, active) =>
  normalizeProfileKey(connection?.profile ?? active)
)

const $settingsTarget = computed(
  [$settingsConnectionId, $settingsScopeProfile, $settingsConnectionOwner, $settingsForegroundProfile],
  (connectionId, profile, connectionOwner, foregroundProfile) => ({
    connectionId,
    profile,
    connectionOwner,
    foregroundProfile
  })
)

const $resolvedSettingsConnection = atom<{
  target: ReturnType<typeof $settingsTarget.get>
  connectionOwner: ConnectionOwner
} | null>(null)

// Resolve only while Settings has a consumer. Electron owns routing; no URL
// from the renderer is used to acquire the selected profile's descriptor.
onMount($resolvedSettingsConnection, () => {
  let current: ReturnType<typeof $settingsTarget.get> | null = null
  let generation = 0

  const acquire = (target: ReturnType<typeof $settingsTarget.get>) => {
    const version = ++generation
    current = target
    $resolvedSettingsConnection.set(null)

    if (!target.connectionId || !target.connectionOwner || target.profile === target.foregroundProfile) {
      return
    }

    const resolve = window.hermesDesktop?.getConnectionFor

    if (!resolve) {
      return
    }

    void resolve({
      connectionId: target.connectionId,
      profile: target.profile,
      priority: 'foreground',
      expectedOwner: { profile: target.foregroundProfile, connectionOwner: target.connectionOwner }
    })
      .then(connection => {
        if (current === target && generation === version && connection.connectionId === target.connectionId) {
          $resolvedSettingsConnection.set({ target, connectionOwner: connectionOwnerFrom(connection) })
        }
      })
      // Unavailable ownership stays disabled; never fall back to the foreground.
      .catch(error => {
        if (current === target && generation === version) {
          notifyError(error, translateNow('settings.config.failedLoad'), {
            action: {
              label: translateNow('skills.refresh'),
              onClick: () => {
                if (current === target && generation === version) {
                  acquire(target)
                }
              }
            }
          })
        }
      })
  }

  const unlisten = $settingsTarget.subscribe(acquire)

  return () => {
    current = null
    unlisten()
  }
})

export const $settingsOwner = computed([$settingsTarget, $resolvedSettingsConnection], (target, resolved) => {
  const { connectionId, profile, connectionOwner, foregroundProfile } = target

  if (connectionId && connectionOwner) {
    const selectedOwner =
      profile === foregroundProfile ? connectionOwner : resolved?.target === target ? resolved.connectionOwner : null

    return selectedOwner ? { connectionId, profile, connectionOwner: selectedOwner } : null
  }

  return connectionOwner?.mode === 'remote' && connectionOwner.baseUrl
    ? { connectionId: null, profile, legacyConnection: connectionOwner }
    : null
})

// ── Request-scope form (THE value to hand to API helpers) ──────────────────
// The store contract and the API contract disagree about `null`:
//   - here, `null` means "follow the app's active profile" (no override);
//   - in api/client.ts `profileScoped()`/`capabilityScoped()`, `null` means
//     "deliberately suppress the active profile and target primary/default" —
//     only `undefined` falls back to the active profile.
// Passing the raw override into an API helper therefore silently retargets
// every read/write to the primary profile whenever no override is set — the
// "model change reverts when I re-enter the tab" class of bug (#90549: the
// page WROTE the right profile but READ primary back).
//
// It must also never be `undefined` for a profile the pages actually display.
// `undefined` makes `profileScoped()` omit `?profile=` altogether, and the
// backend resolves an omitted profile to the home it was LAUNCHED with — not
// the profile this store says we are editing. Those differ the moment a pooled
// desktop backend serves a profile other than its launch home (`hermes
// --profile A serve`, editing B): every settings page then READ A's values and
// WROTE them back to A, while the "Changes on this page apply to 'B'" note —
// which reads the concrete $settingsScopeProfile — kept naming B (#118432).
// Send the concrete key the pages render. Only a name with no profile
// directory behind it (`custom`, a HERMES_HOME outside profiles/) stays
// `undefined`, where the ambient path is the only correct answer.
export const $settingsRequestProfile = computed($settingsScopeProfile, (selected): string | undefined =>
  selected === 'custom' ? undefined : selected
)

// Select the profile the settings pages should edit. Picking the app's active
// profile stores `null` (no override) so the scope keeps following the app on
// profile switches — and requests keep their unscoped default shape.
export function setSettingsScope(name: string): void {
  const key = normalizeProfileKey(name)

  $settingsScopeOverride.set(key === normalizeProfileKey($activeGatewayProfile.get()) ? null : key)
}

// An app-wide profile switch re-homes every settings surface to the new
// backend; a surviving override would silently keep edits pointed at the
// previous target. Same drop-the-override contract as the Capabilities
// selector (app/capabilities useOnProfileSwitch).
let lastActiveProfile = normalizeProfileKey($activeGatewayProfile.get())

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)

  if (key !== lastActiveProfile) {
    lastActiveProfile = key
    $settingsScopeOverride.set(null)
  }
})
