import { atom, computed } from 'nanostores'

import type { ProfileScope } from '@/api/client'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'

export interface SettingsOwner {
  connectionId: string
  profile: string
}

// ── Shared settings "Applies to" scope ──────────────────────────────────────
// One selection shared by every config-backed settings page (Model, Workspace,
// Safety, Memory & Context, Voice, Tools & Keys) and the Messaging overlay, so
// picking a profile on one page carries to the next instead of resetting per
// page. `null` means "follow the app's active profile" — the default, which
// keeps single-profile users on the exact pre-existing code path (requests
// fall back to the app-wide active profile in api/client.ts `profileScoped`).
//
// null follows the workspace; explicit fleet picks always retain both names.
// Legacy unregistered primaries keep string/undefined requests: pinning them to
// 'local' would bypass Electron's per-profile remote overrides.
export const $settingsScopeOverride = atom<null | string | SettingsOwner>(null)

export const $settingsRequestProfile = computed(
  [$settingsScopeOverride, $activeGatewayProfile, $connection],
  (override, active, connection): ProfileScope => {
    if (override && typeof override === 'object') {
      return override
    }

    if (connection?.registryScoped && connection.connectionId) {
      return { connectionId: connection.connectionId, profile: normalizeProfileKey(override ?? active) }
    }

    // API null means primary, not follow-active. Never return null here.
    return override ?? undefined
  }
)

// The profile the settings pages are currently editing (a concrete key).
export const $settingsScopeProfile = computed([$settingsRequestProfile, $activeGatewayProfile], (scope, active) =>
  normalizeProfileKey(scope && typeof scope === 'object' ? scope.profile : (scope ?? active))
)

// Whether the settings pages are editing a profile OTHER than the default
// one. The scope follows the app's active profile when no override is set —
// which, after opening any Bot Mode chat, is the BOT's profile — so an edit
// can land in profiles/<bot>/config.yaml while the user believes they are
// editing their main config (#89190/#89162 class). Surfaces render this
// loudly. Until the roster has loaded (no is_default entry yet) the root
// profile's canonical key is assumed, so an unknown default fails loud, not
// quiet. Fleet owner picks compare their concrete profile name against the
// roster's default; an explicit owner pick always states its target anyway,
// because the override itself names the gateway.
export const $settingsScopeEditsNonDefault = computed([$settingsScopeProfile, $profiles], (selected, profiles) => {
  const defaultProfile = profiles.find(profile => profile.is_default)

  return selected !== normalizeProfileKey(defaultProfile?.name)
})

// A same-named profile on another primary invalidates drafts too. Generation
// prevents an away-and-back switch from reviving an old queued continuation.
const $settingsGeneration = atom(0)
let lastConnection = $connection.get()
let lastProfile = normalizeProfileKey($activeGatewayProfile.get())

$connection.subscribe(connection => {
  if (connection !== lastConnection) {
    lastConnection = connection
    $settingsScopeOverride.set(null)
    $settingsGeneration.set($settingsGeneration.get() + 1)
  }
})

$activeGatewayProfile.subscribe(profile => {
  const key = normalizeProfileKey(profile)

  if (key !== lastProfile) {
    lastProfile = key
    $settingsScopeOverride.set(null)
    $settingsGeneration.set($settingsGeneration.get() + 1)
  }
})

export const $settingsScopeKey = computed(
  [$settingsRequestProfile, $settingsScopeProfile, $settingsGeneration],
  (scope, profile, generation) => JSON.stringify([scope, profile, generation])
)

export function setSettingsScope(scope: string | SettingsOwner): void {
  if (typeof scope === 'object') {
    $settingsScopeOverride.set({ connectionId: scope.connectionId, profile: normalizeProfileKey(scope.profile) })

    return
  }

  const key = normalizeProfileKey(scope)
  $settingsScopeOverride.set(key === normalizeProfileKey($activeGatewayProfile.get()) ? null : key)
}
