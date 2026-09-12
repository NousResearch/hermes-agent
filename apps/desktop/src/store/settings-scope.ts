import { atom, computed } from 'nanostores'

import type { ProfileScope } from '@/api/client'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'

export interface SettingsOwner {
  connectionId: string
  profile: string
}

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

export const $settingsScopeProfile = computed([$settingsRequestProfile, $activeGatewayProfile], (scope, active) =>
  normalizeProfileKey(scope && typeof scope === 'object' ? scope.profile : (scope ?? active))
)

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
