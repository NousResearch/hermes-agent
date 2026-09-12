import { atom, computed } from 'nanostores'

import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

// ── Shared settings "Applies to" scope ──────────────────────────────────────
// One selection shared by every config-backed settings page (Model, Workspace,
// Safety, Memory & Context, Voice, Tools & Keys) and the Messaging overlay, so
// picking a profile on one page carries to the next instead of resetting per
// page. `null` means "follow the app's active profile" in selector state; the
// request store below still resolves that choice to a concrete profile.
export const $settingsScopeOverride = atom<null | string>(null)

// The profile the settings pages are currently editing (a concrete key).
export const $settingsScopeProfile = computed([$settingsScopeOverride, $activeGatewayProfile], (override, active) =>
  normalizeProfileKey(override ?? active)
)

// ── Request-scope form (THE value to hand to API helpers) ──────────────────
// Always send the concrete profile shown by the selector. Falling back to the
// ambient API profile makes Settings writes depend on a second store staying in
// lockstep; registered shared remotes can otherwise drop `?profile=` and write
// the launch profile. This also avoids the raw override's incompatible `null`
// meaning in profileScoped()/capabilityScoped() (target primary/default).
export const $settingsRequestProfile = $settingsScopeProfile

// Select the profile the settings pages should edit. Picking the app's active
// profile stores `null` (no override) so the scope keeps following the app on
// profile switches; requests resolve that selection through the concrete store.
export function setSettingsScope(name: string): void {
  const key = normalizeProfileKey(name)

  $settingsScopeOverride.set(key === normalizeProfileKey($activeGatewayProfile.get()) ? null : key)
}

// An app-wide profile switch re-homes every settings surface to the new
// backend; a surviving override would silently keep edits pointed at the
// previous target. Same drop-the-override contract as the Capabilities
// selector (app/skills useOnProfileSwitch).
let lastActiveProfile = normalizeProfileKey($activeGatewayProfile.get())

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)

  if (key !== lastActiveProfile) {
    lastActiveProfile = key
    $settingsScopeOverride.set(null)
  }
})
