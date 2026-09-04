import { atom, computed } from 'nanostores'

import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

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

// ── Request-scope form (THE value to hand to API helpers) ──────────────────
// The store contract and the API contract disagree about `null`:
//   - here, `null` means "follow the app's active profile" (no override);
//   - in api/client.ts `profileScoped()`/`capabilityScoped()`, `null` means
//     "deliberately suppress the active profile and target primary/default" —
//     only `undefined` falls back to the active profile.
// Passing the raw override into an API helper therefore silently retargets
// every read/write to the primary profile whenever no override is set — the
// "model change reverts when I re-enter the tab" class of bug (#90549: the
// page WROTE the right profile but READ primary back). Always send this
// computed (or `override ?? undefined`) on requests; keep the raw override
// only for UI concerns (selector highlight, cache keys, remount keys).
export const $settingsRequestProfile = computed(
  $settingsScopeOverride,
  (override): string | undefined => override ?? undefined
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
// selector (app/skills useOnProfileSwitch).
let lastActiveProfile = normalizeProfileKey($activeGatewayProfile.get())

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)

  if (key !== lastActiveProfile) {
    lastActiveProfile = key
    $settingsScopeOverride.set(null)
  }
})
