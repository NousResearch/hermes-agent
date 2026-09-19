/**
 * Name-based sync between the desktop skin and the web Dashboard's theme.
 *
 * Three theme name spaces coexist, and only a name that means the same thing
 * to BOTH web surfaces may cross between them:
 *
 *   - The Python skin engine (`display.skin`) knows only default, ares, mono,
 *     slate, daylight, warm-lightmode, poseidon, sisyphus and charizard, and an
 *     unknown name silently resolves to the engine's `default` — so this module
 *     must NEVER write `display.skin`. The Dashboard's `dashboard.theme` key,
 *     reachable over REST, is the sync bus instead.
 *   - The Dashboard knows default, default-large, nous-blue, midnight, ember,
 *     mono, cyberpunk, rose, plus user YAML themes from ~/.hermes/dashboard-themes/.
 *   - The Desktop's built-ins are nous, github, catppuccin, everforest,
 *     solarized, nous-alt, slate, midnight, ember, mono, cyberpunk.
 *
 * The only names both surfaces render meaningfully are `midnight`, `ember`,
 * `mono` and `cyberpunk` — those four are the entire sync surface. Everything
 * else (the Dashboard's `default`/`default-large`/`nous-blue`/`rose`/user YAML
 * themes, the Desktop's `nous`/`github`/`slate`/…) is deliberately left alone:
 * no coercion, no fallback rendering. The server does NOT validate names
 * (`PUT /api/dashboard/theme` persists any string verbatim), so this module is
 * what keeps synced names sane.
 */

import { getDashboardThemes, setDashboardTheme } from '@/api/dashboard-themes'
import { persistString, persistStringRecord, storedString, storedStringRecord } from '@/lib/storage'
import { normalizeProfileKey } from '@/store/profile'

import { $pendingSkinApply } from './backend-sync'

/** Dashboard theme name → desktop skin name, for the four shared palettes. */
export const DASHBOARD_SHARED_THEMES: Readonly<Record<string, string>> = Object.freeze({
  cyberpunk: 'cyberpunk',
  ember: 'ember',
  midnight: 'midnight',
  mono: 'mono'
})

// Last dashboard theme name observed/applied — the sync baseline. It tracks
// the `dashboard.theme` value on the backend this desktop app is connected to
// (the transport is deliberately not profile-scoped, so that is the
// primary/launch-profile backend — one value per desktop app), not any
// profile's own preference. Only the profile that observes a change repaints,
// because the desktop skin itself is intentionally per-profile; making the
// cross-profile effect uniform is a documented open question on the PR, not
// something this key settles.
const BASELINE_KEY = 'hermes-desktop-dashboard-theme-v1'

// Per-profile opt-out. Only the 'off' value is stored, so a profile that never
// touched the toggle reads as enabled (default-on).
const SYNC_KEY = 'hermes-desktop-dashboard-sync-v1'

/**
 * Fold the Dashboard's active theme into the desktop. A first observation only
 * records the baseline — a fresh connect must never stomp the user's existing
 * desktop pick. Only a genuine name change that maps to a desktop skin paints,
 * and it paints through `$pendingSkinApply` (the ThemeProvider drains that atom
 * via `setTheme`, which persists per profile) — never a second apply path.
 * A changed name with no desktop counterpart updates the baseline and paints
 * nothing: unshared names are left alone, not coerced.
 */
export const ingestDashboardTheme = (active: null | string | undefined, { profile }: { profile: string }): void => {
  const name = (active ?? '').trim()

  if (!name) {
    return
  }

  // Disabled profiles record nothing, but disabling never touches the
  // baseline. So a genuine first-ever observation (no baseline: fresh install
  // or cleared storage) only seeds it and paints nothing, while re-enabling
  // later — baseline still present — resumes following the dashboard: the
  // next observation differing from the baseline paints as intended.
  if (!isDashboardSyncEnabled(profile)) {
    return
  }

  const baseline = storedString(BASELINE_KEY)

  if (!baseline) {
    persistString(BASELINE_KEY, name)

    return
  }

  if (baseline === name) {
    return
  }

  persistString(BASELINE_KEY, name)
  const skin = toDesktopSkin(name)

  if (skin) {
    $pendingSkinApply.set(skin)
  }
}

/** Sync is enabled by default; only an explicit 'off' entry disables it. */
export const isDashboardSyncEnabled = (profile: string): boolean =>
  storedStringRecord(SYNC_KEY)[normalizeProfileKey(profile)] !== 'off'

/**
 * Mirror an explicit desktop skin pick to the Dashboard when it has a shared
 * name. A baseline that already equals the mapped dashboard name means this is
 * a value the module just applied from the Dashboard — PUTting it back would
 * echo, so it no-ops. A failed PUT is swallowed (the same way the web
 * dashboard swallows its api failures): the desktop's own pick stands, the
 * Dashboard simply keeps its own theme.
 */
export async function publishDashboardTheme(desktopSkinName: string, { profile }: { profile: string }): Promise<void> {
  if (!isDashboardSyncEnabled(profile)) {
    return
  }

  const theme = toDashboardTheme(desktopSkinName)

  if (!theme || storedString(BASELINE_KEY) === theme) {
    return
  }

  try {
    await setDashboardTheme(theme)
    persistString(BASELINE_KEY, theme)
  } catch {
    // Backend down / not up yet — nothing to reconcile against.
  }
}

// `gateway.ready`, window `focus` and `visibilitychange` can land in the same
// tick; this flag collapses those into one GET. Late callers return immediately
// rather than queueing — the in-flight fetch observes the freshest state the
// backend can serve anyway.
let refreshInFlight = false

/**
 * Pull the Dashboard's active theme and fold it in. Swallows failures: a
 * backend that isn't up yet must not surface an error — the next
 * focus/reconnect retries.
 */
export async function refreshDashboardTheme(profile: string): Promise<void> {
  if (refreshInFlight || !isDashboardSyncEnabled(profile)) {
    return
  }

  refreshInFlight = true

  try {
    const response = await getDashboardThemes()
    ingestDashboardTheme(response.active, { profile })
  } catch {
    // The next refresh picks it up once the backend is answering.
  } finally {
    refreshInFlight = false
  }
}

/** Opt a profile in/out of dashboard sync. Writes a fresh record — never mutates. */
export const setDashboardSyncEnabled = (profile: string, enabled: boolean): void => {
  const key = normalizeProfileKey(profile)
  const next: Record<string, string> = {}

  for (const [name, value] of Object.entries(storedStringRecord(SYNC_KEY))) {
    if (name !== key && value === 'off') {
      next[name] = value
    }
  }

  if (!enabled) {
    next[key] = 'off'
  }

  persistStringRecord(SYNC_KEY, next)
}

/** Shared desktop skin name → dashboard theme name; null for anything else. */
export const toDashboardTheme = (desktopSkinName: null | string | undefined): null | string => {
  const name = (desktopSkinName ?? '').trim()

  if (!name) {
    return null
  }

  // Reverse-scanned rather than a direct key lookup: the map is identity
  // today, so a lookup would pass every test — but scanning by skin value
  // stays correct if a name ever maps to a different desktop skin.
  const entry = Object.entries(DASHBOARD_SHARED_THEMES).find(([, skin]) => skin === name)

  return entry ? entry[0] : null
}

/** Shared dashboard theme name → desktop skin name; null for anything else. */
export const toDesktopSkin = (dashboardThemeName: null | string | undefined): null | string => {
  const name = (dashboardThemeName ?? '').trim()

  return name ? (DASHBOARD_SHARED_THEMES[name] ?? null) : null
}
