/**
 * A profile's appearance as its config.yaml records it (`desktop.theme`,
 * `desktop.theme_mode`).
 *
 * localStorage is per origin, so a pick that lived only there never reached the
 * Webapp (another origin) or another Desktop on the same profile. The backend
 * is now the authority and localStorage the cache the boot paint reads: the
 * config load publishes here, ThemeProvider adopts it for the profile it
 * paints, and a pick writes back through `saveProfileAppearance`.
 *
 * Reads race writes, so every read is stamped on one clock with every local
 * change: a published value speaks for the backend only when its GET began
 * after the profile's last local change and no write for it is in flight. A
 * slower, older GET can never snap a fresh pick back.
 */

import { atom } from 'nanostores'

import { getApiRequestProfile } from '@/api/client'
import { saveHermesConfig } from '@/api/config'
import { translateNow } from '@/i18n'

import type { ThemeMode } from './context'

export interface ProfileAppearance {
  profile: string
  /** Desktop theme name as written; `''` = never picked. */
  theme: string
  /** `''` = never picked. */
  mode: '' | ThemeMode
  /** Clock reading when the GET that produced this began. */
  readAt: number
}

export interface ProfileAppearancePatch {
  theme?: string
  theme_mode?: ThemeMode
}

/** The last config load's appearance (for the profile that load served). */
export const $profileAppearance = atom<null | ProfileAppearance>(null)

let clock = 0
const localChangeAt = new Map<string, number>()
const writesInFlight = new Map<string, number>()

const isThemeMode = (value: unknown): value is ThemeMode => value === 'light' || value === 'dark' || value === 'system'

/** Record a local appearance change for `profile` (a pick, a settled write, a
 *  peer window's pick), so any value read before it no longer counts. */
export function markLocalAppearanceChange(profile: string): void {
  localChangeAt.set(profile, ++clock)
}

/** Call before the config GET: the profile it reads (the ambient request
 *  scope, exactly what the GET is routed by) and when it began. */
export function beginProfileAppearanceRead(): { profile: string; readAt: number } {
  return { profile: (getApiRequestProfile() ?? '').trim() || 'default', readAt: ++clock }
}

export function publishProfileAppearance(read: { profile: string; readAt: number }, desktop: unknown): void {
  const record = desktop && typeof desktop === 'object' ? (desktop as Record<string, unknown>) : {}
  const theme = typeof record.theme === 'string' ? record.theme.trim() : ''

  $profileAppearance.set({ ...read, mode: isThemeMode(record.theme_mode) ? record.theme_mode : '', theme })
}

/** Whether a published appearance still speaks for the backend. */
export function appearanceIsCurrent(appearance: ProfileAppearance): boolean {
  return !writesInFlight.get(appearance.profile) && appearance.readAt > (localChangeAt.get(appearance.profile) ?? 0)
}

/** Write a pick to the profile's config.yaml. Sparse: PUT /api/config
 *  deep-merges, so echoing more would overwrite keys other surfaces changed. */
export async function saveProfileAppearance(profile: string, patch: ProfileAppearancePatch): Promise<void> {
  // A bare renderer (tests, the design preview) has no backend to write to.
  if (!window.hermesDesktop) {
    return
  }

  markLocalAppearanceChange(profile)
  writesInFlight.set(profile, (writesInFlight.get(profile) ?? 0) + 1)

  try {
    const result = await saveHermesConfig({ desktop: patch }, profile)

    if (!result?.ok) {
      throw new Error(translateNow('settings.config.autosaveFailed'))
    }
  } finally {
    writesInFlight.set(profile, (writesInFlight.get(profile) ?? 1) - 1)
    // A GET that began while this write was in flight may have been served
    // before it landed.
    markLocalAppearanceChange(profile)
  }
}
