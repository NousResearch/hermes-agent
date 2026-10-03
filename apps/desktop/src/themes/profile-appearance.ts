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
 * A profile name belongs to ONE gateway, so every read, write and piece of
 * bookkeeping here is keyed by its owner: (connection, profile), captured
 * synchronously when the read or pick happens. A queued write therefore still
 * lands on the gateway it was picked on after the window switches to another.
 *
 * Reads race writes, so every read is stamped on one clock with every local
 * change: a published value speaks for the backend only when its GET began
 * after the owner's last local change and no write for it is in flight. A
 * slower, older GET can never snap a fresh pick back. Writes race each other,
 * so an owner's writes go out one at a time in pick order: the newest pick is
 * always the last PUT to land.
 */

import { atom } from 'nanostores'

import { ambientOwnerConnectionId, connectionScoped, getApiRequestProfile } from '@/api/client'
import { saveHermesConfig } from '@/api/config'
import { translateNow } from '@/i18n'

import type { ThemeMode } from './context'

export interface ProfileAppearance {
  profile: string
  /** `(connection, profile)` key of the gateway the GET was routed to. */
  owner: string
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
// Per owner: the settlement of its latest queued write.
const writeQueues = new Map<string, Promise<unknown>>()

const isThemeMode = (value: unknown): value is ThemeMode => value === 'light' || value === 'dark' || value === 'system'

// The connection an untagged request is served by right now ('local' for the
// local pool). Identity for keys only; never sent as a request pin.
export const profileAppearanceOwner = (profile: string): string => `${ambientOwnerConnectionId() ?? ''}::${profile}`

/** Record a local appearance change for `profile` on its captured or current connection (a
 *  pick, a settled write, a peer window's pick), so any value read before it
 *  no longer counts. */
export function markLocalAppearanceChange(profile: string, owner = profileAppearanceOwner(profile)): void {
  localChangeAt.set(owner, ++clock)
}

/** Call before the config GET: the owner it reads (the ambient request scope,
 *  exactly what the GET is routed by) and when it began. */
export function beginProfileAppearanceRead(): { owner: string; profile: string; readAt: number } {
  const profile = (getApiRequestProfile() ?? '').trim() || 'default'

  return { owner: profileAppearanceOwner(profile), profile, readAt: ++clock }
}

export function publishProfileAppearance(
  read: { owner: string; profile: string; readAt: number },
  desktop: unknown
): void {
  // This atom is the foreground publication, not a cache of every gateway.
  // A late read must not evict the current owner's value, even if both
  // gateways call their profile "default".
  const liveProfile = (getApiRequestProfile() ?? '').trim() || 'default'

  if (read.owner !== profileAppearanceOwner(liveProfile)) {
    return
  }

  const record = desktop && typeof desktop === 'object' ? (desktop as Record<string, unknown>) : {}
  const theme = typeof record.theme === 'string' ? record.theme.trim() : ''

  $profileAppearance.set({ ...read, mode: isThemeMode(record.theme_mode) ? record.theme_mode : '', theme })
}

/** Whether a published appearance still speaks for the backend. */
export function appearanceIsCurrent(appearance: ProfileAppearance): boolean {
  return !writesInFlight.get(appearance.owner) && appearance.readAt > (localChangeAt.get(appearance.owner) ?? 0)
}

/** Write a pick to the profile's config.yaml on the gateway it was picked on,
 *  after that owner's earlier writes settle. Sparse: PUT /api/config
 *  deep-merges, so echoing more would overwrite keys other surfaces changed. */
export async function saveProfileAppearance(profile: string, patch: ProfileAppearancePatch): Promise<void> {
  // A bare renderer (tests, the design preview) has no backend to write to.
  if (!window.hermesDesktop) {
    return
  }

  // Capture the owner now, not when the queue reaches this write: by then the
  // window may be routed to another gateway that also has this profile name.
  // An untagged pick stays untagged (the pin carries exactly the tag the
  // immediate write would have had), so it keeps Electron's untagged routing.
  const key = profileAppearanceOwner(profile)
  const pin = { connectionId: connectionScoped().connectionId, profile }

  localChangeAt.set(key, ++clock)
  writesInFlight.set(key, (writesInFlight.get(key) ?? 0) + 1)

  const write = (writeQueues.get(key) ?? Promise.resolve())
    .catch(() => undefined)
    .then(async () => {
      const result = await saveHermesConfig({ desktop: patch }, pin)

      if (!result?.ok) {
        throw new Error(translateNow('settings.config.autosaveFailed'))
      }
    })

  writeQueues.set(key, write)

  try {
    await write
  } finally {
    if (writeQueues.get(key) === write) {
      writeQueues.delete(key)
    }

    writesInFlight.set(key, (writesInFlight.get(key) ?? 1) - 1)
    // A GET that began while this write was in flight may have been served
    // before it landed.
    localChangeAt.set(key, ++clock)
  }
}
