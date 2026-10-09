/**
 * A profile's appearance as its config.yaml records it (`desktop.theme`,
 * `desktop.theme_mode`).
 *
 * localStorage is per origin, so a pick that lived only there never reached the
 * Webapp (another origin) or another Desktop on the same profile. The backend
 * is the authority and localStorage the cache the boot paint reads: a config
 * load adopts here, ThemeProvider paints what was adopted for the profile it
 * shows, and a pick writes back through `saveProfileAppearance`.
 *
 * A profile name belongs to ONE gateway, so everything here is keyed by its
 * owner, (connection, profile), captured synchronously when the read or pick
 * happens: a queued write still lands on the gateway it was picked on after the
 * window switches to another.
 *
 * Order: GET and PUT /api/config carry the config.yaml revision (its mtime,
 * so any writer moves it: another window, process or editor; every PUT
 * advances it). Each field keeps the answer with the highest revision, so a
 * slower, older read or save never replaces a newer one. A backend that predates revisions sends none; there the
 * last answer wins, except a read begun before this window's latest write on
 * that owner, which may have been served before the write landed. This window's
 * own writes for an owner go out one at a time, in pick order, so its newest
 * pick is the last to land.
 */

import { atom } from 'nanostores'

import { ambientOwnerConnectionId, connectionScoped, getApiRequestProfile } from '@/api/client'
import { getHermesConfig, peekConfigRevision, saveHermesConfig } from '@/api/config'
import { translateNow } from '@/i18n/runtime'

import type { ThemeMode } from './context'

export type AppearanceField = 'theme' | 'theme_mode'

export interface ProfileAppearancePatch {
  theme?: string
  theme_mode?: ThemeMode
}

/** The foreground owner's adopted appearance; `''` = config.yaml leaves it unset. */
export interface ProfileAppearance {
  profile: string
  owner: string
  theme: string
  mode: '' | ThemeMode
}

interface Answer {
  value: string
  revision?: number
}

interface OwnerAppearance {
  /** Bumped when each write starts and settles: without revisions, a read begun before it may predate it. */
  generation: number
  /** This window's writes for the owner, chained in pick order. */
  writes: Promise<unknown>
  answers: Partial<Record<AppearanceField, Answer>>
}

/** The foreground owner's appearance, republished when a config load changes it. */
export const $profileAppearance = atom<null | ProfileAppearance>(null)

const owners = new Map<string, OwnerAppearance>()

export const isThemeMode = (value: unknown): value is ThemeMode =>
  value === 'light' || value === 'dark' || value === 'system'

/** The `(connection, profile)` key every appearance read, write and pick is owned by. */
export const appearanceOwnerKey = (connectionId: string, profile: string): string => `${connectionId}::${profile}`

// The connection an untagged request is served by right now ('local' for the
// local pool). Identity for keys only; never sent as a request pin.
export const profileAppearanceOwner = (profile: string): string =>
  appearanceOwnerKey(ambientOwnerConnectionId() ?? '', profile)

/** What `owner`'s config.yaml last answered for `field`; `''` while unset or unread. */
export const adoptedAppearance = (owner: string, field: AppearanceField): string =>
  owners.get(owner)?.answers[field]?.value ?? ''

// The profile the ambient request scope reads, exactly what a config GET is routed by.
const requestProfile = (): string => (getApiRequestProfile() ?? '').trim() || 'default'

function ownerAppearance(owner: string): OwnerAppearance {
  let state = owners.get(owner)

  if (!state) {
    state = { answers: {}, generation: 0, writes: Promise.resolve() }
    owners.set(owner, state)
  }

  return state
}

function adopt(state: OwnerAppearance, field: AppearanceField, value: string, revision?: number): void {
  const current = state.answers[field]?.revision

  if (revision === undefined || current === undefined || revision >= current) {
    state.answers[field] = { value, revision }
  }
}

export interface ProfileAppearanceRead {
  owner: string
  profile: string
  generation: number
}

/** Call before the config GET: the owner it reads and the writes it began after. */
export function beginProfileAppearanceRead(): ProfileAppearanceRead {
  const profile = requestProfile()
  const owner = profileAppearanceOwner(profile)

  return { generation: ownerAppearance(owner).generation, owner, profile }
}

/** Adopt a config load's appearance (a `getHermesConfig` answer). */
export function publishProfileAppearance(read: ProfileAppearanceRead, config: { desktop?: unknown }): void {
  const { owner, profile } = read
  const state = ownerAppearance(owner)
  const revision = peekConfigRevision(config)

  if (revision === undefined && state.generation !== read.generation) {
    return
  }

  const desktop = (config.desktop && typeof config.desktop === 'object' ? config.desktop : {}) as Record<
    string,
    unknown
  >

  adopt(state, 'theme', typeof desktop.theme === 'string' ? desktop.theme.trim() : '', revision)
  adopt(state, 'theme_mode', isThemeMode(desktop.theme_mode) ? desktop.theme_mode : '', revision)

  // This atom is the foreground publication, not a cache of every gateway: a
  // late answer must not evict the current owner's value, even if both
  // gateways call their profile "default".
  if (owner !== profileAppearanceOwner(requestProfile())) {
    return
  }

  const theme = adoptedAppearance(owner, 'theme')
  const mode = adoptedAppearance(owner, 'theme_mode') as ProfileAppearance['mode']
  const current = $profileAppearance.get()

  // An unchanged answer publishes nothing, so a re-read writes no cache and
  // wakes no peer window.
  if (current?.owner !== owner || current.theme !== theme || current.mode !== mode) {
    $profileAppearance.set({ mode, owner, profile, theme })
  }
}

/** Re-read the foreground owner's appearance: a peer window saved or adopted one. */
export async function refreshProfileAppearance(): Promise<void> {
  const read = beginProfileAppearanceRead()

  try {
    publishProfileAppearance(read, await getHermesConfig())
  } catch {
    // Like any failed config load: the next one reads again.
  }
}

/** Write to the profile's config.yaml on the gateway it was picked on, after
 *  that owner's earlier writes settle, and adopt what it saved (the caller
 *  repaints; only a config load publishes, so an unread field is never taken
 *  for an unset one). Sparse: PUT /api/config deep-merges, so echoing more
 *  would overwrite keys other surfaces changed. Rejects when the save fails. */
export async function saveProfileAppearance(profile: string, patch: ProfileAppearancePatch): Promise<void> {
  // Capture the owner now, not when the queue reaches this write: by then the
  // window may be routed to another gateway that also has this profile name.
  // An untagged pick stays untagged (the pin carries exactly the tag the
  // immediate write would have had), so it keeps Electron's untagged routing.
  const owner = profileAppearanceOwner(profile)
  const pin = { connectionId: connectionScoped().connectionId, profile }
  const state = ownerAppearance(owner)
  const write = state.writes.catch(() => undefined).then(() => saveHermesConfig({ desktop: patch }, pin))

  state.writes = write
  state.generation += 1

  try {
    const result = await write

    if (!result?.ok) {
      throw new Error(translateNow('settings.config.autosaveFailed'))
    }

    for (const [field, value] of Object.entries(patch) as [AppearanceField, string][]) {
      adopt(state, field, value, result.revision)
    }
  } finally {
    state.generation += 1
  }
}

/** A deleted or renamed profile's answers are not its successor's, whose
 *  config.yaml may carry an older revision. Writes in flight settle unseen. */
export function forgetProfileAppearance(owner: string): void {
  owners.delete(owner)
}
