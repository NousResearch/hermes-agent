import { LOCAL_CONNECTION_ID, registryBackendScopeKey } from '@hermes/shared'

import { persistStringRecord, readJson, storedString, storedStringRecord, writeJson } from '@/lib/storage'

import { DEFAULT_SKIN_NAME, RETIRED_SKINS } from './presets'
import { resolveTheme } from './user-themes'

export const LAST_APPEARANCE_ROUTE_KEY = 'hermes-desktop-active-appearance-v2'
// Appearance-only compatibility scope, never a registry route or tunnel URL.
export const LEGACY_REMOTE_APPEARANCE_SCOPE = 'legacy:remote'

interface AppearanceConnection {
  connectionId?: string
  mode?: 'local' | 'remote'
}

export const appearanceConnectionId = (connection: AppearanceConnection | null): string =>
  connection?.connectionId?.trim() ||
  (connection?.mode === 'remote' ? LEGACY_REMOTE_APPEARANCE_SCOPE : LOCAL_CONNECTION_ID)

export const normalizeSkin = (name: string | null): string =>
  name && resolveTheme(name) && !RETIRED_SKINS.has(name) ? name : DEFAULT_SKIN_NAME

// Fresh profiles follow the OS; a stored mode stays independent of the skin.
const normalizeMode = (value: string | null) =>
  value === 'light' || value === 'dark' || value === 'system' ? value : 'system'

export const APPEARANCE_KEYS = new Set([
  'hermes-desktop-theme-v2',
  'hermes-desktop-mode-v1',
  'hermes-desktop-profile-themes-v1',
  'hermes-desktop-profile-modes-v1',
  'hermes-desktop-profile-themes-v2',
  'hermes-desktop-profile-modes-v2'
])

// Legacy records cannot identify a gateway. Leave them untouched as a stable
// upgrade fallback; new picks must never change what an unvisited peer inherits.
const profilePref = <T extends string>(record: string, legacy: string, normalize: (v: string | null) => T) => {
  const scopedRecord = record.replace(/v1$/, 'v2')
  const profileKey = (profile: string) => profile.trim() || 'default'

  // An empty scoped entry masks a renamed/deleted profile's legacy assignment
  // on this gateway only; same-named profiles elsewhere still inherit it.
  const own = (scoped: Record<string, string>, profile: string, connectionId: string): string | undefined => {
    const pick = scoped[registryBackendScopeKey(connectionId, profileKey(profile))]

    return pick === '' ? undefined : (pick ?? storedStringRecord(record)[profileKey(profile)])
  }

  // Only write a mask when a legacy entry exists to hide; otherwise just remove.
  const clear = (entries: Record<string, string>, profile: string, connectionId: string): void => {
    const key = registryBackendScopeKey(connectionId, profileKey(profile))

    if (storedStringRecord(record)[profileKey(profile)] !== undefined) {
      entries[key] = ''
    } else {
      delete entries[key]
    }
  }

  const stored = (profile: string, connectionId = LOCAL_CONNECTION_ID): string | null => {
    const connection = connectionId.trim()
    const scoped = connection ? storedStringRecord(scopedRecord) : {}
    const pick = own(scoped, profile, connection)

    if (pick !== undefined) {
      return pick
    }

    const inherited = scoped[`conn:${connection}`] ?? storedString(legacy)

    if (inherited != null) {
      return inherited
    }

    const values = [...new Set(Object.values(storedStringRecord(record)).filter(Boolean))]

    return values.length === 1 ? values[0] : null
  }

  return {
    stored,
    drop: (profile: string, connectionId: string): void => {
      const entries = storedStringRecord(scopedRecord)
      clear(entries, profile, connectionId)
      persistStringRecord(scopedRecord, entries)
    },
    // Carry only the profile's own pick; inherited fallbacks stay inherited.
    rename: (from: string, to: string, connectionId: string): void => {
      const entries = storedStringRecord(scopedRecord)
      const value = own(entries, from, connectionId)
      clear(entries, from, connectionId)

      if (value !== undefined) {
        entries[registryBackendScopeKey(connectionId, profileKey(to))] = value
      } else {
        clear(entries, to, connectionId)
      }

      persistStringRecord(scopedRecord, entries)
    },
    resolve: (profile: string, connectionId = LOCAL_CONNECTION_ID): T => normalize(stored(profile, connectionId)),
    assign: (profile: string, value: T, connectionId = LOCAL_CONNECTION_ID): void => {
      const connection = connectionId.trim()

      if (!connection) {
        return
      }

      persistStringRecord(scopedRecord, {
        ...storedStringRecord(scopedRecord),
        [registryBackendScopeKey(connection, profileKey(profile))]: value,
        // Preserve last-pick inheritance for unassigned profiles on THIS gateway.
        [`conn:${connection}`]: value
      })
    }
  }
}

export const skinPref = profilePref('hermes-desktop-profile-themes-v1', 'hermes-desktop-theme-v2', normalizeSkin)
export const modePref = profilePref('hermes-desktop-profile-modes-v1', 'hermes-desktop-mode-v1', normalizeMode)

export function dropAppearanceForProfile(profile: string, connectionId = LOCAL_CONNECTION_ID): void {
  if (!connectionId.trim()) {
    return
  }

  skinPref.drop(profile, connectionId)
  modePref.drop(profile, connectionId)
  const route = readJson<{ connectionId: string; profile: string }>(LAST_APPEARANCE_ROUTE_KEY)

  if (route?.connectionId === connectionId && route.profile === profile.trim()) {
    writeJson(LAST_APPEARANCE_ROUTE_KEY, { ...route, profile: 'default' })
  }
}

export function migrateAppearanceForProfile(from: string, to: string, connectionId = LOCAL_CONNECTION_ID): void {
  if (!connectionId.trim() || from.trim() === to.trim()) {
    return
  }

  skinPref.rename(from, to, connectionId)
  modePref.rename(from, to, connectionId)
  const route = readJson<{ connectionId: string; profile: string }>(LAST_APPEARANCE_ROUTE_KEY)

  if (route?.connectionId === connectionId && route.profile === from.trim()) {
    writeJson(LAST_APPEARANCE_ROUTE_KEY, { ...route, profile: to.trim() })
  }
}
