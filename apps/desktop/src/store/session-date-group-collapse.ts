import type { HermesConnection } from '@/global'
import { Codecs, persistentAtom } from '@/lib/persisted'

export type CollapsedSessionDateGroups = Record<string, string[]>

export const SESSION_DATE_GROUP_COLLAPSE_STORAGE_KEY = 'hermes.desktop.sessionDateGroups.collapsed.v1'

const sanitizeCollapsedGroups = (value: unknown): CollapsedSessionDateGroups => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(value).flatMap(([scope, keys]) => {
      if (!Array.isArray(keys)) {
        return []
      }

      const cleanKeys = [...new Set(keys.filter((key): key is string => typeof key === 'string' && key.length > 0))]

      return cleanKeys.length > 0 ? [[scope, cleanKeys]] : []
    })
  )
}

export const $collapsedSessionDateGroups = persistentAtom<CollapsedSessionDateGroups>(
  SESSION_DATE_GROUP_COLLAPSE_STORAGE_KEY,
  {},
  Codecs.json(sanitizeCollapsedGroups)
)

interface SessionDateGroupProfileOptions {
  allProfilesKey: string
  showAllProfiles: boolean
}

export function resolveSessionDateGroupProfile(
  connection: HermesConnection | null,
  requestedProfile: null | string | undefined,
  { allProfilesKey, showAllProfiles }: SessionDateGroupProfileOptions
): string {
  const requested = requestedProfile?.trim()

  if (showAllProfiles) {
    return requested || allProfilesKey
  }

  if (requested && requested !== allProfilesKey) {
    return requested
  }

  return connection?.profile?.trim() || 'default'
}

export function sessionDateGroupScope(connection: HermesConnection | null, activeProfile?: null | string): string {
  const mode = connection?.mode ?? 'disconnected'
  const endpoint = (connection?.baseUrl?.trim() || 'local').replace(/\/+$/, '')
  const profile = activeProfile?.trim() || connection?.profile?.trim() || 'default'

  return JSON.stringify([mode, endpoint, profile])
}

export function getCollapsedSessionDateGroups(scope: string): Set<string> {
  return new Set($collapsedSessionDateGroups.get()[scope] ?? [])
}

const writeCollapsedSessionDateGroups = (scope: string, keys: ReadonlySet<string>) => {
  const current = $collapsedSessionDateGroups.get()
  const next = { ...current }

  if (keys.size === 0) {
    delete next[scope]
  } else {
    next[scope] = [...keys]
  }

  $collapsedSessionDateGroups.set(next)
}

export function setSessionDateGroupCollapsed(scope: string, key: string, collapsed: boolean): void {
  const keys = getCollapsedSessionDateGroups(scope)

  if (collapsed) {
    keys.add(key)
  } else {
    keys.delete(key)
  }

  writeCollapsedSessionDateGroups(scope, keys)
}

export function collapseAllSessionDateGroups(scope: string, knownKeys: readonly string[]): void {
  const keys = getCollapsedSessionDateGroups(scope)

  knownKeys.forEach(key => keys.add(key))
  writeCollapsedSessionDateGroups(scope, keys)
}

export function expandAllSessionDateGroups(scope: string, knownKeys: readonly string[]): void {
  const keys = getCollapsedSessionDateGroups(scope)

  knownKeys.forEach(key => keys.delete(key))
  writeCollapsedSessionDateGroups(scope, keys)
}
