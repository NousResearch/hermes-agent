import { atom } from 'nanostores'

import type { HermesConnection } from '@/global'
import { desktopConnectionScope } from '@/lib/connection-scope'
import { readKey, storedStringArray, writeKey } from '@/lib/storage'

const LEGACY_PINNED_STORAGE_KEY = 'hermes.desktop.pinnedSessions'
const PINNED_STORAGE_KEY_PREFIX = 'hermes.desktop.pinnedSessions.v2.'

let activeStorageKey: null | string = null
let activeScopeInitialized = false
const scopeLoads = new WeakSet<readonly string[]>()

function decodePins(raw: null | string): string[] {
  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw) as unknown

    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === 'string' && item.length > 0)
      : []
  } catch {
    return []
  }
}

export const $pinnedSessionIds = atom<string[]>([])

$pinnedSessionIds.listen(value => {
  if (scopeLoads.delete(value) || !activeStorageKey) {
    return
  }

  activeScopeInitialized = true
  writeKey(activeStorageKey, JSON.stringify(value))
})

export function activatePinnedSessionConnection(connection: HermesConnection | null): void {
  const scope = desktopConnectionScope(connection)
  activeStorageKey = scope ? `${PINNED_STORAGE_KEY_PREFIX}${encodeURIComponent(scope)}` : null
  const raw = activeStorageKey ? readKey(activeStorageKey) : null
  const pins = decodePins(raw)

  activeScopeInitialized = raw !== null
  scopeLoads.add(pins)
  $pinnedSessionIds.set(pins)
}

export function pinnedSessionScopeInitialized(): boolean {
  return activeScopeInitialized
}

export function initializePinnedSessionScope(ids: string[]): void {
  if (activeScopeInitialized || !activeStorageKey) {
    return
  }

  activeScopeInitialized = true
  $pinnedSessionIds.set(ids)
}

export function legacyPinnedSessionIds(): string[] {
  return storedStringArray(LEGACY_PINNED_STORAGE_KEY)
}
