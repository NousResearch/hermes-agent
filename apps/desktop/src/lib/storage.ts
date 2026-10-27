// ── Persistence choke point ─────────────────────────────────────────────────
// Every persisted read/write in the app funnels through readKey/writeKey, so a
// single subscriber (telemetry, cross-window sync, an audit log) can observe all
// of it without instrumenting each call site. No listeners by default → no cost.

import { isSharedPreferenceKey } from './ui-preferences/keys'

export interface PersistenceEvent {
  key: string
  op: 'read' | 'remove' | 'write'
  /** A value this client stored for itself (a migration or a default), not a
   *  choice the user made: the machine's shared preferences never take it. */
  provisional?: boolean
  value: null | string
}

type PersistenceListener = (event: PersistenceEvent) => void

const persistenceListeners = new Set<PersistenceListener>()

/** Observe every persisted get/set (e.g. pipe into telemetry/sync). */
export function onPersistenceEvent(listener: PersistenceListener): () => void {
  persistenceListeners.add(listener)

  return () => void persistenceListeners.delete(listener)
}

function emitPersistence(event: PersistenceEvent) {
  for (const listener of persistenceListeners) {
    listener(event)
  }
}

/** Raw read. Returns null when absent or storage is unavailable. */
export function readKey(key: string): null | string {
  let value: null | string = null

  try {
    value = window.localStorage.getItem(key)
  } catch {
    // Restricted contexts (private mode, disabled storage) read as absent.
  }

  emitPersistence({ key, op: 'read', value })

  return value
}

export interface WriteOptions {
  /** With `provisional`: the value is the store's default, so it stays
   *  provisional even when it is already stored (an older build wrote it). */
  isDefault?: boolean
  /** A value this client derives for itself (a default, a heal, a migration
   *  that only fills a default), not a choice the user made. */
  provisional?: boolean
}

// Shared preference keys whose value here is provisional. Kept at the choke
// point, not in the sync, because stores write their load-time value while
// their module evaluates, before the sync is listening: the machine's shared
// store never takes such a value, by seed or by write, so a fresh client's
// defaults cannot fill it ahead of the client that holds the user's choices
// (#145536 review). Not a shared key itself.
export const PROVISIONAL_PREFERENCES_KEY = 'hermes:ui-preferences:provisional'

function rawRead(key: string): null | string {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function rawWrite(key: string, value: null | string) {
  try {
    if (value === null) {
      window.localStorage.removeItem(key)
    } else {
      window.localStorage.setItem(key, value)
    }
  } catch {
    // Storage is best-effort; never let a quota/permission error break the UI.
  }
}

export function provisionalPreferenceKeys(): Set<string> {
  try {
    const parsed = JSON.parse(rawRead(PROVISIONAL_PREFERENCES_KEY) ?? '[]') as unknown

    return new Set(Array.isArray(parsed) ? parsed.filter((key): key is string => typeof key === 'string') : [])
  } catch {
    return new Set()
  }
}

export function markProvisionalPreference(key: string, provisional: boolean) {
  const keys = provisionalPreferenceKeys()

  if (keys.has(key) === provisional) {
    return
  }

  if (provisional) {
    keys.add(key)
  } else {
    keys.delete(key)
  }

  rawWrite(PROVISIONAL_PREFERENCES_KEY, keys.size === 0 ? null : JSON.stringify([...keys]))
}

/** Raw write. A null value removes the key. Best-effort. A provisional write
 *  that changes nothing is skipped, and one that rewrites the user's own value
 *  (a heal, a migration from a real value) keeps it theirs. */
export function writeKey(
  key: string,
  value: null | string,
  { isDefault = false, provisional = false }: WriteOptions = {}
) {
  const shared = isSharedPreferenceKey(key)

  if (provisional) {
    const previous = rawRead(key)

    if (previous === value) {
      if (shared && isDefault && value !== null) {
        markProvisionalPreference(key, true)
      }

      return
    }

    provisional = value !== null && (isDefault || !shared || previous === null || provisionalPreferenceKeys().has(key))
  }

  rawWrite(key, value)

  if (shared) {
    markProvisionalPreference(key, provisional)
  }

  emitPersistence({ key, op: value === null ? 'remove' : 'write', value, ...(provisional && { provisional }) })
}


/** Parsed JSON read. Returns null on absence, unavailable storage, OR malformed
 *  JSON — callers layer their own shape validation on the parsed value. */
export function readJson<T>(key: string): T | null {
  const raw = readKey(key)

  if (raw === null) {
    return null
  }

  try {
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

/** JSON write; a null value removes the key. Best-effort (see writeKey). */
export function writeJson(key: string, value: unknown) {
  writeKey(key, value === null ? null : JSON.stringify(value))
}

export function storedBoolean(key: string, fallback: boolean): boolean {
  const value = readKey(key)

  return value === null ? fallback : value === 'true'
}

export function persistBoolean(key: string, value: boolean, options?: WriteOptions) {
  writeKey(key, String(value), options)
}

export function storedString(key: string): null | string {
  return readKey(key)
}

export function persistString(key: string, value: null | string, options?: WriteOptions) {
  writeKey(key, value, options)
}

export function storedStringArray(key: string): string[] {
  const value = readKey(key)

  if (!value) {
    return []
  }

  try {
    const parsed = JSON.parse(value)

    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter((item): item is string => typeof item === 'string' && item.length > 0)
  } catch {
    return []
  }
}

export function persistStringArray(key: string, value: string[], options?: WriteOptions) {
  writeKey(key, value.length === 0 ? null : JSON.stringify(value), options)
}

export function storedStringRecord(key: string): Record<string, string> {
  const value = readKey(key)

  if (!value) {
    return {}
  }

  try {
    const parsed = JSON.parse(value)

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
    )
  } catch {
    return {}
  }
}

export function persistStringRecord(key: string, value: Record<string, string>, options?: WriteOptions) {
  writeKey(key, JSON.stringify(value), options)
}

export function arraysEqual(left: string[], right: string[]) {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

export function insertUniqueId(ids: string[], id: string, index: number) {
  const next = ids.filter(item => item !== id)
  const boundedIndex = Math.min(Math.max(index, 0), next.length)
  next.splice(boundedIndex, 0, id)

  return next
}
