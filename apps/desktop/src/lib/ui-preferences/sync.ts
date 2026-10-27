import type { UiPreferencesSnapshot, UiPreferencesWrite } from '../../../electron/ui-preferences'
import { onPersistenceEvent } from '../storage'

import { isSharedPreferenceKey } from './keys'

export type { UiPreferencesSnapshot, UiPreferencesWrite }

/** The host's door to the machine store: preload IPC natively, `/api/ui-preferences` in a browser. */
export interface UiPreferencesBridge {
  /** Native only: the store as main read it before this window painted. */
  snapshot?: null | UiPreferencesSnapshot
  read(): Promise<UiPreferencesSnapshot>
  write(payload: UiPreferencesWrite): Promise<UiPreferencesSnapshot>
  /** Pushes (native) or polls (browser) the store after another writer changed it. */
  onChanged?(callback: (snapshot: UiPreferencesSnapshot) => void): () => void
}

const FLUSH_DELAY_MS = 150
// Local changes not yet confirmed by the store, kept across reloads: a change
// made right before the tab reloads or closes must still win over the older
// store value on the next load, and be written then. Not a shared key itself.
const OUTBOX_KEY = 'hermes:ui-preferences:outbox'
const RETRY_DELAY_MS = 5000
const MAX_RETRY_DELAY_MS = 60_000

function localSharedKeys(): string[] {
  const keys: string[] = []

  try {
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index)

      if (key && isSharedPreferenceKey(key)) {
        keys.push(key)
      }
    }
  } catch {
    // Restricted storage reads as empty.
  }

  return keys
}

function readRaw(key: string): null | string {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

/** Set one cached key to the store's value; true when localStorage changed. Raw on
 *  purpose: the choke point would echo it back up as a local write. */
function writeRaw(key: string, value: null | string): boolean {
  const previous = readRaw(key)

  if (previous === value) {
    return false
  }

  try {
    if (value === null) {
      window.localStorage.removeItem(key)
    } else {
      window.localStorage.setItem(key, value)
    }
  } catch {
    return false
  }

  return true
}

/**
 * Make this window's localStorage cache match the store, before anything has
 * read it: the first-paint path for native Desktop, whose preload hands over
 * a snapshot synchronously. A store that does not exist yet changes nothing
 * (the sync seeds it from this cache instead).
 */
export function hydrateFromSnapshot(snapshot: null | undefined | UiPreferencesSnapshot): void {
  if (!snapshot?.exists) {
    return
  }

  const outbox = readOutbox()

  for (const key of new Set([...localSharedKeys(), ...Object.keys(snapshot.values)])) {
    if (isSharedPreferenceKey(key) && !outbox.has(key)) {
      writeRaw(key, snapshot.values[key] ?? null)
    }
  }
}

function readOutbox(): Map<string, null | string> {
  try {
    const parsed = JSON.parse(readRaw(OUTBOX_KEY) ?? '{}') as unknown

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return new Map()
    }

    return new Map(
      Object.entries(parsed).filter(
        (entry): entry is [string, null | string] =>
          isSharedPreferenceKey(entry[0]) && (entry[1] === null || typeof entry[1] === 'string')
      )
    )
  } catch {
    return new Map()
  }
}

function saveOutbox(pending: Map<string, null | string>) {
  writeRaw(OUTBOX_KEY, pending.size === 0 ? null : JSON.stringify(Object.fromEntries(pending)))
}

/** A write the store answered "no" to (HTTP 400 / validation), as opposed to
 *  one that could not reach it; only the latter is worth retrying. */
function isRefusal(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /^400:|unsupported preference|unsupported value|store is full/.test(message)
}

function announce(key: string, oldValue: null | string, newValue: null | string) {
  // Existing cross-window listeners (theme, translucency, haptics, persistent
  // atoms) already follow `storage` events; a same-window apply fires one too.
  window.dispatchEvent(new StorageEvent('storage', { key, newValue, oldValue }))
}

/**
 * Keep this window's shared preferences and the machine store in step:
 * - first contact with no store imports this window's existing values once
 *   (`seed`), and never overwrites a store that exists;
 * - a local change to a shared key is merged into the store per key, last
 *   write wins, debounced and retried while the store is unreachable;
 * - a change another window, tab or device made lands here live.
 */
export function startUiPreferencesSync(bridge: null | UiPreferencesBridge | undefined): () => void {
  if (!bridge) {
    return () => undefined
  }

  const known = new Map<string, null | string>()
  const pending = readOutbox()
  // Values the store refused, kept in this window until changed again.
  const refused = new Map<string, null | string>()
  let revision = -1
  let flushTimer: ReturnType<typeof setTimeout> | null = null
  let stopped = false
  // Nothing is written until first contact: a write that created the store
  // ahead of the seed would make the seed see an existing store and skip the
  // rest of this window's values.
  let ready = false

  const apply = (snapshot: UiPreferencesSnapshot, { live }: { live: boolean }) => {
    if (!snapshot.exists || snapshot.revision < revision) {
      return
    }

    revision = snapshot.revision

    for (const key of new Set([...localSharedKeys(), ...Object.keys(snapshot.values), ...known.keys()])) {
      if (!isSharedPreferenceKey(key)) {
        continue
      }

      const value = snapshot.values[key] ?? null
      known.set(key, value)

      // A change made here and not yet stored outranks an older store value.
      if (pending.has(key) || (refused.has(key) && refused.get(key) === readRaw(key))) {
        continue
      }

      const previous = readRaw(key)

      if (writeRaw(key, value) && live) {
        announce(key, previous, value)
      }
    }
  }

  let writing = false
  let writeDelay = RETRY_DELAY_MS

  const schedule = (delay: number) => {
    if (flushTimer === null && !writing && ready && !stopped) {
      flushTimer = setTimeout(() => void flush(), delay)
    }
  }

  // One write in flight at a time: two independent requests could commit out
  // of order, and the older batch's later revision would win.
  const flush = async () => {
    flushTimer = null

    if (pending.size === 0) {
      return
    }

    const batch = new Map(pending)
    let retryIn = FLUSH_DELAY_MS
    writing = true

    try {
      const snapshot = await bridge.write({ values: Object.fromEntries(batch) })

      settle(batch)
      writeDelay = RETRY_DELAY_MS
      apply(snapshot, { live: true })
    } catch (error) {
      if (isRefusal(error)) {
        // The store refused this batch outright (oversized value, full store):
        // it stays in this window only, and later changes are not held behind it.
        console.warn('[ui-preferences] the machine store refused these preferences', [...batch.keys()], error)
        batch.forEach((value, key) => refused.set(key, value))
        settle(batch)
      } else {
        retryIn = writeDelay
        writeDelay = Math.min(writeDelay * 2, MAX_RETRY_DELAY_MS)
      }
    } finally {
      writing = false
    }

    if (pending.size > 0) {
      schedule(retryIn)
    }
  }

  const settle = (batch: Map<string, null | string>) => {
    for (const [key, value] of batch) {
      if (pending.get(key) === value) {
        pending.delete(key)
      }
    }

    saveOutbox(pending)
  }

  const stopListening = onPersistenceEvent(event => {
    if (event.op === 'read' || !isSharedPreferenceKey(event.key)) {
      return
    }

    if (!pending.has(event.key) && known.has(event.key) && known.get(event.key) === event.value) {
      return
    }

    refused.delete(event.key)
    pending.set(event.key, event.value)
    saveOutbox(pending)
    schedule(FLUSH_DELAY_MS)
  })

  const seed = async (snapshot: UiPreferencesSnapshot) => {
    if (snapshot.exists) {
      return snapshot
    }

    const values = Object.fromEntries(localSharedKeys().map(key => [key, readRaw(key)]))

    return bridge.write({ seed: true, values })
  }

  let connectTimer: ReturnType<typeof setTimeout> | null = null
  let connectDelay = RETRY_DELAY_MS

  const connect = (initial: null | UiPreferencesSnapshot | undefined) => {
    connectTimer = null

    void (initial ? Promise.resolve(initial) : bridge.read())
      .then(seed)
      .then(snapshot => {
        if (stopped) {
          return
        }

        apply(snapshot, { live: true })
        ready = true
        schedule(FLUSH_DELAY_MS)
      })
      .catch(() => {
        // Unreachable store: this window keeps its cached values and local
        // changes stay pending until first contact succeeds.
        if (!stopped) {
          connectTimer = setTimeout(() => connect(null), connectDelay)
          connectDelay = Math.min(connectDelay * 2, MAX_RETRY_DELAY_MS)
        }
      })
  }

  connect(bridge.snapshot)

  const stopChanges = bridge.onChanged?.(snapshot => apply(snapshot, { live: true })) ?? (() => undefined)

  return () => {
    stopped = true
    stopListening()
    stopChanges()

    if (flushTimer !== null) {
      clearTimeout(flushTimer)
    }

    if (connectTimer !== null) {
      clearTimeout(connectTimer)
    }
  }
}
