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
// store value on the next load, and be written then. One outbox PER TAB (its id
// rides in sessionStorage, which survives a reload): same-origin tabs share
// localStorage, and one shared outbox let each tab overwrite the others'
// unsent changes. Not shared keys themselves.
const OUTBOX_PREFIX = 'hermes:ui-preferences:outbox:'
const TAB_KEY = 'hermes:ui-preferences:tab'
// The store generation this origin has joined (imported its own values into).
// Until then a key the store lacks is never removed here: it may be this
// client's data the store has not received yet, not a peer's deletion.
const JOINED_KEY = 'hermes:ui-preferences:generation'
// Shared keys whose value here is provisional: one this client stored for
// itself (a migration from its own config, a default), not a user's choice.
// Such a value never enters the store, by seed or by write: a fresh browser's
// default would otherwise sit there first and the client with the real value
// (native Desktop's) could not import it. It stays here until the store has a
// value for the key, or the user makes a real change.
const PROVISIONAL_KEY = 'hermes:ui-preferences:provisional'
const RETRY_DELAY_MS = 5000
const MAX_RETRY_DELAY_MS = 60_000

let outboxKey: null | string = null
// The shared keys this window held before the app's module graph evaluated
// (`captureOwnPreferences`, run by boot.ts): this client's own data. Stores
// that persist their default the moment their module loads add keys after
// that, and a first-contact import must not offer those defaults as data, or
// a fresh browser's defaults would fill the store before the client that has
// real values (native Desktop's keybinds, themes) imports them.
let ownKeys: null | ReadonlySet<string> = null

/** Record this window's own shared keys; boot.ts calls it before anything loads. */
export function captureOwnPreferences(): void {
  const provisional = readProvisional()

  ownKeys = new Set(localSharedKeys().filter(key => !provisional.has(key)))
}

function readProvisional(): Set<string> {
  try {
    const parsed = JSON.parse(readRaw(PROVISIONAL_KEY) ?? '[]') as unknown

    return new Set(Array.isArray(parsed) ? parsed.filter((key): key is string => typeof key === 'string') : [])
  } catch {
    return new Set()
  }
}

function markProvisional(key: string, provisional: boolean) {
  const keys = readProvisional()

  if (keys.has(key) === provisional) {
    return
  }

  if (provisional) {
    keys.add(key)
  } else {
    keys.delete(key)
  }

  writeRaw(PROVISIONAL_KEY, keys.size === 0 ? null : JSON.stringify([...keys]))
}

/** Cache one store value here. A key the store lacks is removed only once
 *  this origin joined (`prune`), and never while its value here is
 *  provisional; a value the store has replaces a provisional one for good. */
function adopt(key: string, value: null | string, prune: boolean, live: boolean) {
  if (value === null && (!prune || readProvisional().has(key))) {
    return
  }

  if (value !== null) {
    markProvisional(key, false)
  }

  const previous = readRaw(key)

  if (writeRaw(key, value) && live) {
    announce(key, previous, value)
  }
}

/** This tab's outbox key, claimed once per page load. The previous load of this
 *  tab (same sessionStorage) hands its unsent changes over. */
function claimOutboxKey(): string {
  if (outboxKey) {
    return outboxKey
  }

  const id = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`
  let previous: null | string = null

  try {
    previous = window.sessionStorage.getItem(TAB_KEY)
    window.sessionStorage.setItem(TAB_KEY, id)
  } catch {
    // No sessionStorage: this load's outbox is still its own.
  }

  outboxKey = `${OUTBOX_PREFIX}${id}`

  if (previous) {
    const carried = readRaw(`${OUTBOX_PREFIX}${previous}`)

    if (carried !== null) {
      writeRaw(outboxKey, carried)
      writeRaw(`${OUTBOX_PREFIX}${previous}`, null)
    }
  }

  return outboxKey
}

const generationOf = (snapshot: UiPreferencesSnapshot) => snapshot.generation ?? ''

/** True once this origin imported its values into the snapshot's generation. */
function hasJoined(snapshot: UiPreferencesSnapshot): boolean {
  return snapshot.exists && readRaw(JOINED_KEY) === generationOf(snapshot)
}

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
 * Make this window's localStorage cache match the store, before any store
 * module has read it (`boot.ts` runs this ahead of the app's module graph).
 * A store that does not exist yet changes nothing (the sync seeds it from this
 * cache instead), and until this origin joined the store's generation a key
 * the store lacks is kept: the store may simply not have it yet. Every change
 * is announced, so anything already listening follows it.
 */
export function hydrateFromSnapshot(snapshot: null | undefined | UiPreferencesSnapshot): void {
  if (!snapshot?.exists) {
    return
  }

  const outbox = readOutbox()
  const prune = hasJoined(snapshot)

  for (const key of new Set([...localSharedKeys(), ...Object.keys(snapshot.values)])) {
    if (isSharedPreferenceKey(key) && !outbox.has(key)) {
      adopt(key, snapshot.values[key] ?? null, prune, true)
    }
  }
}

function readOutbox(): Map<string, null | string> {
  try {
    const parsed = JSON.parse(readRaw(claimOutboxKey()) ?? '{}') as unknown

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
  writeRaw(claimOutboxKey(), pending.size === 0 ? null : JSON.stringify(Object.fromEntries(pending)))
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
 * - first contact with a store generation (or with no store) imports this
 *   origin's existing values (`seed`: only keys the store lacks, so nothing is
 *   overwritten); until then no local key is removed for being absent there;
 * - a local change to a shared key is merged into the store per key, last
 *   write wins, debounced and retried while the store is unreachable;
 * - a change another window, tab or device made lands here live;
 * - a provisional value (a write marked `provisional`) never leaves this origin.
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
  let generation: null | string = null
  let flushTimer: ReturnType<typeof setTimeout> | null = null
  let stopped = false
  // Nothing is written until first contact: a write that created the store
  // ahead of the seed would make the seed see an existing store and skip the
  // rest of this window's values.
  let ready = false

  const apply = (snapshot: UiPreferencesSnapshot, { live }: { live: boolean }) => {
    // A replaced store (deleted, or corrupt and kept aside) starts a new
    // generation at a low revision: follow it rather than wait for the count.
    if (!snapshot.exists || (generationOf(snapshot) === generation && snapshot.revision < revision)) {
      return
    }

    revision = snapshot.revision
    generation = generationOf(snapshot)
    const prune = hasJoined(snapshot)

    if (prune) {
      // Joined: this window's cache mirrors the store, so all of it is data
      // worth re-importing should the store ever be replaced.
      ownKeys = null
    }

    for (const key of new Set([...localSharedKeys(), ...Object.keys(snapshot.values), ...known.keys()])) {
      if (!isSharedPreferenceKey(key)) {
        continue
      }

      const value = snapshot.values[key] ?? null
      known.set(key, value)

      // A change made here and not yet stored outranks an older store value.
      if (!pending.has(key) && !(refused.has(key) && refused.get(key) === readRaw(key))) {
        adopt(key, value, prune, live)
      }
    }

    if (!prune && ready) {
      rejoin(snapshot)
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

    // Provisional or not, the latest write decides what this value is.
    markProvisional(event.key, Boolean(event.provisional) && event.op === 'write')

    if (event.provisional) {
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

  // First contact with a generation: add this origin's values the store lacks.
  const join = async (snapshot: UiPreferencesSnapshot) => {
    if (hasJoined(snapshot)) {
      return snapshot
    }

    const provisional = readProvisional()

    const values = Object.fromEntries(
      localSharedKeys()
        .filter(key => (!ownKeys || ownKeys.has(key)) && !provisional.has(key))
        .map(key => [key, readRaw(key)])
    )

    // Nothing to import: an existing store is joined as it is, and no store is
    // created from nothing.
    const joinedSnapshot = Object.keys(values).length > 0 ? await bridge.write({ seed: true, values }) : snapshot

    if (joinedSnapshot.exists) {
      writeRaw(JOINED_KEY, generationOf(joinedSnapshot))
    }

    return joinedSnapshot
  }

  let joining = false
  let joinRefused = false

  // A live snapshot from a generation this origin has not joined (the store
  // was replaced while this window was open): import, then follow it.
  function rejoin(snapshot: UiPreferencesSnapshot) {
    if (joining || joinRefused || stopped) {
      return
    }

    joining = true
    join(snapshot).then(
      joinedSnapshot => {
        joining = false

        if (!stopped) {
          apply(joinedSnapshot, { live: true })
        }
      },
      error => {
        joining = false
        joinRefused = isRefusal(error)
      }
    )
  }

  let connectTimer: ReturnType<typeof setTimeout> | null = null
  let connectDelay = RETRY_DELAY_MS

  const connect = (initial: null | UiPreferencesSnapshot | undefined) => {
    connectTimer = null

    void (initial ? Promise.resolve(initial) : bridge.read())
      .then(join)
      .then(snapshot => {
        if (stopped) {
          return
        }

        ready = true
        apply(snapshot, { live: true })
        schedule(FLUSH_DELAY_MS)
      })
      .catch(error => {
        if (stopped) {
          return
        }

        if (isRefusal(error)) {
          // The store cannot take this origin's import (full): follow it
          // without removing anything here, and try the import on next load.
          joinRefused = true
          ready = true
          void bridge.read().then(
            snapshot => apply(snapshot, { live: true }),
            () => undefined
          )
          schedule(FLUSH_DELAY_MS)
        } else {
          // Unreachable store: this window keeps its cached values and local
          // changes stay pending until first contact succeeds.
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
