import type { UiPreferencesSnapshot, UiPreferencesWrite } from '../../../electron/ui-preferences'
import { markProvisionalPreference, onPersistenceEvent, previousValueOf, provisionalPreferenceKeys } from '../storage'

import { isSharedPreferenceKey } from './keys'
import { rebaseWrite } from './merge'
import { adoptOrphanedOutboxes, readOutbox, saveOutbox, siblingPendingKeys, watchOutbox } from './outbox'

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
// The store generation this origin has joined (imported its own values into).
// Until then a key the store lacks is never removed here: it may be this
// client's data the store has not received yet, not a peer's deletion.
const JOINED_KEY = 'hermes:ui-preferences:generation'
const RETRY_DELAY_MS = 5000
const MAX_RETRY_DELAY_MS = 60_000

// The shared keys this window held before the app's module graph evaluated
// (`captureOwnPreferences`, run by boot.ts): this client's own data. Stores
// that persist their default the moment their module loads add keys after
// that, and a first-contact import must not offer those defaults as data, or
// a fresh browser's defaults would fill the store before the client that has
// real values (native Desktop's keybinds, themes) imports them.
let ownKeys: null | ReadonlySet<string> = null

/** Record this window's own shared keys; boot.ts calls it before anything loads. */
export function captureOwnPreferences(): void {
  const provisional = provisionalPreferenceKeys()

  ownKeys = new Set(localSharedKeys().filter(key => !provisional.has(key)))
}

/** Cache one store value here. A key the store lacks is removed only once
 *  this origin joined (`prune`), and never while its value here is
 *  provisional; a value the store has replaces a provisional one for good. */
function adopt(key: string, value: null | string, prune: boolean, live: boolean) {
  if (value === null && (!prune || provisionalPreferenceKeys().has(key))) {
    return
  }

  if (value !== null) {
    markProvisionalPreference(key, false)
  }

  const previous = readRaw(key)

  if (writeRaw(key, value) && live) {
    announce(key, previous, value)
  }
}

/** `mine`, a change made from `base`, rebased onto `current` (see merge.ts);
 *  the cache takes the result, announced, when it differs from `mine`. */
function rebaseLocal(key: string, base: null | string, current: null | string, mine: null | string): null | string {
  const rebased = rebaseWrite(base, current, mine)

  if (rebased !== mine && writeRaw(key, rebased)) {
    announce(key, mine, rebased)
  }

  return rebased
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

  const outbox = new Set([...readOutbox().keys(), ...siblingPendingKeys()])
  const prune = hasJoined(snapshot)

  for (const key of new Set([...localSharedKeys(), ...Object.keys(snapshot.values)])) {
    if (isSharedPreferenceKey(key) && !outbox.has(key)) {
      adopt(key, snapshot.values[key] ?? null, prune, true)
    }
  }
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
  // What this window's stores last saw per key (read at load, read again, or
  // written), and the cached value each pending change was made on: a store
  // that never re-reads must not undo the entries a newer value brought.
  const loaded = new Map(localSharedKeys().map(key => [key, readRaw(key)]))
  const seen = new Map<string, null | string>()
  const pendingBase = new Map<string, null | string>()
  const seenOf = (key: string) => (seen.has(key) ? (seen.get(key) ?? null) : (loaded.get(key) ?? null))
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

    const siblings = siblingPendingKeys()

    for (const key of new Set([...localSharedKeys(), ...Object.keys(snapshot.values), ...known.keys()])) {
      if (!isSharedPreferenceKey(key)) {
        continue
      }

      const value = snapshot.values[key] ?? null
      known.set(key, value)

      // A change made here and not yet stored outranks an older store value,
      // rebased onto it so the store's other entries survive; so does one a
      // sibling tab has not stored yet.
      if (pending.has(key) && pendingBase.has(key) && pendingBase.get(key) !== value) {
        pending.set(key, rebaseLocal(key, pendingBase.get(key) ?? null, value, pending.get(key) ?? null))
        pendingBase.set(key, value)
        saveOutbox(pending)
      } else if (!pending.has(key) && !siblings.has(key) && !(refused.has(key) && refused.get(key) === readRaw(key))) {
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
        pendingBase.delete(key)
      }
    }

    saveOutbox(pending)
  }

  const stopListening = onPersistenceEvent(event => {
    if (!isSharedPreferenceKey(event.key)) {
      return
    }

    if (event.op === 'read') {
      seen.set(event.key, event.value)

      return
    }

    const base = seenOf(event.key)
    seen.set(event.key, event.value)

    // A provisional value (storage.ts keeps the mark) never leaves this origin.
    if (event.provisional) {
      return
    }

    if (!pending.has(event.key) && known.has(event.key) && known.get(event.key) === event.value) {
      return
    }

    const current = previousValueOf(event) ?? null
    const value = rebaseLocal(event.key, base, current, event.value)

    if (!pendingBase.has(event.key)) {
      pendingBase.set(event.key, current)
    }

    refused.delete(event.key)
    pending.set(event.key, value)
    store()
  })

  // A tab that closed with changes unsent left its outbox: take it over (its
  // values are already this origin's cache). This tab's own outbox stays
  // stamped as alive while it holds changes; a page going away sends at once.
  const outbox = watchOutbox({
    hasPending: () => pending.size > 0 && !stopped,
    onLeave: () => {
      saveOutbox(pending)

      if (flushTimer !== null) {
        clearTimeout(flushTimer)
        flushTimer = null
      }

      if (ready && !writing) {
        void flush()
      }
    },
    onOrphans: () => {
      adoptOrphanedOutboxes().forEach((value, key) => pending.has(key) || pending.set(key, value))
      store()
    },
    onReturn: () => saveOutbox(pending)
  })

  function store() {
    saveOutbox(pending)
    outbox.touch()
    schedule(FLUSH_DELAY_MS)
  }

  outbox.touch()

  // First contact with a generation: add this origin's values the store lacks.
  const join = async (snapshot: UiPreferencesSnapshot) => {
    if (hasJoined(snapshot)) {
      return snapshot
    }

    const provisional = provisionalPreferenceKeys()

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
    outbox.stop()
    // Nothing here sends it any more: the next tab adopts what is unsent.
    saveOutbox(pending, { closed: true })

    if (flushTimer !== null) {
      clearTimeout(flushTimer)
    }

    if (connectTimer !== null) {
      clearTimeout(connectTimer)
    }
  }
}
