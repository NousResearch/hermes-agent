// Local changes not yet confirmed by the machine's store, kept across reloads:
// a change made right before the tab reloads or closes must still win over the
// older store value on the next load, and be written then. One outbox PER TAB
// (its id rides in sessionStorage, which survives a reload): same-origin tabs
// share localStorage, and one shared outbox let each tab overwrite the others'
// unsent changes. A tab that closed with changes unsent (inside the debounce,
// offline, during a lock retry) leaves its outbox behind: the next tab of this
// origin adopts it once its owner is gone, so the change is still written and
// the key is removed (#145536 re-review). Each change keeps the value it was
// made on (`bases`), so whoever sends it rebases it onto the store's current
// value instead of writing it whole. Not shared keys themselves.
import { isSharedPreferenceKey } from './keys'

const OUTBOX_PREFIX = 'hermes:ui-preferences:outbox:'
const TAB_KEY = 'hermes:ui-preferences:tab'
// A live tab with unsent changes re-stamps its outbox this often. A hidden
// tab's timers can be throttled to once a minute, so an owner counts as gone
// only well past that, or once its page was hidden for good (`closed`).
export const OUTBOX_HEARTBEAT_MS = 10_000
const OUTBOX_DEAD_MS = 90_000

interface OutboxDoc {
  /** The tab taking this outbox over: it re-reads the mark before sending,
   *  and the owner, seeing it, stops sending (one sender per change). */
  adopter?: string
  bases: Record<string, null | string>
  beat: number
  closed?: boolean
  values: Record<string, null | string>
}

/** Unsent changes and, per change, the value it was made on. */
export interface Outbox {
  bases: Map<string, null | string>
  values: Map<string, null | string>
}

let outboxKey: null | string = null
// Set once the page is going away (pagehide), cleared if it comes back from the
// back/forward cache: every save from then on marks the outbox closed, since
// pagehide and visibilitychange fire in either order and a send in flight may
// still settle.
let pageClosing = false
// True while this tab's last save stored unsent changes: only then does a
// missing outbox mean another tab took it (not full or blocked storage).
let outboxHeld = false

function readRaw(key: string): null | string {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeRaw(key: string, value: null | string) {
  try {
    if (value === null) {
      window.localStorage.removeItem(key)
    } else {
      window.localStorage.setItem(key, value)
    }
  } catch {
    // Storage is best-effort.
  }
}

function parseOutbox(raw: null | string): null | OutboxDoc {
  try {
    const doc = JSON.parse(raw ?? 'null') as null | Partial<OutboxDoc>
    const values = doc?.values

    if (!doc || !values || typeof values !== 'object' || Array.isArray(values)) {
      return null
    }

    const entries = (record: unknown) =>
      Object.fromEntries(
        Object.entries(record && typeof record === 'object' ? record : {}).filter(
          (entry): entry is [string, null | string] =>
            isSharedPreferenceKey(entry[0]) && (entry[1] === null || typeof entry[1] === 'string')
        )
      )

    return {
      ...(typeof doc.adopter === 'string' && { adopter: doc.adopter }),
      bases: entries(doc.bases),
      beat: typeof doc.beat === 'number' ? doc.beat : 0,
      closed: doc.closed === true,
      values: entries(values)
    }
  } catch {
    return null
  }
}

function outboxKeys(): string[] {
  const keys: string[] = []

  try {
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index)

      if (key?.startsWith(OUTBOX_PREFIX)) {
        keys.push(key)
      }
    }
  } catch {
    // Restricted storage reads as empty.
  }

  return keys
}

/** Claim another load's outbox and remove it: mark this tab as adopter, then
 *  re-read. Null when it is gone or another tab (or its owner, re-saving)
 *  wrote after the mark: that one sends it, not this tab. */
function takeOutbox(key: string): null | OutboxDoc {
  const doc = parseOutbox(readRaw(key))

  if (!doc) {
    writeRaw(key, null)

    return null
  }

  writeRaw(key, JSON.stringify({ ...doc, adopter: outboxKey }))
  const claimed = parseOutbox(readRaw(key))

  if (claimed?.adopter !== outboxKey) {
    return null
  }

  writeRaw(key, null)

  return claimed
}

/**
 * Take over the outboxes of this origin's tabs that are gone, oldest first so
 * a later change to a key wins, and remove them. An entry is taken only while
 * this origin's cache still holds its value: anything since (an edit here, a
 * newer store value a live tab adopted) superseded it.
 */
export function adoptOrphanedOutboxes(): Outbox {
  const now = Date.now()
  const own = outboxKey

  const orphans = outboxKeys()
    .filter(key => key !== own)
    .map(key => ({ doc: parseOutbox(readRaw(key)), key }))
    .filter(({ doc }) => !doc || doc.closed || now - doc.beat > OUTBOX_DEAD_MS)
    .sort((left, right) => (left.doc?.beat ?? 0) - (right.doc?.beat ?? 0))

  const adopted: Outbox = { bases: new Map(), values: new Map() }

  for (const { key } of orphans) {
    const doc = takeOutbox(key)

    for (const [name, value] of Object.entries(doc?.values ?? {})) {
      if (readRaw(name) === value) {
        mergeEntry(adopted, doc as OutboxDoc, name)
      }
    }
  }

  return adopted
}

function mergeEntry(into: Outbox, doc: OutboxDoc, name: string) {
  into.values.set(name, doc.values[name] ?? null)

  if (Object.hasOwn(doc.bases, name)) {
    into.bases.set(name, doc.bases[name] ?? null)
  } else {
    into.bases.delete(name)
  }
}

/** True when this tab's unsent changes were taken over by another tab (its
 *  outbox is gone or carries another adopter): it must not send them. */
export function outboxTakenOver(): boolean {
  const doc = parseOutbox(readRaw(claimOutboxKey()))

  const taken = doc ? doc.adopter !== undefined && doc.adopter !== outboxKey : outboxHeld

  // Reported once: changes saved after this are this tab's again.
  outboxHeld = outboxHeld && !taken

  return taken
}

/** Keys another tab of this origin changed and has not stored yet: its edit
 *  is newer than the store's value, so no tab adopts the older value over it
 *  in the cache they share (and the orphan stays adoptable). */
export function siblingPendingKeys(): Set<string> {
  const own = claimOutboxKey()

  return new Set(
    outboxKeys()
      .filter(key => key !== own)
      .flatMap(key => Object.keys(parseOutbox(readRaw(key))?.values ?? {}))
  )
}

/** This tab's outbox key, claimed once per page load. The previous load of this
 *  tab (same sessionStorage) hands its unsent changes over, and the outboxes
 *  of tabs that are gone join them. */
export function claimOutboxKey(): string {
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
  const carried = previous ? takeOutbox(`${OUTBOX_PREFIX}${previous}`) : null
  const outbox = adoptOrphanedOutboxes()

  for (const name of Object.keys(carried?.values ?? {})) {
    mergeEntry(outbox, carried as OutboxDoc, name)
  }

  saveOutbox(outbox.values, { bases: outbox.bases })

  return outboxKey
}

export function readOutbox(): Outbox {
  const doc = parseOutbox(readRaw(claimOutboxKey()))

  return { bases: new Map(Object.entries(doc?.bases ?? {})), values: new Map(Object.entries(doc?.values ?? {})) }
}

/** Store this tab's unsent changes (and the value each was made on) with a
 *  fresh liveness stamp; `closed` once the page is going away, so the next
 *  tab need not wait for the stamp to age. */
export function saveOutbox(
  pending: ReadonlyMap<string, null | string>,
  { bases = new Map(), closed = pageClosing }: { bases?: ReadonlyMap<string, null | string>; closed?: boolean } = {}
) {
  const key = claimOutboxKey()

  const doc: OutboxDoc = {
    bases: Object.fromEntries([...bases].filter(([name]) => pending.has(name))),
    beat: Date.now(),
    ...(closed && { closed }),
    values: Object.fromEntries(pending)
  }

  writeRaw(key, pending.size === 0 ? null : JSON.stringify(doc))
  outboxHeld = pending.size > 0 && readRaw(key) !== null
}

/** Heartbeat and page-lifecycle hooks for one running sync. `touch()` after
 *  every outbox save arms the heartbeat (`onOrphans`: adopt, re-stamp, touch)
 *  while the outbox holds changes, and only then: an idle tab runs no timer.
 *  Another tab's outbox saved as `closed` is an orphan the moment its page goes. */
export function watchOutbox({
  hasPending,
  onLeave,
  onOrphans,
  onReturn
}: {
  hasPending: () => boolean
  onLeave: () => void
  onOrphans: () => void
  onReturn: () => void
}): { stop: () => void; touch: () => void } {
  let timer: null | ReturnType<typeof setTimeout> = null

  const touch = () => {
    if (timer === null && hasPending()) {
      timer = setTimeout(() => {
        timer = null
        onOrphans()
      }, OUTBOX_HEARTBEAT_MS)
    }
  }

  const hide = () => {
    pageClosing = true
    onLeave()
  }

  const show = () => {
    pageClosing = false
    onReturn()
  }

  const visibility = () => {
    if (document.visibilityState === 'hidden') {
      onLeave()
    }
  }

  const storage = (event: StorageEvent) => {
    if (event.key?.startsWith(OUTBOX_PREFIX) && event.key !== outboxKey && parseOutbox(event.newValue)?.closed) {
      onOrphans()
    }
  }

  window.addEventListener('pagehide', hide)
  window.addEventListener('pageshow', show)
  window.addEventListener('storage', storage)
  document.addEventListener('visibilitychange', visibility)

  return {
    stop: () => {
      if (timer !== null) {
        clearTimeout(timer)
      }

      window.removeEventListener('pagehide', hide)
      window.removeEventListener('pageshow', show)
      window.removeEventListener('storage', storage)
      document.removeEventListener('visibilitychange', visibility)
    },
    touch
  }
}
