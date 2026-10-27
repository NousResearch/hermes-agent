// Local changes not yet confirmed by the machine's store, kept across reloads:
// a change made right before the tab reloads or closes must still win over the
// older store value on the next load, and be written then. One outbox PER TAB
// (its id rides in sessionStorage, which survives a reload): same-origin tabs
// share localStorage, and one shared outbox let each tab overwrite the others'
// unsent changes. A tab that closed with changes unsent (inside the debounce,
// offline, during a lock retry) leaves its outbox behind: the next tab of this
// origin adopts it once its owner is gone, so the change is still written and
// the key is removed (#145536 re-review). Not shared keys themselves.
import { isSharedPreferenceKey } from './keys'

const OUTBOX_PREFIX = 'hermes:ui-preferences:outbox:'
const TAB_KEY = 'hermes:ui-preferences:tab'
// A live tab with unsent changes re-stamps its outbox this often. A hidden
// tab's timers can be throttled to once a minute, so an owner counts as gone
// only well past that, or once its page was hidden for good (`closed`).
export const OUTBOX_HEARTBEAT_MS = 10_000
const OUTBOX_DEAD_MS = 90_000

interface OutboxDoc {
  beat: number
  closed?: boolean
  values: Record<string, null | string>
}

let outboxKey: null | string = null
// Set once the page is going away (pagehide), cleared if it comes back from the
// back/forward cache: every save from then on marks the outbox closed, since
// pagehide and visibilitychange fire in either order and a send in flight may
// still settle.
let pageClosing = false

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

    return {
      beat: typeof doc.beat === 'number' ? doc.beat : 0,
      closed: doc.closed === true,
      values: Object.fromEntries(
        Object.entries(values).filter(
          (entry): entry is [string, null | string] =>
            isSharedPreferenceKey(entry[0]) && (entry[1] === null || typeof entry[1] === 'string')
        )
      )
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

/**
 * Take over the outboxes of this origin's tabs that are gone, oldest first so
 * a later change to a key wins, and remove them. An entry is taken only while
 * this origin's cache still holds its value: anything since (an edit here, a
 * newer store value a live tab adopted) superseded it.
 */
export function adoptOrphanedOutboxes(): Map<string, null | string> {
  const now = Date.now()
  const own = outboxKey

  const orphans = outboxKeys()
    .filter(key => key !== own)
    .map(key => ({ doc: parseOutbox(readRaw(key)), key }))
    .filter(({ doc }) => !doc || doc.closed || now - doc.beat > OUTBOX_DEAD_MS)
    .sort((left, right) => (left.doc?.beat ?? 0) - (right.doc?.beat ?? 0))

  const adopted = new Map<string, null | string>()

  for (const { doc, key } of orphans) {
    for (const [name, value] of Object.entries(doc?.values ?? {})) {
      if (readRaw(name) === value) {
        adopted.set(name, value)
      }
    }

    writeRaw(key, null)
  }

  return adopted
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
  const carried = previous ? parseOutbox(readRaw(`${OUTBOX_PREFIX}${previous}`)) : null

  if (previous) {
    writeRaw(`${OUTBOX_PREFIX}${previous}`, null)
  }

  saveOutbox(new Map([...adoptOrphanedOutboxes(), ...Object.entries(carried?.values ?? {})]))

  return outboxKey
}

export function readOutbox(): Map<string, null | string> {
  return new Map(Object.entries(parseOutbox(readRaw(claimOutboxKey()))?.values ?? {}))
}

/** Store this tab's unsent changes with a fresh liveness stamp; `closed` once
 *  the page is going away, so the next tab need not wait for the stamp to age. */
export function saveOutbox(pending: ReadonlyMap<string, null | string>, { closed = pageClosing } = {}) {
  const key = claimOutboxKey()
  const doc: OutboxDoc = { beat: Date.now(), ...(closed && { closed }), values: Object.fromEntries(pending) }

  writeRaw(key, pending.size === 0 ? null : JSON.stringify(doc))
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
