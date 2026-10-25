// Cross-tab exclusion for origins where `navigator.locks` does not exist.
// Plain HTTP is not a secure context, so two tabs of the webapp share an
// origin and have no Web Locks. A localStorage compare-and-swap lease is the
// claim there. Callers that already have `navigator.locks` (Electron, HTTPS)
// must keep using Web Locks; this module does not replace that path.
//
// The lease outlives one missed renewal on a background tab (Chromium clamps
// those timers to about a minute) and is dropped on `pagehide`, so a reload
// does not wait it out. A crashed tab stops renewing and the other tab takes
// the name once `expires` is past.

const PREFIX = 'hermes.desktop.tabLease.v1:'
const LEASE_MS = 180_000
const RENEW_MS = 30_000
const POLL_MS = 40

interface Lease {
  owner: string
  expires: number
}

type AcquireResult = 'acquired' | 'busy' | 'unavailable'

type Taken =
  | { ok: false; reason: 'busy' | 'unavailable' }
  | { ok: true; release: () => void }

const held = new Set<() => void>()

function localStorageOrNull(): Storage | null {
  try {
    if (typeof window === 'undefined') {
      return null
    }

    return window.localStorage
  } catch {
    return null
  }
}

function storageKey(name: string): string {
  // Journal names carry the message text and can be long. The lease only needs
  // a stable id, and a short key stays inside localStorage limits.
  let high = 0x811c9dc5
  let low = 0x811c9dc5 ^ name.length

  for (let index = 0; index < name.length; index += 1) {
    const code = name.charCodeAt(index)

    high = Math.imul(high ^ code, 0x01000193)
    low = Math.imul(low ^ (code + index), 0x01000193)
  }

  const hex = (value: number) => (value >>> 0).toString(16).padStart(8, '0')

  return `${PREFIX}${hex(high)}${hex(low)}`
}

function readLease(store: Storage, key: string): Lease | null {
  const raw = store.getItem(key)

  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<Lease>

    if (typeof parsed.owner !== 'string' || typeof parsed.expires !== 'number') {
      return null
    }

    return { expires: parsed.expires, owner: parsed.owner }
  } catch {
    return null
  }
}

function compareAndSwap(key: string, owner: string, now: number): AcquireResult {
  const store = localStorageOrNull()

  if (!store) {
    return 'unavailable'
  }

  try {
    const current = readLease(store, key)

    if (current && current.owner !== owner && current.expires > now) {
      return 'busy'
    }

    const next = JSON.stringify({ expires: now + LEASE_MS, owner } satisfies Lease)

    store.setItem(key, next)

    return store.getItem(key) === next ? 'acquired' : 'busy'
  } catch {
    return 'unavailable'
  }
}

function releaseLease(key: string, owner: string): void {
  const store = localStorageOrNull()

  if (!store) {
    return
  }

  try {
    const current = readLease(store, key)

    if (current && current.owner !== owner) {
      return
    }

    store.removeItem(key)
  } catch {
    // Already gone, or storage is locked down. The record expires on its own.
  }
}

function randomOwner(): string {
  const bytes = new Uint8Array(16)
  const cryptoApi = globalThis.crypto

  if (cryptoApi?.getRandomValues) {
    cryptoApi.getRandomValues(bytes)
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256)
    }
  }

  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => {
    setTimeout(resolve, ms)
  })
}

function beginHold(key: string, owner: string): () => void {
  let stopped = false
  let timer: ReturnType<typeof setInterval> | undefined = undefined

  const release = () => {
    if (stopped) {
      return
    }

    stopped = true
    held.delete(release)

    if (timer !== undefined) {
      clearInterval(timer)
    }

    releaseLease(key, owner)
  }

  timer = setInterval(() => {
    if (stopped) {
      return
    }

    const store = localStorageOrNull()
    const current = store ? readLease(store, key) : null

    if (current && current.owner !== owner && current.expires > Date.now()) {
      stopped = true
      held.delete(release)

      if (timer !== undefined) {
        clearInterval(timer)
      }

      return
    }

    compareAndSwap(key, owner, Date.now())
  }, RENEW_MS)

  held.add(release)

  return release
}

function releaseAll(): void {
  for (const release of [...held]) {
    release()
  }
}

if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', event => {
    if (event.persisted) {
      return
    }

    releaseAll()
  })
}

function takeLease(name: string): Taken {
  const owner = randomOwner()
  const key = storageKey(name)
  const wrote = compareAndSwap(key, owner, Date.now())

  if (wrote !== 'acquired') {
    return { ok: false, reason: wrote === 'unavailable' ? 'unavailable' : 'busy' }
  }

  // setItem + getItem is the compare-and-swap. The loser does not start a
  // renewal timer, so it cannot overwrite the winner on the next tick.
  return { ok: true, release: beginHold(key, owner) }
}

/** Take `name` if no live tab holds it. Null means another live tab has it, or storage cannot record a lease. */
export async function tryPlainHttpTabLease(name: string): Promise<null | (() => void)> {
  const taken = takeLease(name)

  return taken.ok ? taken.release : null
}

/**
 * Wait until this tab holds `name`, then run `task`. Matches Web Locks
 * `request` (exclusive, queued). If storage cannot record a lease at all,
 * run anyway so a queue drain is not stuck with no way to claim it.
 */
export async function withPlainHttpTabLease<T>(name: string, task: () => Promise<T>): Promise<T> {
  for (;;) {
    const taken = takeLease(name)

    if (taken.ok) {
      try {
        return await task()
      } finally {
        taken.release()
      }
    }

    if (taken.reason === 'unavailable') {
      return task()
    }

    await delay(POLL_MS)
  }
}
