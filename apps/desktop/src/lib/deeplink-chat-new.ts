/**
 * Pure helpers for hermes://chat/new deep links.
 * Kept free of React so unit tests can cover cwd sanity + sticky slot keys.
 */

export const DEEPLINK_STICKY_PREFIX = 'hermes.desktop.deeplink.sticky.'
export const DEEPLINK_STICKY_PENDING = 'hermes.desktop.deeplink.sticky.pending'

export function deeplinkStickyStorageKey(slot: string): string {
  return `${DEEPLINK_STICKY_PREFIX}${slot.trim().toLowerCase()}`
}

/**
 * Accept absolute workspace paths only. Reject relative and `..` traversal.
 * Unix: `/…` · Windows: `C:\…` / `C:/…` · UNC: `\\server\share\…`
 */
export function cwdLooksSane(cwd: string): boolean {
  const path = cwd.trim()
  if (!path) return false
  if (path.includes('\0')) return false
  // Normalize for traversal checks (also catch encoded weirdness callers might pass)
  const normalized = path.replace(/\\/g, '/')
  if (normalized.includes('/../') || normalized.endsWith('/..') || normalized === '..' || normalized.startsWith('../')) {
    return false
  }
  if (normalized.startsWith('/')) return true
  // Windows drive
  if (/^[A-Za-z]:[\\/]/.test(path)) return true
  // UNC
  if (path.startsWith('\\\\') || normalized.startsWith('//')) return true
  return false
}

export function normalizeStickySlot(raw: string | undefined | null): string {
  return String(raw || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
}

export function readStickySessionId(slot: string, storage: Storage = localStorage): null | string {
  const key = normalizeStickySlot(slot)
  if (!key) return null
  try {
    const id = storage.getItem(deeplinkStickyStorageKey(key))?.trim()
    return id || null
  } catch {
    return null
  }
}

export function writeStickySessionId(slot: string, sessionId: string, storage: Storage = localStorage): void {
  const key = normalizeStickySlot(slot)
  const id = sessionId.trim()
  if (!key || !id) return
  try {
    storage.setItem(deeplinkStickyStorageKey(key), id)
  } catch {
    /* quota / private mode */
  }
}

export function clearStickySessionId(slot: string, storage: Storage = localStorage): void {
  const key = normalizeStickySlot(slot)
  if (!key) return
  try {
    storage.removeItem(deeplinkStickyStorageKey(key))
  } catch {
    /* ignore */
  }
}

export function setStickyPending(slot: string, sessionStore: Storage = sessionStorage): void {
  const key = normalizeStickySlot(slot)
  if (!key) return
  try {
    sessionStore.setItem(DEEPLINK_STICKY_PENDING, key)
  } catch {
    /* ignore */
  }
}

export function takeStickyPending(sessionStore: Storage = sessionStorage): null | string {
  try {
    const slot = normalizeStickySlot(sessionStore.getItem(DEEPLINK_STICKY_PENDING))
    if (slot) sessionStore.removeItem(DEEPLINK_STICKY_PENDING)
    return slot || null
  } catch {
    return null
  }
}
