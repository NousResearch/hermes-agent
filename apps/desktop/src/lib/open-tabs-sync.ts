/**
 * Decide how a Desktop client adopts the backend's open tab strip.
 *
 * The strip used to live only in this device's localStorage. Switching
 * devices (MacBook remote ↔ Mini local, same backend) therefore dropped the
 * tabs even though the sessions were still on the backend. The backend file
 * is the shared list. This function is the guard that keeps a fresh device
 * from pushing its empty strip over the other device's tabs.
 */

export interface OpenTabPlacement {
  anchor?: string
  before?: null | string
  dir?: 'bottom' | 'center' | 'left' | 'right' | 'top'
  storedSessionId: string
}

export interface OpenTabsDocument {
  revision: number
  tiles: OpenTabPlacement[]
  updated_at?: null | string
}

export type OpenTabsSyncDecision =
  | { kind: 'adopt'; revision: number; tiles: OpenTabPlacement[] }
  | { kind: 'noop'; revision: number }
  | { kind: 'seed' }
  | { kind: 'stay' }
  | { kind: 'push-local' }

const DIRS = new Set(['bottom', 'center', 'left', 'right', 'top'])

export function canonicalizeOpenTabs(raw: unknown): OpenTabPlacement[] {
  if (!Array.isArray(raw)) {
    return []
  }

  const tiles: OpenTabPlacement[] = []
  const seen = new Set<string>()

  for (const item of raw) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const record = item as Record<string, unknown>
    const stored = typeof record.storedSessionId === 'string' ? record.storedSessionId.trim() : ''

    if (!stored || stored.length > 200 || seen.has(stored)) {
      continue
    }

    seen.add(stored)

    const tile: OpenTabPlacement = { storedSessionId: stored }

    if (typeof record.dir === 'string' && DIRS.has(record.dir)) {
      tile.dir = record.dir as OpenTabPlacement['dir']
    }

    if (typeof record.anchor === 'string' && record.anchor.trim()) {
      tile.anchor = record.anchor.trim().slice(0, 200)
    }

    if (record.before === null) {
      tile.before = null
    } else if (typeof record.before === 'string' && record.before.trim()) {
      tile.before = record.before.trim().slice(0, 200)
    }

    tiles.push(tile)

    if (tiles.length >= 40) {
      break
    }
  }

  return tiles
}

export function sameOpenTabs(left: OpenTabPlacement[], right: OpenTabPlacement[]): boolean {
  return JSON.stringify(canonicalizeOpenTabs(left)) === JSON.stringify(canonicalizeOpenTabs(right))
}

/**
 * `remote === null` means the GET failed (old backend, network). Stay local
 * and do not PUT — a failed read is not an empty strip.
 *
 * A missing local revision means this device has never synced this backend.
 * Empty remote + local tabs seeds the file. Non-empty remote wins over a
 * device-local leftover, which is the bug: the other device's tabs are the
 * ones the user actually had open.
 */
export function decideOpenTabSync(input: {
  localAppliedRevision: null | number
  localTiles: OpenTabPlacement[]
  remote: null | OpenTabsDocument
}): OpenTabsSyncDecision {
  const remote = input.remote

  if (!remote || !Number.isInteger(remote.revision) || remote.revision < 0) {
    return { kind: 'stay' }
  }

  const remoteTiles = canonicalizeOpenTabs(remote.tiles)
  const localTiles = canonicalizeOpenTabs(input.localTiles)
  const applied = input.localAppliedRevision

  if (applied == null) {
    if (remoteTiles.length === 0 && localTiles.length > 0) {
      return { kind: 'seed' }
    }

    if (remoteTiles.length === 0) {
      return { kind: 'noop', revision: remote.revision }
    }

    return { kind: 'adopt', revision: remote.revision, tiles: remoteTiles }
  }

  if (applied !== remote.revision) {
    return { kind: 'adopt', revision: remote.revision, tiles: remoteTiles }
  }

  if (!sameOpenTabs(localTiles, remoteTiles)) {
    return { kind: 'push-local' }
  }

  return { kind: 'noop', revision: remote.revision }
}
