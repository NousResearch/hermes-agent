/**
 * Roster Section Manual Ordering and Drag Reorder.
 *
 * Provides persistent manual ordering scoped to individual roster sections.
 * Composes with user-made sections (#101290) by scoping order to each section/connection
 * context, preserving pinned-first semantics, deterministic unranked bot placement,
 * and isolated per-section resets.
 */

import { atom } from 'nanostores'

import { botRosterKey } from './data'
import { getPluginCtx } from './shared'
import type { RosterRow } from './types'
import { $draggingBot, BOT_DRAG_MIME, onBotMovedToSection } from './user-sections'

export { $draggingBot, BOT_DRAG_MIME }

export const BOT_ROSTER_ORDER_KEY = 'bot-roster-order-v1'

/** Map of sectionScopeKey -> array of bot roster keys in manual order. */
export type RosterOrderMap = Record<string, string[]>

export const $rosterOrder = atom<RosterOrderMap>({})

/** Section scope of the bot currently in flight. */
export const $draggingBotScope = atom<null | string>(null)

/** Pinned status of the bot currently in flight. */
export const $draggingBotPinned = atom<null | boolean>(null)

/**
 * Scope key for a section context, combining connection and section identity.
 * Unassigned uses 'unassigned'.
 */
export function rosterSectionScope(sectionId: null | string, connectionId?: string | null): string {
  const conn = String(connectionId || 'local').trim() || 'local'
  const sec = sectionId ? String(sectionId).trim() : 'unassigned'

  return `${conn}::${sec}`
}

export function normalizeRosterOrderMap(value: unknown): RosterOrderMap {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: RosterOrderMap = {}

  for (const [scope, list] of Object.entries(value)) {
    if (!Array.isArray(list)) {
      continue
    }

    const seen = new Set<string>()
    const cleanList: string[] = []

    for (const item of list) {
      const key = String(item || '').trim()

      if (!key || seen.has(key)) {
        continue
      }

      seen.add(key)
      cleanList.push(key)
    }

    if (cleanList.length > 0) {
      out[scope] = cleanList
    }
  }

  return out
}

export function persistRosterOrder(next: RosterOrderMap): void {
  const clean = normalizeRosterOrderMap(next)
  $rosterOrder.set(clean)

  try {
    getPluginCtx()?.storage?.set?.(BOT_ROSTER_ORDER_KEY, clean)
  } catch {
    // Storage unavailable - order persists for this window session
  }
}

export function loadRosterOrder(): void {
  try {
    const raw = getPluginCtx()?.storage?.get?.(BOT_ROSTER_ORDER_KEY, {})
    $rosterOrder.set(normalizeRosterOrderMap(raw))
  } catch {
    $rosterOrder.set({})
  }
}

/** Clear custom order for a specific section scope, or all scopes if none passed. */
export function clearRosterOrder(scopeKey?: string): void {
  if (!scopeKey) {
    persistRosterOrder({})

    return
  }

  const current = { ...$rosterOrder.get() }
  delete current[scopeKey]
  persistRosterOrder(current)
}

/** Check whether a section scope has a custom manual order. */
export function hasCustomRosterOrder(scopeKey: string, orderMap: RosterOrderMap = $rosterOrder.get()): boolean {
  return Boolean(orderMap[scopeKey]?.length)
}

/** Check whether any section scope has a custom manual order. */
export function hasAnyCustomRosterOrder(orderMap: RosterOrderMap = $rosterOrder.get()): boolean {
  return Object.values(orderMap).some(list => list.length > 0)
}

/** Remove a bot key from a specific section's order, or all sections if scope omitted. */
export function removeBotFromRosterOrder(botKey: string, scopeKey?: string): void {
  const current = $rosterOrder.get()
  const next: RosterOrderMap = {}
  let changed = false

  for (const [scope, list] of Object.entries(current)) {
    if (scopeKey && scope !== scopeKey) {
      next[scope] = list

      continue
    }

    const filtered = list.filter(k => k !== botKey)

    if (filtered.length !== list.length) {
      changed = true

      if (filtered.length > 0) {
        next[scope] = filtered
      }
    } else if (list.length > 0) {
      next[scope] = list
    }
  }

  if (changed) {
    persistRosterOrder(next)
  }
}

/** Prune deleted or stale keys no longer in the active roster. */
export function pruneRosterOrder(liveRosterKeys: Set<string>): void {
  const current = $rosterOrder.get()
  const next: RosterOrderMap = {}
  let changed = false

  for (const [scope, list] of Object.entries(current)) {
    const filtered = list.filter(k => liveRosterKeys.has(k))

    if (filtered.length !== list.length) {
      changed = true
    }

    if (filtered.length > 0) {
      next[scope] = filtered
    }
  }

  if (changed) {
    persistRosterOrder(next)
  }
}

export function rosterRowKey(row: { bot?: RosterRow } | RosterRow): string {
  const bot = ('bot' in row && row.bot ? row.bot : row) as RosterRow

  return botRosterKey(bot)
}

export interface OrderableRosterRow {
  activity: number
  bot?: RosterRow
  created?: number
  pinned: boolean
}

/**
 * Reorder rows within a section.
 * Pinned rows stay at the top.
 * When orderKeys is empty or undefined, falls back to default recency sort.
 * When orderKeys is present:
 *   - Ranked rows sort by their index in orderKeys.
 *   - Unranked rows (newly discovered bots) sort deterministically by (created, key)
 *     to prevent churn on chat activity, appended after ranked rows in each band.
 */
export function orderRosterRows<T extends OrderableRosterRow>(
  rows: T[],
  getRowKey: (row: T) => string,
  orderKeys?: string[]
): T[] {
  if (!orderKeys || !orderKeys.length) {
    return rows.slice().sort((a, b) => {
      const pa = a.pinned ? 1 : 0
      const pb = b.pinned ? 1 : 0

      if (pa !== pb) {
        return pb - pa
      }

      return b.activity - a.activity
    })
  }

  const orderIndexMap = new Map<string, number>()
  orderKeys.forEach((key, index) => {
    orderIndexMap.set(key, index)
  })

  const sortBucket = (bucket: T[]): T[] => {
    return bucket.slice().sort((a, b) => {
      const keyA = getRowKey(a)
      const keyB = getRowKey(b)
      const idxA = orderIndexMap.has(keyA) ? orderIndexMap.get(keyA)! : -1
      const idxB = orderIndexMap.has(keyB) ? orderIndexMap.get(keyB)! : -1

      if (idxA >= 0 && idxB >= 0) {
        return idxA - idxB
      }

      if (idxA >= 0) {
        return -1
      }

      if (idxB >= 0) {
        return 1
      }

      // Unranked rows: deterministic placement based on creation time then key.
      // Avoids activity churn.
      const ca = a.created || 0
      const cb = b.created || 0

      if (ca !== cb) {
        return cb - ca
      }

      return keyA.localeCompare(keyB)
    })
  }

  const pinnedRows = rows.filter(r => r.pinned)
  const unpinnedRows = rows.filter(r => !r.pinned)

  return [...sortBucket(pinnedRows), ...sortBucket(unpinnedRows)]
}

/**
 * Move fromKey before or after toKey within a section scope.
 * Preserves all existing keys (including hidden/filtered rows).
 */
export function moveRosterItem(
  currentOrder: string[],
  allSectionBotKeys: string[],
  fromKey: string,
  toKey: string,
  position: 'before' | 'after'
): string[] {
  if (fromKey === toKey) {
    return currentOrder
  }

  let baseOrder: string[]

  if (currentOrder.length > 0) {
    baseOrder = [...currentOrder]
    const existing = new Set(baseOrder)

    for (const key of allSectionBotKeys) {
      if (!existing.has(key)) {
        baseOrder.push(key)
        existing.add(key)
      }
    }
  } else {
    baseOrder = [...allSectionBotKeys]
  }

  const fromIndex = baseOrder.indexOf(fromKey)

  if (fromIndex >= 0) {
    baseOrder.splice(fromIndex, 1)
  }

  const targetIndex = baseOrder.indexOf(toKey)

  if (targetIndex < 0) {
    baseOrder.push(fromKey)
  } else {
    const insertIndex = position === 'before' ? targetIndex : targetIndex + 1
    baseOrder.splice(insertIndex, 0, fromKey)
  }

  return baseOrder
}

// Automatically purge bot from old section order whenever moved across sections
onBotMovedToSection(bot => {
  removeBotFromRosterOrder(botRosterKey(bot))
})
