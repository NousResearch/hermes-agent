/**
 * Roster Section Manual Ordering and Drag Reorder.
 *
 * Provides persistent manual ordering scoped to individual roster sections.
 * Composes with user-made sections (#101290) and group-chat slots (#113414) by
 * scoping order to each (connection, section) context, preserving pinned-first
 * outer bands, deterministic unranked bot placement, and isolated resets.
 */

import { atom } from 'nanostores'

import { getPluginCtx } from './shared'
import type { GatewaySource, RosterRow } from './types'

function rowIdentityKey(bot: Partial<RosterRow> | null | undefined): string {
  return `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`
}

export const BOT_ROSTER_ORDER_KEY = 'bot-roster-order-v1'

/** Map of sectionScopeKey -> array of bot roster keys in manual order. */
export type RosterOrderMap = Record<string, string[]>

export const $rosterOrder = atom<RosterOrderMap>({})

/**
 * Scope key for a section context, combining connection and section identity.
 * Unassigned uses 'unassigned'.
 */
export function rosterSectionScope(sectionId: null | string, connectionId?: string | null): string {
  const conn = String(connectionId || 'local').trim() || 'local'
  const sec = sectionId ? String(sectionId).trim() || 'unassigned' : 'unassigned'

  return `${conn}::${sec}`
}

function normalizeScopeList(list: unknown): string[] {
  if (!Array.isArray(list)) {
    return []
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

  return cleanList
}

export function normalizeRosterOrderMap(value: unknown): RosterOrderMap {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: RosterOrderMap = {}

  for (const [scope, list] of Object.entries(value)) {
    const cleanScope = scope.trim()

    if (!cleanScope) {
      continue
    }

    const cleanList = normalizeScopeList(list)

    if (cleanList.length > 0) {
      out[cleanScope] = cleanList
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

function scopeMatchesSectionId(scope: string, sectionId: string): boolean {
  const sep = scope.indexOf('::')

  return sep >= 0 && scope.slice(sep + 2) === sectionId
}

/** Snapshot custom orders across all connections for a section id before deleting it. */
export function snapshotSectionRosterOrder(
  sectionId: string,
  orderMap: RosterOrderMap = $rosterOrder.get()
): RosterOrderMap {
  const cleanId = sectionId.trim()

  if (!cleanId) {
    return {}
  }

  const snapshot: RosterOrderMap = {}

  for (const [scope, list] of Object.entries(orderMap)) {
    if (scopeMatchesSectionId(scope, cleanId) && list.length > 0) {
      snapshot[scope] = [...list]
    }
  }

  return snapshot
}

/** Clear custom orders across every connection for a deleted section id. */
export function clearSectionRosterOrder(sectionId: string): void {
  const cleanId = sectionId.trim()

  if (!cleanId) {
    return
  }

  const current = $rosterOrder.get()
  const next: RosterOrderMap = {}
  let changed = false

  for (const [scope, list] of Object.entries(current)) {
    if (scopeMatchesSectionId(scope, cleanId)) {
      changed = true
    } else if (list.length > 0) {
      next[scope] = list
    }
  }

  if (changed) {
    persistRosterOrder(next)
  }
}

/** Restore snapshotted section orders when a section delete is undone. */
export function restoreSectionRosterOrder(snapshot: RosterOrderMap): void {
  const entries = Object.entries(snapshot)

  if (!entries.length) {
    return
  }

  const next: RosterOrderMap = { ...$rosterOrder.get() }

  for (const [scope, list] of entries) {
    if (list.length > 0) {
      next[scope] = [...list]
    }
  }

  persistRosterOrder(next)
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

/** Remove a bot from its previous section's custom order without touching its destination section. */
export function removeBotFromSectionOrder(bot: Partial<RosterRow>, sectionId: null | string): void {
  const key = rowIdentityKey(bot)
  const sec = sectionId ? String(sectionId).trim() || 'unassigned' : 'unassigned'

  for (const scope of Object.keys($rosterOrder.get())) {
    if (scopeMatchesSectionId(scope, sec)) {
      removeBotFromRosterOrder(key, scope)
    }
  }
}

function deferredSourceConnections(sources?: GatewaySource[]): Set<string> {
  if (!Array.isArray(sources) || !sources.length) {
    return new Set()
  }

  const deferred = new Set<string>()

  for (const source of sources) {
    const id = String(source?.connectionId || '').trim()

    if (id && (source.reachable === false || Boolean(String(source.error || '').trim()))) {
      deferred.add(id)
    }
  }

  return deferred
}

function keyOnDeferredConnection(
  botKey: string,
  deferredConnections: Set<string>,
  hasAuthoritativeSources: boolean
): boolean {
  const sep = botKey.indexOf('::')
  const conn = sep >= 0 ? botKey.slice(0, sep) : ''

  if (!conn) {
    return false
  }

  // Unknown source inventory is not proof that a remote gateway left the registry.
  if (!hasAuthoritativeSources) {
    return conn !== 'local' && conn !== 'legacy'
  }

  return deferredConnections.has(conn)
}

/**
 * Prune deleted keys no longer in the active roster while preserving entries
 * belonging to offline, errored, or connect-on-demand (SSH) connections.
 */
export function pruneRosterOrder(liveRosterKeys: Set<string>, sources?: GatewaySource[]): void {
  const current = $rosterOrder.get()
  const entries = Object.entries(current)

  if (!entries.length) {
    return
  }

  const hasAuthoritativeSources = Array.isArray(sources) && sources.length > 0
  const deferredConnections = deferredSourceConnections(sources)
  const next: RosterOrderMap = {}
  let changed = false

  for (const [scope, list] of entries) {
    const filtered = list.filter(
      k => liveRosterKeys.has(k) || keyOnDeferredConnection(k, deferredConnections, hasAuthoritativeSources)
    )

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

export interface OrderableRosterRow {
  activity: number
  created?: number
  pinned: boolean
}

function compareRankedRows<T extends OrderableRosterRow>(
  a: T,
  b: T,
  getRowKey: (row: T) => string,
  orderIndexMap: Map<string, number>
): number {
  const keyA = getRowKey(a)
  const keyB = getRowKey(b)
  const idxA = orderIndexMap.get(keyA) ?? -1
  const idxB = orderIndexMap.get(keyB) ?? -1

  if (idxA >= 0 && idxB >= 0) {
    return idxA - idxB
  }

  if (idxA >= 0 || idxB >= 0) {
    return idxA >= 0 ? -1 : 1
  }

  // Unranked rows: deterministic placement by creation time then key to avoid activity churn.
  const createdDiff = (b.created || 0) - (a.created || 0)

  return createdDiff !== 0 ? createdDiff : keyA.localeCompare(keyB)
}

/**
 * Reorder bot rows within a section.
 * Pinned rows always remain in the top band.
 * When orderKeys is empty or undefined, falls back to default recency sort.
 */
export function orderRosterRows<T extends OrderableRosterRow>(
  rows: T[],
  getRowKey: (row: T) => string,
  orderKeys?: string[]
): T[] {
  if (!orderKeys || !orderKeys.length) {
    return rows.slice().sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.activity - a.activity)
  }

  const orderIndexMap = new Map<string, number>()
  orderKeys.forEach((key, index) => {
    orderIndexMap.set(key, index)
  })

  const pinnedRows = rows.filter(r => r.pinned).sort((a, b) => compareRankedRows(a, b, getRowKey, orderIndexMap))
  const unpinnedRows = rows.filter(r => !r.pinned).sort((a, b) => compareRankedRows(a, b, getRowKey, orderIndexMap))

  return [...pinnedRows, ...unpinnedRows]
}

/**
 * Apply custom bot ordering onto a mixed list of bot and group-chat rows while
 * preserving pinned bands and group-chat slot positions (mirroring sortGroupRosterRows).
 * When orderKeys is empty, returns rows untouched.
 */
export function orderSectionMixedRows<
  TRow extends { bot: RosterRow; kind?: 'bot' } | { kind: 'group'; pinned?: boolean }
>(
  rows: TRow[],
  toOrderable: (row: Extract<TRow, { bot: RosterRow }>) => Extract<TRow, { bot: RosterRow }> & OrderableRosterRow,
  orderKeys?: string[]
): TRow[] {
  if (!orderKeys || !orderKeys.length) {
    return rows
  }

  type BotEntry = Extract<TRow, { bot: RosterRow }>

  const annotated = rows.map(row => {
    if (row.kind === 'group') {
      return { isGroup: true as const, pinned: Boolean(row.pinned), row }
    }

    const orderable = toOrderable(row as BotEntry)

    return { isGroup: false as const, orderable, pinned: orderable.pinned, row }
  })

  const orderedBots = orderRosterRows(
    annotated.filter(entry => !entry.isGroup).map(entry => entry.orderable),
    row => rowIdentityKey(row.bot),
    orderKeys
  )

  const banded = [...annotated.filter(entry => entry.pinned), ...annotated.filter(entry => !entry.pinned)]
  let index = 0

  return banded.map(entry => (entry.isGroup ? entry.row : orderedBots[index++]!))
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

  const baseOrder = currentOrder.length > 0 ? [...currentOrder] : [...allSectionBotKeys]

  if (currentOrder.length > 0) {
    const existing = new Set(baseOrder)

    for (const key of allSectionBotKeys) {
      if (!existing.has(key)) {
        baseOrder.push(key)
        existing.add(key)
      }
    }
  }

  const fromIndex = baseOrder.indexOf(fromKey)

  if (fromIndex >= 0) {
    baseOrder.splice(fromIndex, 1)
  }

  const targetIndex = baseOrder.indexOf(toKey)

  if (targetIndex < 0) {
    baseOrder.push(fromKey)
  } else {
    baseOrder.splice(position === 'before' ? targetIndex : targetIndex + 1, 0, fromKey)
  }

  return baseOrder
}
