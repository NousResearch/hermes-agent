/**
 * Rehydrate nanostore atoms after desktop state hydration seeds localStorage.
 *
 * This module exists to break circular dependencies: desktop-state.ts
 * can't import from store/* without creating circular imports at module
 * evaluation time, so it dynamically imports this bridge instead.
 */
import { reloadJSON, reloadStringArray } from './storage'

import { $paneStates } from '../store/panes'
import { $pinnedSessionIds, $sidebarAgentsGrouped, SIDEBAR_PINNED_STORAGE_KEY, SIDEBAR_AGENTS_GROUPED_STORAGE_KEY } from '../store/layout'
import { arraysEqual } from './storage'

const PANE_STATES_STORAGE_KEY = 'hermes.desktop.paneStates.v1'

/**
 * Re-read all desktop-state-backed stores from localStorage after hydration
 * has seeded it with durable IPC data. Atoms that only read localStorage at
 * module-init time get updated here.
 */
export function rehydrateStores(): void {
  rehydratePinnedSessions()
  rehydratePaneStates()
  rehydrateAgentsGrouped()
}

function rehydratePinnedSessions(): void {
  const ids = reloadStringArray(SIDEBAR_PINNED_STORAGE_KEY)
  if (ids === undefined) return
  const current = $pinnedSessionIds.get()
  if (!arraysEqual(ids, current)) {
    $pinnedSessionIds.set(ids)
  }
}

function rehydratePaneStates(): void {
  const states = reloadJSON<Record<string, { open: boolean }>>(PANE_STATES_STORAGE_KEY)
  if (!states) return
  const current = $paneStates.get()
  // Only update if something actually changed to avoid unnecessary re-renders.
  let changed = false

  for (const [id, state] of Object.entries(states)) {
    const existing = current[id]
    if (!existing || existing.open !== state.open) {
      changed = true
      break
    }
  }

  if (changed) {
    // Merge restored states into existing, preserving in-memory-only
    // widthOverrides that are not persisted.
    const merged: Record<string, { open: boolean; widthOverride?: number }> = {}

    for (const [id, state] of Object.entries(states)) {
      merged[id] = { open: state.open, widthOverride: current[id]?.widthOverride }
    }

    // Preserve any pane IDs in current that aren't in the restored set.
    for (const [id, state] of Object.entries(current)) {
      if (!(id in merged)) {
        merged[id] = state
      }
    }

    $paneStates.set(merged)
  }
}

function rehydrateAgentsGrouped(): void {
  try {
    const raw = window.localStorage.getItem(SIDEBAR_AGENTS_GROUPED_STORAGE_KEY)
    if (raw === null) return
    const value = raw === 'true'
    if ($sidebarAgentsGrouped.get() !== value) {
      $sidebarAgentsGrouped.set(value)
    }
  } catch {
    // Best-effort
  }
}
