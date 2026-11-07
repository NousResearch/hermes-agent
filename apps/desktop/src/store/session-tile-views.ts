/**
 * Per-view state of the chat area: each profile view (a concrete profile, or
 * "All profiles") keeps its own tab order, front tab per zone and main chat,
 * and gets them back when the user returns to it.
 *
 * The tab SET lives in the session-tile buckets (session-tile-storage.ts);
 * this module adds what a bucket does not carry: strip slots as the user left
 * them (a tab drag reorders the tree, not the tile list), which tab was in
 * front, and which chat the main tab held.
 */
import { backendScopeKey } from '@hermes/shared'

import { sessionRoute } from '@/app/routes'
import { findGroupOfPane, type GroupNode, type LayoutNode } from '@/components/pane-shell/tree/model'
import { $layoutTree, activateTreePane } from '@/components/pane-shell/tree/store'
import { readJson, writeJson } from '@/lib/storage'

import { $tileViewScope, requestFreshSession, setProfileViewRestore } from './profile'
import { requestRoute } from './recovery-requests'
import { $selectedStoredSessionId } from './session'
import { TILE_PANE_PREFIX } from './session-focus'
import { orderTilesByTree } from './session-tile-order'
import { isTileStoreWriter, VIEW_MAIN_KEY } from './session-tile-storage'

const WORKSPACE_PANE = 'workspace'

function groupsOf(node: LayoutNode | null): GroupNode[] {
  if (!node) {
    return []
  }

  return node.type === 'group' ? [node] : node.children.flatMap(groupsOf)
}

const isChatPane = (paneId: string) => paneId === WORKSPACE_PANE || paneId.startsWith(TILE_PANE_PREFIX)

/** The tiles in strip order, each stamped with the slot it holds now (`before`
 *  = the pane right after it in its zone). Replaying the list re-adopts every
 *  tab where it was, including a tab dragged in front of the main tab. */
export function snapshotTileStrip<T extends { before?: null | string; storedSessionId: string }>(
  tree: LayoutNode | null,
  tiles: readonly T[]
): T[] {
  const ordered = orderTilesByTree(tree, tiles) ?? [...tiles]

  if (!tree) {
    return ordered
  }

  return ordered.map(tile => {
    const pane = `${TILE_PANE_PREFIX}${tile.storedSessionId}`
    const group = findGroupOfPane(tree, pane)

    if (!group) {
      return tile
    }

    const before = group.panes[group.panes.indexOf(pane) + 1] ?? null

    return before === tile.before ? tile : { ...tile, before }
  })
}

/** Front pane of every zone that holds chat tabs. */
export function frontChatPanes(tree: LayoutNode | null): string[] {
  return groupsOf(tree)
    .filter(group => group.panes.some(isChatPane) && isChatPane(group.active))
    .map(group => group.active)
}

let pendingFront: readonly string[] = []
let pendingFrontToken = 0

function applyPendingFront(): void {
  for (const pane of pendingFront) {
    const tree = $layoutTree.get()
    const group = tree ? findGroupOfPane(tree, pane) : null

    if (group && group.active !== pane) {
      activateTreePane(group.id, pane)
    }
  }
}

/** Front each remembered pane again. A view swap runs inside a store
 *  listener, so the tabs it brings back are adopted after it returns: the
 *  panes stay pending through that tick and are fronted as they land (an
 *  adoption may front its own newcomer). Panes that never land are dropped. */
export function frontPanes(panes: readonly string[]): void {
  const token = ++pendingFrontToken
  pendingFront = panes
  applyPendingFront()
  queueMicrotask(() => {
    if (token === pendingFrontToken) {
      pendingFront = []
    }
  })
}

$layoutTree.listen(() => {
  if (pendingFront.length > 0) {
    applyPendingFront()
  }
})

// ── Main chat per view ─────────────────────────────────────────────────────

type ViewMainMemory = Record<string, null | string>

function viewMainMemory(): ViewMainMemory {
  const stored = readJson<unknown>(VIEW_MAIN_KEY)

  return stored && typeof stored === 'object' && !Array.isArray(stored) ? (stored as ViewMainMemory) : {}
}

let currentConnectionId: () => null | string = () => null

/** session-states owns the connection the tabs are scoped to. */
export function bindViewConnection(read: () => null | string): void {
  currentConnectionId = read
}

const viewKey = (view: string) => backendScopeKey(currentConnectionId(), view)

function rememberViewMain(view: string, storedSessionId: null | string): void {
  const memory = viewMainMemory()
  const key = viewKey(view)

  if (memory[key] !== storedSessionId) {
    writeJson(VIEW_MAIN_KEY, { ...memory, [key]: storedSessionId })
  }
}

/** `undefined` = this view never had a main chat recorded. */
export function viewMainSession(view: string): null | string | undefined {
  const memory = viewMainMemory()
  const key = viewKey(view)

  return key in memory ? memory[key] : undefined
}

function restoreViewMain(view: string): boolean {
  const storedSessionId = viewMainSession(view)

  if (storedSessionId === undefined) {
    return false
  }

  if (storedSessionId) {
    requestRoute(sessionRoute(storedSessionId))
  } else {
    requestFreshSession()
  }

  return true
}

if (isTileStoreWriter()) {
  // A view switch never changes the selection itself, so every change lands
  // on the view it happened in.
  $selectedStoredSessionId.listen(storedSessionId => rememberViewMain($tileViewScope.get(), storedSessionId))
  setProfileViewRestore(restoreViewMain)
}
