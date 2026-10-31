/**
 * Escape in a side pane (files, review, a preview or Browser tile, logs) folds
 * that pane and hands focus back to the composer, instead of falling through
 * to the chat's global Esc-to-cancel and aborting the running turn.
 *
 * Two presses, two jobs: the first Escape closes the pane you are looking at,
 * the second (now in the composer) stops the reply. Pattern from Claude
 * Cowork's desktop changelog v4.4212.0 (2026-10-29): "Escape closes a side pane
 * … when focus is in the pane, and returns focus to what opened it. While
 * Claude is replying, Escape in a pane closes the pane, and a second Escape
 * stops the reply."
 *
 * The fold is the zone minimize the ⌘J toggle and the zone chevron use, so it
 * is non-destructive: tabs (and any unsaved preview draft) stay on the rail.
 */

import { requestComposerFocus } from '@/app/chat/composer/focus'
import { findGroup } from '@/components/pane-shell/tree/model'
import { $layoutTree, setTreeGroupMinimized } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'

import { isEditableTarget } from './combo'
import { composerFocusBlockedBySurface } from './composer-focus-keys'

// The chat itself (workspace, session/route tiles) is never a side pane.
const isMainSurface = (paneId: string) =>
  paneId === 'workspace' || paneId.startsWith('session-tile:') || paneId.startsWith('route-tile:')

// The navigation column (sessions, bots) is chrome you move through, not a
// pane something opened beside the chat; Esc there keeps its old meaning.
const isNavigationPane = (paneId: string) =>
  (registry.getArea('panes').find(pane => pane.id === paneId)?.data as { placement?: string } | undefined)
    ?.placement === 'left'

/** The side zone that should fold for this Escape, or null to leave it alone. */
export function sidePaneZoneForEscape(event: KeyboardEvent): null | string {
  if (
    event.key !== 'Escape' ||
    event.defaultPrevented ||
    event.isComposing ||
    event.metaKey ||
    event.ctrlKey ||
    event.altKey ||
    event.shiftKey ||
    isEditableTarget(event.target) ||
    // Dialogs, menus, the terminal and full-page routes own their Escape.
    composerFocusBlockedBySurface()
  ) {
    return null
  }

  const zoneId =
    event.target instanceof Element
      ? (event.target.closest<HTMLElement>('[data-tree-group]')?.dataset.treeGroup ?? null)
      : null

  const tree = $layoutTree.get()
  const zone = zoneId && tree ? findGroup(tree, zoneId) : null

  if (!zone || zone.minimized || zone.panes.some(id => isMainSurface(id) || isNavigationPane(id))) {
    return null
  }

  return zone.id
}

export function handleSidePaneEscape(event: KeyboardEvent): boolean {
  const zoneId = sidePaneZoneForEscape(event)

  if (!zoneId) {
    return false
  }

  // Claim the key so the chat's bubble-phase Esc-to-cancel stands down.
  event.preventDefault()
  setTreeGroupMinimized(zoneId, true)
  requestComposerFocus('active')

  return true
}
