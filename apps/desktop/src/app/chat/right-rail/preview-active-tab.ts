/**
 * THE PREVIEW AN AGENT TOOL ACTS ON — one resolution shared by read_preview
 * (preview-reader.ts), drive_preview (the nav / input / script-runner
 * registries) and the preview tour, so a read and the action after it can
 * never land on different tabs.
 *
 * Callers pass the tabs the requesting session may see
 * (`previewTabsFor(owner)`): a background session's agent stays inside its own
 * tabs even while another session holds focus (#73890).
 */

import { findGroup } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $hoveredTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId } from '@/store/layout'
import { type PreviewTab, previewTabsFor } from '@/store/preview'
import { explicitOpenBlocksZone, PREVIEW_TILE_PREFIX } from '@/store/preview-explicit'

function tabIdFromPreviewPane(paneId: string | undefined): null | string {
  if (!paneId?.startsWith(`${PREVIEW_TILE_PREFIX}:`)) {
    return null
  }

  return paneId.slice(PREVIEW_TILE_PREFIX.length + 1)
}

/** Active preview tab in a layout zone, if that tab is still open. */
function openTabInGroup(groupId: null | string, tabs: readonly PreviewTab[]): null | PreviewTab {
  const tree = $layoutTree.get()

  if (!tree || !groupId) {
    return null
  }

  const tabId = tabIdFromPreviewPane(findGroup(tree, groupId)?.active)

  if (!tabId) {
    return null
  }

  return tabs.find(tab => tab.id === tabId) ?? null
}

/**
 * The preview the user is looking at: hovered zone, else focused zone, else
 * the store. A focused zone that is still the pre-open zone does not override
 * an explicit open living in a different group — that is follow()'s clobber,
 * not a look.
 */
export function resolveActivePreviewTab(tabs: readonly PreviewTab[] = previewTabsFor()): null | PreviewTab {
  if (tabs.length === 0) {
    return null
  }

  const hovered = openTabInGroup($hoveredTreeGroup.get(), tabs)

  if (hovered) {
    return hovered
  }

  const focusedId = $activeTreeGroup.get()
  const focused = openTabInGroup(focusedId, tabs)
  const openIds = tabs.map(tab => tab.id)

  if (focused && !explicitOpenBlocksZone(focusedId, openIds)) {
    return focused
  }

  return tabs.find(tab => tab.id === $rightRailActiveTabId.get()) ?? tabs[0] ?? null
}

/** The active preview among the tabs `owner` (a stored id; null = a session
 *  with none yet; omitted = the focused session) may see. */
export function activePreviewTabFor(owner?: null | string): null | PreviewTab {
  return resolveActivePreviewTab(previewTabsFor(owner))
}
