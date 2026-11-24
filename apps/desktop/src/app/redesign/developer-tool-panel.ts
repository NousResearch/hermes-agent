import { computed } from 'nanostores'

import { allPaneIds, findGroupOfPane, type GroupNode, type LayoutNode } from '@/components/pane-shell/tree/model'
import type { PaneSizing } from '@/components/pane-shell/tree/renderer/track-model'
import {
  $collapsedTreeSides,
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  $panesWithCloser,
  activateTreePane,
  closeTreePane,
  isCollapsePane,
  isPaneVisible,
  moveTreePane,
  moveTreePanes,
  restoreTreePane,
  setTreePaneHidden
} from '@/components/pane-shell/tree/store'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { CHAT_ITEMS_PANE_ID } from '@/store/chat-items'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'
import { PREVIEW_TILE_PREFIX } from '@/store/preview-explicit'
import { $portalShell } from '@/store/shell-variant'

const TOOLS = ['files', 'review', 'terminal']

const WIDTH_PROPERTY = '--developer-tool-panel-width'

/** The redesign's tool panel has ONE width, whatever tool it shows: a user's resize (px), or null for
 *  the token default (tokens/space.css `--portal-tool-panel-width`). Global: it is about this
 *  machine's screen, not a profile or a session. */
export const $developerToolPanelWidth = persistentAtom<null | number>(
  'hermes.desktop.developerToolPanelWidth',
  null,
  Codecs.json<null | number>(value => (typeof value === 'number' && value > 0 ? value : null))
)

/** The width every tool pane declares in the redesign. */
export const DEVELOPER_TOOL_PANEL_WIDTH = `var(${WIDTH_PROPERTY}, var(--portal-tool-panel-width))`

// Panes whose visibility an app store owns (bindPaneVisibility / bindToolPaneCollapse): bound, they
// hide through their closer, so the store a later "Show" writes is not left claiming they are open.
// A preview tile's closer closes its tab, so it only hides.
const STORE_BOUND = [...TOOLS, CHAT_ITEMS_PANE_ID]

// The chat's items ride the same panel and width as a tool, like a preview tile.
export const isTool = (paneId: string) => STORE_BOUND.includes(paneId) || paneId.startsWith(`${PREVIEW_TILE_PREFIX}:`)

/** A tool pane's sizing in the redesign: the shared width, never under the tool's own minimum (the
 *  terminal and the browser need more room than the file tree). Classic keeps each pane's own. */
export function developerToolSizing(minWidth: string): PaneSizing {
  return { width: DEVELOPER_TOOL_PANEL_WIDTH, minWidth }
}

/** Mirror the shared width onto the document, and fold a sash resize of whichever tool is showing
 *  (the pane shell writes it to that pane alone) into the one width every tool reads. */
export function installDeveloperToolPanelWidth(root: HTMLElement = document.documentElement): () => void {
  const stopWidth = $developerToolPanelWidth.subscribe(width => {
    if (width === null) {
      root.style.removeProperty(WIDTH_PROPERTY)
    } else {
      root.style.setProperty(WIDTH_PROPERTY, `${width}px`)
    }
  })

  const absorb = () => {
    if (!$portalShell.get()) {
      return
    }

    for (const [paneId, state] of Object.entries($paneStates.get())) {
      if (isTool(paneId) && state.widthOverride !== undefined) {
        $developerToolPanelWidth.set(Math.round(state.widthOverride))
        setPaneWidthOverride(paneId, undefined)
      }
    }
  }

  const stopStates = $paneStates.subscribe(absorb)
  const stopShell = $portalShell.listen(absorb)

  return () => {
    stopWidth()
    stopStates()
    stopShell()
  }
}

const builtInTools = (zone: GroupNode) => zone.panes.filter(id => STORE_BOUND.includes(id)).length

// The tools gathered: a zone beside the chat that holds more than one of them, a built-in among
// them. One tool alone in a zone is where a classic layout (or the user) put it, and is gathered on
// its next reveal. Browser tabs alone never make the panel: each tab is a pane of its own, and a
// zone the user tiled with several is theirs.
const isToolPanel = (zone: GroupNode) =>
  !zone.panes.includes('workspace') && builtInTools(zone) > 0 && zone.panes.filter(isTool).length > 1

/** The redesign's one tool panel: the zone holding the most built-in tools, the first in the tree's
 *  order on a tie. Not "the zone Files is in with every built-in tool": a tool can leave the tree
 *  (its tab's Close), and the tools that remain are still the panel the next one joins. */
function toolPanel(tree: LayoutNode): GroupNode | null {
  let panel: GroupNode | null = null

  for (const id of allPaneIds(tree).filter(isTool)) {
    const zone = findGroupOfPane(tree, id)

    if (zone && isToolPanel(zone) && builtInTools(zone) > (panel ? builtInTools(panel) : 0)) {
      panel = zone
    }
  }

  return panel
}

/** True while `paneId` stands in the redesign's tool panel. The panel is the window's side even when
 *  a Browser tab in it makes the pane tree read its zone as a main surface (paneRootSide), so a
 *  side-bound closer hides the pane there instead of dismissing it. */
export function inDeveloperToolPanel(paneId: string): boolean {
  const tree = $portalShell.get() ? $layoutTree.get() : null
  const zone = tree ? findGroupOfPane(tree, paneId) : null

  return zone !== null && isToolPanel(zone)
}

/** A deliberate tool reveal uses one side panel, retaining every pane and shell. */
export function prepareDeveloperToolPanel(paneId: string): void {
  if (!$portalShell.get() || $narrowViewport.get()) {
    return
  }

  const tree = $layoutTree.get()

  if (!tree) {
    return
  }

  const main = findGroupOfPane(tree, 'workspace')
  const panel = toolPanel(tree)

  if (!main) {
    return
  }

  if (panel) {
    if (!panel.panes.includes(paneId) && findGroupOfPane(tree, paneId)) {
      moveTreePane(paneId, { groupId: panel.id, pos: 'center' })
    }

    return
  }

  const panes = [...new Set([...TOOLS, paneId])].filter(id => findGroupOfPane(tree, id))
  moveTreePanes(panes, { groupId: main.id, pos: 'right' }, paneId)
}

/** Hide the panel's tools, not a shell or a tab that would expose a sibling tool. Only the built-in
 *  tools hide: a plugin pane someone stacked into the panel has no tool tile to bring it back, so
 *  it stays, fronted, and keeps the panel on screen until it leaves. */
export function hideDeveloperToolPanel(paneId: string): boolean {
  if (!$portalShell.get() || $narrowViewport.get()) {
    return false
  }

  const tree = $layoutTree.get()
  const panel = tree && findGroupOfPane(tree, paneId)

  if (!panel || !isToolPanel(panel)) {
    return false
  }

  const staying = panel.panes.find(id => !isTool(id) && !$hiddenTreePanes.get().has(id))
  // A tool panel's closer folds its whole zone while its tab is in front; front a sibling first so the
  // panel only hides, and a later Show unhides into a zone that is not folded.
  const front = staying ?? panel.panes.find(id => isTool(id) && !isCollapsePane(id))

  if (front && isTool(panel.active ?? '')) {
    activateTreePane(panel.id, front)
  }

  panel.panes.filter(isTool).forEach(hideTool)

  return true
}

// Bound, a tool hides through its closer, so the store a later "Show" writes is not left claiming it
// is open. A Browser tab's closer closes the tab, so it only hides.
function hideTool(id: string): void {
  if (STORE_BOUND.includes(id) && $panesWithCloser.get().has(id)) {
    closeTreePane(id)
  } else {
    setTreePaneHidden(id, true)
  }
}

// The side's tools on screen: the built-in ones wherever they stand, and the Browser tabs in the
// panel. Browser tabs the user tiled into a zone of their own are theirs, not the side.
const shownTools = (): string[] => {
  const tree = $layoutTree.get()

  if (!tree) {
    return []
  }

  const panel = toolPanel(tree)

  return allPaneIds(tree).filter(id => (STORE_BOUND.includes(id) || panel?.panes.includes(id)) && isPaneVisible(id))
}

/** Whether a tool is on screen beside the chat: what the band's right sidebar button says and flips
 *  in the redesign. Read from the pane tree, the same answer each tool's tile gives for itself.
 *  Classic never asks, so there the tree is not walked. */
export const $developerToolShown = computed(
  [$portalShell, $layoutTree, $dismissedPanes, $hiddenTreePanes, $collapsedTreeSides],
  portal => portal && shownTools().length > 0
)

/** The band's right sidebar button and its keybind, on a wide redesign window: hide the tool panel,
 *  or show Files in it. Classic's button is positional (it folds the outermost column right of the
 *  chat), which here is whichever tool zone happens to stand last, not the panel. False where that
 *  positional toggle still applies (classic, and the medium layout's sheets). */
export function toggleDeveloperToolPanel(): boolean {
  if (!$portalShell.get() || $narrowViewport.get()) {
    return false
  }

  const shown = shownTools()

  if (shown.length === 0) {
    restoreTreePane('files')
    prepareDeveloperToolPanel('files')
  }

  // Every zone showing a tool, so one press leaves the side empty: the panel as a whole, and a tool
  // standing alone in a zone of its own (not yet gathered) by itself.
  for (const tool of shown) {
    if (isPaneVisible(tool) && !hideDeveloperToolPanel(tool)) {
      hideTool(tool)
    }
  }

  return true
}
