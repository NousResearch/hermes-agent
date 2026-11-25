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
  revealTreePane,
  setTreePaneHidden
} from '@/components/pane-shell/tree/store'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { CHAT_ITEMS_PANE_ID } from '@/store/chat-items'
import { $interfaceMode } from '@/store/interface-mode'
import { $rightRailActiveTabId } from '@/store/layout'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'
import { $visiblePreviewTabs, browserTabFor, openBrowserTab } from '@/store/preview'
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

// The tools the app ships, each with an app store owning its visibility (bindPaneVisibility /
// bindToolPaneCollapse). Browser tabs are tools too (isTool), but each is a pane the user opens.
const BUILT_IN_TOOLS = [...TOOLS, CHAT_ITEMS_PANE_ID]

// The chat's items ride the same panel and width as a tool, like a preview tile.
export const isTool = (paneId: string) => BUILT_IN_TOOLS.includes(paneId) || paneId.startsWith(`${PREVIEW_TILE_PREFIX}:`)

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

const builtInTools = (zone: GroupNode) => zone.panes.filter(id => BUILT_IN_TOOLS.includes(id)).length

// The tools gathered: a zone beside the chat that holds more than one of them, a built-in among
// them. One tool alone in a zone is where a classic layout (or the user) put it, and is gathered on
// its next reveal. A zone the user tiled with Browser tabs alone is theirs.
const isToolPanel = (zone: GroupNode) =>
  !zone.panes.includes('workspace') && builtInTools(zone) > 0 && zone.panes.filter(isTool).length > 1

const zones = (node: LayoutNode): GroupNode[] => (node.type === 'group' ? [node] : node.children.flatMap(zones))

/** The redesign's one tool panel: the zone holding the most built-in tools, the first in the tree's
 *  order on a tie. A tool can leave the tree (its tab's Close); the tools that remain are still the
 *  panel the next one joins. */
function toolPanel(tree: LayoutNode): GroupNode | null {
  let panel: GroupNode | null = null

  for (const zone of zones(tree)) {
    if (isToolPanel(zone) && builtInTools(zone) > (panel ? builtInTools(panel) : 0)) {
      panel = zone
    }
  }

  return panel
}

// Where the tools are one docked panel: the redesign on a wide window. Below wide they are sheets.
const panelDocked = () => $portalShell.get() && !$narrowViewport.get()

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
  if (!panelDocked()) {
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
  if (!panelDocked()) {
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
  if (BUILT_IN_TOOLS.includes(id) && $panesWithCloser.get().has(id)) {
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

  return allPaneIds(tree).filter(
    id => (BUILT_IN_TOOLS.includes(id) || (isTool(id) && panel?.panes.includes(id))) && isPaneVisible(id)
  )
}

/** Whether a tool is on screen beside the chat: what the band's right sidebar button says and flips
 *  in the redesign, read from the pane tree as each tool's tile reads itself. False in classic,
 *  without a walk. */
export const $developerToolShown = computed(
  [$portalShell, $layoutTree, $dismissedPanes, $hiddenTreePanes, $collapsedTreeSides],
  portal => portal && shownTools().length > 0
)

/** The Browser the panel is holding right now, if the user has one open there. A Browser tab IS a
 *  tool of this panel (`isTool`), and in Simple mode the panel is its only door — the developer rail
 *  is Advanced chrome — so the toggle must be able to front it rather than always landing on Files,
 *  which Simple shadows closed (store/interface-mode.ts). Null when no Browser tab is open, or when
 *  the user tiled one into a zone of its own (that zone is theirs, not the side). */
function openBrowserPane(): null | string {
  const tab = browserTabFor($visiblePreviewTabs.get(), $rightRailActiveTabId.get())
  const paneId = tab ? `${PREVIEW_TILE_PREFIX}:${tab.id}` : null

  return paneId && (inDeveloperToolPanel(paneId) || $dismissedPanes.get().has(paneId)) ? paneId : null
}

/** The band's right sidebar button and its keybind, on a wide redesign window: hide the tool panel,
 *  or show Files in it. Classic's button is positional (it folds the outermost column right of the
 *  chat), which here is whichever tool zone happens to stand last, not the panel. False where that
 *  positional toggle still applies (classic, and the medium layout's sheets). */
export function toggleDeveloperToolPanel(): boolean {
  if (!panelDocked()) {
    return false
  }

  const shown = shownTools()

  if (shown.length === 0) {
    // Front what the panel already holds — a Browser the user opened there (including one dismissed
    // via ⌘⇧L) — so Simple mode is not the one mode whose Browser has no way back. In Simple mode
    // with no Browser yet, open a fresh Browser rather than surfacing Files (developer chrome).
    const browser = openBrowserPane()

    if (browser) {
      revealTreePane(browser)
      prepareDeveloperToolPanel(browser)

      return true
    }

    if ($interfaceMode.get() === 'simple') {
      openBrowserTab()
      const opened = openBrowserPane()

      if (opened) {
        prepareDeveloperToolPanel(opened)
      }

      return true
    }

    restoreTreePane('files')
    prepareDeveloperToolPanel('files')

    return true
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
