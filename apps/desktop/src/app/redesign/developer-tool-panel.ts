import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import type { PaneSizing } from '@/components/pane-shell/tree/renderer/track-model'
import {
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  activateTreePane,
  moveTreePane,
  moveTreePanes,
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

// The chat's items ride the same panel and width as a tool, like a preview tile.
const isTool = (paneId: string) =>
  TOOLS.includes(paneId) || paneId === CHAT_ITEMS_PANE_ID || paneId.startsWith(`${PREVIEW_TILE_PREFIX}:`)

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
  const panel = findGroupOfPane(tree, 'files')

  if (!main) {
    return
  }

  const together = panel && TOOLS.every(id => panel.panes.includes(id)) && !panel.panes.includes('workspace')

  if (together) {
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

  if (!panel || panel.panes.includes('workspace') || !TOOLS.every(id => panel.panes.includes(id))) {
    return false
  }

  for (const id of panel.panes.filter(isTool)) {
    setTreePaneHidden(id, true)
  }

  const staying = panel.panes.find(id => !isTool(id) && !$hiddenTreePanes.get().has(id))

  if (staying && isTool(panel.active ?? '')) {
    activateTreePane(panel.id, staying)
  }

  return true
}
