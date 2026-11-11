import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import {
  $layoutTree,
  $narrowViewport,
  moveTreePane,
  moveTreePanes,
  setTreePaneHidden
} from '@/components/pane-shell/tree/store'
import { $portalShell } from '@/store/shell-variant'

const TOOLS = ['files', 'review', 'terminal']

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

/** Hide the panel, not a shell or a tab that would expose a sibling tool. */
export function hideDeveloperToolPanel(paneId: string): boolean {
  if (!$portalShell.get() || $narrowViewport.get()) {
    return false
  }

  const tree = $layoutTree.get()
  const panel = tree && findGroupOfPane(tree, paneId)

  if (!panel || panel.panes.includes('workspace') || !TOOLS.every(id => panel.panes.includes(id))) {
    return false
  }

  for (const id of panel.panes) {
    setTreePaneHidden(id, true)
  }

  return true
}
