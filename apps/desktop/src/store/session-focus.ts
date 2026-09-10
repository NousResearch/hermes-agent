import { atom, computed } from 'nanostores'

import { findGroup, findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $workspaceMode } from '@/components/pane-shell/workspace-scope'

// The sidebar still owns keyboard focus, but navigating its chrome must not
// replace the chat being worked in with the route's (possibly hidden) primary.
const $lastContentGroup = atom<null | string>(null)

$activeTreeGroup.subscribe(groupId => {
  const tree = $layoutTree.get()
  const active = groupId && tree ? findGroup(tree, groupId)?.active : undefined

  if (active !== 'sessions') {
    $lastContentGroup.set(groupId)
  }
})

export const $focusedTreePaneId = computed(
  [$activeTreeGroup, $layoutTree, $workspaceMode, $lastContentGroup],
  (groupId, tree, workspaceMode, lastContentGroup) => {
    let active = groupId && tree ? findGroup(tree, groupId)?.active : undefined

    if (active === 'sessions' && tree) {
      const content = lastContentGroup ? findGroup(tree, lastContentGroup) : null
      active = (content ?? findGroupOfPane(tree, 'workspace'))?.active
    }

    if (active?.startsWith('session-tile:')) {
      return active
    }

    // Bot chats are tiles, never the primary selection. Sidebar roster focus
    // must not publish a null session and let the Bots home reclaim the chat.
    if (workspaceMode === 'bots' && tree) {
      const mainActive = findGroupOfPane(tree, 'workspace')?.active

      if (mainActive?.startsWith('session-tile:')) {
        return mainActive
      }
    }

    return active
  }
)
