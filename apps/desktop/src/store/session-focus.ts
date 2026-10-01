import { atom, computed } from 'nanostores'

import { findGroup, findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $workspaceMode } from '@/components/pane-shell/workspace-scope'

// A chat surface: the primary's workspace or a session tile. Everything else a
// zone can show — the sessions list, Files, Terminal, a preview tab — is chrome.
const isChatPane = (paneId?: string): boolean => paneId === 'workspace' || Boolean(paneId?.startsWith('session-tile:'))

// Chrome can own keyboard focus, but working in it (navigating the sessions
// list, browsing Files, typing in Terminal) must not replace the chat being
// worked in with the route's (possibly hidden) primary — the Files rail and
// statusbar follow this chat, so they would jump projects mid-click.
const $lastContentGroup = atom<null | string>(null)

$activeTreeGroup.subscribe(groupId => {
  const tree = $layoutTree.get()
  const active = groupId && tree ? findGroup(tree, groupId)?.active : undefined

  if (!groupId || isChatPane(active)) {
    $lastContentGroup.set(groupId)
  }
})

export const $focusedTreePaneId = computed(
  [$activeTreeGroup, $layoutTree, $workspaceMode, $lastContentGroup],
  (groupId, tree, workspaceMode, lastContentGroup) => {
    let active = groupId && tree ? findGroup(tree, groupId)?.active : undefined

    if (groupId && tree && !isChatPane(active)) {
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
