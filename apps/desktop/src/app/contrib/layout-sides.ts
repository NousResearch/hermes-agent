import { computed } from 'nanostores'

import { allPaneIds } from '@/components/pane-shell/tree/model'
import { $layoutTree, bindTreeSideVisibility, mirrorLayoutTree } from '@/components/pane-shell/tree/store'
import { modeLayout } from '@/store/interface-mode'
import { $fileBrowserOpen, $panesFlipped, $sidebarOpen, setFileBrowserOpen, setSidebarOpen } from '@/store/layout'

/** Side toggles belong to panes, not physical edges. Derive the flip from the
 * tree so dragging sessions or choosing a mirrored preset remaps the buttons. */
export function bindLayoutSides() {
  const sessionsOnRight = () => {
    const tree = $layoutTree.get()

    if (!tree) {
      return null
    }

    const order = allPaneIds(tree)
    const sessions = order.indexOf('sessions')
    const main = order.indexOf('workspace')

    return sessions >= 0 && main >= 0 ? sessions > main : null
  }

  $layoutTree.subscribe(() => {
    const flipped = sessionsOnRight()

    if (flipped !== null && flipped !== $panesFlipped.get()) {
      $panesFlipped.set(flipped)
    }
  })

  $panesFlipped.listen(flipped => {
    const current = sessionsOnRight()

    // Restoration replaces the tree; a transient mismatch is not a flip gesture.
    if (!modeLayout.restoring && current !== null && current !== flipped) {
      mirrorLayoutTree()
    }
  })

  const $leftEdgeOpen = computed([$panesFlipped, $sidebarOpen, $fileBrowserOpen], (flipped, sidebar, files) =>
    flipped ? files : sidebar
  )

  const $rightEdgeOpen = computed([$panesFlipped, $sidebarOpen, $fileBrowserOpen], (flipped, sidebar, files) =>
    flipped ? sidebar : files
  )

  bindTreeSideVisibility('left', $leftEdgeOpen, open =>
    ($panesFlipped.get() ? setFileBrowserOpen : setSidebarOpen)(open)
  )
  bindTreeSideVisibility('right', $rightEdgeOpen, open =>
    ($panesFlipped.get() ? setSidebarOpen : setFileBrowserOpen)(open)
  )
}
