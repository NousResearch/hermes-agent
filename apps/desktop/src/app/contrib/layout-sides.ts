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

  // Two subscriptions write each other's input: layout edits remap the flag,
  // the flag gesture mirrors the tree. Each must recognize the other's writes
  // or they fight over $panesFlipped (#131701): the tree-derived write reverted
  // a gesture whose mirror kept sessionsOnRight() (column splits keep their
  // order), silently eating every later press. Booleans reset too early —
  // nanostores queues a listener fired mid-drain for the NEXT drain, after the
  // gesture's try/finally has already closed — so each side consumes a ticket
  // from the other instead.
  let mirrorTickets = 0
  let remapTickets = 0

  $layoutTree.subscribe(() => {
    // The gesture's own commit: the flag IS the authority for this tree.
    if (mirrorTickets > 0) {
      mirrorTickets -= 1
      return
    }

    const flipped = sessionsOnRight()

    if (flipped !== null && flipped !== $panesFlipped.get()) {
      // Remap (drag/preset/dock), not a gesture — must not mirror back.
      remapTickets += 1
      $panesFlipped.set(flipped)
    }
  })

  $panesFlipped.listen(() => {
    // Restoration replaces the tree; a transient mismatch is not a flip
    // gesture. Neither is this binding's own tree-derived remap above.
    if (modeLayout.restoring || remapTickets > 0) {
      remapTickets -= 1
      return
    }

    mirrorTickets += 1
    mirrorLayoutTree()
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
