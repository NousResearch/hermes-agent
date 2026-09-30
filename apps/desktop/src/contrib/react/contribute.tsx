/**
 * Mount-scoped contribution — the "reverse portal" companion to `Slot`.
 *
 * `ctx.register` is for PERMANENT contributions (registered at plugin load,
 * alive for the plugin's lifetime). `<Contribute>` is for chrome that belongs
 * to a mounted surface: while the owning component is mounted, `children`
 * render inside the target area's Slot; on unmount the contribution disposes
 * itself. No route sniffing, no `when()` polling — React's lifecycle IS the
 * visibility contract (a page's titlebar control leaves with the page).
 *
 * Children stay LIVE: they're projected through a nanostore the registered
 * render subscribes to, so caller state flows into the slot on every render
 * without re-registering (which would churn every other slot).
 */

import { useStore } from '@nanostores/react'
import { atom, type WritableAtom } from 'nanostores'
import { type ReactNode, useEffect, useRef } from 'react'

import { registry } from '../registry'

function ProjectedNode({ $node }: { $node: WritableAtom<ReactNode> }) {
  return <>{useStore($node)}</>
}

export interface ContributeProps {
  /** Target area id (e.g. `titleBar.center`). */
  area: string
  /** Stable contribution id — namespace it (`kanban:board-switcher`). */
  id: string
  order?: number
  children: ReactNode
}

export function Contribute({ area, children, id, order }: ContributeProps) {
  const $node = useRef<WritableAtom<ReactNode>>(null!)

  $node.current ??= atom<ReactNode>(null)

  // Push the latest children into the projection after every render.
  useEffect(() => {
    $node.current.set(children)
  })

  useEffect(
    () => registry.register({ area, id, order, render: () => <ProjectedNode $node={$node.current} /> }),
    [area, id, order]
  )

  return null
}
