import type { useSensors } from '@dnd-kit/core'
import { closestCenter, DndContext, type DragEndEvent, type DragMoveEvent, type DragStartEvent } from '@dnd-kit/core'
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { atom } from 'nanostores'
import type * as React from 'react'
import { useEffect, useRef } from 'react'

import { PROJECT_ROW_DROP_ATTR } from '../session-drag'

// Sidebar reordering is a strictly vertical list. The dragged item's transform
// is rendered Y-only in useSortableBindings (no x, no scale); this just stops
// dnd-kit's auto-scroll from dragging the rail — or the window — sideways when
// the pointer nears an edge, killing the horizontal "drag to valhalla".
const reorderAutoScroll = { threshold: { x: 0, y: 0.2 } }

// One self-contained, nesting-safe reorderable list. It owns its DndContext, so a
// drag only ever collides with THIS list's own items — drop it at any depth (repos,
// worktrees, sessions) and reordering "just works" without leaking into the lists
// around or inside it. Pair each item with useSortableBindings(id); the list reports
// the new id order and the caller persists it. This is the single generic primitive
// behind every reorderable surface in the sidebar.
/** A place outside the list where an item can be let go instead of being moved within it. */
export interface OutsideDrop {
  accepts: (id: string) => boolean
  onDrop: (id: string) => void
  target: () => HTMLElement | null
}

/** Whether a sidebar-row drag that can land on Pinned is in flight. An empty Pinned shelf stands out
 *  of flow at rest; this flag — or a session's own tab drag, `$treeDragging` — brings it out so the
 *  lifted row has a hit-testable drop target (HADESK-132). DndContext keeps no dragging state of
 *  its own and `elementsFromPoint` skips `visibility: hidden`, so the shelf must be shown the moment
 *  a row lifts. */
export const $sessionListPinDragging = atom(false)

/** Clears the shared pin-drag flag, but only for the list that armed it: a second list unmounting
 *  mid-drag (a search, a fold) must not drop another list's in-flight signal. */
function endPinDrag(armed: { current: boolean }) {
  if (armed.current) {
    armed.current = false
    $sessionListPinDragging.set(false)
  }
}

export function ReorderableList({
  children,
  ids,
  onReorder,
  outside,
  sensors
}: {
  children: React.ReactNode
  ids: string[]
  onReorder: (ids: string[]) => void
  outside?: OutsideDrop
  sensors?: ReturnType<typeof useSensors>
}) {
  // The outside target the dragged item is over, wearing the drop mark.
  const over = useRef<HTMLElement | null>(null)
  // Whether THIS list armed the shared pin-drag flag: only then does it clear it on teardown.
  const pinDrag = useRef(false)

  const mark = (target: HTMLElement | null) => {
    if (target !== over.current) {
      over.current?.removeAttribute(PROJECT_ROW_DROP_ATTR)
      target?.setAttribute(PROJECT_ROW_DROP_ATTR, 'true')
      over.current = target
    }
  }

  // A list that goes away mid-drag (a search starts, the route changes) takes its mark and its pin
  // signal with it.
  useEffect(
    () => () => {
      over.current?.removeAttribute(PROJECT_ROW_DROP_ATTR)
      endPinDrag(pinDrag)
    },
    []
  )

  // The lift that can pin brings Pinned's shelf out at once (HADESK-132): its drop target must be
  // visible for `elementsFromPoint` to hit-test it, and the drag snapshots geometry on the way up.
  const handleDragStart = ({ active }: DragStartEvent) => {
    if (outside?.accepts(String(active.id))) {
      pinDrag.current = true
      $sessionListPinDragging.set(true)
    }
  }

  const handleDragMove = ({ active }: DragMoveEvent) => {
    const { initial, translated } = active.rect.current
    const target = initial && translated && outside?.accepts(String(active.id)) ? outside.target() : null

    if (!initial || !translated || !target) {
      mark(null)

      return
    }

    // Where the item is drawn: moved on Y only (useSortableBindings), but its x follows the pointer
    // (dnd-kit's own transform tracks both axes). Hit-test the REAL horizontal position so carrying
    // a chat sideways out of the sidebar — into a workspace pane to split it there — clears this
    // list's outside target instead of still hovering Pinned, which would pin AND split one drop.
    // Hit-tested, so a target scrolled out of its clip is not under the item where its box would be.
    const under = document.elementsFromPoint(
      translated.left + translated.width / 2,
      translated.top + translated.height / 2
    )

    mark(under.includes(target) ? target : null)
  }

  const handleDragEnd = ({ activatorEvent, active, over: overItem }: DragEndEvent) => {
    const dropped = over.current

    mark(null)
    endPinDrag(pinDrag)

    // dnd-kit only restores focus for keyboard drags; after a pointer drop the
    // browser leaves :focus on the grab handle, which keeps a focus-within
    // grabber/affordance reveal stuck "on". Drop that focus so the row returns
    // to its resting state once the pointer moves away.
    if (!(activatorEvent instanceof KeyboardEvent)) {
      ;(document.activeElement as HTMLElement | null)?.blur()
    }

    if (dropped) {
      outside?.onDrop(String(active.id))

      return
    }

    if (!overItem || active.id === overItem.id) {
      return
    }

    const from = ids.indexOf(String(active.id))
    const to = ids.indexOf(String(overItem.id))

    if (from >= 0 && to >= 0) {
      onReorder(arrayMove(ids, from, to))
    }
  }

  return (
    <DndContext
      autoScroll={reorderAutoScroll}
      collisionDetection={closestCenter}
      onDragCancel={() => {
        mark(null)
        endPinDrag(pinDrag)
      }}
      onDragEnd={handleDragEnd}
      onDragMove={outside && handleDragMove}
      onDragStart={handleDragStart}
      sensors={sensors}
    >
      <SortableContext items={ids} strategy={verticalListSortingStrategy}>
        {children}
      </SortableContext>
    </DndContext>
  )
}

export function useSortableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id })
  // The FULL handle (role/tabIndex + dnd-kit's keyboard and pointer
  // activators) belongs on the grabber only. Row shells forward just
  // `onPointerDown` from it: a keyboard activator on a container makes every
  // focused descendant control (the ⋯ menu button) arm a drag on Space, and
  // an armed KeyboardSensor then eats Space/Enter window-wide — the rename
  // dialog swallowed spaces (#83617).
  const dragHandleProps: React.HTMLAttributes<HTMLElement> = { ...attributes, ...listeners }

  return {
    dragging: isDragging,
    dragHandleProps,
    ref: setNodeRef,
    reorderable: true as const,
    style: {
      // Uniform vertical list: only ever translate on Y. Ignoring x and the
      // scaleX/scaleY that CSS.Transform.toString would emit keeps a dragged
      // group/row from drifting sideways or morphing its size mid-drag.
      transform: transform ? `translate3d(0px, ${transform.y}px, 0)` : undefined,
      transition: isDragging ? undefined : transition,
      willChange: isDragging ? 'transform' : undefined
    }
  }
}

/**
 * A row shell owns the presses that STARTED inside its own DOM, and nothing
 * else. React re-dispatches an event fired in a PORTAL along the REACT tree,
 * so a pointerdown on a dialog's input — `DialogContent` portals into `<body>`
 * — still arrives at the row shell that rendered the dialog, carrying a
 * `target` outside the row. Those presses belong to the dialog: selecting a
 * session title in the rename input must not arm a reorder or lift the row onto
 * the shared drag session (the pointer-side sibling of the Space leak #83617
 * fixed on the keyboard side). Gate the shell's own `onPointerDown` with this
 * BEFORE its `[data-reorder-handle], [data-row-actions]` exclusion — that
 * selector walks the DOM, where a portal's content has neither marker.
 */
export function shellOwnsPress(event: React.PointerEvent<HTMLElement>) {
  const target = event.target

  return target instanceof Node && event.currentTarget.contains(target)
}
