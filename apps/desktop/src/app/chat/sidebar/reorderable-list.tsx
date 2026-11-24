import type { useSensors } from '@dnd-kit/core'
import { closestCenter, DndContext, type DragEndEvent, type DragMoveEvent } from '@dnd-kit/core'
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import type * as React from 'react'
import { useRef } from 'react'

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

  const mark = (target: HTMLElement | null) => {
    if (target !== over.current) {
      over.current?.removeAttribute(PROJECT_ROW_DROP_ATTR)
      target?.setAttribute(PROJECT_ROW_DROP_ATTR, 'true')
      over.current = target
    }
  }

  const handleDragMove = ({ active }: DragMoveEvent) => {
    const item = active.rect.current.translated
    const target = item && outside?.accepts(String(active.id)) ? outside.target() : null
    const box = target?.getBoundingClientRect()
    const y = item ? item.top + item.height / 2 : 0

    mark(box && y >= box.top && y <= box.bottom ? target : null)
  }

  const handleDragEnd = ({ activatorEvent, active, over: overItem }: DragEndEvent) => {
    const dropped = over.current

    mark(null)

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
      onDragCancel={() => mark(null)}
      onDragEnd={handleDragEnd}
      onDragMove={outside && handleDragMove}
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
