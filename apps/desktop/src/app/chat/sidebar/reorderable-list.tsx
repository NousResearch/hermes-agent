import type { useSensors } from '@dnd-kit/core'
import { closestCenter, DndContext, type DragEndEvent } from '@dnd-kit/core'
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import type * as React from 'react'

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
export function ReorderableList({
  children,
  ids,
  onReorder,
  sensors
}: {
  children: React.ReactNode
  ids: string[]
  onReorder: (ids: string[]) => void
  sensors?: ReturnType<typeof useSensors>
}) {
  const handleDragEnd = ({ activatorEvent, active, over }: DragEndEvent) => {
    // dnd-kit only restores focus for keyboard drags; after a pointer drop the
    // browser leaves :focus on the grab handle, which keeps a focus-within
    // grabber/affordance reveal stuck "on". Drop that focus so the row returns
    // to its resting state once the pointer moves away.
    if (!(activatorEvent instanceof KeyboardEvent)) {
      ;(document.activeElement as HTMLElement | null)?.blur()
    }

    if (!over || active.id === over.id) {
      return
    }

    const from = ids.indexOf(String(active.id))
    const to = ids.indexOf(String(over.id))

    if (from >= 0 && to >= 0) {
      onReorder(arrayMove(ids, from, to))
    }
  }

  return (
    <DndContext
      autoScroll={reorderAutoScroll}
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
      sensors={sensors}
    >
      <SortableContext items={ids} strategy={verticalListSortingStrategy}>
        {children}
      </SortableContext>
    </DndContext>
  )
}

/** Pointer-only dnd listeners for a sortable row shell. The full handle
 *  (attributes + keyboard activator) stays on the grabber; the shell must
 *  never receive role/tabIndex or onKeyDown — a focused shell descendant
 *  would otherwise start a dnd-kit keyboard drag or feed its Space/Enter
 *  end codes (#83617). */
export type ShellDragProps = {
  onPointerDown?: (event: React.PointerEvent) => void
}

export function useSortableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id })

  // The grabber carries the FULL handle (role/tabIndex, keyboard + pointer
  // activators) so keyboard reorder stays accessible. The row SHELL must only
  // ever receive the pointer activator: spreading attributes + onKeyDown onto
  // the shell made the whole row a focusable dnd-kit activator. A focused
  // shell started keyboard drags on Space/Enter, and while a drag session was
  // active dnd-kit's KeyboardSensor arms a window keydown listener whose
  // default `end` codes include Space — swallowing the keystroke in ANY text
  // input (session rename dialog, first composer keystroke after launch).
  const shellDragProps: ShellDragProps | undefined = listeners
    ? { onPointerDown: event => listeners.onPointerDown(event) }
    : undefined

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
    shellDragProps,
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
