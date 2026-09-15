import type { useSensors } from '@dnd-kit/core'
import { closestCenter, DndContext, type DragEndEvent, useDraggable, useDroppable } from '@dnd-kit/core'
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
/**
 * A droppable that is NOT one of the list's sortable rows (a session-folder
 * header, say). A drop on it is reported through `onDrop` instead of a reorder,
 * so a list can carry both gestures over one DndContext — and over one press,
 * which is what lets a session row be reordered BY the list and filed into a
 * folder by the same drag session (see session-row.tsx's dual-gesture note).
 */
export interface ReorderDropTarget {
  id: string
  onDrop: (activeId: string) => void
}

/** What a finished drag resolved to. */
export type ReorderDrop =
  { kind: 'drop-target'; targetId: string } | { ids: string[]; kind: 'reorder' } | { kind: 'none' }

/**
 * Arbitrate a completed drag: a drop on a registered target is reported, a drop
 * on another row reorders, anything else (a miss, a row on itself) does nothing.
 * A row the list does NOT order — a session inside a folder — can be dragged but
 * never reorders, so dropping it on a row is a no-op rather than a reorder from
 * an index the list never had.
 *
 * Exported because it is the whole rule: dnd-kit owns the geometry, this owns
 * what the gesture means.
 */
export function resolveReorderDrop(
  activeId: unknown,
  overId: null | unknown,
  ids: string[],
  dropTargets?: ReorderDropTarget[]
): ReorderDrop {
  if (overId === null || overId === undefined) {
    return { kind: 'none' }
  }

  const over = String(overId)

  if (dropTargets?.some(candidate => candidate.id === over)) {
    // A drop target wins over the row it may sit next to: the user aimed at the
    // folder, and the row under the pointer is just what the geometry found.
    return { kind: 'drop-target', targetId: over }
  }

  if (String(activeId) === over) {
    return { kind: 'none' }
  }

  const from = ids.indexOf(String(activeId))
  const to = ids.indexOf(over)

  return from >= 0 && to >= 0 ? { ids: arrayMove(ids, from, to), kind: 'reorder' } : { kind: 'none' }
}

export function ReorderableList({
  children,
  dropTargets,
  ids,
  onReorder,
  sensors
}: {
  children: React.ReactNode
  dropTargets?: ReorderDropTarget[]
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

    const resolved = resolveReorderDrop(active.id, over?.id ?? null, ids, dropTargets)

    if (resolved.kind === 'drop-target') {
      dropTargets?.find(candidate => candidate.id === resolved.targetId)?.onDrop(String(active.id))
    } else if (resolved.kind === 'reorder') {
      onReorder(resolved.ids)
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

export function useSortableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id })

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
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

/** Droppable-only binding: a target that is not a sortable row (a folder
 *  header). A drop on it is routed through the list's `dropTargets`. */
export function useDropTargetBindings(id: string) {
  const { isOver, setNodeRef } = useDroppable({ id })

  return { dropHighlight: isOver, dropRef: setNodeRef }
}

/** Drag-source-only binding: a row that can be dragged onto a drop target but is
 *  NOT a member of the reorder order — a session that lives inside a folder.
 *  Dropping it on a sortable row is a no-op (the list cannot place a row it does
 *  not order); dropping it on a folder files it there. */
export function useDraggableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform } = useDraggable({ id })

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
    ref: setNodeRef,
    reorderable: true as const,
    style: {
      transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
      willChange: isDragging ? 'transform' : undefined
    }
  }
}
