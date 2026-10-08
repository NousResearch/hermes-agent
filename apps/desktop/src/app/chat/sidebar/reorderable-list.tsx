import type { ClientRect, CollisionDetection, UniqueIdentifier, useSensors } from '@dnd-kit/core'
import {
  closestCenter,
  DndContext,
  type DragCancelEvent,
  type DragEndEvent,
  type DragMoveEvent,
  type DragStartEvent
} from '@dnd-kit/core'
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import type * as React from 'react'
import { useRef } from 'react'

// Sidebar reordering is a strictly vertical list. The dragged item's transform
// is rendered Y-only in useSortableBindings (no x, no scale); this just stops
// dnd-kit's auto-scroll from dragging the rail — or the window — sideways when
// the pointer nears an edge, killing the horizontal "drag to valhalla".
const reorderAutoScroll = { threshold: { x: 0, y: 0.2 } }

/** A tree-nesting policy layered on this list's reordering — opt-in, and the only thing that makes a
 *  drop do something other than move the item to a new slot.
 *
 *  Nesting and reordering share one drag, so they need different gestures: the policy sees where the
 *  pointer is and how far it has travelled sideways, and decides. Returning a target from the `drop`
 *  phase takes the drop over from reordering — the caller then owns the outcome and the paint —
 *  while null leaves the ordinary reorder in charge. Every phase fires, nest or not, so the caller
 *  can set up on start, paint on move, commit on drop, and tear down on either ending. `pointer` is
 *  null for keyboard drags: there is no pointer to resolve against. */
export type NestPhase = 'cancel' | 'drop' | 'move' | 'start'

export interface NestDropInfo {
  activeId: string
  /** The row dnd-kit calls the drop target (null = released in the list's empty space). */
  overId: null | string
  phase: NestPhase
  /** Pointer position at this move/release, plus the drag's horizontal travel. */
  pointer: null | { dx: number; x: number; y: number }
}

export type NestResolver = (info: NestDropInfo) => null | NestOutcome

/**
 * What a policy did with a drop, plus the two elements the release `click` may land on: the dragged
 * row itself, and the row the pointer was over when the policy took the drop (a nest target). The list
 * hands both to the drop-click swallow so the release cannot also activate a row — see
 * `createDropClickSwallow`.
 */
export interface NestOutcome {
  /** The nest target's row, when the policy committed a nest. */
  targetEl?: HTMLElement | null
  targetId: null | string
}

/**
 * Opt-in gate on a list's reordering: while the drag is in a "quiet" spot, nothing reflows.
 *
 * dnd-kit's default (`closestCenter`) slides the other items as soon as the pointer nears them, so a
 * target row can move out from under the pointer mid-drag — which reads as the row running away just
 * before you drop. A policy with its own meaning on the pointer's position (a nest target, say) needs
 * that position to mean one thing, so while `quiet` says true the list reports a drop target that is
 * the DRAGGED item itself. Pinning `over` to the active item is what keeps the dragged row under the
 * pointer: dnd-kit's sortable transform is `displaceItem ? strategy(...) : null`, and `displaceItem`
 * needs `over` — so reporting NO target (an empty collision list) does not merely stop the reflow, it
 * nulls the dragged row's transform too and drops the row back into its original slot. `over` equal to
 * the active id makes the strategy yield a zero displacement, so the row holds still instead.
 *
 * The gate has to be HYSTERETIC, not instantaneous. Entering a row pins the crossing; leaving it
 * releases the crossing only once the pointer is clear of the row's box, so passing over a project and
 * coming back does not flip the reorder twice on the way through.
 *
 * `quiet` must be a PURE function of the pointer — a reflow here re-measures the rows the policy just
 * read.
 */
export type ReorderQuietZone = (pointer: null | { x: number; y: number }) => boolean

type NestDragEvent = DragCancelEvent | DragEndEvent | DragMoveEvent | DragStartEvent

/** dnd-kit hands collision detection the pointer's position relative to the droppable container,
 *  which is the viewport box for this list — so it IS client coordinates. A null is the keyboard
 *  drag (no pointer), which no quiet zone can claim. */
const pointerOf = (
  coordinates: null | { x: number; y: number }
): null | { x: number; y: number } => (coordinates ? { x: coordinates.x, y: coordinates.y } : null)

/**
 * Swallow exactly the one `click` a finished pointer drag leaves behind.
 *
 * The release `click` lands on whatever is under the pointer, which is a row in a DIFFERENT list from
 * the one that just dragged — so the swallow has to cover the whole document, but only ever the ONE
 * click a drag produces. Anything broader (a window-level flag consulted later) would eat the user's
 * next real click on an unrelated row.
 *
 * Scope, in order of what the browser can do:
 *  - the click's target is the dragged row or something inside it → always ours (the row never moved)
 *  - the click's target is inside the dragged row's region and the drag ended ON a nest target →
 *    ours; that release is a nest, and the row under the pointer must not also activate
 *  - anything else (a release in a gap, a keyboard drag, a click far from the drag) → not ours
 *
 * The armed flag is per-list: two sidebar lists can have a drag in flight independently, and one
 * list's drag must not disarm or swallow another list's click.
 */
export interface DropClickSwallow {
  /** Arm the swallow for a drag that just ended over `nestTarget` (null when it ended elsewhere). */
  arm: (draggedEl: HTMLElement | null, nestTarget: HTMLElement | null) => void
  /** Whether this list currently owns the swallow. */
  readonly armed: boolean
}

/** The nearest ancestor (inclusive) carrying `attr`, up to `root`. */
const closestWithin = (node: EventTarget | null, attr: string, root: HTMLElement | null): HTMLElement | null => {
  if (!(node instanceof HTMLElement)) {
    return null
  }

  const found = node.closest<HTMLElement>(`[${attr}]`)

  return found && (!root || root.contains(found)) ? found : null
}

/** The row carrying `data-session-row="<id>"`. Matched by attribute value, never through a selector
 *  built out of `id`: sortable ids are opaque, and the gateway/profile groups' are `JSON.stringify`ed
 *  arrays whose quotes made `[data-session-row="${id}"]` a CSS syntax error. */
export const findSessionRow = (id: string): HTMLElement | null =>
  [...document.querySelectorAll<HTMLElement>('[data-session-row]')].find(
    row => row.dataset.sessionRow === id
  ) ?? null

export function createDropClickSwallow(): DropClickSwallow {
  let owner: HTMLElement | null = null
  let nestTarget: HTMLElement | null = null

  const isOurs = (event: MouseEvent): boolean => {
    const target = event.target

    // The dragged row itself.
    if (owner && closestWithin(target, 'data-session-row', owner) === owner) {
      return true
    }

    // A nest release: the pointer was over the target row, so the release click belongs to it.
    return Boolean(nestTarget && closestWithin(target, 'data-project-row', nestTarget) === nestTarget)
  }

  return {
    armed: false,
    arm: (draggedEl, target) => {
      owner = draggedEl
      nestTarget = target

      if (!draggedEl && !target) {
        return
      }

      window.addEventListener(
        'click',
        event => {
          if (isOurs(event)) {
            event.stopPropagation()
            event.preventDefault()
          }

          owner = null
          nestTarget = null
        },
        { capture: true, once: true }
      )
    }
  }
}

/**
 * The collision pass for a list with a `quietZone`, as a resettable object.
 *
 * Returning NO collision is not how you hold a row still — dnd-kit takes `over` from the first
 * collision, and the sortable transform is `displaceItem ? strategy(...) : null` with `displaceItem`
 * requiring `over`. An empty list therefore nulls the dragged row's transform and it snaps back into
 * its original slot. So the dragged item always collides with ITSELF: the strategy sees
 * `index === activeIndex` and yields a zero displacement, which is exactly "held under the pointer".
 *
 * The pin is hysteretic. While the pointer is inside a claimed row the reorder is held; it is released
 * only once the pointer is past that row's box by `slop` pixels, in the direction it was travelling.
 * A pointer on its way back out spends frames inside the row it just left, and an instant gate would
 * resume the reorder inside it — swapping on the way through instead of once past the box. Requiring
 * the crossing to clear the edge is what makes "past a project and back" behave the same as the first
 * pass.
 */
export interface ReorderPin {
  clear: () => void
  detect: CollisionDetection
  /** The policy this pin was built for, so a changed policy can replace it rather than go stale. */
  quiet: ReorderQuietZone | undefined
}

/** How far past a row's edge the pointer must go before a held reorder resumes. */
const EDGE_SLOP = 2

export function createReorderPin(
  quietZone: ReorderQuietZone | undefined,
  { slop = EDGE_SLOP }: { slop?: number } = {}
): ReorderPin {
  // The row whose box the pointer is currently inside, or has only just left.
  let held: null | UniqueIdentifier = null

  /** Has the pointer got clear of this row's box, by `slop`, on EITHER edge? */
  const cleared = (rect: ClientRect | undefined, pointer: null | { x: number; y: number }) => {
    if (!rect || !pointer) {
      return true
    }

    return pointer.y < rect.top - slop || pointer.y > rect.top + rect.height + slop
  }

  return {
    clear: () => {
      held = null
    },
    detect: args => {
      const pointer = pointerOf(args.pointerCoordinates)

      if (pointer && quietZone?.(pointer)) {
        // Claimed. Hold the row the pointer is inside, so the pointer never has to be inside a
        // droppable for this to work: the dragged row's own id is the fallback. Holding the ACTIVE row
        // is what keeps the dragged row under the pointer — `over` stays valid, so the sortable
        // transform survives, and the strategy's `index === activeIndex` branch gives it a zero
        // displacement with no other row disturbed.
        held = rowsAt(args.droppableRects, pointer) ?? args.active.id

        return [{ id: held }]
      }

      if (held !== null) {
        if (!cleared(args.droppableRects.get(held), pointer)) {
          // Inside the row we held on, or within the slop of its edge. A pointer on its way back out
          // spends frames in exactly this band, and releasing here would swap on the way THROUGH the
          // project instead of once the pointer has crossed its box. Holding it is what makes "past a
          // project and back again" cross each box the same way in both directions.
          return [{ id: held }]
        }

        held = null
      }

      return closestCenter(args)
    },
    quiet: quietZone
  }
}

/** The droppable whose rect contains the pointer. */
const rowsAt = (rects: Map<UniqueIdentifier, ClientRect>, pointer: { x: number; y: number }) => {
  for (const [id, rect] of rects) {
    if (
      pointer.x >= rect.left &&
      pointer.x <= rect.left + rect.width &&
      pointer.y >= rect.top &&
      pointer.y <= rect.top + rect.height
    ) {
      return id
    }
  }

  return null
}

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
  quietZone,
  resolveNest,
  sensors
}: {
  children: React.ReactNode
  ids: string[]
  onReorder: (ids: string[]) => void
  quietZone?: ReorderQuietZone
  resolveNest?: NestResolver
  sensors?: ReturnType<typeof useSensors>
}) {
  const nestInfo = (phase: NestPhase, event: NestDragEvent): NestDropInfo => {
    const activator = event.activatorEvent
    const delta = 'delta' in event ? event.delta : { x: 0, y: 0 }

    return {
      activeId: String(event.active.id),
      overId: 'over' in event && event.over ? String(event.over.id) : null,
      phase,
      // A keyboard drag activates on a KeyboardEvent and has no pointer.
      pointer:
        activator instanceof MouseEvent
          ? { dx: delta.x, x: activator.clientX + delta.x, y: activator.clientY + delta.y }
          : null
    }
  }

  // One swallow per list instance (see createDropClickSwallow): the flag must not be shared.
  const swallow = useRef(createDropClickSwallow()).current

  // The pin holds per-drag state, so it is rebuilt whenever the policy changes and cleared when a
  // drag ends: a stale pin would hold the NEXT drag still before it started.
  const pin = useRef(createReorderPin(quietZone))

  if (pin.current.quiet !== quietZone) {
    pin.current = createReorderPin(quietZone)
  }

  const detectCollision = pin.current.detect

  const handleDragEnd = (event: DragEndEvent) => {
    const { activatorEvent, active, over } = event

    // This drag is over; the pin must not reach the next one.
    pin.current.clear()

    // dnd-kit only restores focus for keyboard drags; after a pointer drop the
    // browser leaves :focus on the grab handle, which keeps a focus-within
    // grabber/affordance reveal stuck "on". Drop that focus so the row returns
    // to its resting state once the pointer moves away.
    if (!(activatorEvent instanceof KeyboardEvent)) {
      ;(document.activeElement as HTMLElement | null)?.blur()
    }

    // A pointer drag ends with the pointer still down over whatever is under it, so the browser
    // delivers a `click` to that element on release. On a row whose own press is also a drop target
    // (a nest, say) that click reads as a plain activation and does the wrong thing — entering the
    // project, or running the row's action — the instant you let go. Keyboard drags never produce one.
    if (!(activatorEvent instanceof KeyboardEvent)) {
      swallow.arm(findSessionRow(String(active.id)), null)
    }

    // The policy sees every drop — nest or plain reorder — so it can always tear
    // its paint down; a non-null answer means it also handled the outcome.
    const outcome = resolveNest?.(nestInfo('drop', event))

    if (outcome) {
      swallow.arm(findSessionRow(String(active.id)), outcome.targetEl ?? null)

      return
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
      collisionDetection={detectCollision}
      onDragCancel={event => {
        pin.current.clear()
        void resolveNest?.(nestInfo('cancel', event))
      }}
      onDragEnd={handleDragEnd}
      onDragMove={event => void resolveNest?.(nestInfo('move', event))}
      onDragStart={event => void resolveNest?.(nestInfo('start', event))}
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
