import type { CollisionDetection, UniqueIdentifier, useSensors } from '@dnd-kit/core'
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
  /** The element of the region the pointer was over, when the policy committed a nest: everything the
   *  release `click` can land on inside it is covered by the swallow. */
  targetEl?: HTMLElement | null
  targetId: null | string
}

/**
 * Opt-in reorder policy: which row the dragged item has taken the place of, read from the pointer.
 *
 * dnd-kit's default (`closestCenter`) slides the other items as soon as the pointer NEARS them, so a
 * target row can move out from under the pointer mid-drag — which reads as the row running away just
 * before you drop past it. A list whose pointer carries its own meaning (a nest target, say) needs
 * the reflow to TRAIL the pointer rather than lead it, so it answers this instead: the policy names
 * the row the dragged item is over, and null means "still in its own slot, nothing has moved".
 *
 * Null is not "no collision": an empty collision list nulls the dragged row's transform and snaps it
 * back into its original slot (see `createReorderPin`), so the pin holds the dragged item while the
 * policy answers null.
 *
 * Must be pure in the pointer and the live geometry — the collision pass runs it on every move, and a
 * reflow re-measures the rows it just read. State a policy needs across calls (which row it holds) it
 * keeps itself, and starts a drag with none.
 */
export type ReorderSlotResolver = (pointer: null | { x: number; y: number }) => null | UniqueIdentifier

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

    // A nest release: the pointer was over the target's region, so the release click belongs to it —
    // whichever row inside that region it landed on (the target's own row, one of its session rows, a
    // nested project) must not also run its own press.
    return Boolean(nestTarget && target instanceof Node && nestTarget.contains(target))
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
 * The collision pass for a list with a `resolveSlot` policy.
 *
 * The policy's answer IS `over` — the row it says the dragged item has taken the place of — so the
 * other rows move once the pointer is past them and stay moved until the pointer is past them again.
 * `over` is what dnd-kit feeds the sorting strategy, and the strategy only displaces anything when it
 * has one, so the policy owns the reflow completely.
 *
 * While the policy answers null the dragged item collides with ITSELF instead. That keeps it under
 * the pointer: the strategy's `index === activeIndex` branch yields a zero displacement and no other
 * row is disturbed. Reporting NO collision would not hold it still — an empty list nulls the dragged
 * row's transform, and the row snaps back into its original slot.
 *
 * A list without a policy keeps dnd-kit's own behaviour.
 */
export interface ReorderPin {
  detect: CollisionDetection
  /** The policy this pin was built for, so a changed policy can replace it rather than go stale. */
  slot: ReorderSlotResolver | undefined
}

export function createReorderPin(resolveSlot: ReorderSlotResolver | undefined): ReorderPin {
  return {
    detect: args => {
      if (!resolveSlot) {
        return closestCenter(args)
      }

      const pointer = pointerOf(args.pointerCoordinates)

      return [{ id: (pointer && resolveSlot(pointer)) || args.active.id }]
    },
    slot: resolveSlot
  }
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
  resolveSlot,
  resolveNest,
  sensors
}: {
  children: React.ReactNode
  ids: string[]
  onReorder: (ids: string[]) => void
  resolveSlot?: ReorderSlotResolver
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

  // The pin is rebuilt whenever the policy changes: a stale one would answer for the old list. It
  // holds no per-drag state of its own — a policy that has any starts a drag with fresh state.
  const pin = useRef(createReorderPin(resolveSlot))

  if (pin.current.slot !== resolveSlot) {
    pin.current = createReorderPin(resolveSlot)
  }

  const detectCollision = pin.current.detect

  const handleDragEnd = (event: DragEndEvent) => {
    const { activatorEvent, active, over } = event

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
