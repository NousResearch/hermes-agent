/**
 * Sidebar PROJECT drag — move a project INTO another project, or OUT to the top level.
 *
 * Projects are already draggable: `ReorderableList` sorts them with dnd-kit. Nesting rides that same
 * drag rather than adding a rival one, and the two are told apart by WHERE the pointer is, not by how
 * far it has travelled:
 *
 *   over a project's own area — its row and the session rows under it → add it as a subproject of that
 *   project, with that area outlined
 *   anywhere else: a gap between rows, empty space below the list → a plain reorder, the project
 *   titles moving around the release point
 *
 * So there is no separate nest gesture and nothing to learn: a row says "in here", a gap says "here".
 * The area is the whole row as it is drawn, session rows included, so a project open to show its
 * sessions is as tall a target as it looks. Nesting into an already-nested project happens by pointing
 * at its row.
 *
 * The reorder TRAILS the pointer: dnd-kit's own `closestCenter` moves the other rows as soon as the
 * pointer nears them, so a project shoves its neighbours aside before the pointer has been past them
 * and a row can slide out from under a nest. Here a row is only crossed once the pointer is past it —
 * its whole area, in the direction it is travelling — and once crossed it stays moved until the
 * pointer is past where it went. That is the difference between aiming at a row and falling through
 * it, and it is what makes dragging a project INTO a list of projects possible at all.
 *
 * What the release will do is said before it happens: a chip beside the pointer reads "Add subproject
 * to <name>" or "Top level", and the area it would land in is outlined. Both are repainted every
 * frame of the drag, so a list that scrolls or reorders underneath a still pointer cannot leave either
 * one stale — the outline is clipped to the pane the list lives in, and neither appears while the
 * answer is a reorder or a place a project cannot go: a discovered (auto) row owns no project record,
 * a project cannot be parented to itself or to one of its own descendants, and a project already at
 * the top level has nothing to pull out of. Row tooltips are muted for the length of the drag, so the
 * chip is the only text saying what is about to happen.
 *
 * An outdent clears the parent and then lets the list run its own reorder, so the project lands where
 * it was released instead of being pulled back to wherever it sat inside its old parent.
 *
 * The backend owns the same rule (`projects.set_parent`, which refuses looping moves); this module
 * only decides intent, paints it, and hands the move to the store.
 */

import { queryAllVisible } from '@/components/pane-shell/pane-visibility'
import { rectContains } from '@/components/pane-shell/tree/renderer/drag-session'
import type { ZoneRect } from '@/components/pane-shell/tree/zones-engine'
import { createDragGhost, type DragGhost } from '@/lib/drag-ghost'

import type { NestResolver, ReorderSlotResolver } from '../reorderable-list'

import { projectDescendantIds } from './model'
import type { SidebarProjectTree } from './workspace-groups'

/** Row tag `ProjectOverviewRow` puts on every project wrapper — the row plus its own session rows. */
const ROW_ATTR = 'data-sessions-project'
/** Marks the outline drawn around the area a drop would nest into. */
const ZONE_ATTR = 'data-project-nest-zone'
/** Set on `<body>` for the length of a project drag. */
const DRAG_ATTR = 'data-project-drag'
/** Tooltips (the caret's "Show/Hide … sessions" one especially) are noise mid-drag: the chip is the
 *  thing saying what the release does, and a second label disagreeing with it is worse than none. */
const MUTE_CSS = `body[${DRAG_ATTR}] [role="tooltip"]{display:none!important}`
/** The projects list, so the outline is clipped to the pane it lives in instead of hanging out over
 *  the chat beside it. */
const PANE_ATTR = 'data-project-pane'
/** How far past a row's edge the pointer must go before the row counts as crossed. Without it a
 *  pointer resting exactly on an edge would flip the reorder every frame. */
const EDGE_SLOP = 2

/** What the reorder walk needs of a row: where it is, and where it is drawn. */
export type ProjectNestRowGeometry = Pick<ProjectNestRow, 'box' | 'id' | 'index'>

export interface ProjectNestRow {
  el: HTMLElement
  id: string
  /** Place in the list, read from the DOM: the reorder measures every crossing from the dragged row's
   *  own index, and the dragged row is out of the geometry, so its index has to come from elsewhere. */
  index: number
  /** The project's own row plus its session rows: the whole thing counts as the clickable area, and
   *  it is what a drop is resolved against and what the outline frames. */
  box: ZoneRect
}

export interface ProjectDropPoint {
  /** Pointer position, in client coordinates. */
  x: number
  y: number
}

/** What a release at that point does: nest into `targetId`, or move out to the top level. */
export type ProjectDropIntent =
  | { kind: 'into'; targetId: string }
  /** Top level, landing after `afterId` (null = the very top of the list). */
  | { afterId: null | string; kind: 'top' }

const snapRect = (el: HTMLElement): ZoneRect => {
  const r = el.getBoundingClientRect()

  return { bottom: r.bottom, left: r.left, right: r.right, top: r.top }
}

/** Every visible project row with the geometry of the moment. Re-read on every pointer move and
 *  every frame of the drag, so the answer survives a list that scrolls or reorders underneath the
 *  pointer.
 *
 *  `excludeId` drops the row being dragged: dnd-kit slides it under the pointer, and its own area
 *  would otherwise be the thing the release lands on.
 */
export function readProjectRows(excludeId = ''): ProjectNestRow[] {
  return queryAllVisible<HTMLElement>(`[${ROW_ATTR}]`)
    .map((el, index) => ({
      box: snapRect(el),
      el,
      id: el.dataset.sessionsProject || '',
      index
    }))
    .filter(row => row.id !== excludeId)
}

/** The dragged row's place in the list. It is out of `readProjectRows`, and the reorder counts its
 *  crossings from where the row started, so this is read separately. */
export function projectRowIndex(id: string): number {
  return queryAllVisible<HTMLElement>(`[${ROW_ATTR}]`).findIndex(el => (el.dataset.sessionsProject || '') === id)
}

/**
 * Pure resolution, so the policy can be tested without a DOM: what does a release here do?
 *
 * A project nests when the pointer is over another project's area — the same clickable area that
 * enters it, session rows included, and the same one the outline frames. A release anywhere else is a
 * reorder, and a reorder is not this module's decision: it answers null and lets the list run its own.
 * The space between two projects stays a gap instead of falling into whichever project happens to be
 * nearest.
 *
 * Outdenting is the exception, and only because it cannot be expressed as a reorder: a subproject
 * released in a gap has to leave its parent AND land where it was dropped, and only the list can do
 * the second half. So this answers `top`, the resolver clears the parent, and the drop is reported
 * as unhandled so the list reorders it into place.
 */
export function resolveProjectDropIntent({
  activeId,
  pointer,
  projects,
  rows
}: {
  activeId: string
  pointer: ProjectDropPoint
  projects: SidebarProjectTree[]
  rows: ProjectNestRow[]
}): null | ProjectDropIntent {
  // The dragged row is out of the running — dnd-kit slides it under the pointer.
  const row = rows.find(candidate => candidate.id !== activeId && rectContains(candidate.box, pointer.x, pointer.y))

  if (row) {
    const target = projects.find(project => project.id === row.id)

    // Auto (discovered) rows have no project record to parent to, and nothing nests under itself or
    // one of its own descendants. Refusing leaves the pointer on a row, not in a gap, so the release
    // is a no-op — which is right: there is no other project there to mean.
    if (!target || target.isAuto || projectDescendantIds(projects, activeId).has(target.id)) {
      return null
    }

    return { kind: 'into', targetId: target.id }
  }

  const active = projects.find(project => project.id === activeId)

  if (!active?.parentId) {
    return null
  }

  // Outdent: the released gap, read as a place in the top-level order. A nested row above the gap is
  // skipped — it travels with the parent it is under, and the parent's own position is the answer.
  const nested = projectDescendantIds(projects, activeId)
  const above = rows.filter(candidate => !nested.has(candidate.id) && candidate.box.bottom <= pointer.y)
  const last = above[above.length - 1]

  return { afterId: last?.id ?? null, kind: 'top' }
}

/**
 * Where the pointer has taken the dragged project in the list, as a row it should take the place of —
 * or null while it is still in its own slot. Answered from the rows' own geometry, so a row that has
 * just moved out of the way is stepped over where it now is, and `slot` (the last answer) is the only
 * state: the reorder is a walk, and each call advances or retreats the walk by the crossings the
 * pointer has made since.
 *
 * A row is crossed once the pointer is past its whole area — its own row and the session rows under it
 * — on the side facing the direction the pointer came from. Passing over a project therefore leaves
 * the list alone until the pointer is clear of it, and crossing back is the same gate the other way
 * round, so the row stays where it moved to until the pointer has crossed what it moved to. That is
 * the whole point of the slop: without it a pointer sitting on an edge would swap the rows under
 * themselves every frame.
 */
export function resolveProjectSlot({
  activeIndex,
  pointer,
  rows,
  slot
}: {
  /** The dragged row's place in the list; every crossing is measured from it. */
  activeIndex: number
  pointer: ProjectDropPoint
  /** The rows of the moment, without the dragged one — only their places and their areas are read. */
  rows: ProjectNestRowGeometry[]
  /** The row the last call answered, which is where this one starts walking from. */
  slot: null | string
}): null | string {
  /** Rows below the dragged one, in the order the pointer meets them going down. */
  const down = rows.filter(row => row.index > activeIndex).sort((a, b) => a.index - b.index)
  /** Rows above it, nearest first: the order the pointer meets them going up. */
  const up = rows.filter(row => row.index < activeIndex).sort((a, b) => b.index - a.index)
  const outward = (row: ProjectNestRowGeometry) => (row.index > activeIndex ? down : up)

  // Away from the dragged row, the pointer has to clear the far edge of a row's area; back toward it,
  // the near edge — and while it is over the area in between, the row holds.
  const crossed = (row: ProjectNestRowGeometry) =>
    row.index > activeIndex ? pointer.y >= row.box.bottom + EDGE_SLOP : pointer.y <= row.box.top - EDGE_SLOP

  const recrossed = (row: ProjectNestRowGeometry) =>
    row.index > activeIndex ? pointer.y <= row.box.top - EDGE_SLOP : pointer.y >= row.box.bottom + EDGE_SLOP

  const held = slot ? rows.find(row => row.id === slot) : undefined

  if (held) {
    const ladder = outward(held)
    let at = ladder.indexOf(held)

    if (recrossed(held)) {
      // Coming back over the row it holds, the walk hands the slot to the next row in — which then
      // has to be crossed back in turn, and so on until one is still held (or the dragged row's own
      // place, which falls through to the rows on the other side).
      while (at >= 0 && recrossed(ladder[at])) {
        at -= 1
      }

      if (at >= 0) {
        return ladder[at].id
      }
    } else {
      // Held. A fast drag can still carry the pointer past the rows beyond it.
      while (at + 1 < ladder.length && crossed(ladder[at + 1])) {
        at += 1
      }

      return ladder[at].id
    }
  }

  // Nothing held: the furthest row the pointer has crossed on either side is where the release would
  // land. Each side is monotone — crossing a row means crossing every one before it — so it is the
  // last crossed of each, and only one of them can apply.
  const furthest = (ladder: ProjectNestRowGeometry[]) => {
    for (let at = ladder.length - 1; at >= 0; at -= 1) {
      if (crossed(ladder[at])) {
        return ladder[at]
      }
    }

    return undefined
  }

  const below = furthest(down)
  const above = furthest(up)

  if (below && above) {
    return (pointer.y - below.box.bottom <= above.box.top - pointer.y ? below : above).id
  }

  return (below ?? above)?.id ?? null
}

/** The outline around the pending target area — the "it will land in here" affordance, clamped to
 *  the pane the list lives in so it never hangs out over the chat beside it. */
function createZoneOutline(pane: HTMLElement | null) {
  const el = document.createElement('div')

  el.setAttribute(ZONE_ATTR, '')
  el.style.cssText =
    'position:fixed;z-index:9998;pointer-events:none;border-radius:0.375rem;' +
    'outline:1px solid var(--color-sidebar-ring);outline-offset:-1px'
  document.body.appendChild(el)

  return {
    destroy: () => el.remove(),
    paint: (rect: ZoneRect) => {
      // A row can hang past the pane's own bottom edge on a scrolled list. Clamp, or the outline is
      // drawn over whatever sits beside the sidebar.
      const paneRect = pane?.getBoundingClientRect()
      const left = paneRect ? Math.max(rect.left, paneRect.left) : rect.left
      const top = paneRect ? Math.max(rect.top, paneRect.top) : rect.top
      const right = paneRect ? Math.min(rect.right, paneRect.right) : rect.right
      const bottom = paneRect ? Math.min(rect.bottom, paneRect.bottom) : rect.bottom

      el.style.left = `${left}px`
      el.style.top = `${top}px`
      el.style.width = `${Math.max(0, right - left)}px`
      el.style.height = `${Math.max(0, bottom - top)}px`
      el.style.opacity = right > left && bottom > top ? '1' : '0'
    }
  }
}

/** Injected once, on the first drag: the rule that mutes tooltips while one is in flight. */
let muteStyle: HTMLStyleElement | null = null

const muteTooltips = (on: boolean) => {
  if (on && !muteStyle) {
    muteStyle = document.createElement('style')
    muteStyle.textContent = MUTE_CSS
    document.head.appendChild(muteStyle)
  }

  if (on) {
    document.body.setAttribute(DRAG_ATTR, '')
  } else {
    document.body.removeAttribute(DRAG_ATTR)
  }
}

/**
 * The `ReorderableList` policy for the projects list: reads the rows when a drag engages, re-reads
 * them every frame after that, paints the pending outcome, and commits it on release. Only the drop
 * answers non-null, and only when the outcome was structural — a plain reorder is answered by null so
 * the list handles it itself.
 *
 * `slot` is the half that keeps the list's reflow behind the pointer (see `resolveProjectSlot`): the
 * list moves a row aside when this says the pointer has crossed it, and not before.
 */
export interface ProjectNestPolicy {
  /** Hand this to `ReorderableList`'s `resolveSlot`. */
  slot: ReorderSlotResolver
  /** Hand this to `ReorderableList`'s `resolveNest`. */
  resolve: NestResolver
}

export function createProjectNestResolver(deps: {
  /** The sidebar's projects, read live: labels, nesting and auto flags change between drags. */
  projects: () => SidebarProjectTree[]
  /** Commit a nest through the store (RPC + optimistic tree patch). */
  setParent: (projectId: string, parentId: string) => void
  /**
   * Outdent: clear the project's parent. The ORDER is left to the list's own reorder, so the project
   * lands where it was released instead of being pulled back under its old parent.
   */
  setTopLevel: (projectId: string) => void
  /** Chip text for each outcome. */
  strings: { nestInto: (name: string) => string; topLevel: string }
}): ProjectNestPolicy {
  let rows: ProjectNestRow[] = []
  let activeId = ''
  let activeIndex = -1
  let slot: null | string = null
  let lastPointer: ProjectDropPoint | null = null
  let ghost: DragGhost | null = null
  let zone: ReturnType<typeof createZoneOutline> | null = null
  let frame: null | number = null

  const hidePaint = () => {
    ghost?.destroy()
    ghost = null
    zone?.destroy()
    zone = null
  }

  /** Read the list again — it may have scrolled, or reordered under the pointer — and repaint. */
  const repaint = (pointer: ProjectDropPoint | null) => {
    if (!pointer || !activeId) {
      return
    }

    const projects = deps.projects()

    rows = readProjectRows(activeId)

    const intent = resolveProjectDropIntent({ activeId, pointer, projects, rows })
    const row = intent?.kind === 'into' ? rows.find(candidate => candidate.id === intent.targetId) : null

    if (intent) {
      if (!ghost) {
        ghost = createDragGhost('')
      }

      const target = intent.kind === 'into' ? projects.find(project => project.id === intent.targetId) : null

      ghost.setLabel(intent.kind === 'top' ? deps.strings.topLevel : deps.strings.nestInto(target?.label ?? ''))
      ghost.moveTo(pointer.x, pointer.y)
    } else {
      ghost?.destroy()
      ghost = null
    }

    if (row) {
      // The pane is resolved per drag, not per render: the list can be re-measured while one is in
      // flight, and a stale reference would clamp the outline to a stale box.
      zone ??= createZoneOutline(document.querySelector<HTMLElement>(`[${PANE_ATTR}]`))
      zone.paint(row.box)
    } else {
      zone?.destroy()
      zone = null
    }
  }

  // The paint is re-derived every frame, not just on the frames the pointer moves on: the list scrolls
  // (by hand, or by dnd-kit's edge auto-scroll) and reflows under a still pointer, and an outline or
  // chip left over from the previous layout would point at a row that is no longer there.
  const tick = () => {
    repaint(lastPointer)

    // Nothing left to paint — the list unmounted, or the drag ended without a phase. Stop, rather than
    // waking every frame for the rest of the session; the next move starts the loop again.
    if (rows.length === 0) {
      stopFrames()

      return
    }

    frame = requestAnimationFrame(tick)
  }

  const startFrames = () => {
    frame ??= requestAnimationFrame(tick)
  }

  const stopFrames = () => {
    if (frame !== null) {
      cancelAnimationFrame(frame)
      frame = null
    }
  }

  const teardown = () => {
    stopFrames()
    hidePaint()
    muteTooltips(false)
    rows = []
    slot = null
    activeId = ''
    activeIndex = -1
    lastPointer = null
  }

  /**
   * Where the pointer has taken the dragged project — the list's own reading, because a nest target
   * the list was free to slide away would be a target the release could miss. `resolveProjectSlot`
   * walks the crossings, so a row the pointer has passed stays passed until it is passed again.
   */
  const resolveSlot: ReorderSlotResolver = pointer => {
    if (!pointer || !activeId || activeIndex < 0) {
      return null
    }

    rows = readProjectRows(activeId)
    slot = resolveProjectSlot({ activeIndex, pointer, rows, slot })

    return slot
  }

  const resolve: NestResolver = info => {
    if (info.phase === 'start') {
      activeId = info.activeId
      activeIndex = projectRowIndex(info.activeId)
      rows = readProjectRows(activeId)
      slot = null
      muteTooltips(true)

      return null
    }

    if (info.phase === 'cancel') {
      teardown()

      return null
    }

    if (info.phase === 'move') {
      lastPointer = info.pointer
      startFrames()
      repaint(lastPointer)

      return null
    }

    // Drop. A nest is committed here and reported as handled, so the list does not also reorder.
    // An outdent is committed here and reported as NOT handled: the list's own reorder is what puts
    // the project where it was released, and skipping it would leave the row back under its old
    // parent, wherever that sits in the order.
    const projects = deps.projects()

    rows = readProjectRows(info.activeId)

    const intent = info.pointer
      ? resolveProjectDropIntent({ activeId: info.activeId, pointer: info.pointer, projects, rows })
      : null

    teardown()

    if (!intent) {
      return null
    }

    if (intent.kind === 'top') {
      deps.setTopLevel(info.activeId)

      return null
    }

    const active = projects.find(project => project.id === info.activeId)

    if (active?.parentId !== intent.targetId) {
      deps.setParent(info.activeId, intent.targetId)
    }

    // The element of the area the pointer was over, so the release click can be swallowed: it lands
    // somewhere in that area — the target's own row, one of its session rows, a nested project — and
    // that row's own press is "enter this project".
    const targetRow = rows.find(candidate => candidate.id === intent.targetId)

    return { targetEl: targetRow?.el ?? null, targetId: intent.targetId }
  }

  return { resolve, slot: resolveSlot }
}
