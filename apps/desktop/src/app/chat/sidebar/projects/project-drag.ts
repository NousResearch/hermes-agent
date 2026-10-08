/**
 * Sidebar PROJECT drag — move a project INTO another project, or OUT to the top level.
 *
 * Projects are already draggable: `ReorderableList` sorts them with dnd-kit. Nesting rides that same
 * drag rather than adding a rival one, and the two are told apart by WHERE the pointer is, not by how
 * far it has travelled:
 *
 *   over a project's own block — its row and the session rows under it → add it as a subproject of
 *   that project, outlined whole
 *   anywhere else: a gap between rows, empty space below the list → a plain reorder, the project
 *   titles moving around the release point
 *
 * So there is no separate nest gesture and nothing to learn: a row says "in here", a gap says "here".
 * Nesting into an already-nested project happens by pointing at its row.
 *
 * What the release will do is said before it happens: a chip beside the pointer reads "Add subproject
 * to <name>" or "Top level", and the whole region it would land in is outlined. Both follow the list
 * as it scrolls, the outline is clipped to the pane the list lives in, and neither appears while the
 * answer is a reorder or a place a project cannot go: a discovered (auto) row owns no project
 * record, a project cannot be parented to itself or to one of its own descendants, and a project
 * already at the top level has nothing to pull out of. Row tooltips are muted for the length of the
 * drag, so the chip is the only text saying what is about to happen.
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

import type { NestResolver, ReorderQuietZone } from '../reorderable-list'

import { projectDescendantIds } from './model'
import type { SidebarProjectTree } from './workspace-groups'

/** Row tag `ProjectOverviewRow` puts on every project wrapper — the row plus its own session rows. */
const ROW_ATTR = 'data-sessions-project'
/** The project's own row element inside that wrapper. Pointing here means THIS project; anywhere
 *  else in its block (its session rows) or its region (nested subprojects) means its root. */
const HEADER_ATTR = 'data-project-row'
/** Marks the outline drawn around the region a drop would nest into. */
const ZONE_ATTR = 'data-project-nest-zone'
/** Set on `<body>` for the length of a project drag. */
const DRAG_ATTR = 'data-project-drag'
/** Tooltips (the caret's "Show/Hide … sessions" one especially) are noise mid-drag: the chip is the
 *  thing saying what the release does, and a second label disagreeing with it is worse than none. */
const MUTE_CSS = `body[${DRAG_ATTR}] [role="tooltip"]{display:none!important}`
/** The projects list, so the outline is clipped to the pane it lives in instead of hanging out over
 *  the chat beside it. */
const PANE_ATTR = 'data-project-pane'

export interface ProjectNestRow {
  el: HTMLElement
  id: string
  /** The project's own row — the clickable area, and the only thing that names the project. */
  rect: ZoneRect
  /** The row extended over every row of the region it heads, nested subprojects included. */
  group: ZoneRect
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

const area = (rect: ZoneRect) => (rect.right - rect.left) * (rect.bottom - rect.top)

/** Every visible project row with the geometry of the moment: a `rect` for the row itself, a `group`
 *  for the region it heads. Re-read on every pointer move and on scroll, so the answer survives a
 *  list that scrolls or reorders underneath the pointer.
 *
 *  `excludeId` drops the row being dragged. It is not a boundary: dnd-kit slides it under the pointer,
 *  so leaving it in would end the region above it exactly where the pointer is — the frame would stop
 *  at the dragged row and nothing under the pointer would match at all.
 */
export function readProjectRows(projects: SidebarProjectTree[], excludeId = ''): ProjectNestRow[] {
  return expandRowsToGroups(
    projects,
    queryAllVisible<HTMLElement>(`[${ROW_ATTR}]`)
      .filter(el => (el.dataset.sessionsProject || '') !== excludeId)
      .map(el => {
        const block = snapRect(el)
        const header = el.querySelector<HTMLElement>(`[${HEADER_ATTR}]`)

        return {
          block,
          el,
          id: el.dataset.sessionsProject || '',
          // No header element (an older row, or a skin that drops it): the block stands in, and the
          // whole row counts as the project's own.
          rect: header ? snapRect(header) : block
        }
      })
  )
}

/**
 * Give each project row the extent of the region it heads: its own row plus every row rendered under
 * it while it is open, down to the next row that is NOT nested inside it.
 *
 * Rows arrive as headers only, so no element carries the region's extent — the extent is what the
 * neighbouring header says: this project's region ends where the next project outside its subtree
 * begins, and the last region ends at the list's last row. A nested project's region sits inside its
 * parent's, which keeps going over the child's own subprojects. The region is what the frame outlines
 * — the hit test deliberately does NOT use it, so a gap between two projects stays a gap.
 */
export function expandRowsToGroups(
  projects: SidebarProjectTree[],
  rows: (Pick<ProjectNestRow, 'el' | 'id' | 'rect'> & { block: ZoneRect })[]
): ProjectNestRow[] {
  const sorted = [...rows].sort((a, b) => a.block.top - b.block.top)
  const last = sorted[sorted.length - 1]

  return sorted.map((row, index) => {
    const subtree = projectDescendantIds(projects, row.id)
    const next = sorted.slice(index + 1).find(candidate => !subtree.has(candidate.id))
    const bottom = Math.max(row.block.bottom, next ? next.block.top : (last?.block.bottom ?? row.block.bottom))

    return { ...row, group: { ...row.rect, bottom } }
  })
}

/**
 * Pure resolution, so the policy can be tested without a DOM: what does a release here do?
 *
 * A project nests when the pointer is over another project's ROW — the same clickable area that
 * enters it. A release anywhere else is a reorder, and a reorder is not this module's decision: it
 * answers null and lets the list run its own. The row's region is drawn as the frame, never hit
 * tested, so the space between two projects stays a gap instead of falling into whichever project
 * happens to be nearest.
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
  const row = rows.find(candidate => candidate.id !== activeId && rectContains(candidate.rect, pointer.x, pointer.y))

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
  const above = rows.filter(candidate => !nested.has(candidate.id) && candidate.rect.bottom <= pointer.y)
  const last = above[above.length - 1]

  return { afterId: last?.id ?? null, kind: 'top' }
}

/** The outline around the pending target region — the "it will land in here" affordance, clamped to
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
      // The region's bottom is the end of the last row it heads, which on a scrolled list can be
      // well past the pane's own bottom edge. Clamp, or the outline is drawn over whatever sits
      // beside the sidebar.
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
 * them on every move and on scroll, paints the pending outcome, and commits it on release. Only the
 * drop answers non-null, and only when the outcome was structural — a plain reorder is answered by
 * null so the list handles it itself.
 *
 * `quiet` is the half that keeps a nest target still: while the pointer is over any project row the
 * list stops reflowing, so a row cannot slide out from under the pointer just before the drop.
 */
export interface ProjectNestPolicy {
  /** Hand this to `ReorderableList`'s `quietZone`. */
  quiet: ReorderQuietZone
  /** Hand this to `ReorderableList`'s `resolveNest`. */
  resolve: NestResolver
}

/** The row under the pointer, ignoring the dragged one (which is out of the geometry). */
const rowAt = (rows: ProjectNestRow[], activeId: string, x: number, y: number) =>
  rows.find(row => row.id !== activeId && rectContains(row.rect, x, y))

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
  let lastPointer: ProjectDropPoint | null = null
  let ghost: DragGhost | null = null
  let zone: ReturnType<typeof createZoneOutline> | null = null

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

    rows = readProjectRows(projects, activeId)

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
      zone.paint(row.group)
    } else {
      zone?.destroy()
      zone = null
    }
  }

  // The list can scroll without the pointer moving — the sidebar's own scroller, or dnd-kit's edge
  // auto-scroll — so the geometry is re-read on scroll as well as on move.
  const onScroll = () => repaint(lastPointer)

  const teardown = () => {
    hidePaint()
    muteTooltips(false)
    window.removeEventListener('scroll', onScroll, true)
    rows = []
    lastPointer = null
    activeId = ''
  }

  /**
   * Claim every spot over a project row, not just the spot that currently means a nest. A self, a
   * descendant or an auto row cannot be a nest target — but the row still must not shuffle under the
   * pointer, or the frame and the row would disagree about where the release landed. It reads the
   * geometry rather than resolving an intent, so it stays pure and the collision pass cannot move a
   * row by asking it a question.
   */
  const quiet: ReorderQuietZone = pointer => {
    if (!pointer || !activeId) {
      return false
    }

    return Boolean(rowAt(readProjectRows(deps.projects(), activeId), activeId, pointer.x, pointer.y))
  }

  const resolve: NestResolver = info => {
    if (info.phase === 'start') {
      activeId = info.activeId
      rows = readProjectRows(deps.projects(), activeId)
      muteTooltips(true)
      window.addEventListener('scroll', onScroll, true)

      return null
    }

    if (info.phase === 'cancel') {
      teardown()

      return null
    }

    if (info.phase === 'move') {
      lastPointer = info.pointer
      repaint(lastPointer)

      return null
    }

    // Drop. A nest is committed here and reported as handled, so the list does not also reorder.
    // An outdent is committed here and reported as NOT handled: the list's own reorder is what puts
    // the project where it was released, and skipping it would leave the row back under its old
    // parent, wherever that sits in the order.
    const projects = deps.projects()

    rows = readProjectRows(projects, info.activeId)

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

    // The row the pointer was over, so the release click can be swallowed: it lands on that row, and
    // that row's own press is "enter this project". The swallow scopes by `data-project-row`, so this
    // must be the ROW element, not the wrapper.
    const targetRow = rows.find(candidate => candidate.id === intent.targetId)

    return {
      targetEl: targetRow?.el.querySelector<HTMLElement>(`[${HEADER_ATTR}]`) ?? null,
      targetId: intent.targetId
    }
  }

  return { quiet, resolve }
}
