import { describe, expect, it, vi } from 'vitest'

import {
  createProjectNestResolver,
  type ProjectNestRow,
  resolveProjectDropIntent,
  resolveProjectSlot
} from './project-drag'
import type { SidebarProjectTree } from './workspace-groups'

const project = (id: string, over: Partial<SidebarProjectTree> = {}): SidebarProjectTree =>
  ({ id, isAuto: false, label: id, parentId: null, ...over }) as SidebarProjectTree

/** The sidebar lays project blocks out with a few px between them. Those gaps are the reorder
 *  targets, so the fixture has them: `GAP` of nothing between one block's end and the next row. */
const GAP = 4

// Geometry only matters for the hit test; the element is never read by the pure resolvers. `box` is
// the project's whole area — its row plus its session rows — and it is both the target and the frame.
const row = (
  id: string,
  index: number,
  top: number,
  height: number
): Pick<ProjectNestRow, 'box' | 'el' | 'id' | 'index'> => ({
  box: { bottom: top + height, left: 0, right: 220, top },
  el: null as unknown as HTMLElement,
  id,
  index
})

/** `dev` is open with `align` nested inside it; `other` and `tail` follow. */
const tree = [
  project('dev', { label: 'Dev' }),
  project('align', { label: 'Align', parentId: 'dev' }),
  project('other'),
  project('tail')
]

const laidOut: ProjectNestRow[] = [
  row('dev', 0, 0, 38), //        row 0–18,   session row 18–38
  row('align', 1, 42, 38), //     row 42–60,  session row 60–80
  row('other', 2, 84, 18), //     row 84–102
  row('tail', 3, 106, 18) //      row 106–124
]

/** Release `activeId` at a point. */
const drop = (activeId: string, y: number, over: SidebarProjectTree[] = tree, overRows = laidOut) =>
  resolveProjectDropIntent({ activeId, pointer: { x: 40, y }, projects: over, rows: overRows })

describe('resolveProjectDropIntent', () => {
  it('nests into the project whose whole area the pointer is over — session rows included', () => {
    expect(drop('tail', 10)).toEqual({ kind: 'into', targetId: 'dev' })
    // Its own session row: the area is the target, not just the row that names it.
    expect(drop('tail', 30)).toEqual({ kind: 'into', targetId: 'dev' })
    // The one way to reach a project that is already nested.
    expect(drop('tail', 50)).toEqual({ kind: 'into', targetId: 'align' })
    expect(drop('tail', 90)).toEqual({ kind: 'into', targetId: 'other' })
  })

  it('leaves everything that is not a row to the reorder', () => {
    // The gap between two projects — the other half of the gesture.
    expect(drop('tail', 40)).toBeNull()
    expect(drop('tail', 82)).toBeNull()
    expect(drop('tail', 104)).toBeNull()
    // Empty space below the list.
    expect(drop('tail', 400)).toBeNull()
    // A project's own row while IT is the dragged one: out of the geometry by definition.
    expect(drop('other', 90)).toBeNull()
  })

  it('outdents a subproject released in a gap, naming the project above it', () => {
    // The gap right under its own parent: dev's own row is the last top-level row above it, so the
    // subproject becomes dev's immediate sibling.
    expect(drop('align', 40)).toEqual({ afterId: 'dev', kind: 'top' })
    // Further down, past `other`: after `other`, where it was released.
    expect(drop('align', 104)).toEqual({ afterId: 'other', kind: 'top' })
    // Above everything: the very top of the list.
    expect(drop('align', -5)).toEqual({ afterId: null, kind: 'top' })
  })

  it('never answers an outdent for a project already at the top level', () => {
    expect(drop('other', 104)).toBeNull()
    expect(drop('dev', 104)).toBeNull()
    expect(drop('tail', 104)).toBeNull()
  })

  it('refuses itself, its own descendants, and discovered rows', () => {
    const scanned = project('scanned', { isAuto: true })
    const projects = [...tree, scanned]

    const withAuto: ProjectNestRow[] = [
      row('dev', 0, 0, 38),
      row('align', 1, 42, 38),
      row('other', 2, 84, 18),
      row('tail', 3, 106, 18),
      row('scanned', 4, 128, 18)
    ]

    // Its own row, and a descendant's row (which cannot take its own ancestor).
    expect(drop('dev', 10, projects, withAuto)).toBeNull()
    expect(drop('dev', 50, projects, withAuto)).toBeNull()
    // An auto row has no project record to nest into.
    expect(drop('tail', 138, projects, withAuto)).toBeNull()
  })
})

/**
 * The reorder walk, on the geometry the list actually renders: the rows the dragged project has taken
 * the place of shift up (or down) by its height, so the next crossing is measured against the box
 * where the row now IS. `p` is the dragged row, at index 1 of five.
 */
describe('resolveProjectSlot', () => {
  const ACTIVE_INDEX = 1
  const ROW_H = 40
  const STEP = ROW_H + GAP
  const IDS = ['a', 'p', 'b', 'c', 'd']

  /** The list as it stands with `slot` taken: every row between the dragged one and the slot has
   *  moved one step toward it, exactly as dnd-kit's sorting strategy displaces them. */
  const at = (slot: null | string, activeIndex = ACTIVE_INDEX) => {
    const slotIndex = slot ? IDS.indexOf(slot) : activeIndex

    const rows = IDS.map((id, index) => {
      const shifted =
        index > activeIndex && index <= slotIndex ? index - 1 : index < activeIndex && index >= slotIndex ? index + 1 : index

      return row(id, index, shifted * STEP, ROW_H)
    })

    return rows.filter(entry => entry.index !== activeIndex)
  }

  /** Walk the pointer down the list from the top, one answer at a time. */
  const walk = (points: number[], start: null | string = null) =>
    points.reduce<{ answers: (null | string)[]; slot: null | string }>(
      (state, y) => {
        const slot = resolveProjectSlot({
          activeIndex: ACTIVE_INDEX,
          pointer: { x: 40, y },
          rows: at(state.slot),
          slot: state.slot
        })

        return { answers: [...state.answers, slot], slot }
      },
      { answers: [], slot: start }
    )

  it('moves nothing until the pointer is past the row', () => {
    // Rows sit at 0, 44, 88, 132, 176. The pointer starts inside the dragged row's own place and
    // walks into `b`: entering a row is not crossing it, so the list is left alone.
    expect(walk([0, 60, 88, 110, 128]).answers).toEqual([null, null, null, null, null])
  })

  it('takes the row the pointer has just cleared, one at a time', () => {
    expect(walk([131]).slot).toBe('b')
    // `b` has moved up a step (to 44–84) and `c` is where `b` was: the next crossing is `c`.
    expect(walk([131, 175]).slot).toBe('c')
    expect(walk([131, 175, 219]).slot).toBe('d')
  })

  it('keeps a moved row moved while the pointer is over where it moved to', () => {
    // `b` is at 44–84 now. Over it, and in the slop under it, the answer is still `b`.
    expect(walk([131, 60]).slot).toBe('b')
    expect(walk([131, 85]).slot).toBe('b')
    // Above its new top edge the row is crossed back, and the dragged project is in its own place.
    expect(walk([131, 40]).slot).toBe(null)
  })

  it('crosses a row and back the same way in both directions', () => {
    // Down past `b` and `c`, then back up: `c` is released once the pointer is clear of where `c`
    // now is (88–128), and the position it hands back is `b` — not the dragged row's own place.
    expect(walk([131, 175, 90]).slot).toBe('c')
    expect(walk([131, 175, 85]).slot).toBe('b')
    expect(walk([131, 175, 85, 40]).slot).toBe(null)
    // The same coordinates on the way down again answer the same as they did the first time.
    expect(walk([131, 175, 85, 40, 131, 175]).slot).toBe('c')
  })

  it('counts the rows above the dragged one only when the pointer goes up past them', () => {
    // `a` is above the dragged row: going down over it does nothing (it is already above), and the
    // answer below is `b`.
    expect(walk([131]).slot).toBe('b')
    // Going up, `a` is taken once the pointer is clear of its top edge, and it moves down a step.
    expect(walk([-3]).slot).toBe('a')
    expect(walk([-3, 60]).slot).toBe('a')
    expect(walk([-3, 90]).slot).toBe(null)
  })

  it('carries a fast drag over every row it passed', () => {
    // One frame, from the top of the list to below `d`.
    expect(walk([219]).slot).toBe('d')
    expect(walk([-3]).slot).toBe('a')
  })

  it('survives a row that is no longer there', () => {
    // A re-read can drop a row (the list changed under the drag): the walk starts over rather than
    // answering for a row it cannot place.
    expect(
      resolveProjectSlot({
        activeIndex: ACTIVE_INDEX,
        pointer: { x: 40, y: 131 },
        rows: at('b').filter(entry => entry.id !== 'b'),
        slot: 'b'
      })
    ).toBe(null)
  })
})

describe('the projects list policy', () => {
  const projects = () => [
    project('dev'),
    project('align', { parentId: 'dev' }),
    project('other'),
    project('tail')
  ]

  const policy = () =>
    createProjectNestResolver({
      projects,
      setParent: vi.fn(),
      setTopLevel: vi.fn(),
      strings: { nestInto: (name: string) => `Add subproject to ${name}`, topLevel: 'Top level' }
    })

  /** Stand the rows up in a document the way `readProjectRows` expects to find them: one wrapper per
   *  project, holding the project's own row and a session row — the area is the whole wrapper. */
  const mountRows = () => {
    const tops = new Map<string, { bottom: number; top: number }>()
    let cursor = 0

    for (const node of projects()) {
      tops.set(`block:${node.id}`, { bottom: cursor + 38, top: cursor })
      cursor += 42
    }

    for (const node of projects()) {
      const block = document.createElement('div')

      block.setAttribute('data-sessions-project', node.id)
      block.append(document.createElement('div'))
      document.body.append(block)
    }

    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      // The pane the outline is clamped to is taller than the list, so the clamp is not what a height
      // assertion measures.
      const rect = this.hasAttribute('data-project-pane')
        ? { bottom: 1000, top: 0 }
        : (tops.get(`block:${this.dataset.sessionsProject ?? ''}`) ?? { bottom: 0, top: 0 })

      return {
        ...rect,
        height: rect.bottom - rect.top,
        left: 0,
        right: 220,
        toJSON: () => ({}),
        width: 220,
        x: 0,
        y: rect.top
      } as DOMRect
    })

    return tops
  }

  /** One animation frame, so the paint the drag re-derives each frame has run. */
  const nextFrame = () => new Promise<void>(done => requestAnimationFrame(() => done()))

  it('repaints the pending outcome every frame, not just when the pointer moves', async () => {
    const tops = mountRows()
    const pane = document.createElement('div')

    pane.setAttribute('data-project-pane', '')
    document.body.append(pane)

    const { resolve } = policy()

    resolve({ activeId: 'tail', overId: null, phase: 'start', pointer: { x: 40, y: 130 } })
    // Over `other` (84–122), so the frame outlines ITS area — 38px of it, not the space below it.
    resolve({ activeId: 'tail', overId: null, phase: 'move', pointer: { x: 40, y: 100 } })

    const outline = () => document.querySelector<HTMLElement>('[data-project-nest-zone]')

    expect(outline()?.style.top).toBe('84px')
    expect(outline()?.style.height).toBe('38px')

    // The list moves under a pointer that has not: every row is 20px higher (a scroll, or the reflow
    // the drag itself just caused).
    for (const rect of tops.values()) {
      rect.bottom -= 20
      rect.top -= 20
    }

    await nextFrame()

    // Repainted from the geometry of THIS frame, not from the one the last pointer move saw — which
    // is what left the frame (and the chip) pointing at where a row used to be.
    expect(outline()?.style.top).toBe('64px')
    expect(outline()?.style.height).toBe('38px')

    resolve({ activeId: 'tail', overId: null, phase: 'cancel', pointer: null })

    // The loop goes with the drag, paint and all.
    expect(outline()).toBe(null)

    document.body.replaceChildren()
    vi.restoreAllMocks()
  })

  it('reads the dragged row and its place from the live list', () => {
    mountRows()

    const { resolve, slot } = policy()

    resolve({ activeId: 'tail', overId: null, phase: 'start', pointer: { x: 40, y: 130 } })

    // Every row is at 0, 42, 84, 126 — 38px tall with a 4px gap — and `tail` is the last of them, so
    // its own area has no row below it to take. Walking up, `other` is taken once the pointer is
    // clear of its top edge (84), and `align` once it is clear of `other`'s new place.
    expect(slot({ x: 40, y: 124 })).toBe(null)
    expect(slot({ x: 40, y: 80 })).toBe('other')
    expect(slot({ x: 40, y: 38 })).toBe('align')
    // A keyboard drag has no pointer and nothing to resolve.
    expect(slot(null)).toBe(null)

    resolve({ activeId: 'tail', overId: null, phase: 'cancel', pointer: null })

    // The next drag starts from the dragged row's own place again.
    resolve({ activeId: 'tail', overId: null, phase: 'start', pointer: { x: 40, y: 130 } })

    expect(slot({ x: 40, y: 90 })).toBe(null)

    resolve({ activeId: 'tail', overId: null, phase: 'cancel', pointer: null })

    // Only the elements this test appended.
    document.body.replaceChildren()
    vi.restoreAllMocks()
  })
})
