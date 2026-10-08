import { describe, expect, it, vi } from 'vitest'

import {
  createProjectNestResolver,
  expandRowsToGroups,
  type ProjectNestRow,
  resolveProjectDropIntent
} from './project-drag'
import type { SidebarProjectTree } from './workspace-groups'

const project = (id: string, over: Partial<SidebarProjectTree> = {}): SidebarProjectTree =>
  ({ id, isAuto: false, label: id, parentId: null, ...over }) as SidebarProjectTree

/** The sidebar lays project blocks out with a few px between them. Those gaps are the reorder
 *  targets, so the fixture has them: `GAP` of nothing between one block's end and the next row. */
const GAP = 4

// Geometry only matters for the hit test; the element is never read by the pure resolver. `block` is
// the project's wrapper (its row plus its session rows) and `rect` its own row inside it.
const row = (
  id: string,
  top: number,
  height: number
): Pick<ProjectNestRow, 'el' | 'id' | 'rect'> & { block: { bottom: number; left: number; right: number; top: number } } => ({
  block: { bottom: top + height, left: 0, right: 220, top },
  el: null as unknown as HTMLElement,
  id,
  rect: { bottom: top + 18, left: 0, right: 220, top }
})

/** `dev` is open with `align` nested inside it; `other` and `tail` follow. */
const tree = [
  project('dev', { label: 'Dev' }),
  project('align', { label: 'Align', parentId: 'dev' }),
  project('other'),
  project('tail')
]

const laidOut = expandRowsToGroups(tree, [
  row('dev', 0, 38), //        row 0–18,   session row 18–38
  row('align', 42, 38), //     row 42–60,  session row 60–80
  row('other', 84, 18), //     row 84–102
  row('tail', 106, 18) //      row 106–124
])

/** Release `activeId` at a point. */
const drop = (activeId: string, y: number, over: SidebarProjectTree[] = tree, overRows = laidOut) =>
  resolveProjectDropIntent({ activeId, pointer: { x: 40, y }, projects: over, rows: overRows })

describe('resolveProjectDropIntent', () => {
  it("nests into the project whose own ROW the pointer is on — nested projects included", () => {
    expect(drop('tail', 10)).toEqual({ kind: 'into', targetId: 'dev' })
    // The one way to reach a project that is already nested.
    expect(drop('tail', 50)).toEqual({ kind: 'into', targetId: 'align' })
    expect(drop('tail', 90)).toEqual({ kind: 'into', targetId: 'other' })
  })

  it('leaves everything that is not a row to the reorder', () => {
    // A project's own session row: inside its block, but not on its row.
    expect(drop('tail', 30)).toBeNull()
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

    const withAuto = expandRowsToGroups(projects, [
      row('dev', 0, 38),
      row('align', 42, 38),
      row('other', 84, 18),
      row('tail', 106, 18),
      row('scanned', 128, 18)
    ])

    // Its own row, and a descendant's row (which cannot take its own ancestor).
    expect(drop('dev', 10, projects, withAuto)).toBeNull()
    expect(drop('dev', 50, projects, withAuto)).toBeNull()
    // An auto row has no project record to nest into.
    expect(drop('tail', 138, projects, withAuto)).toBeNull()
  })
})

describe('the reorder quiet zone', () => {
  // The bug it fixes: `closestCenter` slid the rows as the pointer approached, so a nest target moved
  // out from under the pointer just before the drop — the row appeared to run away.
  const policy = () =>
    createProjectNestResolver({
      projects: () => [
        project('dev'),
        project('align', { parentId: 'dev' }),
        project('other'),
        project('tail')
      ],
      setParent: vi.fn(),
      setTopLevel: vi.fn(),
      strings: { nestInto: (name: string) => `Add subproject to ${name}`, topLevel: 'Top level' }
    })

  /** Stand the rows up in a document the way `readProjectRows` expects to find them. */
  const mountRows = (projects: SidebarProjectTree[]) => {
    const tops = new Map<string, { bottom: number; top: number }>()
    let cursor = 0

    for (const id of ['dev', 'align', 'other', 'tail']) {
      tops.set(`block:${id}`, { bottom: cursor + 38, top: cursor })
      tops.set(`row:${id}`, { bottom: cursor + 18, top: cursor })
      cursor += 42
    }

    for (const node of projects) {
      const block = document.createElement('div')
      const row = document.createElement('div')

      block.setAttribute('data-sessions-project', node.id)
      row.setAttribute('data-project-row', node.id)
      block.append(row)
      document.body.append(block)
    }

    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const id = this.dataset.sessionsProject
      const key = id ? `block:${id}` : this.dataset.projectRow ? `row:${this.dataset.projectRow}` : ''
      const rect = tops.get(key) ?? { bottom: 0, top: 0 }

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

  it('claims every row and leaves the gaps and empty space live', () => {
    const projects = [
      project('dev'),
      project('align', { parentId: 'dev' }),
      project('other'),
      project('tail')
    ]

    mountRows(projects)

    const { quiet, resolve } = policy()

    resolve({ activeId: 'tail', overId: null, phase: 'start', pointer: { dx: 0, x: 40, y: 130 } })

    // Every project row is claimed — a nest, and equally a self or a descendant, which must not
    // shuffle either: the frame and the row have to agree about where the release landed.
    expect(quiet({ x: 40, y: 10 })).toBe(true)
    expect(quiet({ x: 40, y: 52 })).toBe(true)
    expect(quiet({ x: 40, y: 94 })).toBe(true)
    expect(quiet({ x: 40, y: 136 })).toBe(false)

    // The gaps between rows, and the space below the list, are where reordering happens.
    expect(quiet({ x: 40, y: 40 })).toBe(false)
    expect(quiet({ x: 40, y: 82 })).toBe(false)
    expect(quiet({ x: 40, y: 400 })).toBe(false)
    // A keyboard drag has no pointer and no rows to claim.
    expect(quiet(null)).toBe(false)

    resolve({ activeId: 'tail', overId: null, phase: 'cancel', pointer: null })

    // Only the elements this test appended.
    document.body.replaceChildren()
    vi.restoreAllMocks()
  })
})

describe('expandRowsToGroups', () => {
  it('stretches each row over the region it heads, from the row down', () => {
    // dev's region ends where `other` — the next project outside its subtree — begins, so it covers
    // its own block, align's, and the gaps between them. Every project owns the gap beneath it, so
    // each one reaches the next row that is outside its subtree (and the last one stops at itself).
    expect(laidOut.map(row => [row.id, row.rect.bottom, row.group.bottom])).toEqual([
      ['dev', 18, 84],
      ['align', 60, 84],
      ['other', 102, 106],
      ['tail', 124, 124]
    ])
  })

  it('contains a nested project inside its parent', () => {
    const dev = laidOut.find(row => row.id === 'dev')!
    const align = laidOut.find(row => row.id === 'align')!

    expect(align.group.top).toBeGreaterThan(dev.group.top)
    expect(align.group.bottom).toBe(dev.group.bottom)
  })
})