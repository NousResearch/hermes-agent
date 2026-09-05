import { render } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  orderProjectsByIds,
  previewWindowMaxHeight,
  PROJECT_PREVIEW_COUNT,
  sortProjectsForOverview,
  usePreviewWindowHeight
} from './model'
import { NO_PROJECT_ID, PROJECT_PREVIEW_LOADED, type SidebarProjectTree } from './workspace-groups'

function makeProject(id: string, sessionCount: number): SidebarProjectTree {
  return {
    id,
    isAuto: true,
    label: id,
    lastActive: 0,
    path: `/repos/${id}`,
    previewSessions: [],
    repos: [],
    sessionCount
  }
}

const home = (): SidebarProjectTree => ({
  ...makeProject(NO_PROJECT_ID, 2),
  isAuto: false,
  isNoProject: true,
  path: null
})

const ids = (projects: SidebarProjectTree[]) => projects.map(project => project.id)

describe('orderProjectsByIds', () => {
  it('leaves the deterministic sort alone when nothing has been dragged', () => {
    const projects = [makeProject('a', 0), makeProject('b', 2)]

    expect(orderProjectsByIds(projects, [])).toBe(projects)
  })

  it('applies the saved manual order', () => {
    const projects = [makeProject('a', 1), makeProject('b', 1), makeProject('c', 1)]

    expect(ids(orderProjectsByIds(projects, ['c', 'a', 'b']))).toEqual(['c', 'a', 'b'])
  })

  it('keeps freshly-scanned zero-session repos below the hand-ordered list', () => {
    // The regression: a disk scan keeps finding git checkouts the user has
    // never opened in Hermes. Surfacing every unsaved id at the top buried the
    // projects they deliberately dragged into place.
    const projects = [makeProject('scanned-1', 0), makeProject('mine', 4), makeProject('scanned-2', 0)]

    expect(ids(orderProjectsByIds(projects, ['mine']))).toEqual(['mine', 'scanned-1', 'scanned-2'])
  })

  it('still surfaces a new project that has real activity', () => {
    // A project you just started working in should not sink beneath the saved
    // order — only the zero-session discoveries do.
    const projects = [makeProject('ordered', 1), makeProject('just-started', 3)]

    expect(ids(orderProjectsByIds(projects, ['ordered']))).toEqual(['just-started', 'ordered'])
  })

  it('drops ids that are no longer present', () => {
    const projects = [makeProject('a', 1)]

    expect(ids(orderProjectsByIds(projects, ['gone', 'a']))).toEqual(['a'])
  })

  it('keeps Home on top of a hand-picked order', () => {
    const projects = [makeProject('a', 1), home(), makeProject('b', 1)]

    expect(ids(orderProjectsByIds(projects, ['b', 'a']))).toEqual([NO_PROJECT_ID, 'b', 'a'])
  })
})

describe('sortProjectsForOverview', () => {
  it('puts Home above the active project', () => {
    const active = { ...makeProject('active', 5), isAuto: false }
    const projects = [makeProject('scanned', 0), active, home()]

    expect(ids(sortProjectsForOverview(projects, 'active'))).toEqual([NO_PROJECT_ID, 'active', 'scanned'])
  })
})

describe('preview window height', () => {
  it('sizes the window in rows off the active density, gaps included', () => {
    // compact 28 / comfortable 45 / detailed 63, plus a 1px gap per seam.
    expect(previewWindowMaxHeight('compact')).toBe('86px')
    expect(previewWindowMaxHeight('comfortable')).toBe('137px')
    expect(previewWindowMaxHeight('detailed')).toBe('191px')
  })

  it('prefers a measured row height over the estimate', () => {
    // The estimate is sized for the virtualizer, which only needs to be close.
    // A hard max-height is stricter: a row that renders taller than its estimate
    // would be cropped mid-glyph by an estimate-derived cap.
    expect(previewWindowMaxHeight('compact', PROJECT_PREVIEW_COUNT, 40)).toBe('122px')
  })

  it('ignores a non-positive measurement (an unmounted or hidden row)', () => {
    expect(previewWindowMaxHeight('compact', PROJECT_PREVIEW_COUNT, 0)).toBe('86px')
    expect(previewWindowMaxHeight('compact', PROJECT_PREVIEW_COUNT, null)).toBe('86px')
  })
})

describe('usePreviewWindowHeight', () => {
  // jsdom reports every box as 0x0 and ships no ResizeObserver, so both are
  // stubbed: the observer fires once on observe (its spec-guaranteed first
  // delivery), and the row reports a height taller than compact's estimate.
  class TestResizeObserver {
    constructor(private readonly callback: ResizeObserverCallback) {}
    disconnect() {}
    observe(target: Element) {
      this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
    }
    unobserve() {}
  }

  const ROW_PX = 40

  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({ height: ROW_PX } as DOMRect)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  function Harness({ rows }: { rows: number }): ReactNode {
    const [ref, maxHeight] = usePreviewWindowHeight('compact', rows)

    return createElement(
      'div',
      { 'data-testid': 'window', ref, style: { maxHeight } },
      Array.from({ length: rows }, (_, i) => createElement('div', { key: i }, `row ${i}`))
    )
  }

  it('sizes the window from the rendered row, not compact\u2019s estimate', () => {
    const { getByTestId } = render(createElement(Harness, { rows: 8 }))

    // 3 * 40 + 2 * 1 — the estimate would have said 86px and clipped the third.
    expect(getByTestId('window').style.maxHeight).toBe('122px')
  })

  it('falls back to the estimate while there is no row to measure', () => {
    const { getByTestId } = render(createElement(Harness, { rows: 0 }))

    expect(getByTestId('window').style.maxHeight).toBe('86px')
  })
})

describe('preview depth', () => {
  it('holds more rows than it shows, so the window has something to scroll', () => {
    expect(PROJECT_PREVIEW_LOADED).toBeGreaterThan(PROJECT_PREVIEW_COUNT)
  })
})
