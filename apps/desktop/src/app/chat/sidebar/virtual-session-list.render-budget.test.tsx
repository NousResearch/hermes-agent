// @vitest-environment jsdom
//
// Render/measure budget for the sortable virtualized session list (#62964).
//
// The production OOM was allocation churn, not retained content: a 15s CPU
// profile on an 11.5k-session profile was ~78% GC with top self time in dnd-kit
// useSortable's items.slice and TanStack Virtual's measureElement. Those costs
// only scale when the list re-renders (or rebuilds its measurements) per tick,
// so the invariants here are COUNT bounds over a real mount of both libraries —
// no module is stubbed into a fake that cannot loop.
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as React from 'react'

import type { SessionInfo } from '@/hermes'
import type { SidebarListRow } from '@/lib/session-date-groups'

import { ReorderableList } from './reorderable-list'
import { VirtualSessionList } from './virtual-session-list'

// ---------------------------------------------------------------------------
// Real-library instrumentation
// ---------------------------------------------------------------------------

// The real virtualizer, wrapped only to count measureElement ref calls and to
// expose the instance for size assertions. The wrap is assigned ONCE per
// instance — a fresh wrapper each render would churn the ref identity itself.
let measures = 0
// A structural slice of the virtualizer the assertions need — the instance's
// generic is internal; only getVirtualItems shape matters here.
let instance: {
  getVirtualItems: () => { index: number; key: unknown; size: number }[]
  measureElement: (node: never) => unknown
  elementsCache: Map<unknown, HTMLElement>
} | null = null

vi.mock('@tanstack/react-virtual', async importOriginal => {
  const original = await importOriginal<typeof import('@tanstack/react-virtual')>()

  return {
    ...original,
    useVirtualizer: (options: Parameters<typeof original.useVirtualizer>[0]) => {
      const virtualizer = original.useVirtualizer(options)
      const seen = virtualizer as unknown as {
        __countedMeasureElement?: boolean
        measureElement: (node: never) => unknown
      }

      if (!seen.__countedMeasureElement) {
        const measure = seen.measureElement.bind(virtualizer)

        seen.__countedMeasureElement = true
        seen.measureElement = (node => {
          measures += 1

          return measure(node)
        }) as typeof seen.measureElement
      }

      instance = virtualizer as unknown as typeof instance

      return virtualizer
    }
  }
})

let rowRenders = 0

vi.mock('./session-row', () => ({
  // forwardRef like the real row: useSortable's setNodeRef must reach a DOM
  // node or the droppable never measures.
  SidebarSessionRow: React.forwardRef(function CountingRow(_props: unknown, ref: React.Ref<HTMLDivElement>) {
    rowRenders += 1

    return <div ref={ref} />
  })
}))

vi.mock('./chrome', () => ({ SidebarDateDivider: () => null }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { sidebar: { dateDivider: {} } } })
}))

// jsdom has no layout. The virtualizer sizes its viewport from
// offsetWidth/offsetHeight (virtual-core getRect) and measures rows through
// getBoundingClientRect; give both stable values so the range and the measured
// sizes are real. Row heights are keyed by data-index.
const VIEWPORT = { height: 600, width: 240 }
const rowHeights = new Map<string, number>()

beforeEach(() => {
  const rect = (height: number, width: number) =>
    ({
      bottom: height,
      height,
      left: 0,
      right: width,
      top: 0,
      width,
      x: 0,
      y: 0,
      toJSON: () => ({})
    }) as DOMRect

  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    get(this: HTMLElement) {
      // The scroller reports the viewport; a virtualized row (keyed by
      // data-index) reports its own height so measurement reads real sizes.
      const index = this.getAttribute('data-index')

      return index != null ? (rowHeights.get(index) ?? 28) : VIEWPORT.height
    }
  })
  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
    configurable: true,
    get(this: HTMLElement) {
      return VIEWPORT.width
    }
  })
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const index = this.getAttribute('data-index')

    if (index != null) {
      return rect(rowHeights.get(index) ?? 28, VIEWPORT.width)
    }

    return rect(VIEWPORT.height, VIEWPORT.width)
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  measures = 0
  rowRenders = 0
  instance = null
  rowHeights.clear()
})

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const session = (id: string, profile = 'default'): SessionInfo =>
  ({ archived: false, id, last_active: 0, profile, started_at: 0 }) as unknown as SessionInfo

const sessionRows = (count: number): SidebarListRow[] =>
  Array.from({ length: count }, (_, i) => ({
    entry: { session: session(`s${i}`) },
    kind: 'session' as const
  }))

// The sortable wrapper consumes ids only; onReorder is inline like a real caller.
const handleReorder = () => {}

// A parent that re-renders on demand — the sidebar's real store ticks
// (heartbeats, session refreshes, dot-state writes) all land as a prop-level
// re-render of the section. Callbacks are inline, like the real caller's.
function Harness({ ids, rows, sortable }: { ids: string[]; rows: SidebarListRow[]; sortable: boolean }) {
  const [tick, setTick] = React.useState(0)
  const bump = () => setTick(t => t + 1)
  ;(window as unknown as { bumpTick: () => void }).bumpTick = bump

  const virtual = (
    <VirtualSessionList
      activeSessionId={null}
      onDeleteSession={() => {}}
      onResumeSession={() => {}}
      onArchiveSession={() => {}}
      onTogglePin={() => {}}
      onToggleUnread={() => {}}
      pinned={false}
      rows={rows}
      sortable={sortable}
    />
  )

  return sortable ? (
    <ReorderableList ids={ids} onReorder={handleReorder}>
      {virtual}
    </ReorderableList>
  ) : (
    virtual
  )
}

describe('VirtualSessionList render/measure budget (#62964)', () => {
  it('an unrelated parent re-render re-renders no sortable rows and measures nothing new', () => {
    const rows = sessionRows(120)
    const ids = rows.map(row => (row.kind === 'session' ? row.entry.session.id : ''))

    const { unmount } = render(<Harness ids={ids} rows={rows} sortable />)

    // Baseline after mount settles: every mounted row rendered and measured
    // exactly once.
    const rowsAfterMount = rowRenders
    const measuresAfterMount = measures

    expect(rowsAfterMount).toBeGreaterThan(0)
    expect(measuresAfterMount).toBeGreaterThan(0)

    // Three no-op store ticks: same rows, inline callbacks (like the real
    // caller) — only the parent re-renders, the way every sidebar store tick
    // lands on the section. A row that re-renders here re-runs dnd-kit's
    // O(items) useSortable work and re-arms the virtualizer's inline option
    // closures — the per-tick churn that made launch OOM at 11.5k sessions.
    const ticks = 3
    for (let i = 0; i < ticks; i++) {
      act(() => {
        ;(window as unknown as { bumpTick: () => void }).bumpTick()
      })
    }

    expect(rowRenders).toBe(rowsAfterMount)
    expect(measures).toBe(measuresAfterMount)

    unmount()
  })

  it('measures each mounted row exactly once for an N-item list', () => {
    const rows = sessionRows(200)
    const ids = rows.map(row => (row.kind === 'session' ? row.entry.session.id : ''))

    render(<Harness ids={ids} rows={rows} sortable />)

    const items = instance?.getVirtualItems() ?? []

    // The mount budget: one measure per rendered row, no remeasure storm —
    // the range plus overscan bounds how many rows exist at all.
    expect(items.length).toBeGreaterThan(0)
    expect(measures).toBe(items.length)
  })

  it('keys measurements by (profile, id): twins with one stored id keep their own sizes (#92454)', () => {
    rowHeights.set('0', 28)
    rowHeights.set('1', 45)

    const rows: SidebarListRow[] = [
      { entry: { session: session('twin', 'alpha') }, kind: 'session' },
      { entry: { session: session('twin', 'beta') }, kind: 'session' }
    ]

    render(<Harness ids={[]} rows={rows} sortable={false} />)

    // The density effect clears the size cache after the refs' first measure
    // (that measure ran BEFORE the clear). In the real app the row
    // ResizeObserver re-delivers the size right after; jsdom's observer is
    // inert, so re-run the same ref measure the RO would fire.
    act(() => {
      instance?.elementsCache.forEach(node => instance?.measureElement(node as never))
    })

    const items = instance?.getVirtualItems() ?? []
    const byIndex = new Map(items.map(item => [item.index, item]))

    expect(byIndex.get(0)?.key).toBe('alpha::twin')
    expect(byIndex.get(1)?.key).toBe('beta::twin')
    expect(byIndex.get(0)?.size).toBe(28)
    expect(byIndex.get(1)?.size).toBe(45)
  })
})
