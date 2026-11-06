import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import type { SidebarListRow } from '@/lib/session-date-groups'
import { $sessionListDensity } from '@/store/session-list-density'

import { SESSION_CARD_ROW_ESTIMATE_PX, sessionRowEstimate } from './session-row-details'
import { VirtualSessionList } from './virtual-session-list'

// The virtualizer is mocked with a STABLE instance (the real hook returns
// one), so the component's measure() effect fires exactly when its deps
// change — which is the behavior under test: `card` must invalidate cached
// measurements, or toggling Inbox style leaves the previous mode's row
// heights in place (#88473).
const measureSpy = vi.fn()
let estimateSize: (index: number) => number = () => 0

const stableVirtualizer = {
  measure: (...args: []) => measureSpy(...args),
  getVirtualItems: () => [],
  getTotalSize: () => 0,
  measureElement: vi.fn()
}

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: (options: { estimateSize: (index: number) => number }) => {
    estimateSize = options.estimateSize

    return stableVirtualizer
  }
}))

vi.mock('./chrome', () => ({ SidebarDateDivider: () => null }))
vi.mock('./session-row', () => ({ SidebarSessionRow: () => null }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { sidebar: { dateDivider: {} } } })
}))

const session = (id: string) =>
  ({ archived: false, id, last_active: 0, profile: 'default', started_at: 0 }) as unknown as SessionInfo

const rows: SidebarListRow[] = [
  { key: 'today', kind: 'divider', label: 'Today' },
  { entry: { session: session('s1') }, kind: 'session' },
  { entry: { session: session('s2') }, kind: 'session' }
]

const defaultProps = {
  activeSessionId: null,
  onDeleteSession: () => {},
  onResumeSession: () => {},
  onArchiveSession: () => {},
  onTogglePin: () => {},
  onToggleUnread: () => {},
  pinned: false,
  rows,
  sortable: false
}

function renderList(props: Partial<Parameters<typeof VirtualSessionList>[0]> = {}) {
  return render(<VirtualSessionList {...defaultProps} {...props} />)
}

describe('VirtualSessionList row measurement', () => {
  beforeEach(() => {
    measureSpy.mockClear()
    $sessionListDensity.set('compact')
  })

  afterEach(() => {
    cleanup()
  })

  it('re-measures when the density changes', () => {
    const { rerender } = renderList()

    const afterMount = measureSpy.mock.calls.length

    $sessionListDensity.set('detailed')
    rerender(<VirtualSessionList {...defaultProps} />)

    expect(measureSpy.mock.calls.length).toBeGreaterThan(afterMount)
  })

  it('re-measures when Inbox card mode toggles — stale compact measurements must not survive the switch (#88473)', () => {
    const { rerender } = renderList()

    const afterMount = measureSpy.mock.calls.length

    rerender(<VirtualSessionList {...defaultProps} card />)

    expect(measureSpy.mock.calls.length).toBeGreaterThan(afterMount)
  })

  it('routes the estimate by row kind and mode', () => {
    const { rerender } = renderList()

    // Dividers keep their own fixed estimate in every mode.
    expect(estimateSize(0)).toBe(28)
    expect(estimateSize(1)).toBe(sessionRowEstimate('compact'))

    rerender(<VirtualSessionList {...defaultProps} card />)

    expect(estimateSize(0)).toBe(28)
    expect(estimateSize(1)).toBe(SESSION_CARD_ROW_ESTIMATE_PX)
  })

  it('estimates a card at or above the tallest four-line card stack (#88473)', () => {
    // A full Inbox card renders four text lines (header, title, preview,
    // model/size) where the tallest inline density renders three — plus the
    // card's own padding, and one more title line when the title wraps on a
    // narrow sidebar. The estimate must cover that worst case: undersized
    // estimates paint rows over their neighbours on cold start, before
    // self-measurement can correct them.
    const onePreviewLine = 13.5
    const oneTitleLine = 17.6

    expect(SESSION_CARD_ROW_ESTIMATE_PX).toBeGreaterThanOrEqual(
      sessionRowEstimate('detailed') + onePreviewLine + oneTitleLine
    )
  })

  it('estimates the next row of a kind from the last REAL measurement, not the constant', () => {
    // The recurring glitch (not #88473's cold-start case, which the estimate
    // constants already cover): THIS machine's actual row height can differ
    // from the cross-platform constant on every row the virtualizer has
    // never measured yet, which is every row a resize/scroll/prop change
    // reveals for the first time — not just the very first one. A row
    // estimated from the previous row's real measurement tracks reality
    // instead of restating the same guess every time.
    const realHeightPx = 41
    const originalRect = Element.prototype.getBoundingClientRect

    Element.prototype.getBoundingClientRect = () => ({ height: realHeightPx }) as DOMRect
    stableVirtualizer.getVirtualItems = () => [{ index: 1, key: 's1', start: 0 }] as never

    try {
      const { container } = renderList()
      const measuredRow = container.querySelector('[data-index="1"]')

      expect(measuredRow).not.toBeNull()
      // estimateSize(2) asks about a same-kind row ('s2') that has never been
      // on screen — it must now read the real measurement, not fall back to
      // the compact constant (26px) that seeded the cache before mount.
      expect(estimateSize(2)).toBe(realHeightPx)
      expect(estimateSize(2)).not.toBe(sessionRowEstimate('compact'))
    } finally {
      Element.prototype.getBoundingClientRect = originalRect
      stableVirtualizer.getVirtualItems = () => []
    }
  })
})
