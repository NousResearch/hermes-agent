import { cleanup, fireEvent, render } from '@testing-library/react'
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

let virtualItems: Array<{ index: number; start: number }> = []

const stableVirtualizer = {
  measure: (...args: []) => measureSpy(...args),
  getVirtualItems: () => virtualItems,
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

const sessionRowPropsHistory = vi.hoisted(() => [] as Array<Record<string, unknown>>)

vi.mock('./session-row', () => ({
  SidebarSessionRow: (props: Record<string, unknown>) => {
    sessionRowPropsHistory.push(props)

    const session = props.session as { id: string }

    return props.onToggleTree ? (
      <button data-testid={`tree-toggle-${session.id}`} onClick={props.onToggleTree as () => void} type="button" />
    ) : null
  }
}))

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
    virtualItems = []
    sessionRowPropsHistory.length = 0
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

  it('renders the virtualized parent with its collapse state and toggle behavior', () => {
    const onToggle = vi.fn()
    const parent = { id: 'parent', profile: 'default', started_at: 1 } as never
    const child = { id: 'child', profile: 'default', started_at: 1 } as never

    const sessionRows: SidebarListRow[] = [
      { entry: { hasChildren: true, session: parent }, kind: 'session' },
      { entry: { branchDepth: 1, branchStem: '└─ ', session: child }, kind: 'session' }
    ]

    virtualItems = sessionRows.map((_, index) => ({ index, start: index * 28 }))

    const { getByTestId } = renderList({ rows: sessionRows, treeToggle: { onToggle, open: () => false } })

    expect(sessionRowPropsHistory[0]).toMatchObject({ treeOpen: false })
    expect(sessionRowPropsHistory[1]?.onToggleTree).toBeUndefined()

    fireEvent.click(getByTestId('tree-toggle-parent'))
    expect(onToggle).toHaveBeenCalledWith(parent)
  })
})
