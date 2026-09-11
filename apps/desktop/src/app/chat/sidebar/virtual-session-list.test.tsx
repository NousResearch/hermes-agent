import { cleanup, render } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SidebarListRow } from '@/lib/session-date-groups'

import { VirtualSessionList } from './virtual-session-list'

const mocks = vi.hoisted(() => ({ useVirtualizer: vi.fn() }))

const virtualizer = {
  getTotalSize: () => 68,
  getVirtualItems: () => [
    { end: 26, index: 0, start: 0 },
    { end: 68, index: 1, start: 26 }
  ],
  measure: vi.fn(),
  measureElement: vi.fn()
}
mocks.useVirtualizer.mockImplementation(() => virtualizer)

vi.mock('@dnd-kit/sortable', () => ({ useSortable: vi.fn() }))
vi.mock('@dnd-kit/utilities', () => ({ CSS: { Transform: { toString: vi.fn() } } }))
vi.mock('@tanstack/react-virtual', () => ({ useVirtualizer: mocks.useVirtualizer }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        dateDivider: {
          earlierThisMonth: 'Earlier this month',
          lastMonth: 'Last month',
          lastWeek: 'Last week',
          older: 'Older',
          today: 'Today',
          yesterday: 'Yesterday'
        }
      }
    }
  })
}))

vi.mock('./chrome', () => ({
  SidebarDateDivider: ({ label, ...props }: { label: string } & React.ComponentProps<'div'>) => (
    <div data-testid={`divider-${label}`} {...props} />
  )
}))

vi.mock('./session-row', () => ({ SidebarSessionRow: () => null }))

afterEach(cleanup)

const rows: SidebarListRow[] = [
  { key: 'today', kind: 'divider', label: 'Today' },
  { key: 'older', kind: 'divider', label: 'Older' }
]
const cardRows = [{ entry: { session: { id: 'card-session' } }, kind: 'session' }] as unknown as SidebarListRow[]

const noop = () => {}

describe('VirtualSessionList', () => {
  it('estimates enough space for the permitted two-line card with preview and footer', () => {
    // Keep this in lockstep with the card's declared CSS geometry:
    // py-1.5; header's size-5 action; two --card-gap: 0.4rem gaps;
    // two 13px leading-none title lines; the 0.15rem title/preview gap;
    // and 10px leading-[1.35] preview + footer lines.
    const tallestCardPx =
      2 * 6 + // py-1.5
      20 + // header action button (size-5)
      2 * 6.4 + // card gaps
      2 * 13 + // line-clamp-2 title
      2.4 + // title/preview gap
      2 * (10 * 1.35) // preview and footer

    const expectedEstimatePx = Math.ceil(tallestCardPx)

    mocks.useVirtualizer.mockClear()
    render(
      <VirtualSessionList
        activeSessionId={null}
        card
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        pinned={false}
        rows={cardRows}
        sortable={false}
      />
    )

    const options = mocks.useVirtualizer.mock.calls.at(-1)?.[0] as { estimateSize: (index: number) => number }

    expect(options.estimateSize(0)).toBe(expectedEstimatePx)
  })

  it('positions measured rows independently within a total-size spacer', () => {
    const { getByTestId } = render(
      <VirtualSessionList
        activeSessionId={null}
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        pinned={false}
        rows={rows}
        sortable={false}
      />
    )

    const firstItem = getByTestId('divider-Today').parentElement
    const secondItem = getByTestId('divider-Older').parentElement
    const spacer = firstItem?.parentElement

    expect(firstItem?.dataset.index).toBe('0')
    expect(firstItem?.style.position).toBe('absolute')
    expect(firstItem?.style.transform).toBe('translateY(0px)')
    expect(secondItem?.dataset.index).toBe('1')
    expect(secondItem?.style.transform).toBe('translateY(26px)')
    expect(spacer?.className).toBe('relative')
    expect(spacer?.style.height).toBe('68px')
    expect(spacer?.style.paddingTop).toBe('')
    expect(spacer?.style.paddingBottom).toBe('')
  })

  it('lets wheel overscroll chain to the outer sidebar scroller (#84964)', () => {
    const { getByTestId } = render(
      <VirtualSessionList
        activeSessionId={null}
        onArchiveSession={noop}
        onDeleteSession={noop}
        onResumeSession={noop}
        onTogglePin={noop}
        onToggleUnread={noop}
        pinned={false}
        rows={rows}
        sortable={false}
      />
    )

    const scroller = getByTestId('divider-Today').parentElement?.parentElement?.parentElement

    // The inner virtualized scroller must NOT contain overscroll: it is nested
    // inside the sidebar's own scroll container, and containing it swallowed
    // wheel events at the inner scroll boundary — the mid-list wheel dead-zone
    // at 25+ sessions. Chaining stays inside the sidebar because the OUTER
    // scroller keeps overscroll-contain.
    expect(scroller?.className).toContain('overflow-y-auto')
    expect(scroller?.className).not.toContain('overscroll-contain')
  })
})
