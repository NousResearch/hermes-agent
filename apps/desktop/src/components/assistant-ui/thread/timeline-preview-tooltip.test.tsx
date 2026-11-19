import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TimelineRail } from './timeline-rail'

const tipProps = vi.hoisted(() => [] as Array<Record<string, unknown>>)

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children, ...props }: { children: ReactNode } & Record<string, unknown>) => {
    tipProps.push(props)

    return children
  }
}))

beforeEach(() => {
  tipProps.length = 0
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    width: 48,
    height: 300,
    top: 0,
    left: 0,
    bottom: 300,
    right: 48,
    x: 0,
    y: 0,
    toJSON: () => ({})
  })
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(300)
  vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(48)
  HTMLElement.prototype.scrollTo = vi.fn()
})

afterEach(() => {
  cleanup()
  window.document.documentElement.dir = ''
  window.document.documentElement.style.fontSize = ''
  vi.restoreAllMocks()
})

describe('TimelineRail preview tooltip', () => {
  it.each([
    { direction: 'rtl', side: 'right' },
    { direction: 'ltr', side: 'left' }
  ])('opens toward content and reaches the hovered bar in $direction', ({ direction, side }) => {
    window.document.documentElement.dir = direction
    window.document.documentElement.style.fontSize = '16px'
    render(<TimelineRail activeIndex={0} entries={[{ id: 'one', preview: 'One' }]} loadingId={null} onJump={vi.fn()} />)

    const props = tipProps.at(-1)!
    expect(props.side).toBe(side)
    // 3rem button − 0.375rem bar inset − 1rem hovered bar = 26px from the
    // content-facing edge. Keep the 5px arrow and a 2px breathing gap.
    expect(props.sideOffset).toBe(-19)
    expect(props.className).toContain('timeline-preview-tooltip')
  })
})
