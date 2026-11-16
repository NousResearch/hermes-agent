import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TimelineRail, timelineTickLabel } from './timeline-rail'

vi.mock('@/components/ui/tooltip', () => ({ Tip: ({ children }: { children: ReactNode }) => children }))

beforeEach(() => {
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
  vi.restoreAllMocks()
})

const entries = Array.from({ length: 5000 }, (_, index) => ({ id: `message-${index}`, preview: `Message ${index}` }))

describe('TimelineRail', () => {
  it('bounds mounted buttons for thousands of prompts', () => {
    const view = render(<TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={vi.fn()} />)

    expect(view.container.querySelectorAll('button').length).toBeGreaterThan(0)
    expect(view.container.querySelectorAll('button').length).toBeLessThan(60)
  })

  it('keeps the active bar compact without hover and routes selection by stable ID', () => {
    const onJump = vi.fn()
    render(<TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={onJump} />)

    const active = screen.getByRole('button', { name: 'Message 0' })

    expect(active.getAttribute('aria-current')).toBe('location')
    fireEvent.click(active)
    expect(onJump).toHaveBeenCalledWith('message-0')
  })

  it('labels and marks ticks by author so themes can style them', () => {
    const mixed = [
      { id: 'u1', preview: 'first', author: 'human' as const, number: 1 },
      { id: 's1', preview: 'Board updated.', author: 'system' as const },
      { id: 'u2', preview: 'second', author: 'human' as const, number: 2 }
    ]

    render(<TimelineRail activeIndex={2} entries={mixed} loadingId={null} onJump={vi.fn()} />)

    const human = screen.getByRole('button', { name: '2 · second' })
    const system = screen.getByRole('button', { name: '🤖 Board updated.' })

    expect(human.dataset.author).toBe('human')
    expect(human.dataset.promptNumber).toBe('2')
    expect(system.dataset.author).toBe('system')
    expect(system.dataset.promptNumber).toBeUndefined()
  })

  it('falls back to the bare preview when a human prompt has no known number', () => {
    expect(timelineTickLabel({ id: 'u', preview: 'tail', author: 'human' })).toBe('tail')
    expect(timelineTickLabel({ id: 'more', preview: 'Show earlier' })).toBe('Show earlier')
  })
})
