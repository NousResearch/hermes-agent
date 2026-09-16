import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TimelineRail } from './timeline-rail'

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
    expect(view.container.querySelector('.thread-timeline-track')?.getAttribute('style')).toContain('35000px')
  })

  it('keeps the active bar at maximum width and routes selection by stable ID', () => {
    const onJump = vi.fn()
    render(<TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={onJump} />)

    const active = screen.getByRole('button', { name: 'Message 0' })

    expect(active.getAttribute('aria-current')).toBe('location')
    expect((active.firstElementChild as HTMLElement).style.width).toBe('1rem')
    fireEvent.click(active)
    expect(onJump).toHaveBeenCalledWith('message-0')
  })
})
