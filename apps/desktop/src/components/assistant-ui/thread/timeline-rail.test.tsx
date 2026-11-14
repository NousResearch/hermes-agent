import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TimelineRail } from './timeline-rail'

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

const entries = Array.from({ length: 5000 }, (_, index) => ({
  id: `message-${index}`,
  label: `Message ${index}`,
  preview: `Message ${index}`
}))

const railOf = (root: HTMLElement) => root.querySelector<HTMLElement>('[data-slot="thread-timeline-ticks"]')!

const tickOf = (root: HTMLElement, index: number) =>
  root.querySelector<HTMLElement>(`[data-timeline-index="${index}"]`)!

const chipOf = (root: HTMLElement) => root.querySelector<HTMLElement>('[data-slot="thread-timeline-dock"]')!

describe('TimelineRail', () => {
  it('bounds mounted buttons for thousands of prompts', () => {
    const view = render(<TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={vi.fn()} />)

    expect(view.container.querySelectorAll('button').length).toBeGreaterThan(0)
    expect(view.container.querySelectorAll('button').length).toBeLessThan(60)
  })

  it('keeps the active bar compact without hover and routes selection by stable ID', () => {
    const onJump = vi.fn()

    const { container } = render(
      <TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={onJump} />
    )

    const active = screen.getByRole('button', { name: 'Message 0' })

    expect(active.getAttribute('aria-current')).toBe('location')
    expect(chipOf(container).textContent).toBe('')
    fireEvent.click(active)
    expect(onJump).toHaveBeenCalledWith('message-0')
  })

  it('magnifies the mark under the pointer and labels it', async () => {
    const { container } = render(
      <TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={vi.fn()} />
    )

    // Pitch is 7px, so clientY 100 lands on prompt 14 (its centre is 101.5).
    fireEvent.pointerMove(railOf(container), { clientY: 100, pointerType: 'mouse' })

    await waitFor(() => expect(chipOf(container).textContent).toBe('Message 14'))

    const magnified = tickOf(container, 14).firstElementChild as HTMLElement
    const neighbour = tickOf(container, 13).firstElementChild as HTMLElement

    expect(Number.parseFloat(magnified.style.width)).toBeGreaterThan(1)
    expect(Number.parseFloat(magnified.style.height)).toBeGreaterThan(0.0625)
    expect(Number.parseFloat(neighbour.style.width)).toBeLessThan(Number.parseFloat(magnified.style.width))
    expect(Number.parseFloat(neighbour.style.width)).toBeGreaterThan(0.5)
    expect(Number.parseFloat(chipOf(container).style.top)).toBeGreaterThan(90)
    expect(chipOf(container).style.opacity).toBe('1')
  })

  it('grows the rail by the magnified height so no row is clipped', async () => {
    const { container } = render(
      <TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={vi.fn()} />
    )

    const rail = railOf(container)

    expect(rail.style.height).toBe('')

    fireEvent.pointerMove(rail, { clientY: 100, pointerType: 'mouse' })
    await waitFor(() => expect(chipOf(container).style.opacity).toBe('1'))

    const extra = Number(rail.style.height.match(/\+ (\d+)px/)?.[1] ?? 0)
    const growth = Array.from(container.querySelectorAll<HTMLElement>('[data-timeline-index]')).reduce(
      (sum, tick) => sum + (Number.parseFloat(tick.style.height) - 7),
      0
    )

    // The rail is given at least as much height as the rows grew by, so the
    // magnified rows stay inside it instead of being clipped away.
    expect(rail.style.height).toMatch(/^calc\(100% \+ \d+px\)$/)
    expect(extra).toBeGreaterThan(0)
    expect(extra).toBeGreaterThanOrEqual(growth - 0.5)

    fireEvent.pointerLeave(rail)
    await waitFor(() => expect(rail.style.height).toBe(''))
  })

  it('paints on pointer entry, without waiting for a move', async () => {
    const { container } = render(
      <TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={vi.fn()} />
    )

    fireEvent.pointerEnter(railOf(container), { clientY: 100, pointerType: 'mouse' })

    await waitFor(() => expect(chipOf(container).textContent).toBe('Message 14'))
    expect(chipOf(container).style.opacity).toBe('1')
  })

  it('drops the magnification and the label when the pointer leaves', async () => {
    const { container } = render(
      <TimelineRail activeIndex={0} entries={entries} loadingId={null} onJump={vi.fn()} />
    )

    fireEvent.pointerMove(railOf(container), { clientY: 100, pointerType: 'mouse' })
    await waitFor(() => expect(chipOf(container).textContent).toBe('Message 14'))

    fireEvent.pointerLeave(railOf(container))

    await waitFor(() => expect(chipOf(container).textContent).toBe(''))
    expect(chipOf(container).style.opacity).toBe('0')
    expect(Number.parseFloat((tickOf(container, 14).firstElementChild as HTMLElement).style.width)).toBeCloseTo(
      0.5
    )
  })
})
