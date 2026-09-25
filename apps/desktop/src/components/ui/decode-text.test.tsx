// The decode scramble is decorative. Once a quiet-surface placeholder (empty
// pane zone, contrib panes) has resolved and held, its interval must be gone:
// a replaying default kept a 22 Hz setState ticker alive for as long as the
// mark was on screen, holding an otherwise idle renderer at ~16 commits/s
// (#98394). Only a caller that asks for `loop` (the boot overlay) replays.
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { DecodeText } from './decode-text'

const TICK_MS = 45
// half a char per tick + the 16-tick hold, with slack for the final tick.
const settleTicks = (text: string) => text.length * 2 + 16 + 4

beforeEach(() => {
  vi.useFakeTimers()
  window.matchMedia = vi
    .fn()
    .mockReturnValue({ addEventListener: vi.fn(), matches: false, removeEventListener: vi.fn() })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('resolves once and stops ticking by default', () => {
  const { container } = render(<DecodeText prefix={1} text="HERMES" />)

  act(() => vi.advanceTimersByTime(settleTicks('HERMES') * TICK_MS))
  expect(container.textContent).toBe('HERMES')

  // Fully resolved and held: the timer must be gone, not replaying.
  expect(vi.getTimerCount()).toBe(0)
  act(() => vi.advanceTimersByTime(TICK_MS * 40))
  expect(container.textContent).toBe('HERMES')
})

it('keeps replaying only when the caller asks for loop', () => {
  const { container } = render(<DecodeText loop prefix={1} text="HERMES" />)

  act(() => vi.advanceTimersByTime(settleTicks('HERMES') * TICK_MS))
  expect(vi.getTimerCount()).toBe(1)

  // The replay scrambles the tail again at some point after the hold.
  const seen = new Set<string>()

  for (let i = 0; i < 40; i++) {
    act(() => vi.advanceTimersByTime(TICK_MS))
    seen.add(container.textContent ?? '')
  }

  expect(seen.size).toBeGreaterThan(1)
})
