import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { IntroRevealSurface } from './intro-reveal-surface'
import { INTRO_DEADMAN_MS } from './timeline'

const close = vi.fn().mockResolvedValue({ ok: true })
const skip = vi.fn()

beforeEach(() => {
  vi.useFakeTimers()
  close.mockClear()
  skip.mockClear()
  vi.stubGlobal('hermesDesktop', { introReveal: { close, skip } })
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
  vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: false })))
})

afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

it('skips locally and restores the main window without a responsive main renderer', () => {
  vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: true })))
  render(<IntroRevealSurface />)
  fireEvent.keyDown(window, { key: 'Escape' })
  expect(skip).toHaveBeenCalledOnce()
  act(() => vi.advanceTimersByTime(1200))
  expect(close).toHaveBeenCalledWith({ showMain: true })
})

it('returns the screen on the deadman even if animation frames never arrive', () => {
  render(<IntroRevealSurface />)
  act(() => vi.advanceTimersByTime(INTRO_DEADMAN_MS))
  expect(skip).not.toHaveBeenCalled()
  expect(close).toHaveBeenCalledWith({ showMain: true })
})
