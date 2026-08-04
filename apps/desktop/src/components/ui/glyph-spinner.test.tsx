import { act, render, screen } from '@testing-library/react'
import { Profiler, type ProfilerOnRenderCallback } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { GlyphSpinner } from './glyph-spinner'

describe('GlyphSpinner', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('advances its glyph without an update-phase React commit', () => {
    let updateCommits = 0

    const onRender: ProfilerOnRenderCallback = (_id, phase) => {
      if (phase !== 'mount') {
        updateCommits += 1
      }
    }

    render(
      <Profiler id="glyph-spinner" onRender={onRender}>
        <GlyphSpinner spinner="braille" />
      </Profiler>
    )

    const status = screen.getByRole('status', { name: 'Loading' })
    expect(status.textContent).toBe('⠋')

    act(() => vi.advanceTimersByTime(80))

    expect(status.textContent).toBe('⠙')
    expect(updateCommits).toBe(0)
  })

  it('does not tick while its kept-alive pane is hidden', () => {
    const { rerender } = render(
      <PaneVisibleContext.Provider value={false}>
        <GlyphSpinner spinner="braille" />
      </PaneVisibleContext.Provider>
    )

    const status = screen.getByRole('status', { name: 'Loading' })

    expect(status.textContent).toBe('⠋')
    expect(vi.getTimerCount()).toBe(0)

    rerender(
      <PaneVisibleContext.Provider value>
        <GlyphSpinner spinner="braille" />
      </PaneVisibleContext.Provider>
    )
    expect(vi.getTimerCount()).toBe(1)

    act(() => vi.advanceTimersByTime(80))
    expect(status.textContent).toBe('⠙')

    rerender(
      <PaneVisibleContext.Provider value={false}>
        <GlyphSpinner spinner="braille" />
      </PaneVisibleContext.Provider>
    )
    expect(vi.getTimerCount()).toBe(0)

    const frozen = status.textContent
    act(() => vi.advanceTimersByTime(800))
    expect(status.textContent).toBe(frozen)
  })

  it('suspends animation while the Desktop window is inactive', () => {
    render(<GlyphSpinner spinner="braille" />)

    const status = screen.getByRole('status', { name: 'Loading' })
    expect(vi.getTimerCount()).toBe(1)

    act(() => window.dispatchEvent(new Event('blur')))
    expect(vi.getTimerCount()).toBe(0)

    const frozen = status.textContent
    act(() => vi.advanceTimersByTime(800))
    expect(status.textContent).toBe(frozen)

    act(() => window.dispatchEvent(new Event('focus')))
    expect(vi.getTimerCount()).toBe(1)

    act(() => vi.advanceTimersByTime(80))
    expect(status.textContent).not.toBe(frozen)
  })

  it('suspends animation while the Electron window is minimized or hidden, then resumes on restore', () => {
    let windowStateCallback: ((payload: { isMinimized?: boolean; isVisible?: boolean }) => void) | null = null

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        onWindowStateChanged: vi.fn((callback: typeof windowStateCallback) => {
          windowStateCallback = callback

          return () => {
            if (windowStateCallback === callback) {
              windowStateCallback = null
            }
          }
        })
      }
    })

    try {
      render(<GlyphSpinner spinner="braille" />)

      const status = screen.getByRole('status', { name: 'Loading' })
      expect(windowStateCallback).not.toBeNull()
      expect(vi.getTimerCount()).toBe(1)

      act(() => windowStateCallback?.({ isMinimized: true, isVisible: false }))
      expect(vi.getTimerCount()).toBe(0)

      const frozen = status.textContent
      act(() => vi.advanceTimersByTime(800))
      expect(status.textContent).toBe(frozen)

      act(() => windowStateCallback?.({ isMinimized: false, isVisible: true }))
      expect(vi.getTimerCount()).toBe(1)

      act(() => vi.advanceTimersByTime(80))
      expect(status.textContent).not.toBe(frozen)
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('suspends animation while the document is hidden', () => {
    render(<GlyphSpinner spinner="braille" />)

    const status = screen.getByRole('status', { name: 'Loading' })
    expect(vi.getTimerCount()).toBe(1)

    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' })
    try {
      act(() => document.dispatchEvent(new Event('visibilitychange')))
      expect(vi.getTimerCount()).toBe(0)

      const frozen = status.textContent
      act(() => vi.advanceTimersByTime(800))
      expect(status.textContent).toBe(frozen)

      Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
      act(() => document.dispatchEvent(new Event('visibilitychange')))
      expect(vi.getTimerCount()).toBe(1)

      act(() => vi.advanceTimersByTime(80))
      expect(status.textContent).not.toBe(frozen)
    } finally {
      Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
    }
  })
})
