import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DecodeText } from './decode-text'

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state })
  document.dispatchEvent(new Event('visibilitychange'))
}

describe('DecodeText', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    setVisibility('visible')
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('stops ticking once a one-shot decode resolves', () => {
    const { container } = render(<DecodeText loop={false} prefix={1} text="HERMES" />)

    act(() => {
      vi.advanceTimersByTime(2_000)
    })

    expect(container.textContent).toBe('HERMES')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('parks a looping decode while the window is hidden and resumes when visible', () => {
    render(<DecodeText prefix={1} text="HERMES" />)
    expect(vi.getTimerCount()).toBe(1)

    act(() => setVisibility('hidden'))
    expect(vi.getTimerCount()).toBe(0)

    act(() => setVisibility('visible'))
    expect(vi.getTimerCount()).toBe(1)
  })
})
