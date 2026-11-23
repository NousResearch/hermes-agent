// @vitest-environment jsdom
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TOAST_FLOOR_HEIGHT, useToastFloor } from './toast-floor'

function Floor() {
  return <footer ref={useToastFloor<HTMLElement>()} />
}

const root = window.document.documentElement

describe('useToastFloor', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('publishes the floor’s height on <html> while mounted and takes it back on unmount', () => {
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(69)

    const view = render(<Floor />)

    expect(root.hasAttribute('data-toast-floor')).toBe(true)
    expect(root.style.getPropertyValue(TOAST_FLOOR_HEIGHT)).toBe('69px')

    view.unmount()

    expect(root.hasAttribute('data-toast-floor')).toBe(false)
    expect(root.style.getPropertyValue(TOAST_FLOOR_HEIGHT)).toBe('')
  })

  it('follows the floor as it changes height', () => {
    const observed: ResizeObserverCallback[] = []

    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          observed.push(callback)
        }

        observe() {}
        disconnect() {}
      }
    )

    const height = vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(69)

    render(<Floor />)
    height.mockReturnValue(113)
    observed.forEach(callback => callback([], {} as ResizeObserver))

    expect(root.style.getPropertyValue(TOAST_FLOOR_HEIGHT)).toBe('113px')
    vi.unstubAllGlobals()
  })
})
