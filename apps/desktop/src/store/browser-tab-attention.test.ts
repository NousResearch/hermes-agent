// The background tab flash must leave nothing behind once the user is back:
// title, favicon and interval all reset however they return, and an icon that
// finishes loading after the return must not re-badge the tab.
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $tabFlashText, flagTabAttention } from './browser-tab-attention'
import { installBrowserTabTitle } from './browser-tab-title'

vi.mock('@/lib/host-capabilities', () => ({ hostHas: (name: string) => name === 'tabTitleStatus' }))

const ICON = 'http://localhost/icon.png'
const BADGED = 'data:image/png;base64,badged'

const images: { onload: (() => void) | null; src: string }[] = []

class FakeImage {
  onload: (() => void) | null = null
  src = ''
  constructor() {
    images.push(this)
  }
}

let away = true
let link: HTMLLinkElement
let uninstallTitle: () => void

function comeBack(how: 'focus' | 'visibilitychange') {
  away = false

  if (how === 'focus') {
    window.dispatchEvent(new Event('focus'))
  } else {
    document.dispatchEvent(new Event('visibilitychange'))
  }
}

function expectFullyReset() {
  expect($tabFlashText.get()).toBeNull()
  expect(document.title).toBe('Hermes')
  expect(link.href).toBe(ICON)
  expect(vi.getTimerCount()).toBe(0)
}

beforeEach(() => {
  vi.useFakeTimers()
  away = true
  images.length = 0
  vi.stubGlobal('Image', FakeImage)
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => away })
  vi.spyOn(document, 'hasFocus').mockImplementation(() => !away)
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
    () =>
      ({
        arc: vi.fn(),
        beginPath: vi.fn(),
        drawImage: vi.fn(),
        fill: vi.fn(),
        stroke: vi.fn()
      }) as unknown as CanvasRenderingContext2D
  )
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(BADGED)
  link = document.createElement('link')
  link.rel = 'icon'
  link.href = ICON
  document.head.appendChild(link)
  uninstallTitle = installBrowserTabTitle()
})

afterEach(() => {
  // Leave the module idle for the next test whatever this one asserted.
  away = false
  window.dispatchEvent(new Event('focus'))
  uninstallTitle()
  link.remove()
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('flagTabAttention', () => {
  it.each(['focus', 'visibilitychange'] as const)(
    'retargets one flash and resets title, favicon and timer on return via %s',
    how => {
      flagTabAttention('Approval needed')
      expect(document.title).toBe('Approval needed')
      images[0].onload?.()
      expect(link.href).toBe(BADGED)

      vi.advanceTimersByTime(1000)
      expect(document.title).toBe('Hermes')

      flagTabAttention('Question for you')
      expect(document.title).toBe('Question for you')
      expect(vi.getTimerCount()).toBe(1)
      expect(images).toHaveLength(1)
      vi.advanceTimersByTime(2000)
      expect(document.title).toBe('Question for you')

      comeBack(how)
      expectFullyReset()

      // Listeners are gone too: a later away/back cycle without a new event changes nothing.
      away = true
      vi.advanceTimersByTime(5000)
      expect(document.title).toBe('Hermes')
    }
  )

  it('does not re-badge the favicon when its load finishes after the user returned', () => {
    flagTabAttention('Approval needed')
    comeBack('focus')

    images[0].onload?.()
    expectFullyReset()
  })

  it('does nothing while the user is looking at the tab', () => {
    away = false
    flagTabAttention('Approval needed')

    expect(document.title).toBe('Hermes')
    expect(images).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(0)
  })
})
