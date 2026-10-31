import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isTouchInteraction } from './touch-interaction'

const coarsePointer = (matches: boolean) =>
  vi.fn().mockReturnValue({ addEventListener: vi.fn(), matches, removeEventListener: vi.fn() })

afterEach(() => {
  vi.unstubAllGlobals()
  delete document.documentElement.dataset.hermesDesktopHost
})

describe('touch input ownership in the browser-hosted Desktop', () => {
  beforeEach(() => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
  })

  it('uses coarse capability only when the event has no input evidence', () => {
    vi.stubGlobal('matchMedia', coarsePointer(true))
    expect(isTouchInteraction(new Event('contextmenu'))).toBe(true)
    expect(isTouchInteraction(Object.assign(new Event('contextmenu'), { pointerType: 'mouse' }))).toBe(false)
    expect(
      isTouchInteraction(Object.assign(new Event('contextmenu'), { sourceCapabilities: { firesTouchEvents: false } }))
    ).toBe(false)
  })

  it('recognizes touch and pen on a hybrid device', () => {
    vi.stubGlobal('matchMedia', coarsePointer(false))

    for (const pointerType of ['touch', 'pen']) {
      expect(isTouchInteraction(Object.assign(new Event('contextmenu'), { pointerType }))).toBe(true)
    }

    expect(
      isTouchInteraction(Object.assign(new Event('dblclick'), { sourceCapabilities: { firesTouchEvents: true } }))
    ).toBe(true)
  })

  it('turns the touch layout on for a coarse-pointer tab', async () => {
    vi.stubGlobal('matchMedia', coarsePointer(true))
    vi.resetModules()

    const { $touchPointer } = await import('./touch-interaction')

    expect($touchPointer.get()).toBe(true)
  })
})

// Touch handling belongs to the browser-hosted Desktop. Native Desktop on a touchscreen laptop or
// a tablet (a coarse pointer) keeps the pointer behaviour and layout it has everywhere else.
describe('native Desktop on a coarse-pointer device', () => {
  it('never reads input as touch, with or without input evidence', () => {
    vi.stubGlobal('matchMedia', coarsePointer(true))

    expect(isTouchInteraction()).toBe(false)
    expect(isTouchInteraction(new Event('contextmenu'))).toBe(false)

    for (const pointerType of ['touch', 'pen']) {
      expect(isTouchInteraction(Object.assign(new Event('contextmenu'), { pointerType }))).toBe(false)
    }
  })

  it('keeps the touch layout off', async () => {
    vi.stubGlobal('matchMedia', coarsePointer(true))
    vi.resetModules()

    const { $touchPointer } = await import('./touch-interaction')

    expect($touchPointer.get()).toBe(false)
  })
})
