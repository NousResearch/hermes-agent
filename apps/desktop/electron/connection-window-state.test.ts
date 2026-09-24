import { describe, expect, it, vi } from 'vitest'

import { liveWindowState, overlayWindowState } from './connection-window-state'

describe('liveWindowState', () => {
  const makeWin = (destroyed = false) => ({ isDestroyed: () => destroyed }) as never

  const deps = (win: unknown) => ({
    fromWebContents: vi.fn().mockReturnValue(win),
    getWindowState: vi.fn().mockReturnValue({ isFullscreen: true }),
    fallback: null as never
  })

  it('reads the SENDER window state, not a cached snapshot (#102451)', () => {
    const senderWin = makeWin()
    const d = deps(senderWin)
    const state = liveWindowState({} as never, d)
    expect(state).toEqual({ isFullscreen: true })
    expect(d.getWindowState).toHaveBeenCalledWith(senderWin)
  })

  it('falls back to the primary window when the reply carries no sender', () => {
    const primary = makeWin()
    const d = {
      fromWebContents: vi.fn(),
      getWindowState: vi.fn().mockReturnValue({ isFullscreen: false }),
      fallback: primary as never
    }
    expect(liveWindowState(undefined, d)).toEqual({ isFullscreen: false })
    expect(d.getWindowState).toHaveBeenCalledWith(primary)
  })

  it('returns undefined when no usable window exists', () => {
    expect(liveWindowState(undefined, deps(null))).toBeUndefined()
    expect(liveWindowState({} as never, deps(makeWin(true)))).toBeUndefined()
  })
})

describe('overlayWindowState', () => {
  it('spreads live state over the connection reply', () => {
    expect(overlayWindowState({ model: 'x', isFullscreen: false }, { isFullscreen: true })).toEqual({
      model: 'x',
      isFullscreen: true
    })
  })

  it('returns the connection untouched when no live state was computable', () => {
    const connection = { model: 'x', isFullscreen: false }
    expect(overlayWindowState(connection, undefined)).toEqual(connection)
  })
})
