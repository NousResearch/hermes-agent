import { describe, expect, it, vi } from 'vitest'

import { destroyTray, quitFromTray, restoreMainWindow } from './tray-actions'

describe('tray actions', () => {
  it('focuses an existing main window', () => {
    const window = { isDestroyed: () => false }
    const createWindow = vi.fn()
    const focusWindow = vi.fn()

    restoreMainWindow({ window, createWindow, focusWindow })

    expect(createWindow).not.toHaveBeenCalled()
    expect(focusWindow).toHaveBeenCalledWith(window)
  })

  it('creates a main window when none can be restored', () => {
    const createWindow = vi.fn()
    const focusWindow = vi.fn()

    restoreMainWindow({ window: null, createWindow, focusWindow })

    expect(createWindow).toHaveBeenCalledOnce()
    expect(focusWindow).not.toHaveBeenCalled()
  })

  it('creates a main window when the previous window was destroyed', () => {
    const createWindow = vi.fn()
    const focusWindow = vi.fn()

    restoreMainWindow({ window: { isDestroyed: () => true }, createWindow, focusWindow })

    expect(createWindow).toHaveBeenCalledOnce()
    expect(focusWindow).not.toHaveBeenCalled()
  })

  it('marks the app as quitting before requesting quit', () => {
    const calls: string[] = []

    quitFromTray({
      markQuitting: () => calls.push('mark'),
      quit: () => calls.push('quit')
    })

    expect(calls).toEqual(['mark', 'quit'])
  })

  it('destroys a live tray and clears its owner reference', () => {
    const tray = { destroy: vi.fn(), isDestroyed: () => false }
    const clear = vi.fn()

    destroyTray({ tray, clear })

    expect(tray.destroy).toHaveBeenCalledOnce()
    expect(clear).toHaveBeenCalledOnce()
  })

  it('only clears an already destroyed tray', () => {
    const tray = { destroy: vi.fn(), isDestroyed: () => true }
    const clear = vi.fn()

    destroyTray({ tray, clear })

    expect(tray.destroy).not.toHaveBeenCalled()
    expect(clear).toHaveBeenCalledOnce()
  })
})
