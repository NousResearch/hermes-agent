import { describe, expect, it, vi } from 'vitest'

import { createHudInputHook, isHudAskGesture } from './hud-hook'

describe('isHudAskGesture', () => {
  it('is Ctrl + right-click off macOS and ⌘ + right-click on it', () => {
    expect(isHudAskGesture({ button: 2, ctrlKey: true, metaKey: false }, 'win32')).toBe(true)
    expect(isHudAskGesture({ button: 2, ctrlKey: false, metaKey: true }, 'win32')).toBe(false)
    expect(isHudAskGesture({ button: 2, ctrlKey: false, metaKey: true }, 'darwin')).toBe(true)
    expect(isHudAskGesture({ button: 2, ctrlKey: true, metaKey: false }, 'darwin')).toBe(false)
  })

  it('never fires on a bare right-click or on other buttons', () => {
    expect(isHudAskGesture({ button: 2, ctrlKey: false, metaKey: false }, 'win32')).toBe(false)
    expect(isHudAskGesture({ button: 1, ctrlKey: true, metaKey: false }, 'win32')).toBe(false)
    expect(isHudAskGesture({ button: 3, ctrlKey: true, metaKey: false }, 'win32')).toBe(false)
  })
})

function fakeUiohook() {
  const listeners = new Set<(event: unknown) => void>()

  const uIOhook = {
    on: vi.fn((_event: string, listener: (event: unknown) => void) => listeners.add(listener)),
    off: vi.fn((_event: string, listener: (event: unknown) => void) => listeners.delete(listener)),
    start: vi.fn(),
    stop: vi.fn()
  }

  return { uIOhook, fire: (event: unknown) => listeners.forEach(listener => listener(event)), listeners }
}

describe('createHudInputHook', () => {
  it('reports unavailable, with the reason, when the module cannot load', async () => {
    const hook = await createHudInputHook(() => Promise.reject(new Error("Cannot find module 'uiohook-napi'")), 'win32')

    expect(hook.available).toBe(false)
    expect(hook.reason).toContain('uiohook-napi')
    expect(hook.start(() => {})).toBe(false)
    hook.stop()
  })

  it('starts the hook and fires only on the ask gesture', async () => {
    const fake = fakeUiohook()
    const hook = await createHudInputHook(() => Promise.resolve({ uIOhook: fake.uIOhook }), 'win32')
    const onAsk = vi.fn()

    expect(hook.available).toBe(true)
    expect(hook.start(onAsk)).toBe(true)
    expect(fake.uIOhook.start).toHaveBeenCalledTimes(1)

    fake.fire({ button: 2, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, x: 0, y: 0 })
    fake.fire({ button: 1, ctrlKey: true, metaKey: false, altKey: false, shiftKey: false, x: 0, y: 0 })
    expect(onAsk).not.toHaveBeenCalled()

    fake.fire({ button: 2, ctrlKey: true, metaKey: false, altKey: false, shiftKey: false, x: 0, y: 0 })
    expect(onAsk).toHaveBeenCalledTimes(1)
  })

  it('stop detaches the listener and stops the hook; restart does not double-subscribe', async () => {
    const fake = fakeUiohook()
    const hook = await createHudInputHook(() => Promise.resolve({ uIOhook: fake.uIOhook }), 'linux')

    hook.start(() => {})
    hook.start(() => {})
    expect(fake.listeners.size).toBe(1)

    hook.stop()
    expect(fake.listeners.size).toBe(0)
    expect(fake.uIOhook.stop).toHaveBeenCalled()
  })
})
