import { expect, test, vi } from 'vitest'

import { createDesktopHeldQuitRuntime } from './desktop-held-quit-runtime'

test('an active turn holds quit once, restores the tray, and confirms a single re-entry', async () => {
  let finishPrompt: ((result: { response: number }) => void) | undefined
  const prompt = new Promise<{ response: number }>(resolve => { finishPrompt = resolve })

  const quit = vi.fn()
  const preventDefault = vi.fn()
  const restore = vi.fn()
  const window = { isDestroyed: () => false, isVisible: () => true }
  const showMessageBox = vi.fn(() => prompt)
  const activeWorkByWebContents = new Map([[1, { count: 1, titles: ['turn'] }]])

  const heldQuitForActiveWork = createDesktopHeldQuitRuntime({
    app: { quit },
    BrowserWindow: { getFocusedWindow: () => window, getAllWindows: () => [window] },
    dialog: { showMessageBox },
    activeWorkByWebContents,
    minimizeToTray: { status: () => ({ available: true }), restore },
    getIsQuittingForHandoff: () => false,
    skipQuitConfirm: false
  } as any)

  expect(heldQuitForActiveWork({ preventDefault } as any)).toBe(true)
  expect(heldQuitForActiveWork({ preventDefault } as any)).toBe(true)
  expect(showMessageBox).toHaveBeenCalledOnce()
  expect(restore).toHaveBeenCalledOnce()
  expect(preventDefault).toHaveBeenCalledTimes(2)

  finishPrompt?.({ response: 1 })
  await Promise.resolve()
  expect(quit).toHaveBeenCalledOnce()
  expect(heldQuitForActiveWork({ preventDefault } as any)).toBe(false)
})

test('a failed prompt cannot trap the process in a held quit', async () => {
  const quit = vi.fn()

  const heldQuitForActiveWork = createDesktopHeldQuitRuntime({
    app: { quit },
    BrowserWindow: {
      getFocusedWindow: () => ({ isDestroyed: () => false, isVisible: () => true }),
      getAllWindows: () => []
    },
    dialog: { showMessageBox: () => Promise.reject(new Error('dialog failed')) },
    activeWorkByWebContents: new Map([[1, { count: 1, titles: [] }]]),
    minimizeToTray: { status: () => ({ available: false }), restore: vi.fn() },
    getIsQuittingForHandoff: () => false,
    skipQuitConfirm: false
  } as any)

  expect(heldQuitForActiveWork({ preventDefault: vi.fn() } as any)).toBe(true)
  await Promise.resolve()
  await Promise.resolve()
  expect(quit).toHaveBeenCalledOnce()
  expect(heldQuitForActiveWork({ preventDefault: vi.fn() } as any)).toBe(false)
})
