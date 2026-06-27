import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $dockedWindow, dockWindow, undockWindow } from './dock'

const dockToWindow = vi.fn(async (_hwnd: number) => ({ ok: true }) as { ok: boolean; error?: string })
const undock = vi.fn(async () => ({ ok: true }))

beforeEach(() => {
  dockToWindow.mockClear()
  undock.mockClear()
  dockToWindow.mockResolvedValue({ ok: true })
  $dockedWindow.set(null)
  vi.stubGlobal('window', { hermesDesktop: { dockToWindow, undockWindow: undock } })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('dockWindow', () => {
  it('docks by exact hwnd and records the title for the banner', async () => {
    const ok = await dockWindow(12345, 'Calculator')

    expect(ok).toBe(true)
    expect(dockToWindow).toHaveBeenCalledWith(12345) // hwnd, never the title
    expect($dockedWindow.get()).toBe('Calculator')
  })

  it('refuses to dock without an hwnd (no fuzzy fallback)', async () => {
    const ok = await dockWindow(null, 'Calculator')

    expect(ok).toBe(false)
    expect(dockToWindow).not.toHaveBeenCalled()
    expect($dockedWindow.get()).toBeNull()
  })

  it('does not enter dock state when the IPC reports failure', async () => {
    dockToWindow.mockResolvedValueOnce({ ok: false, error: 'sidecar unavailable' })

    const ok = await dockWindow(999, 'Notepad')

    expect(ok).toBe(false)
    expect($dockedWindow.get()).toBeNull()
  })

  it('swallows IPC rejection and stays undocked', async () => {
    dockToWindow.mockRejectedValueOnce(new Error('boom'))

    const ok = await dockWindow(7, 'X')

    expect(ok).toBe(false)
    expect($dockedWindow.get()).toBeNull()
  })
})

describe('undockWindow', () => {
  it('clears dock state and calls the IPC', async () => {
    $dockedWindow.set('Calculator')

    await undockWindow()

    expect(undock).toHaveBeenCalledOnce()
    expect($dockedWindow.get()).toBeNull()
  })
})
