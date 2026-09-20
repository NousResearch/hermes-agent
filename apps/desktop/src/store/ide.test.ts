import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { openIdeWindow } from './ide'
import { $currentCwd } from './session'
import { canOpenIdeWindow } from './windows'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const notifyError = vi.fn()

vi.mock('./notifications', () => ({
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

function installIdeBridge(open?: (request?: unknown) => Promise<{ ok: boolean; error?: string }>) {
  desktopWindow.hermesDesktop = {
    ...(open ? { ide: { open } } : {})
  } as unknown as Window['hermesDesktop']
}

beforeEach(() => {
  notifyError.mockClear()
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }

  $currentCwd.set('')
})

describe('canOpenIdeWindow', () => {
  it('is false when the desktop bridge is absent', () => {
    delete desktopWindow.hermesDesktop
    expect(canOpenIdeWindow()).toBe(false)
  })

  it('is false when the bridge lacks ide.open', () => {
    installIdeBridge(undefined)
    expect(canOpenIdeWindow()).toBe(false)
  })

  it('is true when the bridge exposes ide.open', () => {
    installIdeBridge(vi.fn().mockResolvedValue({ ok: true }))
    expect(canOpenIdeWindow()).toBe(true)
  })
})

describe('openIdeWindow', () => {
  it('no-ops gracefully when the bridge is absent (web fallback)', async () => {
    delete desktopWindow.hermesDesktop

    const opened = await openIdeWindow()

    expect(opened).toBe(false)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('carries the current workspace root as the IDE seed', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installIdeBridge(open)
    $currentCwd.set('/repo/one')

    const opened = await openIdeWindow()

    expect(opened).toBe(true)
    expect(open).toHaveBeenCalledWith({ cwd: '/repo/one' })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('opens with no request object when there is no workspace', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installIdeBridge(open)
    $currentCwd.set('')

    await openIdeWindow()

    expect(open).toHaveBeenCalledWith(undefined)
  })

  it('notifies on an ok:false result and reports failure', async () => {
    installIdeBridge(vi.fn().mockResolvedValue({ ok: false, error: 'no-window' }))

    const opened = await openIdeWindow()

    expect(opened).toBe(false)
    expect(notifyError).toHaveBeenCalled()
  })
})
