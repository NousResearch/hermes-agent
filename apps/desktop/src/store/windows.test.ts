import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from './profile'
import { $sessions } from './session'
import {
  canOpenBrowserWindow,
  canOpenNewWindow,
  canOpenSessionWindow,
  isPeerInstanceWindow,
  openBrowserInNewWindow,
  openNewWindow,
  openSessionInNewWindow,
  windowConnectionIdOverride
} from './windows'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const notifyError = vi.fn()

vi.mock('./notifications', () => ({
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

function installBridge(
  openSessionWindow?: Window['hermesDesktop']['openSessionWindow'],
  openWindow?: Window['hermesDesktop']['openWindow'],
  openBrowserWindow?: Window['hermesDesktop']['openBrowserWindow']
) {
  desktopWindow.hermesDesktop = {
    ...(openSessionWindow ? { openSessionWindow } : {}),
    ...(openWindow ? { openWindow } : {}),
    ...(openBrowserWindow ? { openBrowserWindow } : {})
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
})

describe('canOpenSessionWindow', () => {
  it('is false when the desktop bridge is absent', () => {
    delete desktopWindow.hermesDesktop
    expect(canOpenSessionWindow()).toBe(false)
  })

  it('is false when the bridge lacks openSessionWindow', () => {
    installBridge(undefined)
    expect(canOpenSessionWindow()).toBe(false)
  })

  it('is true when the bridge exposes openSessionWindow', () => {
    installBridge(vi.fn().mockResolvedValue({ ok: true }))
    expect(canOpenSessionWindow()).toBe(true)
  })
})

describe('isPeerInstanceWindow', () => {
  it('recognizes only the full peer marker', () => {
    expect(isPeerInstanceWindow('?peer=1')).toBe(true)
    expect(isPeerInstanceWindow('?peer=0')).toBe(false)
    expect(isPeerInstanceWindow('?win=secondary')).toBe(false)
    expect(isPeerInstanceWindow('')).toBe(false)
  })
})

describe('windowConnectionIdOverride', () => {
  it('reads and trims the registry connection from a helper URL', () => {
    expect(windowConnectionIdOverride('?win=hud&profile=default&connectionId=%20alia-linux%20')).toBe('alia-linux')
  })

  it('returns null when the helper URL has no connection override', () => {
    expect(windowConnectionIdOverride('?win=hud&profile=default')).toBeNull()
  })
})

describe('openSessionInNewWindow', () => {
  it('no-ops without a session id', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)

    await openSessionInNewWindow('')

    expect(open).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('no-ops gracefully when the bridge is absent (web fallback)', async () => {
    delete desktopWindow.hermesDesktop

    await openSessionInNewWindow('s1')

    expect(notifyError).not.toHaveBeenCalled()
  })

  it('carries the owning profile: stamped row wins, an unstamped child inherits the viewed profile (#82768)', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)
    $activeGatewayProfile.set('work')
    $sessions.set([{ id: 's1', profile: 'research' } as never])

    await openSessionInNewWindow('s1')
    await openSessionInNewWindow('child-not-listed-yet', { watch: true })

    expect(open).toHaveBeenCalledWith('s1', { profile: 'research' })
    expect(open).toHaveBeenCalledWith('child-not-listed-yet', { profile: 'work', watch: true })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('notifies on an ok:false result', async () => {
    installBridge(vi.fn().mockResolvedValue({ ok: false, error: 'invalid-session-id' }))

    await openSessionInNewWindow('s1')

    expect(notifyError).toHaveBeenCalledTimes(1)
  })

  it('notifies when the bridge throws', async () => {
    installBridge(vi.fn().mockRejectedValue(new Error('boom')))

    await openSessionInNewWindow('s1')

    expect(notifyError).toHaveBeenCalledTimes(1)
  })
})

describe('canOpenNewWindow', () => {
  it('is false when the desktop bridge is absent', () => {
    delete desktopWindow.hermesDesktop
    expect(canOpenNewWindow()).toBe(false)
  })

  it('is false when the bridge lacks openWindow', () => {
    installBridge(vi.fn().mockResolvedValue({ ok: true }))
    expect(canOpenNewWindow()).toBe(false)
  })

  it('is true when the bridge exposes openWindow', () => {
    installBridge(undefined, vi.fn().mockResolvedValue({ ok: true }))
    expect(canOpenNewWindow()).toBe(true)
  })
})

describe('openNewWindow', () => {
  it('no-ops gracefully when the bridge is absent (web fallback)', async () => {
    delete desktopWindow.hermesDesktop

    await openNewWindow()

    expect(notifyError).not.toHaveBeenCalled()
  })

  it('no-ops when openWindow is missing', async () => {
    installBridge(vi.fn().mockResolvedValue({ ok: true }))

    await openNewWindow()

    expect(notifyError).not.toHaveBeenCalled()
  })

  it('invokes the bridge', async () => {
    const openWindow = vi.fn().mockResolvedValue({ ok: true })
    installBridge(undefined, openWindow)

    await openNewWindow()

    expect(openWindow).toHaveBeenCalledTimes(1)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('notifies on an ok:false result', async () => {
    installBridge(undefined, vi.fn().mockResolvedValue({ ok: false, error: 'nope' }))

    await openNewWindow()

    expect(notifyError).toHaveBeenCalledTimes(1)
  })
})

describe('canOpenBrowserWindow', () => {
  it('is false when the desktop bridge is absent', () => {
    delete desktopWindow.hermesDesktop
    expect(canOpenBrowserWindow()).toBe(false)
  })

  it('is false when the bridge lacks openBrowserWindow', () => {
    installBridge(vi.fn().mockResolvedValue({ ok: true }))
    expect(canOpenBrowserWindow()).toBe(false)
  })

  it('is true when the bridge exposes openBrowserWindow', () => {
    installBridge(undefined, undefined, vi.fn().mockResolvedValue({ ok: true }))
    expect(canOpenBrowserWindow()).toBe(true)
  })
})

describe('openBrowserInNewWindow', () => {
  it('returns false without a tab id', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(undefined, undefined, open)

    expect(await openBrowserInNewWindow('')).toBe(false)
    expect(open).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('returns false when the bridge is absent', async () => {
    delete desktopWindow.hermesDesktop

    expect(await openBrowserInNewWindow('tab-1')).toBe(false)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('invokes the bridge with the tab id', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(undefined, undefined, open)

    expect(await openBrowserInNewWindow('tab-1')).toBe(true)
    expect(open).toHaveBeenCalledWith('tab-1')
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('returns false and notifies on an ok:false result', async () => {
    installBridge(undefined, undefined, vi.fn().mockResolvedValue({ ok: false, error: 'invalid-tab-id' }))

    expect(await openBrowserInNewWindow('tab-1')).toBe(false)
    expect(notifyError).toHaveBeenCalledTimes(1)
  })

  it('returns false and notifies when the bridge throws', async () => {
    installBridge(undefined, undefined, vi.fn().mockRejectedValue(new Error('boom')))

    expect(await openBrowserInNewWindow('tab-1')).toBe(false)
    expect(notifyError).toHaveBeenCalledTimes(1)
  })
})
