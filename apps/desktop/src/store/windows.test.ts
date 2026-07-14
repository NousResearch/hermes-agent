import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { canOpenSessionWindow, openNewSessionInNewWindow, openSessionInNewWindow } from './windows'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const notifyError = vi.fn()

vi.mock('./notifications', () => ({
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

function installBridge(
  openSessionWindow?: Window['hermesDesktop']['openSessionWindow'],
  openNewSessionWindow?: Window['hermesDesktop']['openNewSessionWindow']
) {
  desktopWindow.hermesDesktop = {
    ...(openSessionWindow ? { openSessionWindow } : {}),
    ...(openNewSessionWindow ? { openNewSessionWindow } : {})
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

describe('windowProfile', () => {
  const originalSearch = window.location.search

  afterEach(() => {
    // Reset the module-level cache by restoring search; windowProfile caches
    // once, so re-import is needed for a true reset. Prefer spying via history.
    window.history.replaceState({}, '', `${window.location.pathname}${originalSearch || ''}${window.location.hash}`)
  })

  it('reads profile from the secondary window query string', async () => {
    // Force a fresh module so the cache is cold for this search string.
    vi.resetModules()
    window.history.replaceState({}, '', '/?win=secondary&profile=app_factory#/s1')
    const { windowProfile: readProfile } = await import('./windows')

    expect(readProfile()).toBe('app_factory')
  })

  it('returns null when profile is absent', async () => {
    vi.resetModules()
    window.history.replaceState({}, '', '/?win=secondary#/s1')
    const { windowProfile: readProfile } = await import('./windows')

    expect(readProfile()).toBeNull()
  })
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

  it('invokes the bridge with the session id', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)

    await openSessionInNewWindow('s1')

    expect(open).toHaveBeenCalledWith('s1', undefined)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('forwards the watch flag for spectator (subagent) windows', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)

    await openSessionInNewWindow('s1', { watch: true })

    expect(open).toHaveBeenCalledWith('s1', { watch: true })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('forwards the session owning profile for multi-profile New Window', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)

    await openSessionInNewWindow('s1', { profile: 'app_factory' })

    expect(open).toHaveBeenCalledWith('s1', { profile: 'app_factory' })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('forwards profile together with watch', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)

    await openSessionInNewWindow('s1', { watch: true, profile: 'ovnova' })

    expect(open).toHaveBeenCalledWith('s1', { watch: true, profile: 'ovnova' })
  })

  it('omits empty profile from bridge opts', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    installBridge(open)

    await openSessionInNewWindow('s1', { profile: '  ' })

    expect(open).toHaveBeenCalledWith('s1', undefined)
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

describe('openNewSessionInNewWindow', () => {
  it('no-ops gracefully when the bridge is absent (web fallback)', async () => {
    delete desktopWindow.hermesDesktop

    await openNewSessionInNewWindow()

    expect(notifyError).not.toHaveBeenCalled()
  })

  it('no-ops when openNewSessionWindow is missing', async () => {
    installBridge(vi.fn().mockResolvedValue({ ok: true }))

    await openNewSessionInNewWindow()

    expect(notifyError).not.toHaveBeenCalled()
  })

  it('invokes the bridge', async () => {
    const openNew = vi.fn().mockResolvedValue({ ok: true })
    installBridge(vi.fn().mockResolvedValue({ ok: true }), openNew)

    await openNewSessionInNewWindow()

    expect(openNew).toHaveBeenCalledTimes(1)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('notifies on an ok:false result', async () => {
    installBridge(vi.fn().mockResolvedValue({ ok: true }), vi.fn().mockResolvedValue({ ok: false, error: 'nope' }))

    await openNewSessionInNewWindow()

    expect(notifyError).toHaveBeenCalledTimes(1)
  })
})
