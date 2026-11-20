import { describe, expect, it, vi } from 'vitest'

import { dispatchPluginNativeNotification } from '@/store/native-notifications'

import { emitGatewayEvent } from './events'
import { createPluginContext } from './plugin'

vi.mock('@/store/native-notifications', () => ({ dispatchPluginNativeNotification: vi.fn() }))

describe('createPluginContext.onDispose', () => {
  it('collects arbitrary cleanups so the host runs them on deactivate', () => {
    const disposers: Array<() => void> = []
    const ctx = createPluginContext('demo', dispose => disposers.push(dispose))

    let cleaned = false
    ctx.onDispose(() => {
      cleaned = true
    })

    // The cleanup is tracked alongside contribution/socket disposers, so the
    // loader's deactivate (which runs every collected disposer) tears it down.
    expect(disposers).toHaveLength(1)
    disposers.forEach(dispose => dispose())
    expect(cleaned).toBe(true)
  })
})

describe('createPluginContext.onEvent', () => {
  it('retires the gateway listener with the plugin, even when subscribed after register()', () => {
    // Runtime plugins are re-imported as a fresh module on every hot reload; a
    // gateway subscription the loader cannot see outlives its incarnation and
    // one event then fires once per reload (#112366). The ctx door is tracked.
    const disposers: Array<() => void> = []
    const ctx = createPluginContext('demo', dispose => disposers.push(dispose))
    const seen: string[] = []

    // Deliberately outside any loader register() scope — the late-subscription case.
    ctx.onEvent('gateway.reconnecting', event => seen.push(event.type))
    emitGatewayEvent({ type: 'gateway.reconnecting', payload: { attempt: 1 } } as never)
    expect(seen).toEqual(['gateway.reconnecting'])

    disposers.forEach(dispose => dispose())
    emitGatewayEvent({ type: 'gateway.reconnecting', payload: { attempt: 2 } } as never)
    expect(seen).toEqual(['gateway.reconnecting'])
  })
})

describe('createPluginContext.os', () => {
  it('dispatches a native notification attributed to the plugin', () => {
    const ctx = createPluginContext('demo')
    ctx.os.notify({ body: 'b', title: 't' })
    expect(dispatchPluginNativeNotification).toHaveBeenCalledWith('demo', { body: 'b', title: 't' })
  })

  it('resolves false (never throws) when the desktop bridge is missing', async () => {
    const ctx = createPluginContext('demo')

    // jsdom has no window.hermesDesktop — the exact older-shell/browser case.
    await expect(ctx.os.openExternal('https://example.com')).resolves.toBe(false)
    await expect(ctx.os.revealPath('/tmp')).resolves.toBe(false)
    await expect(ctx.os.writeClipboard('hi')).resolves.toBe(false)
    // The pickers answer with a path, so their "unavailable" is null.
    await expect(ctx.os.pickSavePath()).resolves.toBeNull()
    await expect(ctx.os.pickOpenPath()).resolves.toBeNull()
  })

  it('file pickers return the chosen path, and null on cancel', async () => {
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue(['/tmp/board.tar.gz']),
      selectSavePath: vi.fn().mockResolvedValue('/tmp/out.tar.gz')
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickSavePath({ title: 'Save' })).resolves.toBe('/tmp/out.tar.gz')
      expect(bridge.selectSavePath).toHaveBeenCalledWith({ title: 'Save' })

      await expect(ctx.os.pickOpenPath({ title: 'Open' })).resolves.toBe('/tmp/board.tar.gz')
      expect(bridge.selectPaths).toHaveBeenCalledWith({ multiple: false, title: 'Open' })

      // Cancel: the save dialog resolves null, the open dialog an empty list.
      bridge.selectSavePath.mockResolvedValue(null)
      bridge.selectPaths.mockResolvedValue([])
      await expect(ctx.os.pickSavePath()).resolves.toBeNull()
      await expect(ctx.os.pickOpenPath()).resolves.toBeNull()
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('pickOpenPath answers null in browser-hosted Desktop, and stages nothing', async () => {
    // The browser bridge has no OS dialog: its picker POSTs the file to
    // /api/chat/file-upload and resolves the copy the gateway staged inside the
    // Hermes home. Handing that back as "the chosen path" is a pick that never
    // happened — plugin backends refuse a path inside the home (#156470), and
    // every attempt left another staged copy behind. The capability is
    // unavailable there, which is the documented null.
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue(['/home/u/.hermes/uploads/web-fcd110fdaff0a5df-hk_tasks.csv'])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickOpenPath({ filters: [{ extensions: ['csv'], name: 'CSV' }] })).resolves.toBeNull()
      expect(bridge.selectPaths).not.toHaveBeenCalled()
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
      delete document.documentElement.dataset.hermesDesktopHost
    }
  })

  it('pickBrowserFile resolves the staged copy and the name the file had', async () => {
    // The honest null above leaves a browser plugin with no way to reach a
    // picked file at all. This door answers what a browser pick actually is:
    // a copy the gateway staged in the Hermes home, named as the user's file
    // was. A plugin opts into it deliberately instead of pattern-matching
    // <home>/uploads/web-<hex>-* (#156470).
    const bridge = {
      getStagedFileDisplayName: vi.fn().mockReturnValue('hk_tasks.csv'),
      selectPaths: vi.fn().mockResolvedValue(['/home/u/.hermes/uploads/web-fcd110fdaff0a5df-hk_tasks.csv'])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickBrowserFile?.({ filters: [{ extensions: ['csv'], name: 'CSV' }] })).resolves.toEqual({
        name: 'hk_tasks.csv',
        path: '/home/u/.hermes/uploads/web-fcd110fdaff0a5df-hk_tasks.csv'
      })
      expect(bridge.selectPaths).toHaveBeenCalledWith({ filters: [{ extensions: ['csv'], name: 'CSV' }], multiple: false })
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
      delete document.documentElement.dataset.hermesDesktopHost
    }
  })

  it('pickBrowserFile answers null in the native app, where pickOpenPath is the door', async () => {
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue(['/Users/u/Downloads/hk_tasks.csv'])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      // No staging happens in the native shell, so there is nothing for this
      // door to describe: the two doors each answer only where they apply.
      await expect(ctx.os.pickBrowserFile?.()).resolves.toBeNull()
      expect(bridge.selectPaths).not.toHaveBeenCalled()
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('pickBrowserFile answers null on cancel, and falls back to the staged name', async () => {
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue([])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickBrowserFile?.()).resolves.toBeNull()

      // An older browser bridge has no display-name lookup, so the staging
      // prefix is stripped here: the user is shown the name their file had,
      // never the `web-<hex>-` one the gateway wrote.
      bridge.selectPaths.mockResolvedValue(['/home/u/.hermes/uploads/web-fcd110fdaff0a5df-hk_tasks.csv'])
      await expect(ctx.os.pickBrowserFile?.()).resolves.toEqual({
        name: 'hk_tasks.csv',
        path: '/home/u/.hermes/uploads/web-fcd110fdaff0a5df-hk_tasks.csv'
      })
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
      delete document.documentElement.dataset.hermesDesktopHost
    }
  })

  it('pickBrowserFile rejects when the upload fails, so a failure is not a cancel', async () => {
    // The bridge rejects for reasons the user did not choose: over the 16 MiB
    // browser cap, a 413 from the gateway, a dropped connection. Flattening
    // those to null would make them read as "no file chosen", which is how a
    // failed import becomes a button that does nothing.
    const bridge = {
      selectPaths: vi.fn().mockRejectedValue(new Error('The file is larger than 16 MiB, the largest file the browser can attach.'))
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickBrowserFile?.()).rejects.toThrow('larger than 16 MiB')
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
      delete document.documentElement.dataset.hermesDesktopHost
    }
  })

  it('file pickers degrade to null on an older shell that lacks them', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}

    try {
      const ctx = createPluginContext('demo')
      await expect(ctx.os.pickSavePath()).resolves.toBeNull()
      await expect(ctx.os.pickOpenPath()).resolves.toBeNull()
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('routes through the bridge and turns a bridge throw into false', async () => {
    const bridge = {
      openExternal: vi.fn().mockResolvedValue(undefined),
      revealPath: vi.fn().mockResolvedValue(true),
      writeClipboard: vi.fn().mockRejectedValue(new Error('nope'))
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')
      await expect(ctx.os.openExternal('https://example.com')).resolves.toBe(true)
      expect(bridge.openExternal).toHaveBeenCalledWith('https://example.com')
      await expect(ctx.os.revealPath('/tmp')).resolves.toBe(true)
      await expect(ctx.os.writeClipboard('hi')).resolves.toBe(false)
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })
})
