import { describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { dispatchPluginNativeNotification } from '@/store/native-notifications'

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

describe('createPluginContext.mediaUrl', () => {
  it('returns null when the desktop media bridge is unavailable', () => {
    const ctx = createPluginContext('studio-rail')

    expect(ctx.mediaUrl('/outputs/clip/stream')).toBeNull()
  })

  it('creates a confined plugin-media URL scoped to the active profile and registry connection', () => {
    const host = window as unknown as { hermesDesktop?: unknown }

    host.hermesDesktop = {}
    setApiRequestProfile('video')
    setApiRequestConnection('mac-mini')

    try {
      const ctx = createPluginContext('studio-rail')

      expect(ctx.mediaUrl('/outputs/clip 1/stream')).toBe(
        'hermes-media://plugin/studio-rail/outputs/clip%201/stream?profile=video&connectionId=mac-mini'
      )
    } finally {
      setApiRequestProfile(null)
      setApiRequestConnection(null)
      delete host.hermesDesktop
    }
  })

  it('rejects encoded traversal in a media path', () => {
    const host = window as unknown as { hermesDesktop?: unknown }

    host.hermesDesktop = {}

    try {
      const ctx = createPluginContext('studio-rail')

      expect(() => ctx.mediaUrl('/outputs/%2e%2e/stream')).toThrow('illegal path traversal')
      expect(() => ctx.mediaUrl('/outputs/%252e%252e/stream')).toThrow('illegal path traversal')
    } finally {
      delete host.hermesDesktop
    }
  })

  it('rejects a plugin id that could escape its namespace', () => {
    const host = window as unknown as { hermesDesktop?: unknown }

    host.hermesDesktop = {}

    try {
      const ctx = createPluginContext('../other-plugin')

      expect(() => ctx.mediaUrl('/outputs/clip/stream')).toThrow('invalid plugin id')
      expect(() => createPluginContext('%252e%252e').mediaUrl('/outputs/clip/stream')).toThrow('invalid plugin id')
    } finally {
      delete host.hermesDesktop
    }
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
