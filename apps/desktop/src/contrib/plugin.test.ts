import { describe, expect, it, vi } from 'vitest'

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
    await expect(ctx.os.pickOpenPaths()).resolves.toEqual([])
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

  it('pickOpenPaths returns every selected path and enables native multiple selection', async () => {
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue(['/tmp/source-a.docx', '/tmp/source-b.pdf'])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickOpenPaths({ directories: true, title: 'Choose sources' })).resolves.toEqual([
        '/tmp/source-a.docx',
        '/tmp/source-b.pdf'
      ])
      expect(bridge.selectPaths).toHaveBeenCalledWith({
        directories: true,
        multiple: true,
        title: 'Choose sources'
      })
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('pickOpenPath selects a directory when requested', async () => {
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue(['/tmp/output'])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickOpenPath({ directories: true, title: 'Choose output folder' })).resolves.toBe('/tmp/output')
      expect(bridge.selectPaths).toHaveBeenCalledWith({
        directories: true,
        multiple: false,
        title: 'Choose output folder'
      })
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('pickOpenPaths returns an empty list when the native bridge fails', async () => {
    const bridge = {
      selectPaths: vi.fn().mockRejectedValue(new Error('native dialog failed'))
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickOpenPaths()).resolves.toEqual([])
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('pickOpenPaths returns an empty list when the user cancels', async () => {
    const bridge = {
      selectPaths: vi.fn().mockResolvedValue([])
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickOpenPaths()).resolves.toEqual([])
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('pickOpenPaths filters malformed bridge results', async () => {
    const bridge = {
      selectPaths: vi
        .fn()
        .mockResolvedValueOnce(['/tmp/source.docx', '', null, 42])
        .mockResolvedValueOnce(null)
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      const ctx = createPluginContext('demo')

      await expect(ctx.os.pickOpenPaths()).resolves.toEqual(['/tmp/source.docx'])
      await expect(ctx.os.pickOpenPaths()).resolves.toEqual([])
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
      await expect(ctx.os.pickOpenPaths()).resolves.toEqual([])
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
