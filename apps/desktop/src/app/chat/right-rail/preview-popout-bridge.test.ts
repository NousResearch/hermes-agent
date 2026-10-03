import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as WindowsStore from '@/store/windows'

const isBrowserWindow = vi.hoisted(() => vi.fn(() => false))
const actOnActivePreview = vi.hoisted(() => vi.fn())
const readActivePreview = vi.hoisted(() => vi.fn())
const activePreviewScriptRunner = vi.hoisted(() => vi.fn(() => null))
const activePreviewNav = vi.hoisted(() => vi.fn(() => null))

type Listener = (event: MessageEvent) => void

/** Same-origin BroadcastChannel never delivers to the posting window, so the
 *  unit test needs a bus that fans out to every subscriber including the sender. */
class LoopbackChannel {
  static listeners = new Map<string, Set<Listener>>()

  name: string

  constructor(name: string) {
    this.name = name
    const set = LoopbackChannel.listeners.get(name) ?? new Set()
    LoopbackChannel.listeners.set(this.name, set)
  }

  addEventListener(_type: 'message', listener: Listener) {
    const set = LoopbackChannel.listeners.get(this.name) ?? new Set()
    set.add(listener)
    LoopbackChannel.listeners.set(this.name, set)
  }

  removeEventListener(_type: 'message', listener: Listener) {
    LoopbackChannel.listeners.get(this.name)?.delete(listener)
  }

  postMessage(data: unknown) {
    const snapshot = [...(LoopbackChannel.listeners.get(this.name) ?? [])]

    for (const listener of snapshot) {
      listener({ data } as MessageEvent)
    }
  }

  close() {}
}

vi.stubGlobal('BroadcastChannel', LoopbackChannel)

vi.mock('@/store/windows', async importOriginal => {
  const actual = await importOriginal<typeof WindowsStore>()

  return {
    ...actual,
    isBrowserWindow: () => isBrowserWindow()
  }
})

vi.mock('./preview-act', () => ({
  actOnActivePreview: (...args: unknown[]) => actOnActivePreview(...args)
}))

vi.mock('./preview-reader', () => ({
  readActivePreview: (...args: unknown[]) => readActivePreview(...args)
}))

vi.mock('./preview-script-runner', () => ({
  activePreviewScriptRunner: () => activePreviewScriptRunner()
}))

vi.mock('./preview-nav', () => ({
  activePreviewNav: () => activePreviewNav()
}))

/** Loopback stand-in for the preload IPC relay (window.hermesDesktop.windowRelay). */
function installDesktopRelay() {
  const listeners = new Set<(payload: unknown) => void>()

  const desktopWindow = window as unknown as { hermesDesktop?: Record<string, unknown> }

  desktopWindow.hermesDesktop = {
    windowRelay: {
      onMessage: (callback: (payload: unknown) => void) => {
        listeners.add(callback)

        return () => listeners.delete(callback)
      },
      send: (payload: unknown) => {
        for (const listener of [...listeners]) {
          listener(payload)
        }
      }
    }
  }

  return () => {
    delete desktopWindow.hermesDesktop
    listeners.clear()
  }
}

describe('preview pop-out bridge', () => {
  afterEach(() => {
    LoopbackChannel.listeners.clear()
    vi.resetModules()
    isBrowserWindow.mockReturnValue(false)
    actOnActivePreview.mockReset()
    readActivePreview.mockReset()
    activePreviewScriptRunner.mockReturnValue(null)
    activePreviewNav.mockReturnValue(null)
  })

  it('reports a live surface when a script runner is registered', { timeout: 60_000 }, async () => {
    activePreviewScriptRunner.mockReturnValue((async () => null) as never)
    const { hasLivePreviewSurface } = await import('./preview-popout-bridge')

    expect(hasLivePreviewSurface()).toBe(true)
  })

  it('round-trips an act request to the browser pop-out responder', async () => {
    isBrowserWindow.mockReturnValue(true)
    actOnActivePreview.mockResolvedValue({ acted: 'click', success: true })

    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()

    try {
      const result = await requestPopoutPreviewAct({ kind: 'click', ref: 'btn-1' })

      expect(actOnActivePreview).toHaveBeenCalledWith({ kind: 'click', ref: 'btn-1' })
      expect(result).toEqual({ acted: 'click', success: true })
    } finally {
      stop()
    }
  })

  it('round-trips a read request over the preload IPC relay', async () => {
    isBrowserWindow.mockReturnValue(true)
    readActivePreview.mockResolvedValue({ kind: 'url', text: 'page' })
    const removeRelay = installDesktopRelay()

    try {
      const { installPopoutPreviewResponder, requestPopoutPreviewRead } = await import('./preview-popout-bridge')
      const stop = installPopoutPreviewResponder()

      try {
        const result = await requestPopoutPreviewRead({ count: 10, start: 0 })

        expect(readActivePreview).toHaveBeenCalledWith({ count: 10, start: 0 })
        expect(result).toEqual({ kind: 'url', text: 'page' })
      } finally {
        stop()
      }
    } finally {
      removeRelay()
    }
  })

  it('resolves null when no pop-out answers within the timeout', async () => {
    vi.useFakeTimers()
    readActivePreview.mockResolvedValue(null)

    try {
      const { requestPopoutPreviewRead } = await import('./preview-popout-bridge')
      const pending = requestPopoutPreviewRead({})

      await vi.advanceTimersByTimeAsync(8_100)

      expect(await pending).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('installs no responder outside the browser pop-out window', async () => {
    isBrowserWindow.mockReturnValue(false)

    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')

    expect(installPopoutPreviewResponder()).not.toThrow()
  })
})
