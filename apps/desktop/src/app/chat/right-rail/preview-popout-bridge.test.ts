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

      expect(actOnActivePreview).toHaveBeenCalledWith({ kind: 'click', ref: 'btn-1' }, expect.any(AbortSignal))
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

  it('answers only for the session whose tab the pop-out shows (#73890)', async () => {
    isBrowserWindow.mockReturnValue(true)
    actOnActivePreview.mockResolvedValue({ acted: 'click', success: true })
    window.history.replaceState(null, '', '/?win=browser&tab=url:browser-a')
    const { $previewTabs } = await import('@/store/preview')
    const url = (u: string) => ({ kind: 'url' as const, label: u, source: u, url: u })

    $previewTabs.set([
      { id: 'url:browser-a', pinned: false, sessionId: 'sess-a', target: url('https://a.example') },
      { id: 'url:browser-b', pinned: false, sessionId: 'sess-b', target: url('https://b.example') }
    ])

    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    vi.useFakeTimers()

    try {
      // sess-b's agent: the pop-out shows sess-a's tab, so it stays silent.
      const refused = requestPopoutPreviewAct({ kind: 'click', ref: 'btn-1' }, 'sess-b')
      await vi.advanceTimersByTimeAsync(20_100)

      expect(await refused).toMatchObject({ success: false, error: expect.stringContaining('uncertain') })
      expect(actOnActivePreview).not.toHaveBeenCalled()

      // sess-a's agent is answered.
      expect(await requestPopoutPreviewAct({ kind: 'click', ref: 'btn-1' }, 'sess-a')).toEqual({
        acted: 'click',
        success: true
      })
    } finally {
      vi.useRealTimers()
      stop()
      $previewTabs.set([])
      window.history.replaceState(null, '', '/')
    }
  })

  it.each(['abort', 'timeout', 'teardown'])('propagates %s to a pending gesture and ignores its late answer', async mode => {
    vi.useFakeTimers()
    isBrowserWindow.mockReturnValue(true)
    let received: AbortSignal | undefined
    let complete!: (result: unknown) => void
    actOnActivePreview.mockImplementation((_action, signal: AbortSignal) => {
      received = signal

      return new Promise(resolve => { complete = resolve })
    })
    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    const controller = new AbortController()
    const pending = requestPopoutPreviewAct({ kind: 'drag', selector: '#handle', dx: 40, dy: 20 }, undefined, controller.signal)

    try {
      expect(received?.aborted).toBe(false)

      if (mode === 'abort') {controller.abort()}

      if (mode === 'timeout') {await vi.advanceTimersByTimeAsync(20_100)}

      if (mode === 'teardown') {
        stop()
        window.dispatchEvent(new Event('pagehide'))
      }

      expect(received?.aborted).toBe(true)
      expect(await pending).toMatchObject({ success: false })
      complete({ success: true })
      await Promise.resolve()
      expect(await pending).toMatchObject({ success: false })
      expect(actOnActivePreview).toHaveBeenCalledTimes(1)
    } finally {
      complete({ success: false })
      stop()
      vi.useRealTimers()
    }
  })

  it('an already cancelled request never reaches the responder', async () => {
    isBrowserWindow.mockReturnValue(true)
    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    const controller = new AbortController()
    controller.abort()

    try {
      expect(await requestPopoutPreviewAct({ kind: 'drag', selector: '#handle', dx: 1, dy: 0 }, undefined, controller.signal)).toMatchObject({ success: false })
      expect(actOnActivePreview).not.toHaveBeenCalled()
    } finally { stop() }
  })

  it('installs no responder outside the browser pop-out window', async () => {
    isBrowserWindow.mockReturnValue(false)

    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')

    expect(installPopoutPreviewResponder()).not.toThrow()
  })
})
