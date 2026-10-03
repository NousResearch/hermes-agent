import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as WindowsStore from '@/store/windows'

import { typeText } from './preview-drive'

const target = vi.hoisted(() => ({
  windowId: 'workspace-a',
  tabId: 'url:second',
  selectionVersion: 2,
  owner: { connectionId: 'a', profile: 'alpha', scope: 'alpha', destination: null,
    conversation: { kind: 'session' as const, id: 'stored-a', connectionId: 'a', profile: 'alpha' } }
}))

const selectedPopoutTarget = vi.hoisted(() => vi.fn((_requester?: unknown, _ids?: readonly string[]): typeof target | null => target))
const validPopoutTarget = vi.hoisted(() => vi.fn((value: unknown) => JSON.stringify(value) === JSON.stringify(target)))
vi.mock('@/store/browser-workspaces', () => ({ selectedPopoutTarget, validPopoutTarget }))

const isBrowserWindow = vi.hoisted(() => vi.fn(() => false))
const actOnActivePreview = vi.hoisted(() => vi.fn())
const readActivePreview = vi.hoisted(() => vi.fn())
const activePreviewScriptRunner = vi.hoisted(() => vi.fn(() => null))
const activePreviewNav = vi.hoisted(() => vi.fn(() => null))

function withDeadline<T extends { id: string }>(packet: T, deadline = Date.now() + 20_000) {
  return { ...packet, id: `${packet.id}-${deadline}`, deadline }
}

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
    isBrowserWindow: () => isBrowserWindow(),
    windowBrowserWorkspaceId: () => isBrowserWindow() ? target.windowId : null
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
  it('bounds settled request history without replaying unexpired IDs when a long-lived responder fills up', async () => {
    vi.useFakeTimers()
    isBrowserWindow.mockReturnValue(true)
    actOnActivePreview.mockResolvedValue({ success: true })
    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')
    let stop = installPopoutPreviewResponder()
    const bus = new LoopbackChannel('hermes:preview-popout')

    const packet = (id: string) => withDeadline({ id, kind: 'act', target, payload: { kind: 'click' },
      tabIds: [target.tabId] })

    try {
      for (let batch = 0; batch < 3; batch++) {
        actOnActivePreview.mockClear()
        const first = packet(`first-${batch}`)
        bus.postMessage(first)
        await Promise.resolve()

        for (let index = 1; index < 1024; index++) {
          bus.postMessage(packet(`${batch}-${index}`))
          await Promise.resolve()
        }

        expect(actOnActivePreview).toHaveBeenCalledTimes(1024)
        bus.postMessage(packet(`overflow-${batch}`))
        expect(actOnActivePreview).toHaveBeenCalledTimes(1024)
        // Remounting the responder cannot open a replay window either.
        stop()
        stop = installPopoutPreviewResponder()
        bus.postMessage(first)
        await vi.advanceTimersByTimeAsync(19_999)
        bus.postMessage(first)
        expect(actOnActivePreview).toHaveBeenCalledTimes(1024)
        await vi.advanceTimersByTimeAsync(1)
        bus.postMessage(first)
        bus.postMessage({ ...first, deadline: Date.now() + 20_000 })
        expect(actOnActivePreview).toHaveBeenCalledTimes(1024)
      }
    } finally {stop(); vi.useRealTimers()}
  })

  it('never invokes a reader for an expired or unbounded request deadline', async () => {
    vi.useFakeTimers()
    isBrowserWindow.mockReturnValue(true)
    readActivePreview.mockResolvedValue({ text: 'page' })
    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    const bus = new LoopbackChannel('hermes:preview-popout')

    try {
      for (const deadline of [Date.now(), Date.now() - 1, Date.now() + 8_001, Infinity, NaN]) {
        bus.postMessage({ id: `invalid-${deadline}-${deadline}`, kind: 'read', target, payload: {}, tabIds: [target.tabId], deadline })
      }

      expect(readActivePreview).not.toHaveBeenCalled()
      expect(vi.getTimerCount()).toBe(0)
    } finally {stop(); vi.useRealTimers()}
  })

  beforeEach(async () => {
    const { $previewTabs } = await import('@/store/preview')
    $previewTabs.set([{ id: 'url:second', pinned: true, sessionId: 'stored-a', target: {
      kind: 'url', label: 'Second', source: 'https://example.test', url: 'https://example.test'
    } }])
  })
  it.each(['abort', 'deadline', 'teardown'] as const)('stops stable-target native typing on %s, never falling back or reviving a replay', async reason => {
    vi.useFakeTimers()
    isBrowserWindow.mockReturnValue(true)
    const input = { focus: vi.fn(), send: vi.fn() }
    let responderSignal: AbortSignal | undefined
    actOnActivePreview.mockImplementation(async (action, signal) => {
      responderSignal = signal
      await typeText(input, action.text, signal)

      return { success: !signal?.aborted }
    })
    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    const controller = new AbortController()

    try {
      const pending = requestPopoutPreviewAct({ kind: 'type', text: 'x'.repeat(10_000) }, target.owner.conversation, controller.signal)
      await vi.advanceTimersByTimeAsync(100)
      expect(input.send).toHaveBeenCalled()
      expect(responderSignal?.aborted).toBe(false)

      if (reason === 'abort') {controller.abort('interrupted')}

      if (reason === 'deadline') {await vi.advanceTimersByTimeAsync(20_000)}

      if (reason === 'teardown') {stop()}
      expect(responderSignal?.aborted).toBe(true)
      const sent = input.send.mock.calls.length
      await vi.advanceTimersByTimeAsync(21_000)
      expect(input.send).toHaveBeenCalledTimes(sent)
      expect(await pending).toMatchObject({ success: false })
      expect(validPopoutTarget(target)).toBe(true)
      expect(actOnActivePreview).toHaveBeenCalledOnce()
      expect(vi.getTimerCount()).toBe(0)
    } finally {stop(); vi.useRealTimers()}
  })

  it('matches cancellation to the captured target even after invalidation and keeps terminal replay tombstones', async () => {
    vi.useFakeTimers()
    isBrowserWindow.mockReturnValue(true)
    let signal: AbortSignal | undefined
    let finish!: () => void
    actOnActivePreview.mockImplementation((_action, current) => {
      signal = current

      return new Promise(resolve => {finish = () => resolve({ success: false })})
    })
    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    const bus = new LoopbackChannel('hermes:preview-popout')
    const packet = withDeadline({ id: 'cancel-original', kind: 'act', target, payload: { kind: 'type' }, tabIds: [target.tabId] })

    try {
      bus.postMessage(packet)
      bus.postMessage({ kind: 'cancel', id: 'other', target })
      bus.postMessage({ kind: 'cancel', id: packet.id, target: { ...target, selectionVersion: 99 } })
      expect(signal?.aborted).toBe(false)
      validPopoutTarget.mockReturnValue(false)
      bus.postMessage({ kind: 'cancel', id: packet.id, target })
      expect(signal?.aborted).toBe(true)
      finish()
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(31_000)
      validPopoutTarget.mockReturnValue(true)
      bus.postMessage(packet)
      expect(actOnActivePreview).toHaveBeenCalledOnce()
      expect(vi.getTimerCount()).toBe(0)
      const input = { focus: vi.fn(), send: vi.fn() }
      actOnActivePreview.mockImplementation(async (action, current) => {
        await typeText(input, action.text, current)

        return { success: true }
      })
      bus.postMessage({ ...packet, id: `expired-in-transit-${packet.deadline}`, payload: { kind: 'type', text: 'too late' } })
      await vi.advanceTimersByTimeAsync(0)
      expect(input.send).not.toHaveBeenCalled()
      expect(vi.getTimerCount()).toBe(0)
    } finally {stop(); vi.useRealTimers()}
  })

  afterEach(() => {
    LoopbackChannel.listeners.clear()
    vi.resetModules()
    selectedPopoutTarget.mockReturnValue(target)
    validPopoutTarget.mockImplementation(value => JSON.stringify(value) === JSON.stringify(target))
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

      expect(actOnActivePreview).toHaveBeenCalledWith({ kind: 'click', ref: 'btn-1' }, expect.any(AbortSignal), expect.objectContaining({ sessionId: 'stored-a' }), {
        tabId: target.tabId,
        valid: expect.any(Function)
      })
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

        expect(readActivePreview).toHaveBeenCalledWith({ count: 10, start: 0 }, expect.objectContaining({ sessionId: 'stored-a' }), target.tabId, [target.tabId])
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

  it('returns a failure instead of allowing a timed-out mutation to fall through to another local tab', async () => {
    vi.useFakeTimers()

    try {
      const { requestPopoutPreviewAct } = await import('./preview-popout-bridge')
      const pending = requestPopoutPreviewAct({ kind: 'click', ref: 'button' })
      await vi.advanceTimersByTimeAsync(20_100)
      expect(await pending).toMatchObject({ success: false, error: expect.stringContaining('not redirected') })
      expect(LoopbackChannel.listeners.get('hermes:preview-popout')?.size).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('shares one responder and does not execute a duplicate request twice', async () => {
    isBrowserWindow.mockReturnValue(true)
    actOnActivePreview.mockResolvedValue({ success: true })
    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')
    const first = installPopoutPreviewResponder()
    const second = installPopoutPreviewResponder()
    const bus = new LoopbackChannel('hermes:preview-popout')
    const packet = withDeadline({ id: 'duplicate', kind: 'act', payload: { kind: 'click' }, target, tabIds: [target.tabId] })
    bus.postMessage(packet)
    bus.postMessage(packet)
    await Promise.resolve()
    expect(actOnActivePreview).toHaveBeenCalledOnce()
    first()
    expect(LoopbackChannel.listeners.get('hermes:preview-popout')?.size).toBe(1)
    second()
    second()
    expect(LoopbackChannel.listeners.get('hermes:preview-popout')?.size).toBe(0)
  })

  it('rejects wrong-target packets before invoking a reader or action', async () => {
    isBrowserWindow.mockReturnValue(true)
    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    const bus = new LoopbackChannel('hermes:preview-popout')

    for (const badTarget of [
      undefined,
      { ...target, windowId: 'other' },
      { ...target, tabId: 'url:seed' },
      { ...target, selectionVersion: 1 }
    ]) {
      bus.postMessage(withDeadline({ id: 'bad', kind: 'act', payload: { kind: 'click' }, target: badTarget, tabIds: [target.tabId] }))
    }

    await Promise.resolve()
    expect(actOnActivePreview).not.toHaveBeenCalled()
    expect(readActivePreview).not.toHaveBeenCalled()
    stop()
  })

  it('passes a live target guard so a switched/closed tab cannot receive later input', async () => {
    isBrowserWindow.mockReturnValue(true)
    actOnActivePreview.mockImplementation(async (_action, _signal, _owner, pinned) => {
      expect(pinned.valid()).toBe(true)
      validPopoutTarget.mockReturnValue(false)
      expect(pinned.valid()).toBe(false)

      return { success: false, error: 'target retired' }
    })
    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    expect(await requestPopoutPreviewAct({ kind: 'type', text: 'safe fixture' })).toMatchObject({ success: false })
    stop()
  })

  it('answers only for the session whose tab the pop-out shows (#73890)', async () => {
    isBrowserWindow.mockReturnValue(true)
    actOnActivePreview.mockResolvedValue({ acted: 'click', success: true })
    window.history.replaceState(null, '', '/')
    const { $previewTabs } = await import('@/store/preview')
    const url = (u: string) => ({ kind: 'url' as const, label: u, source: u, url: u })

    $previewTabs.set([
      { id: 'url:second', pinned: false, sessionId: 'sess-a', target: url('https://a.example') },
      { id: 'url:browser-b', pinned: false, sessionId: 'sess-b', target: url('https://b.example') }
    ])

    const { installPopoutPreviewResponder, requestPopoutPreviewAct } = await import('./preview-popout-bridge')
    const stop = installPopoutPreviewResponder()
    vi.useFakeTimers()

    try {
      selectedPopoutTarget.mockImplementation((_requester, ids) => ids?.includes(target.tabId) ? target : null)
      // sess-b's agent cannot select sess-a's tab.
      const refused = requestPopoutPreviewAct({ kind: 'click', ref: 'btn-1' }, { ...target.owner.conversation, id: 'sess-b' }, undefined, 'sess-b')
      await vi.advanceTimersByTimeAsync(20_100)

      expect(await refused).toBeNull()
      expect(actOnActivePreview).not.toHaveBeenCalled()

      // sess-a's agent is answered.
      expect(await requestPopoutPreviewAct({ kind: 'click', ref: 'btn-1' }, { ...target.owner.conversation, id: 'sess-a' }, undefined, 'sess-a')).toEqual({
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

  it('installs no responder outside the browser pop-out window', async () => {
    isBrowserWindow.mockReturnValue(false)

    const { installPopoutPreviewResponder } = await import('./preview-popout-bridge')

    expect(installPopoutPreviewResponder()).not.toThrow()
  })
})
