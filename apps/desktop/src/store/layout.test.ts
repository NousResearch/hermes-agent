import { afterEach, describe, expect, it, vi } from 'vitest'

const PIN_STORAGE_KEY = 'hermes.desktop.pinnedSessions'

type PinnedPayload = { exists: boolean; ids: string[]; path: string }

function flushMicrotasks() {
  return new Promise(resolve => setTimeout(resolve, 0))
}

async function loadLayoutWithPinnedBridge(snapshot: PinnedPayload = { exists: false, ids: [], path: '/tmp/pins.json' }) {
  vi.resetModules()
  window.localStorage.clear()

  let onChanged: ((payload: PinnedPayload) => void) | null = null

  const bridge = {
    get: vi.fn().mockResolvedValue(snapshot),
    set: vi.fn().mockResolvedValue(snapshot),
    onChanged: vi.fn((callback: (payload: PinnedPayload) => void) => {
      onChanged = callback

      return () => {
        onChanged = null
      }
    })
  }

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { pinnedSessions: bridge }
  })

  const layout = await import('./layout')
  await flushMicrotasks()

  return {
    bridge,
    emitPinnedSessionsChanged: (payload: PinnedPayload) => onChanged?.(payload),
    layout
  }
}

afterEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: undefined
  })
  window.localStorage.clear()
  vi.resetModules()
})

describe('external pinned sessions bridge', () => {
  it('imports an existing external pin snapshot and sanitizes duplicate ids', async () => {
    const { bridge, layout } = await loadLayoutWithPinnedBridge({
      exists: true,
      ids: ['alpha', 'beta', 'alpha', '', '  gamma  '],
      path: '/tmp/pins.json'
    })

    expect(bridge.get).toHaveBeenCalledTimes(1)
    expect(bridge.onChanged).toHaveBeenCalledTimes(1)
    expect(layout.$pinnedSessionIds.get()).toEqual(['alpha', 'beta', 'gamma'])
    expect(JSON.parse(window.localStorage.getItem(PIN_STORAGE_KEY) || '[]')).toEqual(['alpha', 'beta', 'gamma'])
    expect(bridge.set).not.toHaveBeenCalled()
  })

  it('mirrors local pin changes back to the external bridge after initial sync', async () => {
    const { bridge, layout } = await loadLayoutWithPinnedBridge({
      exists: true,
      ids: ['alpha'],
      path: '/tmp/pins.json'
    })

    bridge.set.mockClear()
    layout.pinSession('beta')

    expect(layout.$pinnedSessionIds.get()).toEqual(['alpha', 'beta'])
    expect(bridge.set).toHaveBeenCalledWith(['alpha', 'beta'])
  })

  it('applies external change events without writing them back in a loop', async () => {
    const { bridge, emitPinnedSessionsChanged, layout } = await loadLayoutWithPinnedBridge({
      exists: true,
      ids: ['alpha'],
      path: '/tmp/pins.json'
    })

    bridge.set.mockClear()
    emitPinnedSessionsChanged({ exists: true, ids: ['delta', 'delta', 'epsilon'], path: '/tmp/pins.json' })

    expect(layout.$pinnedSessionIds.get()).toEqual(['delta', 'epsilon'])
    expect(bridge.set).not.toHaveBeenCalled()
  })
})
