import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import type { PlayerSnapshot } from '@/lib/player-contract'

import { createDesktopMedia } from './desktop-media'

const agent = {
  scope: null,
  linked: false,
  link: vi.fn(async () => 'linked'),
  unlink: vi.fn(async () => 'unlinked'),
  request: vi.fn(async () => ({ operation: 'context' })),
  seedPrompt: vi.fn(() => true),
  setupMediaHelpers: vi.fn(async () => ({ available: true })),
  revealMediaJob: vi.fn(async () => true),
  clearLocalData: vi.fn(async () => closedSnapshot())
}

vi.mock('@/lib/player-agent-bridge', () => ({ usePlayerAgentBridge: () => agent }))
vi.mock('@/components/pane-shell/pane-visibility', () => ({
  isElementInHiddenPane: () => false,
  usePaneVisible: () => true
}))
vi.mock('@/hooks/use-resize-observer', () => ({ useResizeObserver: () => undefined }))

function closedSnapshot(): PlayerSnapshot {
  return {
    status: 'closed',
    guestId: null,
    source: null,
    presentation: 'docked',
    pinned: false,
    pinStatus: 'unknown',
    pinMode: 'unsupported',
    fullscreen: false,
    currentTime: null,
    duration: null,
    paused: null,
    muted: null,
    canGoBack: false,
    canGoForward: false,
    awarenessLinked: false,
    safeOrigin: null,
    runtime: { state: 'not-required' },
    error: null
  }
}

function readySnapshot(overrides: Partial<PlayerSnapshot> = {}): PlayerSnapshot {
  return { ...closedSnapshot(), status: 'ready', guestId: 5, source: 'youtube', ...overrides }
}

const disposers: Array<() => void> = []
const create = (id = 'media-test') => createDesktopMedia(id, fn => disposers.push(fn))
const bridgeState = { stateListener: undefined as ((snapshot: PlayerSnapshot) => void) | undefined }

const installBridge = (request: (command: any) => Promise<PlayerSnapshot>) => {
  bridgeState.stateListener = undefined

  const native = {
    request: vi.fn(request),
    onState: vi.fn((listener: (snapshot: PlayerSnapshot) => void) => {
      bridgeState.stateListener = listener

      return () => {
        bridgeState.stateListener = undefined
      }
    })
  }

  ;(window as any).hermesDesktop = { player: native }

  return native
}

const flushFrame = async () =>
  act(async () => {
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
  })

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  disposers.splice(0).forEach(fn => fn())
  bridgeState.stateListener = undefined
  delete (window as any).hermesDesktop
})

describe('desktop media adapter', () => {
  it('does not expose or create the media adapter when the plugin has not declared it', () => {
    const tracked: Array<() => void> = []
    const ctx = createPluginContext('context-test', fn => tracked.push(fn))

    expect(tracked).toHaveLength(0)
    expect(ctx.media).toBeUndefined()
    expect(tracked).toHaveLength(0)
  })

  it('lazily exposes one memoized adapter when the plugin declares media.player', () => {
    const tracked: Array<() => void> = []
    const ctx = createPluginContext('context-test', fn => tracked.push(fn), ['media.player'])

    expect(tracked).toHaveLength(0)
    expect(ctx.media).toBe(ctx.media)
    expect(tracked).toHaveLength(1)
    tracked.forEach(fn => fn())
  })

  it('feature-detects a missing bridge and rejects untyped, malformed, or scope-forging commands', async () => {
    const noBridge = create()
    const missing = renderHook(() => noBridge.usePlayer())
    expect(missing.result.current.available).toBe(false)
    await expect(missing.result.current.command({ type: 'status' })).rejects.toThrow('unavailable')

    const native = installBridge(async () => closedSnapshot())
    const media = create()
    const { result } = renderHook(() => media.usePlayer())
    const controller = result.current

    const invalid = [
      { type: 'bounds', bounds: {}, visible: true, revision: 1 },
      { type: 'open', source: 'https://evil.example', session_id: 'forged' },
      { type: 'open', source: 'https://youtube.com' },
      { type: 'fullscreen', fullscreen: 'true' },
      { type: 'pin', pinned: false, extra: true },
      { type: 'media-control', action: 'seek', positionSeconds: Number.NaN },
      { type: 'media-control', action: { toString: () => 'play' } },
      { type: 'media-control', action: 'pause', positionSeconds: 3 }
    ]

    for (const command of invalid) {await expect(controller.command(command as any)).rejects.toThrow('Unsupported')}
    expect(native.request).not.toHaveBeenCalled()
  })

  it('keeps closed and guestless players hidden and does not flash over chat', async () => {
    const native = installBridge(async () => closedSnapshot())
    const media = create()
    const view = render(<media.Surface />)
    await waitFor(() => expect(native.request).toHaveBeenCalledWith({ type: 'status' }))
    await flushFrame()
    expect(native.request.mock.calls.filter(([cmd]) => cmd.type === 'bounds' && cmd.visible)).toHaveLength(0)
    view.unmount()
  })

  it('measures a live slot, suppresses it when hidden/occluded, and hides it on release', async () => {
    const native = installBridge(async command => (command.type === 'status' ? readySnapshot() : readySnapshot()))
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () =>
        ({
          x: 12,
          y: 24,
          left: 12,
          top: 24,
          right: 312,
          bottom: 224,
          width: 300,
          height: 200,
          toJSON: () => ({})
        }) as DOMRect
    )
    const media = create()
    const view = render(<media.Surface />)
    await waitFor(() => expect(native.request).toHaveBeenCalledWith({ type: 'status' }))
    await act(async () => {
      bridgeState.stateListener?.(readySnapshot())
    })
    await flushFrame()
    expect(native.request).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'bounds', bounds: { x: 12, y: 24, width: 300, height: 200 }, visible: true })
    )

    view.rerender(<media.Surface hidden />)
    await flushFrame()
    expect(native.request).toHaveBeenCalledWith(expect.objectContaining({ type: 'bounds', visible: false }))
    view.rerender(<media.Surface />)
    await flushFrame()
    const callsBeforeOverlay = native.request.mock.calls.length
    const overlay = window.document.createElement('div')
    overlay.dataset.slot = 'dialog-content'
    window.document.body.append(overlay)
    await waitFor(() =>
      expect(
        native.request.mock.calls.slice(callsBeforeOverlay).some(([cmd]) => cmd.type === 'bounds' && !cmd.visible)
      ).toBe(true)
    )
    overlay.remove()
    view.unmount()
    expect(native.request).toHaveBeenLastCalledWith(
      expect.objectContaining({ type: 'bounds', visible: false, bounds: { x: 0, y: 0, width: 0, height: 0 } })
    )
  })

  it('allows only one simultaneous surface lease per adapter', async () => {
    const native = installBridge(async command => (command.type === 'status' ? readySnapshot() : readySnapshot()))
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () =>
        ({ x: 1, y: 2, left: 1, top: 2, right: 101, bottom: 52, width: 100, height: 50, toJSON: () => ({}) }) as DOMRect
    )
    const media = create()
    const first = render(<media.Surface />)
    const second = render(<media.Surface />)
    await act(async () => {
      bridgeState.stateListener?.(readySnapshot())
    })
    await flushFrame()
    expect(native.request.mock.calls.filter(([cmd]) => cmd.type === 'bounds' && cmd.visible)).toHaveLength(1)
    first.unmount()
    second.unmount()
  })

  it('does not let a disposed old lease hide a new owner, or let retired controllers act', async () => {
    const native = installBridge(async command => (command.type === 'status' ? readySnapshot() : readySnapshot()))
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () =>
        ({ x: 1, y: 2, left: 1, top: 2, right: 101, bottom: 52, width: 100, height: 50, toJSON: () => ({}) }) as DOMRect
    )
    const oldDisposers: Array<() => void> = []
    const old = createDesktopMedia('old', fn => oldDisposers.push(fn))
    const oldView = render(<old.Surface />)
    const hook = renderHook(() => old.usePlayer())
    await act(async () => {
      bridgeState.stateListener?.(readySnapshot())
    })
    await flushFrame()
    oldDisposers.forEach(fn => fn())

    const next = create('next')
    const nextView = render(<next.Surface />)
    await act(async () => {
      bridgeState.stateListener?.(readySnapshot())
    })
    await flushFrame()
    const beforeOldUnmount = native.request.mock.calls.length
    oldView.unmount()
    expect(native.request.mock.calls.slice(beforeOldUnmount).filter(([cmd]) => cmd.type === 'bounds')).toHaveLength(0)
    await expect(hook.result.current.command({ type: 'status' })).rejects.toThrow('disposed')
    await expect(hook.result.current.request({ type: 'context' })).rejects.toThrow('disposed')
    nextView.unmount()
  })

  it('does not let a late status response roll back a newer state event', async () => {
    let resolveStatus!: (snapshot: PlayerSnapshot) => void

    const native = installBridge(command =>
      command.type === 'status'
        ? new Promise(resolve => {
            resolveStatus = resolve
          })
        : Promise.resolve(readySnapshot())
    )

    const media = create()
    const view = render(<media.Surface />)
    const hook = renderHook(() => media.usePlayer())
    await act(async () => {
      bridgeState.stateListener?.(readySnapshot({ currentTime: 70 }))
    })
    await act(async () => {
      resolveStatus(closedSnapshot())
    })
    await waitFor(() => expect(hook.result.current.snapshot?.currentTime).toBe(70))
    view.unmount()
  })
})
