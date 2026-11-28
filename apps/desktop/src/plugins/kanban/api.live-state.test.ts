import { type PluginStorage, queryClient } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

// The board header must tell a live event stream from a
// polling fallback and from silence. The chip state is derived from two
// clocks — the transport (socket open) and freshness (frames / poll answers)
// — and the fallback poll cadence follows the same verdict. These tests pin
// the pure derivation, then the bindApi wiring: frame → live, dropped socket
// → polling, a sleeping/background window (keepalives and polls silent) →
// offline, a landing poll → back to polling, and the resync invalidations on
// window return / focus / network return.

const {
  $lastSyncAt,
  $liveChannel,
  $liveState,
  LIVE_POLL_INTERVAL_MS,
  POLL_FALLBACK_INTERVAL_MS,
  bindApi,
  boardKey,
  boardKeyPrefix,
  boardRefetchInterval,
  boardsKey,
  deriveLiveState
} = await import('./api')

const { setConnection } = await import('@/store/session')

const BOARD = (latest_event_id: number) => ({
  assignees: [],
  columns: [],
  latest_event_id,
  now: 0,
  tenants: []
})

const storage = (slug = ''): PluginStorage => ({
  get: <T>(key: string, fallback: T) => (key === 'boardSlug' ? (slug as T) : fallback),
  remove: vi.fn(),
  set: vi.fn()
})

const refuseRest = async <T>(path: string): Promise<T> => {
  throw new Error(`cached snapshot must not refetch: ${path}`)
}

afterEach(() => {
  setConnection(null)
  vi.useRealTimers()
  vi.restoreAllMocks()
  queryClient.clear()
  $liveChannel.set('polling')
  $liveState.set('polling')
  $lastSyncAt.set(null)
})

describe('kanban live chip derivation', () => {
  it('runs the fallback poll cadence when the socket is down, the slow heartbeat when live', () => {
    expect(boardRefetchInterval('live')).toBe(LIVE_POLL_INTERVAL_MS)
    expect(boardRefetchInterval('polling')).toBe(POLL_FALLBACK_INTERVAL_MS)
    expect(boardRefetchInterval('offline')).toBe(POLL_FALLBACK_INTERVAL_MS)
    expect(POLL_FALLBACK_INTERVAL_MS).toBeLessThan(LIVE_POLL_INTERVAL_MS)
  })

  it('derives live/polling/offline from the transport and freshness clocks', () => {
    const now = 1_000_000

    // Boot claims nothing: no backend answer yet.
    expect(deriveLiveState({ channel: 'polling', lastFrameAt: null, lastSyncAt: null }, now)).toBe('polling')

    // Open channel with fresh frames (keepalive every 25 s) is live.
    expect(deriveLiveState({ channel: 'live', lastFrameAt: now - 5_000, lastSyncAt: now - 5_000 }, now)).toBe('live')

    // Three missed keepalives demote even an open channel — polls carry it.
    expect(deriveLiveState({ channel: 'live', lastFrameAt: now - 76_000, lastSyncAt: now - 9_000 }, now)).toBe('polling')

    // A dropped channel never claims live, no matter how fresh the last frame was.
    expect(deriveLiveState({ channel: 'polling', lastFrameAt: now - 1_000, lastSyncAt: now - 1_000 }, now)).toBe('polling')

    // Silence past the fallback bar (3 × 10 s + grace) is offline.
    expect(deriveLiveState({ channel: 'polling', lastFrameAt: null, lastSyncAt: now - 36_000 }, now)).toBe('offline')
    expect(deriveLiveState({ channel: 'polling', lastFrameAt: null, lastSyncAt: now - 34_000 }, now)).toBe('polling')

    // While live, the slow heartbeat (60 s + grace) sets the silence bar.
    expect(deriveLiveState({ channel: 'live', lastFrameAt: now - 66_000, lastSyncAt: now - 66_000 }, now)).toBe('offline')
  })
})

describe('kanban bindApi live wiring', () => {
  const dial = () => {
    let onMessage: (data: unknown) => void = () => undefined
    let onStatus: (status: 'live' | 'polling') => void = () => undefined

    const socket = vi.fn(
      (
        _path: string,
        message: (data: unknown) => void,
        status?: (state: 'live' | 'polling') => void
      ) => {
        onMessage = message
        onStatus = status ?? onStatus

        return vi.fn()
      }
    )

    return { onFrame: (data: unknown) => onMessage(data), onStatus: (s: 'live' | 'polling') => onStatus(s), socket }
  }

  it('reports the socket status and flips the chip on frames', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { onFrame, onStatus, socket } = dial()
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(10))

    const dispose = bindApi(refuseRest, storage('ops'), socket)
    await vi.advanceTimersByTimeAsync(0)

    expect(socket).toHaveBeenCalledWith('/events?board=ops&since=10', expect.any(Function), expect.any(Function))
    expect($liveState.get()).toBe('polling')

    onStatus('live')
    onFrame({ cursor: 10, events: [] })
    expect($liveState.get()).toBe('live')
    expect($lastSyncAt.get()).not.toBeNull()

    // Socket dropped — polling carries the view, shown at once.
    onStatus('polling')
    expect($liveState.get()).toBe('polling')

    dispose()
  })

  it('goes offline when a sleeping/background window hears nothing (fake timers)', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { onFrame, onStatus, socket } = dial()
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(10))

    const dispose = bindApi(refuseRest, storage('ops'), socket)
    await vi.advanceTimersByTimeAsync(0)

    onStatus('live')
    onFrame({ cursor: 10, events: [] })
    expect($liveState.get()).toBe('live')

    // Sleep / covered window: no keepalive frames and no poll answers land.
    await vi.advanceTimersByTimeAsync(80_000)
    expect($liveState.get()).toBe('offline')

    // The fallback poll landing proves reachability again — polling, not live:
    // the frame clock is stale, the socket has to re-prove itself.
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(11))
    expect($lastSyncAt.get()).not.toBeNull()
    expect($liveState.get()).toBe('polling')

    dispose()
  })

  it('invalidates the board cache on window return, focus and network return', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { socket } = dial()
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(10))

    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')
    const dispose = bindApi(refuseRest, storage('ops'), socket)
    await vi.advanceTimersByTimeAsync(0)
    expect(invalidate).not.toHaveBeenCalled()

    const visible = () => {
      Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'visible' })
      document.dispatchEvent(new Event('visibilitychange'))
    }

    visible()
    expect(invalidate).toHaveBeenCalledWith({ queryKey: boardKeyPrefix('local') })
    expect(invalidate).toHaveBeenCalledWith({ queryKey: boardsKey('local') })

    window.dispatchEvent(new Event('focus'))
    window.dispatchEvent(new Event('online'))
    expect(invalidate).toHaveBeenCalledTimes(6)

    // A hidden visibility flip must not thrash the backend.
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' })
    document.dispatchEvent(new Event('visibilitychange'))
    expect(invalidate).toHaveBeenCalledTimes(6)

    dispose()
  })

  it('resets the live atoms on dispose', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { onFrame, onStatus, socket } = dial()
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(10))

    const dispose = bindApi(refuseRest, storage('ops'), socket)
    await vi.advanceTimersByTimeAsync(0)
    onStatus('live')
    onFrame({ cursor: 10, events: [] })
    expect($liveState.get()).toBe('live')

    dispose()
    expect($liveChannel.get()).toBe('polling')
    expect($liveState.get()).toBe('polling')
    expect($lastSyncAt.get()).toBeNull()
  })

  // After the socket (re)opens, the chip must not wait for the first keepalive
  // (up to 25 s): the handshake answer itself proves freshness.
  it('counts an opened socket as fresh — live before the first keepalive', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    const { onStatus, socket } = dial()
    queryClient.setQueryData(boardKey('local', 'ops', false), BOARD(10))

    const dispose = bindApi(refuseRest, storage('ops'), socket)
    await vi.advanceTimersByTimeAsync(0)
    expect($liveState.get()).toBe('polling')

    onStatus('live')
    expect($liveState.get()).toBe('live')
    expect($lastSyncAt.get()).not.toBeNull()

    dispose()
  })
})
