// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  createKanbanLiveConnection,
  type KanbanLiveHandle,
  type KanbanLiveSocket,
  LIVE_RECONNECT_BASE_MS,
  LIVE_RECONNECT_MAX_MS,
  LIVE_REFRESH_COALESCE_MS
} from './live-events'

class FakeSocket implements KanbanLiveSocket {
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: unknown }) => void) | null = null
  onopen: (() => void) | null = null
  closed = false

  close(): void {
    this.closed = true
  }

  open(): void {
    this.onopen?.()
  }

  frame(events: unknown[]): void {
    this.onmessage?.({ data: JSON.stringify({ events, cursor: 99 }) })
  }

  drop(): void {
    this.onclose?.()
  }
}

describe('createKanbanLiveConnection', () => {
  let sockets: FakeSocket[]
  let mintCalls: Array<{ board: null | string; since: number }>
  let refresh: ReturnType<typeof vi.fn<() => void>>
  let setConnected: ReturnType<typeof vi.fn<(connected: boolean) => void>>
  let handle: KanbanLiveHandle | null
  let since: number

  const deps = () => ({
    mintWsUrl: (options: { board: null | string; since: number }) => {
      mintCalls.push(options)

      return Promise.resolve(`ws://test/api/plugins/kanban/events?token=t&since=${options.since}`)
    },
    openSocket: () => {
      const socket = new FakeSocket()

      sockets.push(socket)

      return socket
    },
    refresh,
    setConnected,
    getSince: () => since
  })

  // Let the async mint promise inside connect() settle without advancing timers.
  const settle = async () => {
    await Promise.resolve()
    await Promise.resolve()
  }

  beforeEach(() => {
    vi.useFakeTimers()
    sockets = []
    mintCalls = []
    refresh = vi.fn<() => void>()
    setConnected = vi.fn<(connected: boolean) => void>()
    handle = null
    since = 17
  })

  afterEach(() => {
    handle?.stop()
    vi.useRealTimers()
  })

  it('mints with the store cursor + board and reports connected on open', async () => {
    handle = createKanbanLiveConnection('ops', deps())
    await settle()

    expect(mintCalls).toEqual([{ board: 'ops', since: 17 }])
    expect(sockets).toHaveLength(1)

    sockets[0].open()
    expect(setConnected).toHaveBeenCalledWith(true)

    // Catch-up refetch for the mint→open gap, coalesced like event frames.
    vi.advanceTimersByTime(LIVE_REFRESH_COALESCE_MS)
    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('coalesces a burst of event frames into one refetch', async () => {
    handle = createKanbanLiveConnection(null, deps())
    await settle()
    sockets[0].open()
    vi.advanceTimersByTime(LIVE_REFRESH_COALESCE_MS)
    refresh.mockClear()

    sockets[0].frame([{ id: 18 }])
    sockets[0].frame([{ id: 19 }, { id: 20 }])
    sockets[0].frame([{ id: 21 }])
    expect(refresh).not.toHaveBeenCalled()

    vi.advanceTimersByTime(LIVE_REFRESH_COALESCE_MS)
    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('ignores empty and malformed frames', async () => {
    handle = createKanbanLiveConnection(null, deps())
    await settle()
    sockets[0].open()
    vi.advanceTimersByTime(LIVE_REFRESH_COALESCE_MS)
    refresh.mockClear()

    sockets[0].frame([])
    sockets[0].onmessage?.({ data: 'not json' })
    vi.advanceTimersByTime(LIVE_REFRESH_COALESCE_MS)
    expect(refresh).not.toHaveBeenCalled()
  })

  it('flags disconnected on close (poll fallback) and reconnects with capped backoff', async () => {
    handle = createKanbanLiveConnection('ops', deps())
    await settle()
    sockets[0].open()
    setConnected.mockClear()
    since = 23

    sockets[0].drop()
    expect(setConnected).toHaveBeenCalledWith(false)

    // First retry after the base delay, minted with the CURRENT cursor.
    vi.advanceTimersByTime(LIVE_RECONNECT_BASE_MS)
    await settle()
    expect(sockets).toHaveLength(2)
    expect(mintCalls[1]).toEqual({ board: 'ops', since: 23 })

    // Repeated failures double the delay up to the cap.
    for (let i = 0; i < 8; i += 1) {
      sockets[sockets.length - 1].drop()
      await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_MAX_MS)
    }

    expect(sockets.length).toBeGreaterThan(2)

    const before = sockets.length

    sockets[sockets.length - 1].drop()
    // At the cap: no reconnect just before 30s…
    await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_MAX_MS - 1)
    expect(sockets).toHaveLength(before)
    // …and exactly one at 30s.
    await vi.advanceTimersByTimeAsync(1)
    expect(sockets).toHaveLength(before + 1)
  })

  it('resets the backoff after a successful open', async () => {
    handle = createKanbanLiveConnection(null, deps())
    await settle()
    sockets[0].open()

    sockets[0].drop()
    await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_MAX_MS)
    sockets[1].open()

    sockets[1].drop()
    // Back to the base delay, not the doubled one.
    await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_BASE_MS)
    expect(sockets).toHaveLength(3)
  })

  it('retries when minting the URL fails', async () => {
    let fail = true
    handle = createKanbanLiveConnection(null, {
      ...deps(),
      mintWsUrl: options => {
        mintCalls.push(options)

        if (fail) {
          return Promise.reject(new Error('backend down'))
        }

        return Promise.resolve('ws://test')
      }
    })
    await settle()

    expect(sockets).toHaveLength(0)
    expect(setConnected).toHaveBeenCalledWith(false)

    fail = false
    await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_BASE_MS)
    expect(mintCalls).toHaveLength(2)
    expect(sockets).toHaveLength(1)
  })

  it('stop() closes the socket, clears timers, and never reconnects', async () => {
    handle = createKanbanLiveConnection(null, deps())
    await settle()
    sockets[0].open()
    sockets[0].frame([{ id: 18 }])

    handle.stop()
    handle = null

    expect(sockets[0].closed).toBe(true)
    expect(setConnected).toHaveBeenLastCalledWith(false)

    // Neither the pending coalesced refresh nor any reconnect survives stop.
    refresh.mockClear()
    await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_MAX_MS * 4)
    expect(refresh).not.toHaveBeenCalled()
    expect(sockets).toHaveLength(1)
  })

  it('stop() during an in-flight mint prevents a late socket from opening', async () => {
    let resolveMint: ((url: string) => void) | undefined
    handle = createKanbanLiveConnection(null, {
      ...deps(),
      mintWsUrl: () =>
        new Promise<string>(resolve => {
          resolveMint = resolve
        })
    })

    handle.stop()
    handle = null
    resolveMint!('ws://test')
    await settle()

    expect(sockets).toHaveLength(0)
  })

  it('a socket replaced mid-reconnect does not double-handle close', async () => {
    handle = createKanbanLiveConnection(null, deps())
    await settle()
    sockets[0].open()
    sockets[0].drop()
    await vi.advanceTimersByTimeAsync(LIVE_RECONNECT_BASE_MS)
    expect(sockets).toHaveLength(2)
    setConnected.mockClear()

    // The dead first socket firing close again must not touch state.
    sockets[0].drop()
    expect(setConnected).not.toHaveBeenCalled()
  })
})
