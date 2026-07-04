import { $kanbanLiveConnected, kanbanLatestEventId, refreshKanbanBoard } from '@/store/kanban'

// Live board updates over the kanban /events WebSocket. The strategy is
// deliberately cheap: any event frame schedules one coalesced board refetch
// (the drawer piggybacks on that — it already reloads when the board's
// latest_event_id moves), so this module never interprets event kinds. While
// the socket is down the view's fast polling is the fallback; the
// $kanbanLiveConnected atom is how the poller knows which cadence to run.

// First retry after 1s, doubling to a 30s cap: quick recovery from a blip
// without hammering a dead backend (polling covers freshness meanwhile).
export const LIVE_RECONNECT_BASE_MS = 1_000
export const LIVE_RECONNECT_MAX_MS = 30_000

// A dispatcher pass appends several events in one poll tick; batch the frames
// into a single refetch instead of one per frame.
export const LIVE_REFRESH_COALESCE_MS = 250

/** Structural WebSocket surface so tests can drive a fake socket. */
export interface KanbanLiveSocket {
  close: () => void
  onclose: (() => void) | null
  onerror: (() => void) | null
  onmessage: ((event: { data: unknown }) => void) | null
  onopen: (() => void) | null
}

export interface KanbanLiveDeps {
  /** Mint the authed WS URL (main process picks token vs ticket). */
  mintWsUrl?: (options: { board: null | string; since: number }) => Promise<string>
  openSocket?: (url: string) => KanbanLiveSocket
  /** Applies an event batch — defaults to the store's board refetch. */
  refresh?: () => void
  setConnected?: (connected: boolean) => void
  getSince?: () => number
}

export interface KanbanLiveHandle {
  stop: () => void
}

/** Open + supervise one /events socket for `board` (null → backend's current
 *  board). Reconnects with capped exponential backoff until stopped. */
export function createKanbanLiveConnection(board: null | string, deps: KanbanLiveDeps = {}): KanbanLiveHandle {
  const mintWsUrl =
    deps.mintWsUrl ??
    (options => {
      const mint = window.hermesDesktop?.getKanbanEventsWsUrl

      if (!mint) {
        // Stale preload without the channel: stay in polling mode.
        return Promise.reject(new Error('kanban events WS bridge unavailable'))
      }

      return mint(options)
    })

  const openSocket = deps.openSocket ?? (url => new WebSocket(url) as unknown as KanbanLiveSocket)
  const refresh = deps.refresh ?? (() => void refreshKanbanBoard())
  const setConnected = deps.setConnected ?? (connected => $kanbanLiveConnected.set(connected))
  const getSince = deps.getSince ?? kanbanLatestEventId

  let stopped = false
  let socket: KanbanLiveSocket | null = null
  let reconnectTimer: null | number = null
  let refreshTimer: null | number = null
  let attempt = 0
  // Bumped on stop so a mint resolving after teardown can't open a socket.
  let generation = 0

  const scheduleRefresh = () => {
    if (stopped || refreshTimer !== null) {
      return
    }

    refreshTimer = window.setTimeout(() => {
      refreshTimer = null
      refresh()
    }, LIVE_REFRESH_COALESCE_MS)
  }

  const scheduleReconnect = () => {
    if (stopped || reconnectTimer !== null) {
      return
    }

    const delay = Math.min(LIVE_RECONNECT_BASE_MS * 2 ** attempt, LIVE_RECONNECT_MAX_MS)

    attempt += 1
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null
      void connect()
    }, delay)
  }

  const handleDown = () => {
    if (stopped) {
      return
    }

    socket = null
    setConnected(false)
    scheduleReconnect()
  }

  async function connect(): Promise<void> {
    if (stopped) {
      return
    }

    const gen = generation
    let url: string

    try {
      url = await mintWsUrl({ board, since: getSince() })
    } catch {
      handleDown()

      return
    }

    if (stopped || gen !== generation) {
      return
    }

    let ws: KanbanLiveSocket

    try {
      ws = openSocket(url)
    } catch {
      handleDown()

      return
    }

    socket = ws

    ws.onopen = () => {
      if (stopped || socket !== ws) {
        return
      }

      attempt = 0
      setConnected(true)
      // Anything that happened while we were minting/reconnecting is not in
      // the stream (since= was sampled earlier) — refetch once to catch up.
      scheduleRefresh()
    }

    ws.onmessage = event => {
      if (stopped || socket !== ws) {
        return
      }

      let frame: { events?: unknown } | null = null

      try {
        frame = JSON.parse(String(event.data)) as { events?: unknown }
      } catch {
        return
      }

      if (Array.isArray(frame?.events) && frame.events.length > 0) {
        scheduleRefresh()
      }
    }

    ws.onclose = () => {
      if (socket === ws) {
        handleDown()
      }
    }

    // The browser always follows error with close; onclose owns recovery.
    ws.onerror = () => undefined
  }

  void connect()

  return {
    stop: () => {
      if (stopped) {
        return
      }

      stopped = true
      generation += 1

      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }

      if (refreshTimer !== null) {
        window.clearTimeout(refreshTimer)
        refreshTimer = null
      }

      const ws = socket

      socket = null

      if (ws) {
        ws.onopen = null
        ws.onmessage = null
        ws.onclose = null
        ws.onerror = null

        try {
          ws.close()
        } catch {
          // already closed
        }
      }

      setConnected(false)
    }
  }
}

// ── Module singleton ────────────────────────────────────────────────────────
// The board view is the only subscriber and there must never be two live
// sockets, so start/stop manage one shared connection (a new start replaces
// the previous one — that's also the board-switch path: close + reopen with
// the new ?board).

let active: KanbanLiveHandle | null = null

export function startKanbanLiveEvents(board: null | string): void {
  active?.stop()
  active = createKanbanLiveConnection(board)
}

export function stopKanbanLiveEvents(): void {
  active?.stop()
  active = null
}
