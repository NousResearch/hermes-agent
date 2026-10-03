import { isStableOpen, reconnectBackoffDelayMs } from '@hermes/shared'

import type { HermesTerminalExit, HermesTerminalSession, HermesTerminalState } from '@/global'

import { readJson, writeJson } from './storage'

type TerminalApi = Window['hermesDesktop']['terminal']
type DataListener = Parameters<TerminalApi['onData']>[1]

interface SavedTerminal {
  terminalId: string
  profile: string | null
}

interface BrowserTerminalOptions {
  basePath: string
  currentProfile: () => string | null
  websocketUrl: (profile: string | null) => Promise<string>
  defaultCwd: (profile: string | null) => Promise<string>
}

interface TerminalState {
  id: string
  restoreKey?: string
  profile: string | null
  terminalId?: string
  cwd: string
  shell: string
  cols: number
  rows: number
  socket: WebSocket | null
  generation: number
  stopped: boolean
  status: HermesTerminalState
  retry: number
  retryTimer: number
  openedAt: number | null
  cancelDial?: () => void
  started?: Promise<HermesTerminalSession>
  pendingData: Array<{ data: string; replay: boolean }>
  dataListeners: Set<DataListener>
  exit: HermesTerminalExit | null
  exitListeners: Set<(exit: HermesTerminalExit) => void>
  stateListeners: Set<(state: HermesTerminalState) => void>
}

const META = '\u0000HERMES_TERMINAL_META:'
const MAX_RETRIES = 5
const RETRY_BACKOFF = { baseDelayMs: 500, capMs: 8000 }
const START_TIMEOUT = 10_000
const CLOSE_SIGNALS: Record<number, string> = { 4403: 'denied', 4409: 'superseded', 4410: 'expired', 1013: 'capacity' }

class TerminalConnectionError extends Error {
  constructor(message: string, readonly signal = 'disconnected') {
    super(message)
  }
}

/** Only an attachment that stayed up resets the retry ladder (shared stable-open rule). */
function settleOpen(state: TerminalState): void {
  if (isStableOpen(state.openedAt)) {state.retry = 0}
  state.openedAt = null
}

async function closeSavedShell(saved: SavedTerminal, websocketUrl: BrowserTerminalOptions['websocketUrl']): Promise<boolean> {
  const url = new URL(await websocketUrl(saved.profile))
  url.searchParams.set('attach', saved.terminalId)
  url.searchParams.set('action', 'close')
  const socket = new WebSocket(url)

  return new Promise((resolve, reject) => {
    let acknowledged = false

    const timer = window.setTimeout(() => {
      socket.onclose = null
      socket.onmessage = null
      socket.close(1000, 'close timeout')
      reject(new Error('Terminal close timed out; retry closing this tab'))
    }, START_TIMEOUT)

    socket.onmessage = event => {
      if (typeof event.data !== 'string' || !event.data.startsWith(META)) {return}

      try {
        const meta = JSON.parse(event.data.slice(META.length)) as { closed?: boolean; terminalId?: string }
        acknowledged = meta.closed === true && meta.terminalId === saved.terminalId
      } catch { /* A close is confirmed only by valid metadata and normal closure. */ }
    }

    socket.onerror = () => {
      window.clearTimeout(timer)
      reject(new Error('Terminal close failed to connect; retry closing this tab'))
    }

    socket.onclose = event => {
      window.clearTimeout(timer)

      if (event.code === 1000 && acknowledged) {resolve(true)} else {
        reject(new Error(event.reason || 'Terminal close was not confirmed; retry closing this tab'))
      }
    }
  })
}

interface DialContext {
  websocketUrl: BrowserTerminalOptions['websocketUrl']
  storageKey: (key: string) => string
  isPaused: () => boolean
  stopSocket: (state: TerminalState) => void
  emitData: (state: TerminalState, data: string, replay?: boolean) => void
  setStatus: (state: TerminalState, status: HermesTerminalState) => void
  forget: (state: TerminalState) => void
  finish: (state: TerminalState, signal: string | null) => void
  reconnect: (state: TerminalState) => void
}

async function dialTerminal(state: TerminalState, context: DialContext): Promise<HermesTerminalSession> {
  const { stopSocket, emitData, setStatus, forget, finish, reconnect } = context
  stopSocket(state)
  const generation = ++state.generation
  const url = new URL(await context.websocketUrl(state.profile))

  if (generation !== state.generation || context.isPaused() || state.stopped) {
    throw new TerminalConnectionError('Terminal attachment cancelled')
  }

  url.searchParams.set('cols', String(state.cols))
  url.searchParams.set('rows', String(state.rows))

  if (state.cwd) {url.searchParams.set('cwd', state.cwd)}

  if (state.terminalId) {url.searchParams.set('attach', state.terminalId)}
  const resuming = Boolean(state.terminalId)
  const socket = state.socket = new WebSocket(url)
  socket.binaryType = 'arraybuffer'
  const decoder = new TextDecoder()

  return new Promise((resolve, reject) => {
    let settled = false
    let handshakeComplete = false
    let replayNext = false
    const current = () => state.generation === generation && !context.isPaused() && !state.stopped

    const fail = (error: Error) => {
      if (settled) {return}
      settled = true
      window.clearTimeout(timer)
      state.cancelDial = undefined
      reject(error)
    }

    const timer = window.setTimeout(() => {
      fail(new TerminalConnectionError('Host terminal did not provide startup metadata'))
      socket.onclose = null
      socket.close(1000, 'startup timeout')
    }, START_TIMEOUT)

    state.cancelDial = () => fail(new TerminalConnectionError('Terminal attachment cancelled'))

    const output = (data: string) => {
      if (!current()) {return}
      emitData(state, data, replayNext)
      replayNext = false
    }

    socket.onmessage = event => {
      if (!current()) {return}

      if (typeof event.data === 'string' && event.data.startsWith(META)) {
        try {
          const meta = JSON.parse(event.data.slice(META.length)) as {
            shell?: string; cwd?: string; terminalId?: string; reconnected?: boolean; truncated?: boolean
          }

          if (typeof meta.shell !== 'string' || !meta.shell.trim()) {
            throw new Error('Host terminal returned an empty shell identity')
          }

          if (resuming && meta.terminalId !== state.terminalId) {
            throw new Error('Host terminal did not resume the requested shell')
          }

          state.shell = meta.shell

          if (typeof meta.cwd === 'string') {state.cwd = meta.cwd}

          if (typeof meta.terminalId === 'string' && meta.terminalId) {
            state.terminalId = meta.terminalId
            // The first binary frame is the snapshot, including an empty frame
            // when there is no history. Suppress replies only while parsing it.
            replayNext = true

            if (state.restoreKey) {
              writeJson(context.storageKey(state.restoreKey), { terminalId: state.terminalId, profile: state.profile })
            }

            if (resuming) {
              // Rebuild this xterm from the server tail, never append a second copy.
              state.pendingData = []
              emitData(state, '\u001bc' + (meta.truncated ? '[Earlier terminal output was trimmed from the retained history]\r\n' : ''), true)
            }
          }

          handshakeComplete = true

          // Resume query dimensions do not resize an existing server PTY.
          if (resuming) {socket.send(`\u001b[RESIZE:${state.cols};${state.rows}]`)}
          setStatus(state, 'open')
          state.openedAt = Date.now()

          if (!settled) {
            settled = true
            window.clearTimeout(timer)
            state.cancelDial = undefined
            resolve({ id: state.id, cwd: state.cwd, shell: state.shell, persistent: Boolean(state.terminalId) })
          }
        } catch (error) {
          fail(error instanceof Error ? error : new Error('Host terminal returned invalid startup metadata'))
          socket.close(1000, 'invalid metadata')
        }

        return
      }

      if (typeof event.data === 'string') {
        output(event.data)
      } else if (event.data instanceof Blob) {
        void event.data.arrayBuffer().then(buffer => {
          if (current()) {output(decoder.decode(buffer, { stream: true }))}
        })
      } else {
        output(decoder.decode(event.data as ArrayBuffer, { stream: true }))
      }
    }

    socket.onerror = () => {
      if (!settled) {fail(new TerminalConnectionError('Host terminal WebSocket failed to connect'))}
    }

    socket.onclose = event => {
      if (!current()) {return}
      settleOpen(state)
      const signal = CLOSE_SIGNALS[event.code] ?? 'disconnected'

      if (signal === 'expired') {forget(state)}

      if (!handshakeComplete) {
        fail(new TerminalConnectionError(event.reason || `Host terminal closed (${event.code})`, signal))

        return
      }

      emitData(state, decoder.decode())

      if (!state.terminalId) {
        finish(state, event.code === 4410 ? null : 'disconnected')
      } else if (signal !== 'disconnected') {
        finish(state, signal)
      } else {
        reconnect(state)
      }
    }
  })
}

/** A renderer attachment is disposable; only explicit tab close kills its shell. */
export function createBrowserTerminal(options: BrowserTerminalOptions): TerminalApi {
  const terminals = new Map<string, TerminalState>()
  let sequence = 0
  let paused = false
  const storageKey = (key: string) => `hermes.webapp.terminal.v1:${JSON.stringify([window.location.origin, options.basePath, key])}`

  const load = (key?: string): SavedTerminal | null => {
    const saved = key ? readJson<SavedTerminal>(storageKey(key)) : null

    return saved && typeof saved.terminalId === 'string' && saved.terminalId.length > 0 &&
      (saved.profile === null || typeof saved.profile === 'string') ? saved : null
  }

  const forget = (state: Pick<TerminalState, 'restoreKey'> & { terminalId?: string }) => {
    if (state.restoreKey && (!state.terminalId || load(state.restoreKey)?.terminalId === state.terminalId)) {
      writeJson(storageKey(state.restoreKey), null)
    }
  }

  const setStatus = (state: TerminalState, status: HermesTerminalState) => {
    if (state.status === status) {return}
    state.status = status
    state.stateListeners.forEach(listener => listener(status))
  }

  const emitData = (state: TerminalState, data: string, replay = false) => {
    if (!data) {return}

    if (!state.dataListeners.size) {
      state.pendingData.push({ data, replay })
      // A snapshot is at most 1 MiB; leave space for live output before subscribe.
      let size = state.pendingData.reduce((total, frame) => total + frame.data.length, 0)

      while (size > 2 * 1024 * 1024 && state.pendingData.length > 1) {
        size -= state.pendingData.shift()!.data.length
      }

      return
    }

    state.dataListeners.forEach(listener => replay ? listener(data, { replay }) : listener(data))
  }

  const stopSocket = (state: TerminalState) => {
    ++state.generation
    window.clearTimeout(state.retryTimer)
    settleOpen(state)
    state.cancelDial?.()
    state.cancelDial = undefined
    const socket = state.socket
    state.socket = null

    if (socket) {
      socket.onclose = null
      socket.onerror = null
      socket.onmessage = null
      socket.close(1000, 'detached')
    }
  }

  const finish = (state: TerminalState, signal: string | null) => {
    state.stopped = true
    window.clearTimeout(state.retryTimer)
    setStatus(state, 'disconnected')
    state.exit = { code: null, signal }
    state.exitListeners.forEach(listener => listener(state.exit!))
  }

  const reconnect = (state: TerminalState) => {
    if (paused || state.stopped) {return}

    if (!state.terminalId || state.retry >= MAX_RETRIES) {
      finish(state, 'disconnected')

      return
    }

    setStatus(state, 'reconnecting')
    state.retryTimer = window.setTimeout(() => {
      const attempt = dial(state)
      const generation = state.generation
      void attempt.catch(error => {
        if (paused || state.stopped || generation !== state.generation) {return}
        stopSocket(state)

        if (error instanceof TerminalConnectionError && error.signal !== 'disconnected') {
          finish(state, error.signal)
        } else {
          reconnect(state)
        }
      })
    }, reconnectBackoffDelayMs(state.retry++, RETRY_BACKOFF))
  }

  const dial = (state: TerminalState) => dialTerminal(state, {
    websocketUrl: options.websocketUrl, storageKey, isPaused: () => paused, stopSocket, emitData, setStatus, forget, finish, reconnect
  })

  const detach = async (id: string) => {
    const state = terminals.get(id)

    if (!state) {return false}
    state.stopped = true
    stopSocket(state)
    terminals.delete(id)

    return true
  }

  const dispose = async (id: string) => {
    const state = terminals.get(id)

    if (!state) {return false}
    // Closing a tab during startup must wait for its server-minted identity.
    await state.started?.catch(() => undefined)
    state.stopped = true
    stopSocket(state)

    try {
      if (state.terminalId) {await closeSavedShell({ terminalId: state.terminalId, profile: state.profile }, options.websocketUrl)}
    } catch (error) {
      finish(state, 'disconnected')
      throw error
    }

    forget(state)
    terminals.delete(id)

    return true
  }

  const pause = () => {
    paused = true
    terminals.forEach(state => {
      stopSocket(state)

      if (!state.stopped) {setStatus(state, 'reconnecting')}
    })
  }

  const resume = (event: PageTransitionEvent) => {
    if (!event.persisted || !paused || window.hermesDesktop?.terminal !== api) {return}
    paused = false
    terminals.forEach(state => {
      if (state.stopped) {return}

      if (state.terminalId) {reconnect(state)} else {finish(state, 'disconnected')}
    })
  }

  window.addEventListener('beforeunload', pause)
  window.addEventListener('pagehide', pause)
  window.addEventListener('pageshow', resume)

  const api: TerminalApi = {
    attach: async id => {
      const state = terminals.get(id)

      return Boolean(state && !state.stopped && (state.terminalId || state.socket?.readyState === WebSocket.OPEN))
    },
    // OSC output owns live cwd; returning the launch cwd here would clobber it.
    cwd: async () => null,
    detach,
    dispose,
    closeSaved: async restoreKey => {
      const state = [...terminals.values()].find(state => state.restoreKey === restoreKey)

      if (state) {return dispose(state.id)}
      const saved = load(restoreKey)

      if (saved) {await closeSavedShell(saved, options.websocketUrl)}
      forget({ restoreKey })

      return true
    },
    onData: (id, callback) => {
      const state = terminals.get(id)

      if (!state) {return () => undefined}
      state.dataListeners.add(callback)
      const pending = state.pendingData.splice(0)

      if (pending.length) {queueMicrotask(() => {
        if (state.dataListeners.has(callback)) {pending.forEach(frame => frame.replay ? callback(frame.data, { replay: true }) : callback(frame.data))}
      })}

      return () => { state.dataListeners.delete(callback) }
    },
    onExit: (id, callback) => {
      const state = terminals.get(id)

      if (!state) {return () => undefined}
      state.exitListeners.add(callback)

      if (state.exit) {queueMicrotask(() => { if (state.exitListeners.has(callback)) {callback(state.exit!)} })}

      return () => { state.exitListeners.delete(callback) }
    },
    onState: (id, callback) => {
      const state = terminals.get(id)

      if (!state) {return () => undefined}
      state.stateListeners.add(callback)
      callback(state.status)

      return () => { state.stateListeners.delete(callback) }
    },
    resize: async (id, size) => {
      const state = terminals.get(id)

      if (!state) {return false}
      state.cols = Math.max(1, Math.round(size.cols))
      state.rows = Math.max(1, Math.round(size.rows))

      if (state.status !== 'open' || state.socket?.readyState !== WebSocket.OPEN) {return false}
      state.socket.send(`\u001b[RESIZE:${state.cols};${state.rows}]`)

      return true
    },
    start: async startOptions => {
      const existing = startOptions?.restoreKey
        ? [...terminals.values()].find(state => state.restoreKey === startOptions.restoreKey) : undefined

      if (existing) {
        // StrictMode/remount can overlap startup before a capability is minted.
        // Serialize those starts so a single tab cannot spawn two shells.
        await existing.started?.catch(() => undefined)
        await detach(existing.id)
      }

      const saved = load(startOptions?.restoreKey)

      if (startOptions?.resumeOnly && !saved) {
        throw new TerminalConnectionError('Terminal expired; press Enter to create a new shell', 'expired')
      }

      const state: TerminalState = {
        id: `browser-${Date.now().toString(36)}-${++sequence}`,
        restoreKey: startOptions?.restoreKey,
        profile: saved ? saved.profile : options.currentProfile(),
        terminalId: saved?.terminalId,
        cwd: startOptions?.cwd || '', shell: 'host-shell',
        cols: Math.max(2, Math.round(startOptions?.cols || 80)),
        rows: Math.max(2, Math.round(startOptions?.rows || 24)),
        socket: null, generation: 0, stopped: false, status: 'reconnecting',
        retry: 0, retryTimer: 0, openedAt: null, pendingData: [],
        dataListeners: new Set(), exit: null, exitListeners: new Set(), stateListeners: new Set()
      }

      terminals.set(state.id, state)
      state.started = (async () => {
        if (!state.cwd && !saved) {
          state.cwd = await options.defaultCwd(state.profile).catch(() => '')
        }

        return dial(state)
      })()

      try {
        return await state.started
      } catch (error) {
        stopSocket(state)
        terminals.delete(state.id)
        throw error
      }
    },
    write: async (id, data) => {
      const state = terminals.get(id)

      if (!state || state.stopped || state.status !== 'open' || state.socket?.readyState !== WebSocket.OPEN) {return false}
      state.socket.send(data)

      return true
    }
  }

  return api
}
