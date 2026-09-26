import type { Terminal } from '@xterm/xterm'
import type { RefObject } from 'react'

import type { HermesTerminalSession } from '@/global'

import { $terminals, markTerminalPersistent, removeExitedTerminal } from './terminals'

type TerminalApi = Window['hermesDesktop']['terminal']

export type TerminalStatus = 'closed' | 'open' | 'starting' | 'reconnecting'

// Status line written when a persistent PTY stream ends, keyed by exit signal.
const TERMINAL_EXIT_MESSAGES: Record<string, string> = {
  disconnected: 'Terminal disconnected. Press Enter to reconnect to the same shell.',
  expired: 'Terminal expired or exited. Press Enter to create a new shell.',
  superseded: 'Terminal is attached in another window. Press Enter to take it back.',
  denied: 'Terminal access denied. Check your sign-in and profile, then press Enter to retry.',
  capacity: 'Terminal capacity reached. Close another terminal, then press Enter to retry.'
}

// True once the page/app is tearing down (Cmd+Q, Alt+F4, window close, reload).
// App quit kills the PTYs from the main process, which fires onExit in the
// renderer — but React skips effect cleanups on teardown, so the per-instance
// `disposed` flag never flips. Without this guard those teardown exits would call
// closeTerminal() and wipe the persisted terminal list right before relaunch
// reads it. A real `exit`/Ctrl-D still closes the tab (flag stays false).
let appTearingDown = false

if (typeof window !== 'undefined') {
  const markTearingDown = () => {
    appTearingDown = true
  }

  window.addEventListener('pagehide', markTearingDown)
  window.addEventListener('beforeunload', markTearingDown)
  window.addEventListener('pageshow', () => { appTearingDown = false })
}

interface SessionAttemptOptions {
  isDisposed: () => boolean
  /** True while restored local history is parsing; its query replies must not reach the shell. */
  isReplayingHistory: () => boolean
  /** The attempt adopted a live session: seed size, shell name and selection. */
  onStarted: (session: HermesTerminalSession) => void
  /** Cwd for the next shell; read at each start so a retry follows a `cd`. */
  resolveCwd: () => string
  /** Runs once start metadata says whether the server replays history itself. */
  restoreHistory: () => void
  sessionIdRef: RefObject<string | null>
  setStatus: (status: TerminalStatus) => void
  term: Terminal
  terminalApi: TerminalApi
  /** Live output of a renderer-owned (non-persistent) shell. */
  writeOutput: (data: string) => void
}

export interface SessionAttemptController {
  /** Keystrokes and xterm replies from `term.onData`. */
  input: (data: string) => void
  /** Whether the current shell is server-owned (replayed and resumed by the server). */
  isPersistent: () => boolean
  /** Release the current attempt: unsubscribe, then detach or dispose its shell. */
  release: () => void
  start: () => void
}

// One shell attachment per terminal tab at a time. Each start supersedes the
// previous attempt; an attempt that fails, disconnects or loses its persistent
// stream arms Enter to start (or resume) the next one in the same scrollback.
export function createSessionAttemptController(
  id: string,
  {
    isDisposed,
    isReplayingHistory,
    onStarted,
    resolveCwd,
    restoreHistory,
    sessionIdRef,
    setStatus,
    term,
    terminalApi,
    writeOutput
  }: SessionAttemptOptions
): SessionAttemptController {
  let persistent = Boolean($terminals.get().find(tab => tab.id === id)?.persistent)
  let resumeOnly = persistent
  let replayWrites = 0
  let retrySession: (() => void) | null = null
  let cleanupAttempt: (() => void) | null = null

  const startSession = () => {
    cleanupAttempt?.()
    let current = true
    let attemptSessionId: string | null = null

    const releaseSession = (sid: string) => persistent && terminalApi.detach
      ? terminalApi.detach(sid) : terminalApi.dispose(sid)

    const subscriptions: Array<() => void> = []

    const release = () => {
      current = false
      subscriptions.splice(0).forEach(unsubscribe => unsubscribe())

      if (attemptSessionId) {
        const sid = attemptSessionId
        attemptSessionId = null

        if (sessionIdRef.current === sid) {
          sessionIdRef.current = null
        }

        void releaseSession(sid)
      }
    }

    cleanupAttempt = release

    void terminalApi
      // Prefer the last observed cwd so retry/relaunch stays in the same directory.
      .start({ cols: term.cols, cwd: resolveCwd(), rows: term.rows, restoreKey: id, resumeOnly })
      .then(async session => {
        persistent = Boolean(session.persistent)
        resumeOnly = persistent

        if (persistent) {markTerminalPersistent(id)}

        if (isDisposed() || !current) {
          void releaseSession(session.id)

          return
        }

        restoreHistory()
        attemptSessionId = session.id
        sessionIdRef.current = session.id
        onStarted(session)

        subscriptions.push(
          terminalApi.onData(session.id, (data, options) => {
            if (!current || isDisposed()) {return}

            if (options?.replay) {
              ++replayWrites
              term.write(data, () => { --replayWrites })
            } else if (persistent) {
              term.write(data)
            } else {
              writeOutput(data)
            }
          }),
          terminalApi.onExit(session.id, exit => {
            if (!current || isDisposed() || appTearingDown) {
              return
            }

            release()

            if (persistent && exit.signal) {
              setStatus('closed')
              resumeOnly = exit.signal !== 'expired'
              retrySession = startSession
              term.write(`\r\n${TERMINAL_EXIT_MESSAGES[exit.signal] || 'Terminal disconnected. Press Enter to reconnect.'}\r\n`)

              return
            }

            if (exit.signal === 'disconnected') {
              setStatus('closed')
              retrySession = startSession
              term.write('\r\nTerminal disconnected. Press Enter to start a new shell; scrollback is preserved.\r\n')

              return
            }

            // Only a current process exit removes the persisted tab.
            removeExitedTerminal(id)
          })
        )

        if (persistent && terminalApi.onState) {
          subscriptions.push(terminalApi.onState(session.id, state => {
            if (!current || isDisposed() || appTearingDown) {return}
            setStatus(state === 'disconnected' ? 'closed' : state)

            if (state === 'reconnecting') {
              term.write('\r\nTerminal disconnected. Reconnecting to the same shell…\r\n')
            }
          }))
        }

        // onExit may replay a buffered exit before returning its unsubscribe.
        if (!current) {
          release()

          return
        }

        const attached = await terminalApi.attach(session.id)

        if (!attached) {
          throw new Error('Terminal session disappeared before its output stream attached')
        }

        if (isDisposed() || !current) {
          return
        }

        if (!persistent) {setStatus('open')}

        window.requestAnimationFrame(() => {
          if (current && !isDisposed()) {
            term.clearSelection()
          }
        })
      })
      .catch(error => {
        if (isDisposed() || !current) {
          return
        }

        release()
        retrySession = startSession
        setStatus('closed')
        const expired = error && typeof error === 'object' && 'signal' in error && error.signal === 'expired'

        if (expired) {
          resumeOnly = false
          term.write(`${TERMINAL_EXIT_MESSAGES.expired}\r\n`)
        } else {
          term.write(`Terminal failed to start: ${error instanceof Error ? error.message : String(error)}. Press Enter to retry.\r\n`)
        }
      })
  }

  const input = (data: string) => {
    if (replayWrites || isReplayingHistory()) {return}
    const sessionId = sessionIdRef.current

    if (!sessionId && retrySession && data === '\r') {
      const retry = retrySession
      retrySession = null
      setStatus('starting')
      retry()

      return
    }

    if (sessionId) {
      void terminalApi.write(sessionId, data)
    }
  }

  return {
    input,
    isPersistent: () => persistent,
    release: () => cleanupAttempt?.(),
    start: startSession
  }
}
