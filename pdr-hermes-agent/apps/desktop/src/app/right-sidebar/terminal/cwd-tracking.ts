import type { Terminal } from '@xterm/xterm'
import type { RefObject } from 'react'

import { updateTerminalRestoreCwd } from './terminals'

type TerminalApi = Window['hermesDesktop']['terminal']

// Minimum gap between main-side PTY cwd probes. The probe spawns lsof on macOS,
// so keep it well throttled — cwd only changes on a `cd`, which the reporter
// already reads off the next output snapshot anyway.
const CWD_PROBE_THROTTLE_MS = 2000

// Parse a working directory out of a cwd-reporting OSC payload. Covers OSC 7
// (`file://host/path`, emitted by many bash/zsh integrations) and OSC 9;9
// (`9;<path>`, ConEmu/Windows-Terminal style some PowerShell profiles emit).
// Returns null for anything unrecognized so callers can ignore it.
export function parseOscCwd(code: 7 | 9, payload: string): string | null {
  if (code === 9) {
    // OSC 9;9;<path> — the leading "9;" selects the cwd sub-command.
    if (!payload.startsWith('9;')) {
      return null
    }

    const raw = payload.slice(2).trim().replace(/^"|"$/g, '')

    return raw || null
  }

  // OSC 7 — a file URI. Strip the scheme + authority and percent-decode.
  const match = /^file:\/\/[^/]*(\/.*)$/.exec(payload.trim())

  if (!match) {
    return null
  }

  let raw = match[1]

  try {
    raw = decodeURIComponent(raw)
  } catch {
    // Keep the undecoded path if it isn't valid percent-encoding.
  }

  // Windows file URIs carry a leading slash before the drive (`/C:/Users`).
  const windows = /^\/[A-Za-z]:[\\/]/.exec(raw)

  return (windows ? raw.slice(1) : raw) || null
}

interface TerminalCwdTrackerOptions {
  getSessionId: () => string | null
  /** Latest cwd seen this session; de-dupes redundant store writes and seeds
   *  the cwd a retried shell starts in. */
  lastObservedCwdRef: RefObject<string | null>
  term: Terminal
  terminalApi: TerminalApi
}

// Track the shell's working directory so a reopened tab restarts where the
// user last `cd`'d. Two independent signals feed it: cwd-reporting OSC
// sequences (immediate, for shells configured to emit them) and a periodic
// PTY cwd probe on the main side (shell-agnostic on POSIX). The store
// updater de-dupes, so both feeding it is harmless.
export function trackTerminalCwd(
  id: string,
  { getSessionId, lastObservedCwdRef, term, terminalApi }: TerminalCwdTrackerOptions
): { dispose: () => void; probe: () => void } {
  const recordCwd = (next: string | null | undefined) => {
    const value = (next ?? '').trim()

    if (!value || value === lastObservedCwdRef.current) {
      return
    }

    lastObservedCwdRef.current = value
    updateTerminalRestoreCwd(id, value)
  }

  const cwdOscHandlers = ([7, 9] as const).map(code =>
    term.parser.registerOscHandler(code, payload => {
      recordCwd(parseOscCwd(code, payload))

      return false // let the sequence propagate; we only observe it
    })
  )

  let cwdProbeAt = 0

  const probeCwd = () => {
    const sessionId = getSessionId()

    if (!sessionId || !terminalApi.cwd || Date.now() - cwdProbeAt < CWD_PROBE_THROTTLE_MS) {
      return
    }

    cwdProbeAt = Date.now()
    void terminalApi
      .cwd(sessionId)
      .then(recordCwd)
      .catch(() => {
        // Best-effort: no cwd probe on this platform (e.g. Windows).
      })
  }

  return { dispose: () => cwdOscHandlers.forEach(handler => handler.dispose()), probe: probeCwd }
}
