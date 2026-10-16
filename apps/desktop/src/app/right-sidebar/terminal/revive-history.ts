import type { IMarker, Terminal } from '@xterm/xterm'

export interface ReviveHistory {
  dispose: () => void
  isReplaying: () => boolean
  liveStartMarker: () => IMarker | undefined
  /** Settles once the live-start boundary is marked. */
  ready: Promise<void>
  /** Idempotent; skipped when the server owns the shell and replays it. */
  restore: () => void
}

// Replay last session's scrollback before the fresh shell boots. The process
// is NOT revived — a new shell starts one line below the restored history.
// A marker at that boundary lets persistence append only new PTY output;
// prior history is never reparsed or rewritten based on text heuristics.
export function createReviveHistory(term: Terminal, reviveBuffer: string, isPersistent: () => boolean): ReviveHistory {
  let liveStartMarker: IMarker | undefined
  let markHistoryReady: () => void = () => undefined
  // Parser replies to queries inside the replayed bytes are not user input.
  let replaying = false

  const historyReady = new Promise<void>(resolve => {
    markHistoryReady = resolve
  })

  const markLiveStart = () => {
    liveStartMarker = term.registerMarker(0)
    markHistoryReady()
  }

  // Browser capability is negotiated by start metadata. Delay local history
  // until then, otherwise the server replay would duplicate it.
  let historyRestored = false

  const restoreHistory = () => {
    if (historyRestored) {return}
    historyRestored = true

    if (reviveBuffer && !isPersistent()) {
      replaying = true
      term.write(reviveBuffer)
      term.write('\r\n', () => { replaying = false; markLiveStart() })
    } else {
      markLiveStart()
    }
  }

  return {
    dispose: () => liveStartMarker?.dispose(),
    isReplaying: () => replaying,
    liveStartMarker: () => liveStartMarker,
    ready: historyReady,
    restore: restoreHistory
  }
}
