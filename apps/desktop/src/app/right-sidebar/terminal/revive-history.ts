import type { Terminal } from '@xterm/xterm'

export interface ReviveHistory {
  isReplaying: () => boolean
  /** Settles once the restored history (if any) has been written. */
  ready: Promise<void>
  /** Idempotent; skipped when the server owns the shell and replays it. */
  restore: () => void
}

// Replay last session's scrollback before the fresh shell boots. The process
// is NOT revived — a new shell starts one line below the restored history.
// Stripping the boot gap still applies to the live shell output that follows,
// so the fresh prompt lands flush under the restored block.
export function createReviveHistory(term: Terminal, reviveBuffer: string, isPersistent: () => boolean): ReviveHistory {
  let markHistoryReady: () => void = () => undefined
  // Parser replies to queries inside the replayed bytes are not user input.
  let replaying = false

  const historyReady = new Promise<void>(resolve => {
    markHistoryReady = resolve
  })

  // Browser capability is negotiated by start metadata. Delay local history
  // until then, otherwise the server replay would duplicate it.
  let historyRestored = false

  const restoreHistory = () => {
    if (historyRestored) {
      return
    }

    historyRestored = true

    if (reviveBuffer && !isPersistent()) {
      replaying = true
      term.write(reviveBuffer)
      term.write('\r\n', () => {
        replaying = false
        markHistoryReady()
      })
    } else {
      markHistoryReady()
    }
  }

  return {
    isReplaying: () => replaying,
    ready: historyReady,
    restore: restoreHistory
  }
}
