// Shell notice strings: the remote-display toast, the butterbar, send-outcome toasts and the
// keep-awake offer; `Translations` spreads this in.
export interface NoticeTranslations {
  remoteDisplayBanner: {
    message: (reason: string) => string
  }

  butterbar: {
    goTo: (index: number, total: number) => string
    legal: { before: string; terms: string; between: string; privacy: string; after: string }
  }

  promptNotices: {
    /** An explicit retry of a send whose identityless (legacy) acknowledgement was lost. */
    legacySendUnconfirmed: string
    /** Typed /undo /retry /steer /queue lines (the Ink TUI's wording on the same gateway). */
    sessionCommands: SessionCommandNotices
  }

  /** Once per computer, after a timed sleep interrupted a turn with keep-awake off. */
  keepAwakeOffer: { title: string; message: string; action: string }
}

export interface SessionCommandNotices {
  undoUsage: string
  undoNothing: string
  undidOne: (count: string) => string
  undidOther: (count: string) => string
  retryNothing: string
  retryUnsupported: string
  steerUsage: string
  steerQueued: (preview: string) => string
  steerNoActiveTurn: (preview: string) => string
  steerRejected: string
  queued: (preview: string) => string
  queueFailed: string
  countOne: (count: string) => string
  countOther: (count: string) => string
}
