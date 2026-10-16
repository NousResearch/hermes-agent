import type { DesktopCommandNotices } from './types_desktop_commands'

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
    /** reload.mcp left the chat's servers as they were: a turn was running or waiting. */
    mcpReloadDeferred: string
    /** A reloaded MCP server the chat does not run (no_mcp, an explicit pick, a platform allowlist). */
    mcpReloadNotInChat: (server: string) => string
    /** Configured MCP servers that did not connect on reload. */
    mcpReloadUnavailable: (servers: string) => string
    /** An edit/restore/regenerate whose target compaction archived: the owner refused the rewind. */
    compactedTurnUnavailable: string
    /** The same refusal for any other stale target: undone in another window, not a user turn, or gone. */
    staleTurnUnavailable: string
    /** Typed /undo /retry /steer /queue lines (the Ink TUI's wording on the same gateway). */
    sessionCommands: SessionCommandNotices
    /** Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines. */
    desktopCommands: DesktopCommandNotices
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
  /** A rewind refused while a reply is running: main's stop-first guidance (tui_gateway busy_message). */
  busy: (command: string) => string
  steerUsage: string
  steerQueued: (preview: string) => string
  steerNoActiveTurn: (preview: string) => string
  steerRejected: string
  queued: (preview: string) => string
  queueFailed: string
  countOne: (count: string) => string
  countOther: (count: string) => string
  /** A canonical queued row's edit the owner refused, or whose row started mid-edit. */
  queueEditKeptTitle: string
  queueEditKept: string
  /** A canonical queued row's save whose answer was lost: the edit stays open for a retry. */
  queueEditUnconfirmedTitle: string
  queueEditUnconfirmed: string
}
