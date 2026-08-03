export type ToolTone = 'agent' | 'browser' | 'default' | 'file' | 'image' | 'terminal' | 'web'
export type ToolStatus = 'error' | 'running' | 'success' | 'warning'

export interface ToolPart {
  args?: unknown
  isError?: boolean
  result?: unknown
  toolCallId?: string
  toolName: string
  type: 'tool-call'
}

export interface SearchResultRow {
  snippet: string
  title: string
  url: string
}

export interface ToolTitleAction {
  prefix: string
  suffix: string
  text: string
}

export interface CountMetric {
  count: number
  noun: string
}

export interface ToolView {
  countLabel?: string
  detail: string
  detailLabel: string
  durationLabel?: string
  icon?: string
  imageUrl?: string
  inlineDiff: string
  previewTarget?: string
  /** Set for tools whose output naturally contains ANSI escape codes
   *  (terminal/execute_code) so the renderer knows to run them through
   *  the ANSI parser instead of printing them as literals. */
  rendersAnsi?: boolean
  /** Original query, shown above structured web-search results. */
  searchQuery?: string
  searchHits?: SearchResultRow[]
  /** When the backend reports stderr as a separate stream (terminal /
   *  execute_code), the renderer shows it as its own labeled, neutrally
   *  tinted block under stdout — distinct from an error tone. */
  stderr?: string
  /** Terminal-only command shown as the prompt in the expanded transcript. */
  terminalCommand?: string
  /** Terminal-only process exit code, when the backend reported one. */
  terminalExitCode?: number
  /** When set, the renderer uses stdout+stderr as separate sections and
   *  ignores the merged `detail`. */
  stdout?: string
  status: ToolStatus
  subtitle: string
  title: string
  titleAction?: ToolTitleAction
  tone: ToolTone
}

export interface ToolMeta {
  done: string
  icon?: string
  pending: string
  pendingAction: string
  tone: ToolTone
}

export interface ToolMetaSpec {
  icon?: string
  tone: ToolTone
}

export interface MessageRunningStateSlice {
  message: {
    status?: {
      type?: string
    }
  }
  thread: {
    isRunning: boolean
  }
}
