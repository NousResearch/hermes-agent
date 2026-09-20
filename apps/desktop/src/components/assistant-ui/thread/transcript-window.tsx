import { createContext, type ReactNode, useContext, useMemo } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'

export interface TranscriptWindowValue {
  /** Store holds older messages the runtime window has not materialized. */
  olderAvailable: boolean
  /** Direct row addressing replaces only the display page; null means no jump. */
  revealRow?: (rowId: number, signal: AbortSignal) => Promise<string | null>
  /** Cancel any pending jump and reselect the current live tail. */
  returnToLatest?: () => void
  isHistorical?: boolean
  newerAvailable?: boolean
  /** Exactly the selected bounded source slice, not the full live store. */
  currentMessages?: readonly ChatMessage[]
  /** Pull a page, capturing the reader immediately before the prepend commits.
   * A remote page resolves false on failure; callers can retry without growing
   * an empty render window or holding the reader still during network I/O. */
  expandWindow: (beforePrepend?: () => void) => void | Promise<boolean>
}

const DEFAULT_TRANSCRIPT_WINDOW: Required<TranscriptWindowValue> = {
  olderAvailable: false,
  expandWindow: () => {},
  revealRow: async () => null,
  returnToLatest: () => {},
  isHistorical: false,
  newerAvailable: false,
  currentMessages: []
}

const TranscriptWindowContext = createContext<Required<TranscriptWindowValue>>(DEFAULT_TRANSCRIPT_WINDOW)

export function TranscriptWindowProvider({ children, value }: {
  children: ReactNode
  value: Pick<TranscriptWindowValue, 'olderAvailable' | 'expandWindow'> & Partial<TranscriptWindowValue>
}) {
  const complete = useMemo(() => ({ ...DEFAULT_TRANSCRIPT_WINDOW, ...value }), [value])

  return <TranscriptWindowContext.Provider value={complete}>{children}</TranscriptWindowContext.Provider>
}

export function useTranscriptWindow(): Required<TranscriptWindowValue> {
  return useContext(TranscriptWindowContext)
}

/**
 * "Show earlier" pages the DOM budget first and only then asks the store for
 * more messages — the DOM page is already-materialized content, so spending it
 * first keeps the click cheap and the store window as small as it can be.
 */
export function resolveShowEarlierAction(hiddenCount: number, olderAvailable: boolean): 'dom' | 'window' | null {
  if (hiddenCount > 0) {
    return 'dom'
  }

  return olderAvailable ? 'window' : null
}

/**
 * Slack (px) within which a reader counts as "at the top edge". Wide enough
 * that a wheel notch landing a few pixels short of 0 still pages; well under
 * the RUN_START_SNAP-style thresholds so a mid-transcript reader never does.
 */
export const TOP_EDGE_PX = 48

export interface ShouldAutoShowEarlierInput {
  action: 'dom' | 'window' | null
  isAtBottom: boolean
  loadSettled: boolean
  restorePending: boolean
  scrollTop: number
  /** Present only for `wheel` events; omitted for `scroll`. */
  wheelDeltaY?: number
}

/**
 * Whether reading at the viewport top should page older turns through the
 * same `showEarlier()` path as the button. An unsettled load, a prepend
 * restore still pending, a reader following the bottom, or a mid-transcript
 * scroll must never page on its own — each of those has scrollTop near 0 or
 * changing for reasons that are not "I want to read earlier".
 */
export function shouldAutoShowEarlier({
  action,
  isAtBottom,
  loadSettled,
  restorePending,
  scrollTop,
  wheelDeltaY
}: ShouldAutoShowEarlierInput): boolean {
  if (action == null || !loadSettled || restorePending || isAtBottom || scrollTop > TOP_EDGE_PX) {
    return false
  }

  // A wheel at the clamped top is intent only when it points up.
  return wheelDeltaY === undefined || wheelDeltaY < 0
}
