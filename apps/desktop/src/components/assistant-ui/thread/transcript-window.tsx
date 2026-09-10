import { createContext, type ReactNode, useContext } from 'react'

export interface TranscriptWindowValue {
  /** Store holds older messages the runtime window has not materialized. */
  olderAvailable: boolean
  /** Pull one more page of older messages out of the session store. */
  expandWindow: () => void
}

const TranscriptWindowContext = createContext<TranscriptWindowValue>({
  olderAvailable: false,
  expandWindow: () => {}
})

export function TranscriptWindowProvider({ children, value }: { children: ReactNode; value: TranscriptWindowValue }) {
  return <TranscriptWindowContext.Provider value={value}>{children}</TranscriptWindowContext.Provider>
}

export function useTranscriptWindow(): TranscriptWindowValue {
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
