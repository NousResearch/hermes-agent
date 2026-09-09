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

/** Slack (px) treated as "already at the top edge" for auto Show-earlier. */
export const TOP_EDGE_PX = 48

export interface ShouldAutoShowEarlierInput {
  action: 'dom' | 'window' | null
  isAtBottom: boolean
  loadSettled: boolean
  restorePending: boolean
  scrollTop: number
  topEdgePx?: number
  /** Present only for `wheel` events; omitted for `scroll`. */
  wheelDeltaY?: number
}

/**
 * Auto Show-earlier is the button's `showEarlier()` path, triggered when the
 * reader is at the viewport top and older content exists. Fail-open: missing
 * action, an unsettled load, a pending prepend restore, following the bottom,
 * or a mid-transcript scroll must not page.
 */
export function shouldAutoShowEarlier({
  action,
  isAtBottom,
  loadSettled,
  restorePending,
  scrollTop,
  topEdgePx = TOP_EDGE_PX,
  wheelDeltaY
}: ShouldAutoShowEarlierInput): boolean {
  if (action == null || !loadSettled || restorePending || isAtBottom) {
    return false
  }

  if (scrollTop > topEdgePx) {
    return false
  }

  if (wheelDeltaY !== undefined && !(wheelDeltaY < 0)) {
    return false
  }

  return true
}
