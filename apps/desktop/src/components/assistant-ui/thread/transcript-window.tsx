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

export const THREAD_TOP_EDGE_PX = 48

export type AutoShowEarlierInput = {
  atBottom: boolean
  direction: 'up' | 'down' | null
  hasOlderContent: boolean
  loadSettled: boolean
  restorePending: boolean
  scrollTop: number
}

/**
 * Page backward only after the reader deliberately reaches the transcript's
 * top edge. This is shared by scroll and wheel listeners: a wheel at a
 * clamped `scrollTop === 0` does not emit a scroll event.
 */
export function shouldAutoShowEarlier({
  atBottom,
  direction,
  hasOlderContent,
  loadSettled,
  restorePending,
  scrollTop
}: AutoShowEarlierInput): boolean {
  return (
    loadSettled &&
    !restorePending &&
    !atBottom &&
    hasOlderContent &&
    direction === 'up' &&
    scrollTop <= THREAD_TOP_EDGE_PX
  )
}
