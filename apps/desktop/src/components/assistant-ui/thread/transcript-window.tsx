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
