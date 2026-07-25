import { createContext, useContext } from 'react'

import type { SearchField } from '@/lib/search-match'

/** Per-session search UI metadata for sidebar results. */
export interface SessionSearchMeta {
  /** Best-matching field id. */
  field: SearchField
  /** Localized short label for the chip (标题 / 正文 / …). */
  fieldLabel: string
  /** Optional ranges into the displayed title string. */
  titleRanges?: Array<[number, number]>
  /** Full tooltip: marked snippet or field value. */
  tooltip?: string
  /** Raw snippet with FTS markers when match is body text. */
  markedSnippet?: string
}

export const SessionSearchMetaContext = createContext<Map<string, SessionSearchMeta>>(new Map())

export function useSessionSearchMeta(sessionId: string): SessionSearchMeta | undefined {
  return useContext(SessionSearchMetaContext).get(sessionId)
}
