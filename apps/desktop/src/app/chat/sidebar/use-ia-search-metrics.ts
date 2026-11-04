import { useEffect, useRef } from 'react'

import { captureIaRecorder, recordIaAction } from '@/store/ia-metrics'

/** Measure settled searches, never keystrokes or query text. Counts include local matches. */
export function useIaSearchMetrics(query: string, pending: boolean, failed: boolean, count: number): void {
  const landedAt = useRef(Date.now())
  const previous = useRef('')
  const reported = useRef('')
  // Visit/query history, not a mirror of current reactive state.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (!query && previous.current) {
      recordIaAction('ia.search.cleared')
    }

    previous.current = query

    if (!query) {
      reported.current = ''

      return
    }

    if (pending || reported.current === query) {
      return
    }

    const record = captureIaRecorder()

    const timer = window.setTimeout(() => {
      reported.current = query
      record('ia.search.used')

      if (Date.now() - landedAt.current < 10_000) {
        record('ia.search.early')
      }

      record(failed ? 'ia.search.partial_failure' : count ? 'ia.search.results' : 'ia.search.empty')
    }, 500)

    return () => window.clearTimeout(timer)
  }, [query, pending, failed, count])
}
