import { useEffect } from 'react'

import { sessionTileDelegate } from '@/store/session-states'
import { rewindTranscriptTail, type TranscriptProfileScope, transcriptTailState } from '@/store/transcript-tail'

import { boundRetainedTranscript } from './transcript-retention'

interface TranscriptRetentionOptions {
  /** First message of the live window; null while the window covers the whole
   *  in-memory transcript (nothing to release). */
  anchorId: null | string
  /** False for a static history page or a suppressed transcript — neither is
   *  the live session store this hook bounds. */
  enabled: boolean
  profile?: TranscriptProfileScope
  runtimeId: null | string
  storedSessionId: null | string
}

/**
 * Release the store's paged-through history once it is off the live window.
 *
 * Runs when the window's own cut moves — the only moment the set of rows older
 * than the window changes (streaming grows the tail, "Show earlier" prepends, a
 * re-cut moves the anchor) — so the weight walk never happens per token.
 *
 * The released rows are persisted and stay on the backend, so this is a cache
 * eviction, not data loss: the tail bookkeeping is rewound first (the fetch
 * route must exist before anything is dropped) and the next "Show earlier"
 * fetches them back through the existing older-page backfill.
 *
 * The cut is computed inside the store updater, from the array it is about to
 * replace: a view snapshot can be a flush behind the store, and writing a
 * trimmed version of it back would drop the newest rows of a live turn.
 *
 * `profile` must be referentially stable across renders (the chat view memoizes
 * it): it is a dependency here, and the rewind has to address the same route
 * the render resolved.
 */
export function useTranscriptRetention({
  anchorId,
  enabled,
  profile,
  runtimeId,
  storedSessionId
}: TranscriptRetentionOptions): void {
  useEffect(() => {
    if (!enabled || !runtimeId || !storedSessionId || anchorId === null) {
      return
    }

    // Cheap pre-check: with no recorded tail entry there is no route to fetch
    // released rows back from, so the plan below would be discarded anyway.
    // The rewind inside the updater stays the authoritative arbiter.
    if (!transcriptTailState(storedSessionId, profile)) {
      return
    }

    sessionTileDelegate()?.updateSession(runtimeId, state => {
      const retention = boundRetainedTranscript(state.messages, anchorId)

      if (!retention.released) {
        return state
      }

      // Nothing may be released until the rows are fetchable again.
      if (!rewindTranscriptTail(storedSessionId, retention.retainedPersistedRows, profile)) {
        return state
      }

      return { ...state, messages: retention.messages }
    })
  }, [anchorId, enabled, profile, runtimeId, storedSessionId])
}
