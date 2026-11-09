import { type MutableRefObject, useEffect } from 'react'

import {
  hydrateStoredSessionTranscript,
  profileScopeForTranscriptSession,
  resolveActiveTranscriptSession
} from '@/app/contrib/hooks/use-background-sync'
import type { ClientSessionState } from '@/app/types'
import { $sessionTiles } from '@/store/session-states'
import { onTranscriptChanged } from '@/store/transcript-sync'

interface TranscriptPeerSyncOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/**
 * When another window finishes a turn on a stored session this window is
 * showing, re-pull that transcript. The completing window does not receive
 * its own broadcast. Submit still refuses if this pull has not landed yet.
 */
export function useTranscriptPeerSync({
  activeSessionIdRef,
  busyRef,
  selectedStoredSessionIdRef,
  updateSessionState
}: TranscriptPeerSyncOptions): void {
  useEffect(() => {
    let cancelled = false

    const unsubscribe = onTranscriptChanged(payload => {
      const storedSessionId = payload.sessionId

      if (!storedSessionId) {
        return
      }
      void (async () => {
        const selected = selectedStoredSessionIdRef.current
        const runtimeId = activeSessionIdRef.current

        if (!cancelled && selected === storedSessionId && runtimeId && !busyRef.current) {
          await hydrateStoredSessionTranscript({
            attempts: 1,
            runtimeSessionId: runtimeId,
            storedProfile: profileScopeForTranscriptSession(resolveActiveTranscriptSession(storedSessionId, runtimeId)),
            storedSessionId,
            updateSessionState
          })
        }

        if (cancelled) {
          return
        }

        for (const tile of $sessionTiles.get()) {
          if (tile.storedSessionId !== storedSessionId || !tile.runtimeId || tile.runtimeId === runtimeId) {
            continue
          }

          const owner = tile.ownerRoute
            ? { ownerRoute: tile.ownerRoute, profile: tile.ownerProfile ?? tile.ownerRoute.profile }
            : tile.ownerProfile
              ? { profile: tile.ownerProfile }
              : resolveActiveTranscriptSession(storedSessionId, tile.runtimeId)

          await hydrateStoredSessionTranscript({
            attempts: 1,
            runtimeSessionId: tile.runtimeId,
            storedProfile: profileScopeForTranscriptSession(owner),
            storedSessionId,
            updateSessionState
          })

          if (cancelled) {
            return
          }
        }
      })()
    })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [activeSessionIdRef, busyRef, selectedStoredSessionIdRef, updateSessionState])
}
