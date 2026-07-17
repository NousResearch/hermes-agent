import { type MutableRefObject, useEffect } from 'react'

import type { ClientSessionState } from '@/app/types'
import { getSessionMessages } from '@/hermes'
import { preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { $busy, $messages, $sessions, sessionMatchesStoredId } from '@/store/session'
import { onTranscriptChanged, type TranscriptChangedPayload } from '@/store/transcript-sync'

interface TranscriptSyncOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/**
 * Soft live-sync for multi-window same-session views (#65047 option b):
 * when another window finishes a turn (or this window regains focus), compare
 * the authoritative transcript length and rehydrate if we are behind. Does not
 * fan out gateway events; just a best-effort refresh so the user is not left
 * staring at an open-time snapshot.
 */
export function useTranscriptSync({
  activeSessionIdRef,
  selectedStoredSessionIdRef,
  updateSessionState
}: TranscriptSyncOptions): void {
  useEffect(() => {
    let cancelled = false

    const refreshIfBehind = async (hint?: TranscriptChangedPayload) => {
      const storedSessionId = selectedStoredSessionIdRef.current
      const runtimeSessionId = activeSessionIdRef.current

      if (!storedSessionId || !runtimeSessionId || $busy.get() || cancelled) {
        return
      }

      if (hint && hint.sessionId !== storedSessionId) {
        return
      }

      const localCount = $messages.get().length

      if (hint && hint.messageCount <= localCount) {
        return
      }

      const storedProfile = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))?.profile

      try {
        const latest = await getSessionMessages(storedSessionId, storedProfile)

        if (cancelled || selectedStoredSessionIdRef.current !== storedSessionId) {
          return
        }

        if (latest.messages.length <= localCount) {
          return
        }

        const messages = toChatMessages(latest.messages)

        updateSessionState(
          runtimeSessionId,
          state => ({
            ...state,
            messages: preserveLocalAssistantErrors(messages, state.messages)
          }),
          storedSessionId
        )
      } catch {
        // Best-effort: submit-time guard still blocks a stale send.
      }
    }

    const unsubscribe = onTranscriptChanged(payload => {
      void refreshIfBehind(payload)
    })

    const onFocus = () => {
      void refreshIfBehind()
    }

    window.addEventListener('focus', onFocus)

    return () => {
      cancelled = true
      unsubscribe()
      window.removeEventListener('focus', onFocus)
    }
  }, [activeSessionIdRef, selectedStoredSessionIdRef, updateSessionState])
}
