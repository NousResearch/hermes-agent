import { type MutableRefObject, useCallback } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages } from '@/hermes'
import { preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { latestSessionTodos } from '@/lib/todos'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'

import type { ClientSessionState } from '../../types'

type UpdateSessionState = (
  sessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState,
  storedSessionId?: string | null
) => ClientSessionState

interface StoredSessionHydrationOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionProfileRef: MutableRefObject<string | null>
  updateSessionState: UpdateSessionState
}

export type StoredSessionHydrator = (
  attempts?: number,
  storedSessionId?: string | null,
  runtimeSessionId?: string | null,
  storedSessionProfile?: string | null
) => Promise<void>

/** Re-read persisted history for one profile-qualified runtime owner. */
export function useStoredSessionHydration({
  activeSessionIdRef,
  selectedStoredSessionIdRef,
  selectedStoredSessionProfileRef,
  updateSessionState
}: StoredSessionHydrationOptions): StoredSessionHydrator {
  return useCallback(
    async (
      attempts = 1,
      storedSessionId = selectedStoredSessionIdRef.current,
      runtimeSessionId = activeSessionIdRef.current,
      storedSessionProfile = selectedStoredSessionProfileRef.current ?? $activeGatewayProfile.get()
    ) => {
      if (!storedSessionId || !runtimeSessionId) {
        return
      }

      // Capture profile + stored id together before the first await. Stored ids
      // are profile-local, so looking the owner up by bare id can read a same-id
      // transcript from another profile and graft it onto this runtime.
      const ownerProfile = normalizeProfileKey(storedSessionProfile)

      const isCurrentOwner = () =>
        selectedStoredSessionIdRef.current === storedSessionId &&
        activeSessionIdRef.current === runtimeSessionId &&
        normalizeProfileKey(selectedStoredSessionProfileRef.current ?? $activeGatewayProfile.get()) === ownerProfile

      for (let index = 0; index < Math.max(1, attempts); index += 1) {
        try {
          const latest = await getLatestSessionMessages(storedSessionId, ownerProfile)
          const messages = toChatMessages(latest.messages)

          // The request belongs to the exact profile + durable id + runtime
          // captured before its first await. A same-id profile switch can reuse
          // either id, so all three axes must still match at publication time.
          if (!isCurrentOwner()) {
            return
          }

          updateSessionState(
            runtimeSessionId,
            state => ({
              ...state,
              // Post-turn rehydrate reads only the newest tail page — graft it
              // onto any backfilled older pages instead of dropping them.
              messages: preserveLocalAssistantErrors(
                graftRefreshedTailOntoBackfill(messages, state.messages),
                state.messages
              )
            }),
            storedSessionId
          )

          const restored = todosForHydration(latestSessionTodos(messages))

          // updateSessionState can synchronously notify subscribers. Revalidate
          // again before publishing the sibling todo projection so a switch in
          // that notification cannot graft this owner's tasks onto the next one.
          if (!isCurrentOwner()) {
            return
          }

          if (restored) {
            setSessionTodos(runtimeSessionId, restored)
          } else {
            clearSessionTodos(runtimeSessionId)
          }

          return
        } catch {
          // Best-effort fallback when live stream payloads are empty.
        }

        if (index < attempts - 1) {
          await new Promise(resolve => window.setTimeout(resolve, 250))
        }
      }
    },
    [activeSessionIdRef, selectedStoredSessionIdRef, selectedStoredSessionProfileRef, updateSessionState]
  )
}
