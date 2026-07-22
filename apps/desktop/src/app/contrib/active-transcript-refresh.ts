import type { MutableRefObject } from 'react'

import { getSessionMessages } from '@/hermes'
import { preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { sessionMessagesSignature } from '@/lib/session-signatures'
import type { SessionInfo } from '@/types/hermes'

import type { ClientSessionState } from '../types'

interface ActiveTranscriptRefreshRefs {
  activeSessionIdRef: MutableRefObject<null | string>
  busyRef: MutableRefObject<boolean>
  generationRef: MutableRefObject<number>
  selectedStoredSessionIdRef: MutableRefObject<null | string>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  signatureRef: MutableRefObject<Map<string, string>>
}

interface RefreshActiveTranscriptOptions extends ActiveTranscriptRefreshRefs {
  findStoredSession: (storedSessionId: string) => SessionInfo | undefined
  loadMessages?: typeof getSessionMessages
  updateSessionState: (
    runtimeSessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => ClientSessionState
}

interface TranscriptScopeRefs {
  generationRef: MutableRefObject<number>
  scopeKeyRef: MutableRefObject<string>
}

export function advanceTranscriptScope({ generationRef, scopeKeyRef }: TranscriptScopeRefs, scopeKey: string): void {
  if (scopeKeyRef.current === scopeKey) {
    return
  }

  scopeKeyRef.current = scopeKey
  generationRef.current += 1
}

function isIdleOwnedState(
  state: ClientSessionState | undefined,
  storedSessionId: string
): state is ClientSessionState {
  return Boolean(
    state &&
      state.storedSessionId === storedSessionId &&
      !state.busy &&
      !state.awaitingResponse &&
      state.streamId == null
  )
}

export async function refreshActiveTranscript({
  activeSessionIdRef,
  busyRef,
  findStoredSession,
  generationRef,
  loadMessages = getSessionMessages,
  selectedStoredSessionIdRef,
  sessionStateByRuntimeIdRef,
  signatureRef,
  updateSessionState
}: RefreshActiveTranscriptOptions): Promise<void> {
  const storedSessionId = selectedStoredSessionIdRef.current
  const runtimeSessionId = activeSessionIdRef.current
  const generation = generationRef.current

  if (!storedSessionId || !runtimeSessionId || busyRef.current) {
    return
  }

  const initialState = sessionStateByRuntimeIdRef.current.get(runtimeSessionId)

  if (!isIdleOwnedState(initialState, storedSessionId)) {
    return
  }

  const stored = findStoredSession(storedSessionId)

  if (!stored) {
    return
  }

  try {
    const latest = await loadMessages(storedSessionId, stored.profile)

    if (
      generationRef.current !== generation ||
      selectedStoredSessionIdRef.current !== storedSessionId ||
      activeSessionIdRef.current !== runtimeSessionId ||
      busyRef.current
    ) {
      return
    }

    const currentState = sessionStateByRuntimeIdRef.current.get(runtimeSessionId)

    if (!isIdleOwnedState(currentState, storedSessionId)) {
      return
    }

    const signatureKey = `${stored.profile ?? 'default'}:${storedSessionId}`
    const signature = sessionMessagesSignature(latest.messages)

    if (signatureRef.current.get(signatureKey) === signature) {
      return
    }

    const remoteMessages = toChatMessages(latest.messages)
    const messages = preserveLocalAssistantErrors(remoteMessages, currentState.messages)
    let applied = false

    updateSessionState(
      runtimeSessionId,
      state => {
        if (
          generationRef.current !== generation ||
          selectedStoredSessionIdRef.current !== storedSessionId ||
          activeSessionIdRef.current !== runtimeSessionId ||
          busyRef.current ||
          !isIdleOwnedState(state, storedSessionId)
        ) {
          return state
        }

        applied = true

        return { ...state, messages }
      },
      storedSessionId
    )

    if (applied) {
      signatureRef.current.set(signatureKey, signature)
    }
  } catch {
    // Non-fatal: the next poll or a manual resume can hydrate the transcript.
  }
}
