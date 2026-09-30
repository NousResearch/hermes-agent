import type { SessionStartChatResult, StartChatArgs } from '@hermes/shared'
import { atom } from 'nanostores'

import { parseMaybeObject } from '@/components/assistant-ui/tool/fallback-model/format'
import { $gateway } from '@/store/gateway'
import { $focusedStoredSessionId, isSessionInForeground, requestForOwnedSession } from '@/store/session-states'

export type StartChatOutcome =
  | { profile: string; sessionId: string; status: 'started'; title: null | string }
  | { reason: string; status: 'rejected' }

export function readStartChatResult(result: unknown): null | StartChatOutcome {
  const row = parseMaybeObject(result)

  if (row.status === 'started' && typeof row.session_id === 'string' && row.session_id) {
    return {
      profile: typeof row.profile === 'string' ? row.profile : '',
      sessionId: row.session_id,
      status: 'started',
      title: typeof row.title === 'string' && row.title.trim() ? row.title.trim() : null
    }
  }

  if (row.status === 'rejected') {
    return { reason: typeof row.reason === 'string' ? row.reason : '', status: 'rejected' }
  }

  return null
}

const liveStarts = new Set<string>()

export function markLiveStartChat(toolCallId: string): void {
  liveStarts.add(toolCallId)
}

export function takeLiveStartChat(toolCallId: string): boolean {
  return liveStarts.delete(toolCallId)
}

export function isStartChatCallerWatched(storedId: string): boolean {
  return $focusedStoredSessionId.get() !== null && isSessionInForeground(storedId)
}

export const $startChatRetries = atom<Record<string, 'pending' | StartChatOutcome>>({})

function setRetry(toolCallId: string, value: 'pending' | null | StartChatOutcome): void {
  const { [toolCallId]: _previous, ...rest } = $startChatRetries.get()

  $startChatRetries.set(value ? { ...rest, [toolCallId]: value } : rest)
}

export async function retryStartChat(
  toolCallId: string,
  callerRuntimeId: string,
  args: StartChatArgs
): Promise<null | StartChatOutcome> {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Gateway not connected')
  }

  setRetry(toolCallId, 'pending')

  try {
    const outcome = readStartChatResult(
      await requestForOwnedSession<SessionStartChatResult>(
        callerRuntimeId,
        gateway.request.bind(gateway) as typeof gateway.request,
        'session.start_chat',
        { args, session_id: callerRuntimeId }
      )
    )

    setRetry(toolCallId, outcome)

    return outcome
  } catch (error) {
    setRetry(toolCallId, null)

    throw error
  }
}
