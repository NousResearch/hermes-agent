import type { SessionStartChatResult, StartChatArgs } from '@hermes/shared'
import { atom } from 'nanostores'

import { parseMaybeObject } from '@/components/assistant-ui/tool/fallback-model/format'
import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages/types'
import { $gateway } from '@/store/gateway'
import { $focusedStoredSessionId, isSessionInForeground, requestForOwnedSession } from '@/store/session-states'

export type StartChatOutcome =
  | { profile: string; sessionId: string; status: 'started'; title: null | string }
  | { reason: string; retryable: boolean; status: 'rejected' }

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
    return {
      reason: typeof row.reason === 'string' ? row.reason : '',
      retryable: row.retryable === true,
      status: 'rejected'
    }
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

type StartChatRetry = 'pending' | StartChatOutcome | undefined

/** The card's Retry (live in this window, else recorded on the saved tool row) wins over the call's own result. */
export function startChatOutcome(
  part: Pick<ChatMessagePart, 'toolResultMetadata'> & { result?: unknown },
  retry: StartChatRetry
): null | StartChatOutcome {
  return (
    (retry && retry !== 'pending' ? retry : null) ??
    readStartChatResult(part.toolResultMetadata?.retried) ??
    readStartChatResult(part.result)
  )
}

/** A later start_chat in this chat already started: a Retry here could only start the task twice. */
export function startChatSuperseded(messages: ChatMessage[], toolCallId: string): boolean {
  const retries = $startChatRetries.get()
  let after = false

  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type !== 'tool-call' || part.toolName !== 'start_chat') {
        continue
      }

      if (
        after &&
        startChatOutcome(part, part.toolCallId ? retries[part.toolCallId] : undefined)?.status === 'started'
      ) {
        return true
      }

      after ||= part.toolCallId === toolCallId
    }
  }

  return false
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
        { args, session_id: callerRuntimeId, tool_call_id: toolCallId }
      )
    )

    setRetry(toolCallId, outcome)

    return outcome
  } catch (error) {
    setRetry(toolCallId, null)

    throw error
  }
}
