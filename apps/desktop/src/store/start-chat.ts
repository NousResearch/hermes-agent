import type { SessionStartChatResult, StartChatArgs } from '@hermes/shared'
import { atom } from 'nanostores'

import { parseMaybeObject } from '@/components/assistant-ui/tool/fallback-model/format'
import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages/types'
import { $gateway } from '@/store/gateway'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { isSessionInForeground, requestForOwnedSession } from '@/store/session-states'

type StartChatOutcome =
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

// Tool call ids are unique only within one session (providers reuse ids like `call_0`), so per-call state is keyed by the calling session too.
function callKey(callerId: string, toolCallId: string): string {
  return `${callerId}\u0000${toolCallId}`
}

const liveStarts = new Set<string>()

export function markLiveStartChat(callerId: string, toolCallId: string): void {
  liveStarts.add(callKey(callerId, toolCallId))
}

export function takeLiveStartChat(callerId: string, toolCallId: string): boolean {
  return liveStarts.delete(callKey(callerId, toolCallId))
}

export function isStartChatCallerWatched(storedId: string): boolean {
  return $focusedStoredSessionId.get() !== null && isSessionInForeground(storedId)
}

export const $startChatRetries = atom<Record<string, 'pending' | StartChatOutcome>>({})

type StartChatRetry = 'pending' | StartChatOutcome | undefined

export function startChatRetry(
  retries: Record<string, 'pending' | StartChatOutcome>,
  callerId: string,
  toolCallId: string
): StartChatRetry {
  return retries[callKey(callerId, toolCallId)]
}

function setRetry(key: string, value: 'pending' | null | StartChatOutcome): void {
  const { [key]: _previous, ...rest } = $startChatRetries.get()

  $startChatRetries.set(value ? { ...rest, [key]: value } : rest)
}

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
export function startChatSuperseded(messages: ChatMessage[], callerId: string, toolCallId: string): boolean {
  const retries = $startChatRetries.get()
  let after = false

  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type !== 'tool-call' || part.toolName !== 'start_chat') {
        continue
      }

      if (
        after &&
        startChatOutcome(part, part.toolCallId ? startChatRetry(retries, callerId, part.toolCallId) : undefined)
          ?.status === 'started'
      ) {
        return true
      }

      after ||= part.toolCallId === toolCallId
    }
  }

  return false
}

export async function retryStartChat(
  callerId: string,
  toolCallId: string,
  callerRuntimeId: string,
  args: StartChatArgs
): Promise<null | StartChatOutcome> {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Gateway not connected')
  }

  const key = callKey(callerId, toolCallId)

  setRetry(key, 'pending')

  try {
    const outcome = readStartChatResult(
      await requestForOwnedSession<SessionStartChatResult>(
        callerRuntimeId,
        gateway.request.bind(gateway) as typeof gateway.request,
        'session.start_chat',
        { args, session_id: callerRuntimeId, tool_call_id: toolCallId }
      )
    )

    setRetry(key, outcome)

    return outcome
  } catch (error) {
    setRetry(key, null)

    throw error
  }
}
