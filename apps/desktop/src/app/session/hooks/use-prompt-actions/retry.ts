import { JsonRpcGatewayError } from '@hermes/shared'

import type { ClientSessionState } from '@/app/types'
import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import { chatMessageText } from '@/lib/chat-messages'

import { withSessionNotFoundResume } from './utils'

/** Error-card intent is explicit; normal Reload remains a deliberate regenerate. */
export interface RetryRunConfig {
  runConfig?: { custom?: Record<string, unknown> }
}

export function retryMessageId(config?: RetryRunConfig): string | undefined {
  const id = config?.runConfig?.custom?.retryMessageId

  return typeof id === 'string' ? id : undefined
}

export function planFailedRetry(messages: ChatMessage[], messageId: string) {
  const index = messages.findIndex(message => message.id === messageId && message.role === 'assistant' && message.error)

  if (index < 0 || messages.slice(index + 1).some(message => !message.hidden)) {
    return null
  }

  const failed = messages[index]
  const userIndex = messages.findLastIndex((message, i) => i < index && message.role === 'user')
  const user = messages[userIndex]

  if (!user) {
    return null
  }

  if (failed.retryToken) {
    return { retry_token: failed.retryToken }
  }

  // Only a submit rejected before durable admission may be sent as fresh input.
  if (failed.admissionRejected && user.rowId === undefined && index === userIndex + 1) {
    return { text: chatMessageText(user).trim() }
  }

  throw new Error('Retry checkpoint unavailable. Review completed work before continuing.')
}

export function applyRetryOptimistic(state: ClientSessionState, messageId: string): ClientSessionState {
  return {
    ...state,
    busy: true,
    awaitingResponse: true,
    sawAssistantPayload: false,
    turnLive: false,
    turnStartedAt: Date.now(),
    streamId: null,
    pendingBranchGroup: null,
    messages: state.messages.map(message =>
      message.id === messageId
        ? {
            ...message,
            error: undefined,
            retryToken: undefined,
            interim: true,
            hidden: !chatMessageText(message).trim()
          }
        : message
    )
  }
}

export async function submitFailedRetry(
  sessionId: string,
  storedSessionId: string | null,
  plan: { retry_token: string } | { text: string },
  requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>,
  onRecovered: (sessionId: string) => void,
  stillOwned: () => boolean = () => true
): Promise<void> {
  const method = 'retry_token' in plan ? 'prompt.retry' : 'prompt.submit'
  await withSessionNotFoundResume(
    sessionId,
    storedSessionId,
    async targetId => {
      try {
        return await requestGateway(method, { session_id: targetId, ...plan }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
      } catch (error) {
        if (error instanceof JsonRpcGatewayError && error.code === -32601) {
          throw new Error('This backend cannot continue a failed turn. Review completed work before continuing.')
        }

        throw error
      }
    },
    { requestGateway, onRecovered, driftReason: () => (stillOwned() ? null : 'Retry session changed') }
  )
}

/** Only definitive pre-admission refusals; a lost/timeout response may already have started work. */
export function promptAdmissionRejected(error: unknown): boolean {
  return (
    error instanceof JsonRpcGatewayError &&
    new Set([4000, 4001, 4004, 4009, 4090, 4122, 4124, 5035, -32601]).has(error.code ?? 0)
  )
}

/** A late RPC failure must not overwrite output or a newer user turn that already arrived. */
export function rollbackFailedRetry(
  state: ClientSessionState,
  before: ChatMessage[],
  optimistic?: ChatMessage[]
): ClientSessionState {
  return state.messages === optimistic
    ? {
        ...state,
        busy: false,
        awaitingResponse: false,
        turnLive: false,
        turnStartedAt: null,
        messages: before
      }
    : state
}

/** A shared recovered runtime may already have newer hydrated or streaming state. */
export function adoptRecoveredRetry(state: ClientSessionState, optimistic: ClientSessionState): ClientSessionState {
  return state.messages.length === 0 && !state.busy ? { ...state, ...optimistic } : state
}
