import { JsonRpcGatewayError } from '@hermes/shared'

import type { ClientSessionState } from '@/app/types'
import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import { chatMessageText } from '@/lib/chat-messages'

import { isSessionNotFoundError, withSessionNotFoundResume } from './utils'

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

  if (failed.errorSurface?.retryable === false) {
    throw new Error('Retry checkpoint unavailable. Review completed work before continuing.')
  }

  if (failed.retryToken) {
    return { retry_token: failed.retryToken }
  }

  // Only a submit rejected before durable admission may be sent as fresh input.
  if (failed.admissionRejected && user.rowId === undefined && index === userIndex + 1) {
    if (user.attachmentRefs?.length || user.parts.some(part => part.type !== 'text')) {
      // BUSY can unstage uploaded images. The restored composer owns re-uploading them.
      throw new Error('Review the restored attachments and send the draft again before continuing.')
    }
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
  const call = async (targetId: string) => {
    try {
      return await requestGateway(method, { session_id: targetId, ...plan }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
    } catch (error) {
      if (error instanceof JsonRpcGatewayError && error.code === -32601) {
        throw new Error('This backend cannot continue a failed turn. Review completed work before continuing.')
      }

      throw error
    }
  }
  if ('text' in plan) {
    // A refusal proves no admission only on its original runtime, never on a recovered owner.
    try {
      await call(sessionId)
    } catch (error) {
      if (
        isSessionNotFoundError(error) ||
        (error instanceof JsonRpcGatewayError &&
          (error.data as { reason?: unknown } | undefined)?.reason === 'not_found')
      ) {
        throw new Error('Retry session changed. Review the conversation before sending the draft again.')
      }
      throw error
    }
    return
  }
  await withSessionNotFoundResume(sessionId, storedSessionId, call, {
    requestGateway,
    onRecovered,
    driftReason: () => (stillOwned() ? null : 'Retry session changed')
  })
}

/** Only definitive pre-admission refusals; a lost/timeout response may already have started work. */
export function promptAdmissionRejected(error: unknown): boolean {
  if (!(error instanceof JsonRpcGatewayError)) {
    return false
  }
  const reason = (error.data as { reason?: unknown } | undefined)?.reason
  if (reason === 'SESSION_OWNER_UNCONFIRMED' || reason === 'storage_unavailable') {
    return false
  }
  if (error.code === 4001) {
    // Canonical 4001 also carries post-write failures: only named pre-admission refusals qualify.
    return (
      [
        'not_found',
        'permission_denied',
        'session_busy',
        'invalid_params',
        'profile_mismatch',
        'runtime_draining',
        'session_superseded'
      ].includes(String(reason)) || /^session not found$/i.test(error.message)
    )
  }
  return new Set([4000, 4004, 4009, 4090, 4122, 4124, 5035, -32601]).has(error.code ?? 0)
}

/** A late failure preserves newer output. An uncertain fresh send consumes its old refusal proof. */
export function rollbackFailedRetry(
  state: ClientSessionState,
  before: ChatMessage[],
  optimistic?: ChatMessage[],
  unconfirmedMessageId?: string
): ClientSessionState {
  const restored =
    state.messages === optimistic
      ? {
          ...state,
          busy: false,
          awaitingResponse: false,
          turnLive: false,
          turnStartedAt: null,
          messages: before
        }
      : state
  return unconfirmedMessageId ? revokeFreshRetry(restored, unconfirmedMessageId) : restored
}

function revokeFreshRetry(state: ClientSessionState, messageId: string): ClientSessionState {
  let changed = false
  const messages = state.messages.map(message => {
    if (message.id !== messageId || !message.error || !message.admissionRejected || message.retryToken) {
      return message
    }
    changed = true
    return {
      ...message,
      admissionRejected: false,
      error: 'The retry was not confirmed. Check the conversation before sending again.',
      errorSurface: { layer: 'gateway' as const, code: 'SESSION_OWNER_UNCONFIRMED', retryable: false }
    }
  })
  return changed ? { ...state, messages } : state
}

/** A shared recovered runtime may already have newer hydrated or streaming state. */
export function adoptRecoveredRetry(state: ClientSessionState, optimistic: ClientSessionState): ClientSessionState {
  return state.messages.length === 0 && !state.busy ? { ...state, ...optimistic } : state
}
