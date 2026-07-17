import { type RefObject, useEffect, useRef } from 'react'

import { getSessionMessages, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import {
  clearUpdateContinuation,
  markUpdateContinuationAttempt,
  readUpdateContinuation,
  type UpdateContinuation,
  updateContinuationPrompt,
  updateContinuationToken
} from '@/store/update-continuation'

import type { GatewayRequester } from '../types'

interface UpdateContinuationParams {
  activeSessionId: null | string
  awaitingResponse: boolean
  busy: boolean
  gatewayState: string
  routedSessionId: null | string
  runtimeIdByStoredSessionIdRef: RefObject<Map<string, string>>
  selectedStoredSessionId: null | string
  requestGateway: GatewayRequester
}

type DeliveryResult = 'already-present' | 'submitted' | false
type MessagesLoader = (sessionId: string) => Promise<{ messages: unknown[] }>
type ContinuationSubmitter = (text: string) => Promise<unknown> | unknown

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export async function deliverUpdateContinuation(
  continuation: UpdateContinuation,
  loadMessages: MessagesLoader,
  submitContinuation: ContinuationSubmitter,
  opts: { markAttempt?: () => void; now?: () => number; sleep?: (ms: number) => Promise<void> } = {}
): Promise<DeliveryResult> {
  const wait = opts.sleep ?? sleep

  // A prior renderer may have died after prompt.submit reached the gateway but
  // before localStorage was cleared. Give persistence a moment, then use the
  // request token in the stored transcript as the idempotency acknowledgement.
  if (continuation.attemptedAt) {
    const remaining = 3_000 - ((opts.now ?? Date.now)() - continuation.attemptedAt)

    if (remaining > 0) {
      await wait(remaining)
    }
  }

  const token = updateContinuationToken(continuation.requestId)
  const prompt = updateContinuationPrompt(continuation.requestId)

  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const transcript = await loadMessages(continuation.sessionId)

      if (JSON.stringify(transcript.messages).includes(token)) {
        return 'already-present'
      }
    } catch {
      // Without the transcript check we cannot distinguish a retry from a
      // prompt accepted just before a renderer crash. Fail closed and retry
      // the read on a bounded backoff.
      await wait(750 * (attempt + 1))

      continue
    }

    opts.markAttempt?.()

    try {
      const accepted = await submitContinuation(prompt)

      if (accepted !== false) {
        return 'submitted'
      }
    } catch {
      // A transport timeout is ambiguous: prompt.submit may already have
      // reached the gateway before its user message is durable. Keep the
      // marker and stop this delivery run. A later, freshly confirmed idle
      // phase (or next launch) re-enters through the transcript check above.
    }

    return false
  }

  return false
}

/** Continue only after the normal route-resume loaded the exact, idle chat. */
export function useUpdateContinuation({
  activeSessionId,
  awaitingResponse,
  busy,
  gatewayState,
  routedSessionId,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  requestGateway
}: UpdateContinuationParams): void {
  const inFlightRequestRef = useRef<null | string>(null)

  useEffect(() => {
    const continuation = readUpdateContinuation()

    const runtimeId = continuation
      ? runtimeIdByStoredSessionIdRef.current.get(continuation.sessionId) ?? null
      : null

    const exactIdleSessionReady =
      continuation !== null &&
      gatewayState === 'open' &&
      routedSessionId === continuation.sessionId &&
      selectedStoredSessionId === continuation.sessionId &&
      runtimeId !== null &&
      runtimeId === activeSessionId &&
      !busy &&
      !awaitingResponse

    if (!exactIdleSessionReady || inFlightRequestRef.current === continuation.requestId) {
      return
    }

    inFlightRequestRef.current = continuation.requestId
    void deliverUpdateContinuation(
      continuation,
      sessionId => getSessionMessages(sessionId),
      prompt =>
        requestGateway(
          'prompt.submit',
          { session_id: runtimeId, text: prompt },
          PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
        ),
      {
        markAttempt: () => markUpdateContinuationAttempt(continuation)
      }
    )
      .then(delivered => {
        if (delivered) {
          clearUpdateContinuation()
        }
      })
      .finally(() => {
        inFlightRequestRef.current = null
      })
  }, [
    activeSessionId,
    awaitingResponse,
    busy,
    gatewayState,
    routedSessionId,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    requestGateway
  ])
}
