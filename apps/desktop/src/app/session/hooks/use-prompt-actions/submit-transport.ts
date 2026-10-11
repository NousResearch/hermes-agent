import type { PromptSubmitResult } from '@hermes/shared'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'

import {
  type GatewayRequest,
  type SessionRecoveryDeps,
  type SubmitTextOptions,
  withSessionBusyRetry,
  withSessionNotFoundResume
} from './utils'

/** One prompt.submit wire format for normal sends and both queue drains.
 * Confirmed sends deliberately omit all recovery/retry wrappers: a missing or
 * delayed acknowledgement is an unknown outcome, not permission to replay. */
export async function submitPromptTransport({
  options,
  params,
  recovery,
  requestGateway,
  sessionId,
  storedSessionId
}: {
  options?: SubmitTextOptions
  params: (liveId: string) => Record<string, unknown>
  recovery: SessionRecoveryDeps
  requestGateway: GatewayRequest
  sessionId: string
  storedSessionId: null | string
}): Promise<{ result: PromptSubmitResult | undefined; sessionId: string }> {
  const submitOnce = (liveId: string) =>
    requestGateway<PromptSubmitResult>('prompt.submit', params(liveId), PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)

  if (options?.confirmedExternal) {
    const result = await submitOnce(sessionId)

    if (result?.status !== 'streaming' && result?.status !== 'queued') {
      throw new Error('Native submission outcome unknown')
    }

    options.onExternalAccepted?.(result.status === 'queued')

    return { result, sessionId }
  }

  // A starved backend loop can time out before acceptance even with a valid
  // stored session; preserve the established normal-send recovery policy.
  return withSessionNotFoundResume(
    sessionId,
    storedSessionId,
    liveId => withSessionBusyRetry(() => submitOnce(liveId)),
    recovery,
    { alsoTimeout: true }
  )
}
