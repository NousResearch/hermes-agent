import { JsonRpcGatewayError, type PromptSubmitResult } from '@hermes/shared'

import type { GatewayRequest } from './utils'

export function createPromptSubmitIntent(storedSessionId: null | string) {
  return {
    client_request_id: crypto.randomUUID(),
    ...(storedSessionId && { expected_stored_session_id: storedSessionId })
  }
}

export async function requestPromptSubmit(
  requestGateway: GatewayRequest,
  params: Record<string, unknown>,
  timeoutMs: number
): Promise<PromptSubmitResult> {
  try {
    return await requestGateway<PromptSubmitResult>('prompt.submit', params, timeoutMs)
  } catch (error) {
    // Strict pre-v9 gateways reject these fields before invoking the handler.
    // Only that explicit refusal is safe to retry without the new contract.
    if (
      !(error instanceof JsonRpcGatewayError) ||
      error.code !== 4000 ||
      !/^invalid params for prompt\.submit: (client_request_id|expected_stored_session_id): Extra inputs are not permitted/.test(error.message)
    ) {
      throw error
    }

    const { client_request_id: _requestId, expected_stored_session_id: _storedId, ...legacyParams } = params

    return requestGateway<PromptSubmitResult>('prompt.submit', legacyParams, timeoutMs)
  }
}
