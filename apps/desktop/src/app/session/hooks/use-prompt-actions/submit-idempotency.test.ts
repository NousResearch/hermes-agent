import { JsonRpcGatewayError } from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

import { createPromptSubmitIntent, requestPromptSubmit } from './submit-idempotency'
import type { GatewayRequest } from './utils'

const TIMEOUT_MS = 1_800_000

describe('prompt submit contract compatibility', () => {
  it('retries only a strict pre-handler rejection with the legacy parameters', async () => {
    const intent = createPromptSubmitIntent('stored-root')
    const params = { session_id: 'runtime-a', text: 'hello', ...intent }

    const request = vi.fn()
      .mockRejectedValueOnce(new JsonRpcGatewayError(
        'invalid params for prompt.submit: client_request_id: Extra inputs are not permitted — the client and the Hermes backend are out of sync',
        { code: 4000 }
      ))
      .mockResolvedValueOnce({ status: 'streaming' })

    expect(await requestPromptSubmit(request as GatewayRequest, params, TIMEOUT_MS)).toEqual({ status: 'streaming' })
    expect(request.mock.calls).toEqual([
      ['prompt.submit', params, TIMEOUT_MS],
      ['prompt.submit', { session_id: 'runtime-a', text: 'hello' }, TIMEOUT_MS]
    ])
  })

  it.each([
    new Error('request timed out: prompt.submit'),
    new JsonRpcGatewayError('invalid params for prompt.submit: other_field: Extra inputs are not permitted', { code: 4000 }),
    new JsonRpcGatewayError('client_request_id was already used for a different prompt', { code: 4020 })
  ])('preserves an ambiguous or unrelated failure without a second submit: %s', async error => {
    const request = vi.fn().mockRejectedValue(error)

    await expect(requestPromptSubmit(request as GatewayRequest, {
      session_id: 'runtime-a', text: 'hello', ...createPromptSubmitIntent('stored-root')
    }, TIMEOUT_MS)).rejects.toBe(error)
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('keeps one intent and the captured durable root across runtime replacement', async () => {
    const intent = createPromptSubmitIntent('stored-root')
    const request = vi.fn().mockResolvedValue({ status: 'streaming' })

    await requestPromptSubmit(request as GatewayRequest, { session_id: 'runtime-a', text: 'hello', ...intent }, TIMEOUT_MS)
    await requestPromptSubmit(request as GatewayRequest, { session_id: 'runtime-b', text: 'hello', ...intent }, TIMEOUT_MS)
    expect(intent.client_request_id).toEqual(expect.any(String))
    expect(request.mock.calls.map(call => call[1])).toEqual([
      { session_id: 'runtime-a', text: 'hello', ...intent },
      { session_id: 'runtime-b', text: 'hello', ...intent }
    ])
    expect(intent.expected_stored_session_id).toBe('stored-root')
  })
})
