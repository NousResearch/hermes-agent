import { JsonRpcGatewayError } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { formatGatewayRpcError, isMissingPendingPromptRequest, isMissingRpcMethod } from './gateway-rpc'

describe('isMissingRpcMethod', () => {
  it('detects JSON-RPC method-not-found errors', () => {
    expect(isMissingRpcMethod(new Error('unknown method: projects.create'))).toBe(true)
    expect(isMissingRpcMethod(new Error('Method not found'))).toBe(true)
    expect(isMissingRpcMethod(new Error('RPC failed: -32601'))).toBe(true)
  })

  it('ignores unrelated failures', () => {
    expect(isMissingRpcMethod(new Error('Hermes gateway is not connected'))).toBe(false)
    expect(isMissingRpcMethod(new Error('no such project'))).toBe(false)
  })
})

describe('formatGatewayRpcError', () => {
  it('prefers the server message on projects.create 5063 failures', () => {
    expect(
      formatGatewayRpcError(
        new JsonRpcGatewayError(
          "folder already belongs to project 'demo' (p_1); switch to it instead of creating a duplicate",
          { code: 5063 }
        ),
        'Could not create project'
      )
    ).toBe("folder already belongs to project 'demo' (p_1); switch to it instead of creating a duplicate")
  })

  it('falls back to an opaque code label only when the message is empty', () => {
    expect(formatGatewayRpcError(new JsonRpcGatewayError('', { code: 5063 }), 'Could not create project')).toBe(
      'Hermes RPC request failed (5063)'
    )
  })
})

describe('isMissingPendingPromptRequest', () => {
  it('detects stale prompt response errors from the gateway', () => {
    expect(isMissingPendingPromptRequest(new Error('no pending password request'), 'password')).toBe(true)
    expect(isMissingPendingPromptRequest(new Error('RPC failed: no pending value request'), 'value')).toBe(true)
  })

  it('ignores unrelated gateway failures', () => {
    expect(isMissingPendingPromptRequest(new Error('gateway not connected'), 'password')).toBe(false)
    expect(isMissingPendingPromptRequest(new Error('no pending value request'), 'password')).toBe(false)
  })
})
