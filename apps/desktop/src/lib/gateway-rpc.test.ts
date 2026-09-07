import { describe, expect, it } from 'vitest'

import { isMissingPendingPromptRequest, isMissingRpcMethod } from './gateway-rpc'

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
