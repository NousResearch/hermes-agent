import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GatewayEventContext } from './types'

const mocks = vi.hoisted(() => ({
  ambientRequest: vi.fn(),
  requestForOwnedSession: vi.fn(
    (_sessionId: unknown, _ambient: unknown, _method: string, _params: Record<string, unknown>) =>
      Promise.resolve({ status: 'ok' })
  )
}))

vi.mock('@/store/gateway', () => ({
  $gateway: {
    get: () => ({ request: mocks.ambientRequest }),
    listen: vi.fn(() => () => {}),
    set: vi.fn(),
    subscribe: vi.fn(() => () => {})
  }
}))

vi.mock('@/store/session-states', () => ({
  requestForOwnedSession: mocks.requestForOwnedSession
}))

vi.mock('@/app/right-sidebar/terminal/buffer', () => ({
  readActiveTerminal: vi.fn(() => ({ text: 'visible terminal' }))
}))

import { handleDesktopBridgeEvent } from './desktop-bridge'

function bridgeEvent(type: string, payload: Record<string, unknown>): GatewayEventContext {
  return {
    event: { session_id: 'runtime-owned', type },
    isActiveEvent: false,
    payload,
    sessionId: 'runtime-owned'
  } as unknown as GatewayEventContext
}

describe('handleDesktopBridgeEvent response routing', () => {
  beforeEach(() => {
    mocks.ambientRequest.mockReset()
    mocks.requestForOwnedSession.mockClear()
  })

  it('returns a preview action response through the session owner', () => {
    const handled = handleDesktopBridgeEvent(
      bridgeEvent('preview.act.request', { action: 'elements', request_id: 'request-1' })
    )

    expect(handled).toBe(true)
    expect(mocks.requestForOwnedSession).toHaveBeenCalledOnce()
    expect(mocks.requestForOwnedSession.mock.calls[0]?.[0]).toBe('runtime-owned')
    expect(mocks.requestForOwnedSession.mock.calls[0]?.[2]).toBe('preview.act.respond')
    expect(mocks.requestForOwnedSession.mock.calls[0]?.[3]).toEqual({
      request_id: 'request-1',
      text: JSON.stringify({
        error: 'The in-app browser only takes actions in the session the user is looking at.',
        success: false
      })
    })
    expect(mocks.ambientRequest).not.toHaveBeenCalled()
  })

  it('returns a terminal read response through the session owner', () => {
    const handled = handleDesktopBridgeEvent(
      bridgeEvent('terminal.read.request', { count: 20, request_id: 'request-2', start: 0 })
    )

    expect(handled).toBe(true)
    expect(mocks.requestForOwnedSession).toHaveBeenCalledOnce()
    expect(mocks.requestForOwnedSession.mock.calls[0]?.[0]).toBe('runtime-owned')
    expect(mocks.requestForOwnedSession.mock.calls[0]?.[2]).toBe('terminal.read.respond')
    expect(mocks.requestForOwnedSession.mock.calls[0]?.[3]).toEqual({
      request_id: 'request-2',
      text: JSON.stringify({ text: 'visible terminal' })
    })
    expect(mocks.ambientRequest).not.toHaveBeenCalled()
  })
})
