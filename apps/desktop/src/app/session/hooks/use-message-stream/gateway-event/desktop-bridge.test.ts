import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/app/chat/right-rail/preview-act', () => ({
  actOnActivePreview: vi.fn(async () => ({ success: true, typed: 'hello' }))
}))

import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'
import { $sessionTiles, clearAllSessionStates } from '@/store/session-states'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

function actCtx(sessionId: string, isActiveEvent: boolean): GatewayEventContext {
  return {
    deps: {} as GatewayEventContext['deps'],
    event: { session_id: sessionId, type: 'preview.act.request' },
    explicitSid: sessionId,
    fromActiveSource: () => true,
    isActiveEvent,
    occurredAt: Date.now() / 1000,
    payload: { action: 'type', ref: 'inp-prompt', request_id: 'req-97848', text: 'hello' },
    scheduleConfigRefresh: () => undefined,
    sessionId
  }
}

describe('handleDesktopBridgeEvent preview.act visibility', () => {
  const request = vi.fn(async () => ({}))

  beforeEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $sessionTiles.set([])
    request.mockClear()
    $gateway.set({ request } as never)
  })

  afterEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $sessionTiles.set([])
    $gateway.set(null)
  })

  it('rejects a hidden background session', async () => {
    $activeSessionId.set('runtime-primary')
    handleDesktopBridgeEvent(actCtx('runtime-hidden', false))
    await vi.waitFor(() => expect(request).toHaveBeenCalled())

    expect(request).toHaveBeenCalledWith(
      'preview.act.respond',
      expect.objectContaining({
        request_id: 'req-97848',
        text: expect.stringContaining('only takes actions in the session the user is looking at')
      })
    )
  })

  it('allows a visible session tile that is not the primary active session', async () => {
    $activeSessionId.set('runtime-primary')
    $sessionTiles.set([{ runtimeId: 'runtime-tile', storedSessionId: 'stored-tile' }])

    handleDesktopBridgeEvent(actCtx('runtime-tile', false))
    await vi.waitFor(() => expect(request).toHaveBeenCalled())

    expect(request).toHaveBeenCalledWith(
      'preview.act.respond',
      expect.objectContaining({
        request_id: 'req-97848',
        text: expect.not.stringContaining('only takes actions in the session the user is looking at')
      })
    )
  })
})
