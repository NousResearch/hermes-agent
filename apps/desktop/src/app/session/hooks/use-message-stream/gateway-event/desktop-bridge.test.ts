import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

function previewActContext({
  explicitSid,
  isActiveEvent
}: {
  explicitSid: string
  isActiveEvent: boolean
}): GatewayEventContext {
  return {
    event: { session_id: explicitSid || undefined, type: 'preview.act.request' },
    explicitSid,
    isActiveEvent,
    payload: { action: 'elements', request_id: 'request-1' }
  } as GatewayEventContext
}

describe('preview action bridge routing', () => {
  afterEach(() => {
    $gateway.set(null)
  })

  it('leaves a scoped action request unanswered in a window showing another session', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)

    expect(handleDesktopBridgeEvent(previewActContext({ explicitSid: 'session-a', isActiveEvent: false }))).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })

  it('keeps the legacy fail-fast response for an unscoped inactive request', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)

    expect(handleDesktopBridgeEvent(previewActContext({ explicitSid: '', isActiveEvent: false }))).toBe(true)
    expect(request).toHaveBeenCalledWith('preview.act.respond', {
      request_id: 'request-1',
      text: JSON.stringify({
        error: 'The in-app browser only takes actions in the session the user is looking at.',
        success: false
      })
    })
  })
})
