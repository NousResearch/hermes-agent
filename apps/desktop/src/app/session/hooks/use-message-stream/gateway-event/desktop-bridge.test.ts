import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { $toursEnabled } from '@/store/tours'

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

function tourContext({
  explicitSid,
  isActiveEvent
}: {
  explicitSid: string
  isActiveEvent: boolean
}): GatewayEventContext {
  return {
    event: { session_id: explicitSid || undefined, type: 'tour.request' },
    explicitSid,
    isActiveEvent,
    payload: { action: 'discover', request_id: 'tour-request-1' }
  } as GatewayEventContext
}

describe('tour bridge routing', () => {
  afterEach(() => {
    $gateway.set(null)
    $toursEnabled.set(true)
  })

  it('leaves a scoped request unanswered in another session even when tours are disabled', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    $toursEnabled.set(false)

    expect(handleDesktopBridgeEvent(tourContext({ explicitSid: 'session-a', isActiveEvent: false }))).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })

  it('keeps the legacy fail-fast response for an unscoped inactive request', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)

    expect(handleDesktopBridgeEvent(tourContext({ explicitSid: '', isActiveEvent: false }))).toBe(true)
    expect(request).toHaveBeenCalledWith('tour.respond', {
      request_id: 'tour-request-1',
      text: JSON.stringify({
        error: 'Tours only run in the session the user is looking at.',
        success: false
      })
    })
  })
})
