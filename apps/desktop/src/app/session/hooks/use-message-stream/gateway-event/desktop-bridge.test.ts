import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

const hasLivePreviewSurface = vi.hoisted(() => vi.fn(() => false))
const requestPopoutPreviewAct = vi.hoisted(() => vi.fn())
const requestPopoutPreviewRead = vi.hoisted(() => vi.fn())
const isBrowserWindow = vi.hoisted(() => vi.fn(() => false))
const readActivePreview = vi.hoisted(() => vi.fn())

vi.mock('@/store/windows', async importOriginal => {
  const actual = await importOriginal<typeof import('@/store/windows')>()

  return {
    ...actual,
    isBrowserWindow: () => isBrowserWindow()
  }
})

vi.mock('@/app/chat/right-rail/preview-popout-bridge', () => ({
  hasLivePreviewSurface: () => hasLivePreviewSurface(),
  requestPopoutPreviewAct: (...args: unknown[]) => requestPopoutPreviewAct(...args),
  requestPopoutPreviewRead: (...args: unknown[]) => requestPopoutPreviewRead(...args)
}))

vi.mock('@/app/chat/right-rail/preview-reader', () => ({
  readActivePreview: (...args: unknown[]) => readActivePreview(...args)
}))

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

function previewReadContext(): GatewayEventContext {
  return {
    event: { type: 'preview.read.request' },
    explicitSid: '',
    isActiveEvent: true,
    payload: { request_id: 'read-1', start: 0, count: 100 }
  } as GatewayEventContext
}

describe('preview action bridge routing', () => {
  beforeEach(() => {
    hasLivePreviewSurface.mockReturnValue(false)
    isBrowserWindow.mockReturnValue(false)
    requestPopoutPreviewAct.mockReset()
    requestPopoutPreviewRead.mockReset()
    readActivePreview.mockReset()
  })

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

  it('stays silent on preview.act in the browser pop-out window', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    isBrowserWindow.mockReturnValue(true)

    expect(handleDesktopBridgeEvent(previewActContext({ explicitSid: 'session-a', isActiveEvent: false }))).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })

  it('forwards an active-session act to the pop-out when this window has no live surface', async () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    hasLivePreviewSurface.mockReturnValue(false)
    requestPopoutPreviewAct.mockResolvedValue({ acted: 'elements', success: true })

    expect(handleDesktopBridgeEvent(previewActContext({ explicitSid: 'session-a', isActiveEvent: true }))).toBe(true)

    await vi.waitFor(() => {
      expect(requestPopoutPreviewAct).toHaveBeenCalledWith(expect.objectContaining({ kind: 'elements' }))
      expect(request).toHaveBeenCalledWith('preview.act.respond', {
        request_id: 'request-1',
        text: JSON.stringify({ acted: 'elements', success: true })
      })
    })
  })

  it('stays silent on preview.read in the browser pop-out window', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    isBrowserWindow.mockReturnValue(true)

    expect(handleDesktopBridgeEvent(previewReadContext())).toBe(true)
    expect(request).not.toHaveBeenCalled()
    expect(readActivePreview).not.toHaveBeenCalled()
  })

  it('reads from the pop-out when the chat window has no live surface', async () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    hasLivePreviewSurface.mockReturnValue(false)
    requestPopoutPreviewRead.mockResolvedValue({
      end: 4,
      kind: 'url',
      start: 0,
      text: 'page',
      title: 'Example',
      total_chars: 4,
      url: 'https://example.com'
    })

    expect(handleDesktopBridgeEvent(previewReadContext())).toBe(true)

    await vi.waitFor(() => {
      expect(requestPopoutPreviewRead).toHaveBeenCalled()
      expect(request).toHaveBeenCalledWith('preview.read.respond', {
        request_id: 'read-1',
        text: JSON.stringify({
          end: 4,
          kind: 'url',
          start: 0,
          text: 'page',
          title: 'Example',
          total_chars: 4,
          url: 'https://example.com'
        })
      })
    })
  })
})
