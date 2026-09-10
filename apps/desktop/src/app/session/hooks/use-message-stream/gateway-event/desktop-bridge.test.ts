import { afterEach, describe, expect, it, vi } from 'vitest'

import { readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { $gateway } from '@/store/gateway'
import { $toursEnabled } from '@/store/tours'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

vi.mock('@/app/right-sidebar/terminal/buffer', () => ({ readActiveTerminal: vi.fn() }))
vi.mock('@/app/chat/right-rail/preview-reader', () => ({ readActivePreview: vi.fn() }))

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

function terminalReadContext({
  explicitSid,
  isActiveEvent
}: {
  explicitSid: string
  isActiveEvent: boolean
}): GatewayEventContext {
  return {
    event: { session_id: explicitSid || undefined, type: 'terminal.read.request' },
    explicitSid,
    isActiveEvent,
    payload: { request_id: 'terminal-request-1' }
  } as GatewayEventContext
}

describe('terminal read bridge routing', () => {
  afterEach(() => {
    $gateway.set(null)
    vi.mocked(readActiveTerminal).mockReset()
  })

  it('leaves a scoped read request unanswered in a window showing another session', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    vi.mocked(readActiveTerminal).mockReturnValue({ lines: ['this window is on a different session'] } as never)

    expect(handleDesktopBridgeEvent(terminalReadContext({ explicitSid: 'session-a', isActiveEvent: false }))).toBe(true)

    // The window must not read (let alone report) ITS OWN terminal buffer for
    // a request scoped to a session it isn't showing — that content would be
    // wrong for the caller and could race the owning window's real answer.
    expect(readActiveTerminal).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })

  it('still answers from the owning window', () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    vi.mocked(readActiveTerminal).mockReturnValue({ lines: ['hello'] } as never)

    expect(handleDesktopBridgeEvent(terminalReadContext({ explicitSid: 'session-a', isActiveEvent: true }))).toBe(true)

    expect(request).toHaveBeenCalledWith('terminal.read.respond', {
      request_id: 'terminal-request-1',
      text: JSON.stringify({ lines: ['hello'] })
    })
  })
})

function previewReadContext({
  explicitSid,
  isActiveEvent
}: {
  explicitSid: string
  isActiveEvent: boolean
}): GatewayEventContext {
  return {
    event: { session_id: explicitSid || undefined, type: 'preview.read.request' },
    explicitSid,
    isActiveEvent,
    payload: { request_id: 'preview-request-1' }
  } as GatewayEventContext
}

describe('preview read bridge routing', () => {
  afterEach(() => {
    $gateway.set(null)
    vi.mocked(readActivePreview).mockReset()
  })

  it('leaves a scoped read request unanswered in a window showing another session', async () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    vi.mocked(readActivePreview).mockResolvedValue({ text: 'this window is on a different session' } as never)

    expect(handleDesktopBridgeEvent(previewReadContext({ explicitSid: 'session-a', isActiveEvent: false }))).toBe(true)
    await Promise.resolve()
    await Promise.resolve()

    expect(readActivePreview).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })

  it('still answers from the owning window', async () => {
    const request = vi.fn()
    $gateway.set({ request } as never)
    vi.mocked(readActivePreview).mockResolvedValue({ text: 'hello' } as never)

    expect(handleDesktopBridgeEvent(previewReadContext({ explicitSid: 'session-a', isActiveEvent: true }))).toBe(true)
    await Promise.resolve()
    await Promise.resolve()

    expect(request).toHaveBeenCalledWith('preview.read.respond', {
      request_id: 'preview-request-1',
      text: JSON.stringify({ text: 'hello' })
    })
  })
})

function windowReadContext({
  explicitSid,
  isActiveEvent
}: {
  explicitSid: string
  isActiveEvent: boolean
}): GatewayEventContext {
  return {
    event: { session_id: explicitSid || undefined, type: 'window.read.request' },
    explicitSid,
    isActiveEvent,
    payload: { request_id: 'window-request-1' }
  } as GatewayEventContext
}

describe('window read bridge routing', () => {
  afterEach(() => {
    $gateway.set(null)
  })

  it('leaves a scoped read request unanswered in a window showing another session', async () => {
    const request = vi.fn()
    const readWindowBelow = vi.fn(async () => 'this window is on a different session')
    $gateway.set({ request } as never)
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { readWindowBelow } })

    expect(handleDesktopBridgeEvent(windowReadContext({ explicitSid: 'session-a', isActiveEvent: false }))).toBe(true)
    await Promise.resolve()
    await Promise.resolve()

    // Must not even ask main what sits behind THIS (non-owning) window.
    expect(readWindowBelow).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })

  it('still answers from the owning window', async () => {
    const request = vi.fn()
    const readWindowBelow = vi.fn(async () => 'hello')
    $gateway.set({ request } as never)
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { readWindowBelow } })

    expect(handleDesktopBridgeEvent(windowReadContext({ explicitSid: 'session-a', isActiveEvent: true }))).toBe(true)
    await Promise.resolve()
    await Promise.resolve()

    expect(request).toHaveBeenCalledWith('window.read.respond', {
      request_id: 'window-request-1',
      text: JSON.stringify('hello')
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
