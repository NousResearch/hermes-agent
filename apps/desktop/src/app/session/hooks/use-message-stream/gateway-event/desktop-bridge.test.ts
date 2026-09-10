import { beforeEach, describe, expect, it, vi } from 'vitest'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

const mocks = vi.hoisted(() => ({
  ambientRequest: vi.fn(),
  readActivePreview: vi.fn(async () => ({ text: 'page', title: 'Preview', url: 'https://example.com' })),
  readActiveTerminal: vi.fn(() => ({ text: 'shell' })),
  requestForOwnedSession: vi.fn(async (..._args: unknown[]) => ({ status: 'ok' }))
}))

vi.mock('@/app/chat/right-rail/preview-reader', () => ({
  readActivePreview: mocks.readActivePreview
}))
vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/buffer', () => ({ readActiveTerminal: mocks.readActiveTerminal }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => ({ request: mocks.ambientRequest }) }
}))
vi.mock('@/store/pane-focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/reactions-local', () => ({ recordAgentReaction: vi.fn() }))
vi.mock('@/store/session', () => ({ setMessages: vi.fn() }))
vi.mock('@/store/session-gone-latch', () => ({ ambientRequestFor: () => mocks.ambientRequest }))
vi.mock('@/store/session-states', () => ({ requestForOwnedSession: mocks.requestForOwnedSession }))
vi.mock('@/store/tips', () => ({ $tipsEnabled: { get: () => false }, showTip: vi.fn() }))
vi.mock('@/store/tours', () => ({ $toursEnabled: { get: () => false } }))

function event(type: string, requestId: string, isActiveEvent = true): GatewayEventContext {
  return {
    event: {
      connectionId: 'homelab',
      profile: 'creative-design',
      session_id: 'runtime-creative',
      type
    },
    isActiveEvent,
    payload: { request_id: requestId },
    sessionId: 'runtime-creative'
  } as unknown as GatewayEventContext
}

describe('desktop bridge response routing', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('routes every blocking response through the requesting session owner', async () => {
    expect(handleDesktopBridgeEvent(event('terminal.read.request', 'terminal-1'))).toBe(true)
    expect(handleDesktopBridgeEvent(event('preview.read.request', 'preview-read-1'))).toBe(true)
    expect(handleDesktopBridgeEvent(event('preview.act.request', 'preview-act-1', false))).toBe(true)
    expect(handleDesktopBridgeEvent(event('window.read.request', 'window-1'))).toBe(true)
    expect(handleDesktopBridgeEvent(event('tour.request', 'tour-1'))).toBe(true)

    await vi.waitFor(() => expect(mocks.requestForOwnedSession).toHaveBeenCalledTimes(5))

    expect(mocks.requestForOwnedSession.mock.calls.map(call => [call[0], call[2]])).toEqual(
      expect.arrayContaining([
        ['runtime-creative', 'terminal.read.respond'],
        ['runtime-creative', 'preview.act.respond'],
        ['runtime-creative', 'tour.respond'],
        ['runtime-creative', 'window.read.respond'],
        ['runtime-creative', 'preview.read.respond']
      ])
    )
    expect(mocks.ambientRequest).not.toHaveBeenCalled()
  })
})
