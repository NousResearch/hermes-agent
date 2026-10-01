import { beforeEach, describe, expect, it, vi } from 'vitest'

import { recordAgentReaction } from '@/store/reactions-local'
import { setMessages } from '@/store/session'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/pane-focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/reactions-local', async () => {
  const { registryBackendScopeKey } = await import('@hermes/shared')

  return {
    recordAgentReaction: vi.fn(),
    // Same derivation the real store exports, so the scope asserted below is
    // the scope the handler produces in the app.
    reactionOverlayScope: (event: { connectionId?: string; profile?: string }) =>
      registryBackendScopeKey(event.connectionId ?? null, event.profile ?? null)
  }
})
vi.mock('@/store/session', () => ({ setMessages: vi.fn() }))
vi.mock('@/store/tips', () => ({
  $tipsEnabled: { get: () => false },
  agentTipId: vi.fn(),
  showTip: vi.fn()
}))

const AGENT_THUMBS_UP = [{ emoji: '👍', author: 'agent' }]

const reactionEvent = (overrides: Partial<GatewayEventContext> = {}): GatewayEventContext =>
  ({
    // A registry-tagged event from connection "conn-a" — the shape the
    // gateway registry stamps before fan-in.
    event: { connectionId: 'conn-a', profile: 'default', type: 'message.reaction' },
    isActiveEvent: true,
    fromActiveSource: () => true,
    payload: { row_id: 4242, role: 'assistant', reactions: AGENT_THUMBS_UP },
    ...overrides
  }) as unknown as GatewayEventContext

describe('message.reaction bridge session scope', () => {
  beforeEach(() => {
    vi.mocked(setMessages).mockClear()
    vi.mocked(recordAgentReaction).mockClear()
  })

  it('a background session event never mutates the visible transcript or the overlay', () => {
    expect(handleDesktopBridgeEvent(reactionEvent({ isActiveEvent: false }))).toBe(true)
    expect(setMessages).not.toHaveBeenCalled()
    expect(recordAgentReaction).not.toHaveBeenCalled()
  })

  it('an active event from a non-active source is dropped, not painted onto the visible transcript', () => {
    // isActiveEvent only proves the runtime id matches; two connections can
    // report the same session id. A reaction from source B must not mutate
    // the transcript source A is showing — the owning session paints it
    // from the persisted write on its next load.
    expect(handleDesktopBridgeEvent(reactionEvent({ fromActiveSource: () => false }))).toBe(true)
    expect(setMessages).not.toHaveBeenCalled()
    expect(recordAgentReaction).not.toHaveBeenCalled()
  })

  it('an active event stamps row id and reactions onto the optimistic bubble', () => {
    expect(handleDesktopBridgeEvent(reactionEvent())).toBe(true)
    expect(setMessages).toHaveBeenCalledTimes(1)

    const updater = vi.mocked(setMessages).mock.calls[0][0]

    if (typeof updater !== 'function') {
      throw new Error('expected an updater function')
    }
    const optimistic = { id: 'm1', role: 'assistant', rowId: undefined }
    const next = updater([optimistic] as never)

    expect(next).toEqual([{ ...optimistic, rowId: 4242, reactions: AGENT_THUMBS_UP }])
    expect(recordAgentReaction).toHaveBeenCalledWith(4242, AGENT_THUMBS_UP, 'conn:conn-a::default')
  })

  it('an active event matches the byRowId leg without touching optimistic rows', () => {
    expect(handleDesktopBridgeEvent(reactionEvent())).toBe(true)

    const updater = vi.mocked(setMessages).mock.calls[0][0]

    if (typeof updater !== 'function') {
      throw new Error('expected an updater function')
    }
    const durable = { id: 'm1', role: 'assistant', rowId: 4242 }
    const optimistic = { id: 'm2', role: 'assistant', rowId: undefined }
    const next = updater([durable, optimistic] as never)

    expect(next[0]).toEqual({ ...durable, reactions: AGENT_THUMBS_UP })
    expect(next[1]).toBe(optimistic)
    expect(recordAgentReaction).toHaveBeenCalledWith(4242, AGENT_THUMBS_UP, 'conn:conn-a::default')
  })
})
