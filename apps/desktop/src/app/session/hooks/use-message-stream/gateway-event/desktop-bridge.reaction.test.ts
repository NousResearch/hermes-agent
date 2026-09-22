import { beforeEach, describe, expect, it, vi } from 'vitest'

import { recordAgentReaction } from '@/store/reactions-local'
import { setMessages } from '@/store/session'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/pane-focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/reactions-local', () => ({ recordAgentReaction: vi.fn() }))
vi.mock('@/store/session', () => ({ setMessages: vi.fn() }))
vi.mock('@/store/tips', () => ({
  $tipsEnabled: { get: () => false },
  agentTipId: vi.fn(),
  showTip: vi.fn()
}))

const reactionEvent = (isActiveEvent: boolean): GatewayEventContext =>
  ({
    event: { type: 'message.reaction' },
    isActiveEvent,
    payload: {
      row_id: 4242,
      role: 'assistant',
      reactions: [{ emoji: '👍', author: 'agent' }]
    }
  }) as unknown as GatewayEventContext

describe('message.reaction bridge session scope', () => {
  beforeEach(() => {
    vi.mocked(setMessages).mockClear()
    vi.mocked(recordAgentReaction).mockClear()
  })

  it('a background session event never mutates the visible transcript or the overlay', () => {
    expect(handleDesktopBridgeEvent(reactionEvent(false))).toBe(true)
    expect(setMessages).not.toHaveBeenCalled()
    expect(recordAgentReaction).not.toHaveBeenCalled()
  })

  it('an active event stamps row id and reactions onto the optimistic bubble', () => {
    expect(handleDesktopBridgeEvent(reactionEvent(true))).toBe(true)
    expect(setMessages).toHaveBeenCalledTimes(1)

    const updater = vi.mocked(setMessages).mock.calls[0][0]

    if (typeof updater !== 'function') {throw new Error('expected an updater function')}
    const optimistic = { id: 'm1', role: 'assistant', rowId: undefined }
    const next = updater([optimistic] as never)

    expect(next).toEqual([
      { ...optimistic, rowId: 4242, reactions: [{ emoji: '👍', author: 'agent' }] }
    ])
    expect(recordAgentReaction).toHaveBeenCalledWith(4242, [{ emoji: '👍', author: 'agent' }])
  })

  it('an active event matches the byRowId leg without touching optimistic rows', () => {
    expect(handleDesktopBridgeEvent(reactionEvent(true))).toBe(true)

    const updater = vi.mocked(setMessages).mock.calls[0][0]

    if (typeof updater !== 'function') {throw new Error('expected an updater function')}
    const durable = { id: 'm1', role: 'assistant', rowId: 4242 }
    const optimistic = { id: 'm2', role: 'assistant', rowId: undefined }
    const next = updater([durable, optimistic] as never)

    expect(next[0]).toEqual({ ...durable, reactions: [{ emoji: '👍', author: 'agent' }] })
    expect(next[1]).toBe(optimistic)
    expect(recordAgentReaction).toHaveBeenCalledWith(4242, [{ emoji: '👍', author: 'agent' }])
  })
})
