import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { MessageReaction } from '@/types/hermes'

const reactionTestState = vi.hoisted(() => {
  const store = <T>(initial: T) => {
    let value = initial

    return {
      get: () => value,
      set: (next: T | ((current: T) => T)) => {
        value = typeof next === 'function' ? (next as (current: T) => T)(value) : next
      }
    }
  }
  const gateway = { request: vi.fn(async (..._args: any[]) => ({ row_id: 7, reactions: [] })) }

  return {
    activeSessionId: store<string | null>(null),
    gateway,
    gatewayStore: store(gateway),
    messages: store<any[]>([]),
    requestForOwnedSession: vi.fn(
      async <T>(
        _sessionId: string | null,
        request: typeof gateway.request,
        method: string,
        params: Record<string, unknown>
      ) => request(method, params) as Promise<T>
    ),
    selectedStoredSessionId: store<string | null>(null)
  }
})

vi.mock('@/store/gateway', () => ({ $gateway: reactionTestState.gatewayStore }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/session-states', () => ({ requestForOwnedSession: reactionTestState.requestForOwnedSession }))
vi.mock('@/store/session', () => ({
  $activeSessionId: reactionTestState.activeSessionId,
  $messages: reactionTestState.messages,
  $selectedStoredSessionId: reactionTestState.selectedStoredSessionId,
  setMessages: (update: (messages: any[]) => any[]) =>
    reactionTestState.messages.set(update(reactionTestState.messages.get()))
}))

import { applyReaction, QUICK_REACTIONS, toggleMessageReaction } from '@/store/reactions'
const at = 1_700_000_000

beforeEach(() => {
  reactionTestState.activeSessionId.set(null)
  reactionTestState.selectedStoredSessionId.set(null)
  reactionTestState.messages.set([])
  reactionTestState.gateway.request.mockClear()
  reactionTestState.requestForOwnedSession.mockClear()
})

function reaction(emoji: string, author: MessageReaction['author']): MessageReaction {
  return { emoji, author, at }
}

describe('applyReaction', () => {
  it('adds a reaction to an empty message', () => {
    expect(applyReaction(undefined, '❤️', 'user')).toMatchObject([{ emoji: '❤️', author: 'user' }])
  })

  it('replaces the same author’s existing reaction (one per author)', () => {
    const next = applyReaction([reaction('❤️', 'user')], '😂', 'user')

    expect(next).toHaveLength(1)
    expect(next[0].emoji).toBe('😂')
  })

  it('retracts when the live reaction is re-sent', () => {
    expect(applyReaction([reaction('👍', 'user')], '👍', 'user')).toEqual([])
  })

  it('clears on an explicit null', () => {
    expect(applyReaction([reaction('👍', 'user')], null, 'user')).toEqual([])
  })

  it('keeps authors independent', () => {
    const next = applyReaction([reaction('🔥', 'agent')], '❤️', 'user')

    expect(next.map(r => r.author).sort()).toEqual(['agent', 'user'])
  })

  it('retracting one author leaves the other intact', () => {
    const next = applyReaction([reaction('🔥', 'agent'), reaction('❤️', 'user')], null, 'user')

    expect(next).toMatchObject([{ emoji: '🔥', author: 'agent' }])
  })

  it('never mutates the input array', () => {
    const before = [reaction('❤️', 'user')]
    const snapshot = [...before]

    applyReaction(before, '😂', 'user')

    expect(before).toEqual(snapshot)
  })
})

describe('QUICK_REACTIONS', () => {
  it('is the six iOS Tapback defaults, each distinct', () => {
    expect(QUICK_REACTIONS).toHaveLength(6)
    expect(new Set(QUICK_REACTIONS).size).toBe(6)
  })
})

describe('toggleMessageReaction session binding', () => {
  it('uses the selected stored session while its runtime id is unset', async () => {
    reactionTestState.selectedStoredSessionId.set('stored-session')

    await toggleMessageReaction({ id: 'message-1', role: 'assistant', parts: [] }, '❤️')

    expect(reactionTestState.requestForOwnedSession).toHaveBeenCalledWith(
      'stored-session',
      expect.any(Function),
      'message.react',
      expect.objectContaining({ session_id: 'stored-session', newest_role: 'assistant', emoji: '❤️' })
    )
    expect(reactionTestState.gateway.request).toHaveBeenCalledTimes(1)
  })

  it('still rejects a genuinely new draft with no session identity', async () => {
    await toggleMessageReaction({ id: 'message-1', role: 'assistant', parts: [] }, '❤️')

    expect(reactionTestState.requestForOwnedSession).not.toHaveBeenCalled()
    expect(reactionTestState.gateway.request).not.toHaveBeenCalled()
  })
})
