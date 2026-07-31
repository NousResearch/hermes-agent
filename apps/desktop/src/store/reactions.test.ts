import { describe, expect, it } from 'vitest'

import { applyReaction, QUICK_REACTIONS } from '@/store/reactions'
import type { MessageReaction } from '@/types/hermes'

const at = 1_700_000_000

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
