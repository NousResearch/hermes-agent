import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { focusedChatMessages } from './session-files-store'

const msgs = (id: string) => [{ id, parts: [], role: 'assistant' }] as unknown as ChatMessage[]

describe('focusedChatMessages', () => {
  const main = msgs('main')
  const tile = msgs('tile')
  const primaryFallback = msgs('primary-atom')

  it("uses the focused tile's own messages, never the main chat's", () => {
    expect(
      focusedChatMessages(
        'tile-rt',
        'main-rt',
        { 'main-rt': { messages: main }, 'tile-rt': { messages: tile } },
        primaryFallback
      )
    ).toBe(tile)
  })

  it('gives a tile without messages yet an empty list instead of the main chat', () => {
    expect(focusedChatMessages('tile-rt', 'main-rt', { 'main-rt': { messages: main } }, primaryFallback)).toEqual([])
  })

  it("uses the main chat's state when the main chat is focused", () => {
    expect(focusedChatMessages('main-rt', 'main-rt', { 'main-rt': { messages: main } }, primaryFallback)).toBe(main)
  })

  it('falls back to the primary transcript only for the main chat', () => {
    expect(focusedChatMessages('main-rt', 'main-rt', {}, primaryFallback)).toBe(primaryFallback)
  })

  it('is empty when nothing is focused', () => {
    expect(focusedChatMessages(null, 'main-rt', {}, primaryFallback)).toEqual([])
  })
})
