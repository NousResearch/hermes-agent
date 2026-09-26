import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { RENDER_WEIGHT_CHARS } from '@/lib/render-weight'

import {
  advanceSessionTranscriptWindow,
  type SessionWindowMemo,
  TRANSCRIPT_WINDOW_MIN_MESSAGES
} from './transcript-window'

const transcript = (count: number, chars: number): ChatMessage[] =>
  Array.from({ length: count }, (_, index) => ({
    id: `message-${index}`,
    parts: [{ type: 'text', text: 'x'.repeat(chars) }],
    role: 'assistant'
  }))

describe('session transcript-window retention', () => {
  it('weakly remembers a windowed source array without breaking warm slice reuse', () => {
    const memos = new Map<string, SessionWindowMemo>()
    const messages = transcript(80, RENDER_WEIGHT_CHARS * 100)

    const first = advanceSessionTranscriptWindow(memos, 'session-a', messages)
    const memo = memos.get('session-a')

    expect(first.window.windowed).toBe(true)
    expect(first.window.messages).not.toBe(messages)
    expect(memo?.messages).toBeInstanceOf(WeakRef)
    expect(memo?.messages.deref()).toBe(messages)

    const warm = advanceSessionTranscriptWindow(memos, 'session-a', messages)

    expect(warm).toBe(first)
    expect(warm.window.messages).toBe(first.window.messages)
  })

  it('does not memoize an unwindowed source array that would remain strongly retained', () => {
    const memos = new Map<string, SessionWindowMemo>()
    // The minimum-message floor deliberately keeps this enormous short
    // transcript whole. Its pass-through identity needs no memo.
    const messages = transcript(TRANSCRIPT_WINDOW_MIN_MESSAGES - 1, RENDER_WEIGHT_CHARS * 10_000)

    const state = advanceSessionTranscriptWindow(memos, 'session-short-heavy', messages)

    expect(state.window.windowed).toBe(false)
    expect(state.window.messages).toBe(messages)
    expect(memos.has('session-short-heavy')).toBe(false)
  })

  it('drops an old window memo when paging expands the session to pass-through', () => {
    const memos = new Map<string, SessionWindowMemo>()
    const messages = transcript(80, RENDER_WEIGHT_CHARS * 20)

    const first = advanceSessionTranscriptWindow(memos, 'session-expanded', messages)
    expect(first.window.windowed).toBe(true)
    expect(memos.has('session-expanded')).toBe(true)

    const expanded = advanceSessionTranscriptWindow(memos, 'session-expanded', messages, 100)

    expect(expanded.window.windowed).toBe(false)
    expect(expanded.window.messages).toBe(messages)
    expect(memos.has('session-expanded')).toBe(false)
  })
})
