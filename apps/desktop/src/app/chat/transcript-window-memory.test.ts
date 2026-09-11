import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { RENDER_WEIGHT_CHARS } from '@/lib/render-weight'

import {
  advanceSessionTranscriptWindow,
  type SessionWindowMemo,
  TRANSCRIPT_WINDOW_MIN_MESSAGES
} from './transcript-window'

const message = (id: string, chars: number): ChatMessage => ({
  id,
  parts: [{ type: 'text', text: 'x'.repeat(chars) }],
  role: 'assistant'
})

const heavyTranscript = (): ChatMessage[] =>
  Array.from({ length: TRANSCRIPT_WINDOW_MIN_MESSAGES * 3 }, (_, index) =>
    message(`m-${index}`, RENDER_WEIGHT_CHARS * 200)
  )

describe('SessionWindowMemo memory ownership', () => {
  it('holds the complete source transcript weakly while preserving warm slice identity', () => {
    const memos = new Map<string, SessionWindowMemo>()
    const messages = heavyTranscript()

    const first = advanceSessionTranscriptWindow(memos, 'session-a', messages)
    const memo = memos.get('session-a')

    expect(memo).toBeDefined()
    expect(first.window.windowed).toBe(true)
    expect(first.window.messages).not.toBe(messages)
    expect(memo?.messagesRef.deref()).toBe(messages)
    expect(Object.hasOwn(memo!, 'messages')).toBe(false)

    const second = advanceSessionTranscriptWindow(memos, 'session-a', messages)

    expect(second).toBe(first)
    expect(second.window.messages).toBe(first.window.messages)
  })
})
