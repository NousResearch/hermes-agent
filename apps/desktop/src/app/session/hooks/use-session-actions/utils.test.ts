import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import type { SessionInflightSnapshot, SessionInfo } from '@/types/hermes'

import {
  chatMessageArraysEquivalent,
  isSessionGoneError,
  reconcileResumeInflight,
  reconcileResumeMessages,
  sessionMatchesStoredId,
  sessionShouldHaveTranscript,
  toBranchMessages
} from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

describe('isSessionGoneError', () => {
  it('is true for 404 / session-not-found, false otherwise', () => {
    expect(isSessionGoneError(new Error('Request failed 404'))).toBe(true)
    expect(isSessionGoneError(new Error('Session not found'))).toBe(true)
    expect(isSessionGoneError(new Error('ECONNREFUSED'))).toBe(false)
    expect(isSessionGoneError(null)).toBe(false)
  })
})

describe('sessionMatchesStoredId', () => {
  it('matches on live id or lineage root', () => {
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'a')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'live', _lineage_root_id: 'root' }), 'root')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'b')).toBe(false)
  })
})

describe('sessionShouldHaveTranscript', () => {
  it('is true only when the session has messages', () => {
    expect(sessionShouldHaveTranscript(session({ message_count: 3 }))).toBe(true)
    expect(sessionShouldHaveTranscript(session({ message_count: 0 }))).toBe(false)
    expect(sessionShouldHaveTranscript(undefined)).toBe(false)
  })
})

describe('toBranchMessages', () => {
  it('keeps only user/assistant turns that carry text', () => {
    const out = toBranchMessages([
      msg('u', 'user', 'hi'),
      msg('blank', 'assistant', '   '),
      msg('sys', 'system', 'ignored'),
      msg('a', 'assistant', 'hello')
    ])

    expect(out.map(b => b.source.id)).toEqual(['u', 'a'])
    expect(out[0]).toMatchObject({ content: 'hi', role: 'user' })
  })
})

describe('chatMessageArraysEquivalent', () => {
  it('compares length and per-message equivalence', () => {
    const a = [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')]
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')])).toBe(true)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x')])).toBe(false)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'changed')])).toBe(false)
  })
})

describe('reconcileResumeMessages', () => {
  it('returns next untouched when there is no previous transcript', () => {
    const next = [msg('1', 'user', 'hi')]
    expect(reconcileResumeMessages(next, [])).toBe(next)
  })

  it('re-grafts reasoning parts onto a matching assistant turn', () => {
    const next = [msg('a', 'assistant', 'answer')]

    const previous = [
      msg('a', 'assistant', 'answer', {
        parts: [
          { type: 'reasoning', text: 'thinking' },
          { type: 'text', text: 'answer' }
        ]
      } as Partial<ChatMessage>)
    ]

    const [out] = reconcileResumeMessages(next, previous)
    expect(out.parts.some(p => p.type === 'reasoning')).toBe(true)
  })
})

describe('reconcileResumeInflight', () => {
  const inflight = (over: Partial<SessionInflightSnapshot> = {}): SessionInflightSnapshot => ({
    assistant: 'partial answer',
    streaming: true,
    user: 'keep working',
    ...over
  })

  it('restores the active user turn and partial assistant as one pending stream', () => {
    const history = [msg('a0', 'assistant', 'previous answer')]
    const result = reconcileResumeInflight(history, inflight(), 'runtime-1')

    expect(result.messages.map(message => [message.role, chatMessageText(message), message.pending])).toEqual([
      ['assistant', 'previous answer', undefined],
      ['user', 'keep working', undefined],
      ['assistant', 'partial answer', true]
    ])
    expect(result.streamId).toBe('assistant-inflight-runtime-1')
    expect(result.sawAssistantPayload).toBe(true)
  })

  it('does not duplicate a user prompt already painted by the REST prefetch', () => {
    const history = [msg('u1', 'user', 'keep working')]
    const result = reconcileResumeInflight(history, inflight({ assistant: '' }), 'runtime-2')

    expect(result.messages).toBe(history)
    expect(result.streamId).toBe('assistant-inflight-runtime-2')
    expect(result.sawAssistantPayload).toBe(false)
  })

  it('preserves a repeated prompt when it starts a later turn', () => {
    const history = [msg('u0', 'user', 'keep working'), msg('a0', 'assistant', 'done once')]
    const result = reconcileResumeInflight(history, inflight({ assistant: '' }), 'runtime-3')

    expect(result.messages).toHaveLength(3)
    expect(result.messages.at(-1)).toMatchObject({ role: 'user' })
    expect(chatMessageText(result.messages.at(-1)!)).toBe('keep working')
  })

  it('marks an already-materialized matching assistant tail pending and reuses its id', () => {
    const history = [msg('u1', 'user', 'keep working'), msg('a1', 'assistant', 'partial answer')]
    const result = reconcileResumeInflight(history, inflight(), 'runtime-4')

    expect(result.messages).toHaveLength(2)
    expect(result.messages[1]).toMatchObject({ id: 'a1', pending: true })
    expect(result.streamId).toBe('a1')
  })

  it('leaves a completed transcript untouched when there is no live snapshot', () => {
    const history = [msg('a0', 'assistant', 'done')]
    const result = reconcileResumeInflight(history, null, 'runtime-5')

    expect(result.messages).toBe(history)
    expect(result).toMatchObject({ sawAssistantPayload: false, streamId: null })
  })
})
