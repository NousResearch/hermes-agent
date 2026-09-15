import { afterEach, describe, expect, it } from 'vitest'

import {
  absorbSpokenReplyRewrite,
  adoptSpokenReplySession,
  assistantReplyOrdinal,
  clearSpokenRepliesForTests,
  isLiveTailReplyId,
  latestSpeechUser,
  markAssistantIdSpoken,
  resolveSpokenReply,
  sameSpeechUser,
  speechSourceDelta,
  spokenReplyOf
} from './spoken-reply'

const assistant = (id: string) => ({ id, role: 'assistant' as const })
const user = (id: string) => ({ id, role: 'user' as const })
const hidden = (id: string) => ({ hidden: true, id, role: 'assistant' as const })

afterEach(() => {
  clearSpokenRepliesForTests()
})

describe('speech source reconciliation', () => {
  it.each([
    ['一文。  \n二文。', '一文。\n二文。次です。', '次です。'],
    ['First.\n\nSecond.', 'First. Second.\n\nThird.', '\n\nThird.'],
    ['🌸 一文。  \n二文。', '🌸 一文。\n二文。', ''],
    ['前の内容です。', '別の内容です。', null],
    ['一文。二文。', '一文。', null]
  ])('feeds only a new suffix across display formatting', (previous, next, delta) => {
    expect(speechSourceDelta(previous!, next!)).toBe(delta)
  })

  it('distinguishes a hydrated prompt from a new identical prompt', () => {
    const parts = [{ type: 'text', text: '同じ質問' }]
    const live = latestSpeechUser([{ ...user('user-live'), parts }])
    const stored = latestSpeechUser([{ ...user('100-3-user'), rowId: 42, parts }])
    expect(sameSpeechUser(live, stored)).toBe(true)
    expect(sameSpeechUser(stored, latestSpeechUser([{ ...user('100-8-user'), rowId: 42, parts }]))).toBe(true)
    expect(sameSpeechUser(live, latestSpeechUser([{ ...user('user-new'), parts }]))).toBe(false)
    expect(sameSpeechUser(stored, latestSpeechUser([{ ...user('101-4-user'), rowId: 43, parts }]))).toBe(false)
  })

  it('does not swallow an identical answer in a later turn after history reconciliation', () => {
    const parts = [{ type: 'text', text: '同じ質問' }]
    const answerParts = [{ type: 'text', text: '同じ回答' }]
    markAssistantIdSpoken(
      's',
      [
        { ...user('user-live'), parts },
        { ...assistant('assistant-stream-1'), parts: answerParts }
      ],
      'assistant-stream-1'
    )

    const stored = [
      { ...user('stored-user'), rowId: 42, parts },
      { ...assistant('stored-answer'), rowId: 43, parts: answerParts }
    ]

    expect(resolveSpokenReply('s', stored)?.id).toBe('stored-answer')
    expect(
      resolveSpokenReply('s', [
        ...stored,
        { ...user('user-next'), parts },
        { ...assistant('assistant-stream-next'), parts: answerParts }
      ])?.id
    ).toBe('stored-answer')
  })
})

describe('isLiveTailReplyId', () => {
  it('matches renderer stream and inflight ids only', () => {
    expect(isLiveTailReplyId('assistant-stream-s1')).toBe(true)
    expect(isLiveTailReplyId('inflight-assistant-9')).toBe(true)
    expect(isLiveTailReplyId('42')).toBe(false)
    expect(isLiveTailReplyId('1770-3-assistant')).toBe(false)
  })
})

describe('assistantReplyOrdinal', () => {
  it('counts visible assistant bubbles and skips hidden ones', () => {
    const messages = [user('u1'), assistant('a1'), hidden('skip'), assistant('a2')]

    expect(assistantReplyOrdinal(messages, 'a1')).toBe(0)
    expect(assistantReplyOrdinal(messages, 'a2')).toBe(1)
    expect(assistantReplyOrdinal(messages, 'missing')).toBe(-1)
  })
})

describe('absorbSpokenReplyRewrite', () => {
  it('stays silent when the live-tail id is rewritten at the same ordinal', () => {
    const spoken = { id: 'assistant-stream-s', ordinal: 1 }
    const after = [user('u1'), assistant('a0'), assistant('42')]

    expect(absorbSpokenReplyRewrite(spoken, after)).toEqual({ id: '42', ordinal: 1 })
  })

  it('does not treat a later same-slot-looking turn as the rewrite when ordinal moved', () => {
    const spoken = { id: 'assistant-stream-s', ordinal: 0 }
    const after = [user('u1'), assistant('durable-1'), user('u2'), assistant('assistant-stream-next')]

    expect(absorbSpokenReplyRewrite(spoken, after)).toEqual(spoken)
  })

  it('does not migrate a durable id that simply vanished', () => {
    const spoken = { id: 'durable-old', ordinal: 0 }
    const after = [assistant('durable-new')]

    expect(absorbSpokenReplyRewrite(spoken, after)).toEqual(spoken)
  })

  it('keeps the anchor when the spoken id is still in the list', () => {
    const spoken = { id: 'assistant-stream-s', ordinal: 0 }
    const messages = [assistant('assistant-stream-s')]

    expect(absorbSpokenReplyRewrite(spoken, messages)).toBe(spoken)
  })
})

describe('resolveSpokenReply', () => {
  it('migrates per session and does not leak across sessions', () => {
    const before = [assistant('assistant-stream-s')]
    markAssistantIdSpoken('session-a', before, 'assistant-stream-s')

    const after = [assistant('42')]
    expect(resolveSpokenReply('session-a', after)?.id).toBe('42')
    expect(spokenReplyOf('session-b')).toBeNull()
    expect(resolveSpokenReply('session-b', after)).toBeNull()
  })

  it('lets a second turn at the next ordinal stay unspoken', () => {
    markAssistantIdSpoken('s', [assistant('assistant-stream-1')], 'assistant-stream-1')
    resolveSpokenReply('s', [assistant('durable-1')])

    const nextTurn = [assistant('durable-1'), assistant('assistant-stream-2')]
    const spoken = resolveSpokenReply('s', nextTurn)

    expect(spoken?.id).toBe('durable-1')
    expect(assistantReplyOrdinal(nextTurn, 'assistant-stream-2')).toBe(1)
    expect(spoken?.ordinal).toBe(0)
  })
})

describe('adoptSpokenReplySession', () => {
  it('moves the null-session anchor onto the created session id', () => {
    markAssistantIdSpoken(null, [assistant('a1')], 'a1')
    adoptSpokenReplySession(null, 'session-created')

    expect(spokenReplyOf('session-created')?.id).toBe('a1')
    // Moved, not copied: the next new chat (null session again) starts clean.
    expect(spokenReplyOf(null)).toBeNull()
  })

  it('does not overwrite an anchor the created session already has', () => {
    markAssistantIdSpoken(null, [assistant('a1')], 'a1')
    markAssistantIdSpoken('session-created', [assistant('a1'), assistant('a2')], 'a2')
    adoptSpokenReplySession(null, 'session-created')

    expect(spokenReplyOf('session-created')?.id).toBe('a2')
  })

  it('does not leak a spoken anchor from one real session into another', () => {
    markAssistantIdSpoken('session-a', [assistant('a1')], 'a1')
    adoptSpokenReplySession('session-a', 'session-b')

    expect(spokenReplyOf('session-b')).toBeNull()
    expect(spokenReplyOf('session-a')?.id).toBe('a1')
  })
})
