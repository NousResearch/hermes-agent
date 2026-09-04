import { afterEach, describe, expect, it } from 'vitest'

import {
  absorbSpokenReplyRewrite,
  assistantReplyOrdinal,
  clearSpokenRepliesForTests,
  isLiveTailReplyId,
  markAssistantIdSpoken,
  resolveSpokenReply,
  spokenReplyOf
} from './spoken-reply'

const assistant = (id: string) => ({ id, role: 'assistant' as const })
const user = (id: string) => ({ id, role: 'user' as const })
const hidden = (id: string) => ({ hidden: true, id, role: 'assistant' as const })

afterEach(() => {
  clearSpokenRepliesForTests()
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

  it('uses durable history when a stale snapshot marks an older durable reply', () => {
    const current = [assistant('durable-1'), assistant('durable-2')]
    markAssistantIdSpoken('s', current, 'durable-2')

    const stale = [assistant('durable-1')]
    markAssistantIdSpoken('s', stale, 'durable-1')

    expect(resolveSpokenReply('s', stale)).toEqual({ id: 'durable-1', ordinal: 0 })
    expect(resolveSpokenReply('s', current)).toEqual({ id: 'durable-2', ordinal: 1 })
  })

  it('lets a new low-ordinal live reply become consumed after compaction', () => {
    const beforeCompaction = [
      assistant('durable-1'),
      assistant('durable-2'),
      assistant('durable-3')
    ]

    markAssistantIdSpoken('s', beforeCompaction, 'durable-3')

    const compacted = [assistant('assistant-stream-new')]
    markAssistantIdSpoken('s', compacted, 'assistant-stream-new')

    expect(spokenReplyOf('s')).toEqual({ id: 'assistant-stream-new', ordinal: 0 })
    expect(resolveSpokenReply('s', compacted)).toEqual({ id: 'assistant-stream-new', ordinal: 0 })
  })

  it('keeps repeated notifications for the same durable reply consumed', () => {
    const messages = [assistant('durable-1')]
    markAssistantIdSpoken('s', messages, 'durable-1')
    markAssistantIdSpoken('s', messages, 'durable-1')

    expect(resolveSpokenReply('s', messages)).toEqual({ id: 'durable-1', ordinal: 0 })
  })

  it('leaves a later distinct durable reply speakable', () => {
    const first = [assistant('durable-1')]
    markAssistantIdSpoken('s', first, 'durable-1')

    const later = [assistant('durable-1'), assistant('durable-2')]
    expect(resolveSpokenReply('s', later)).toEqual({ id: 'durable-1', ordinal: 0 })
    expect(later.at(-1)?.id).not.toBe(resolveSpokenReply('s', later)?.id)
  })

  it('keeps the existing live-to-durable rewrite consumed', () => {
    markAssistantIdSpoken('s', [assistant('assistant-stream-1')], 'assistant-stream-1')

    expect(resolveSpokenReply('s', [assistant('durable-1')])).toEqual({ id: 'durable-1', ordinal: 0 })
    expect(spokenReplyOf('s')).toEqual({ id: 'durable-1', ordinal: 0 })
  })
})
