import { afterEach, describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { assistantTextPart, type ChatMessage, collectUnspokenTurnSpeech, lastTextPartText, toChatMessages } from './chat-messages'
import {
  absorbSpokenReplyRewrite,
  adoptSpokenReplySession,
  assistantReplyOrdinal,
  assistantTurnIndex,
  assistantTurnKey,
  clearSpokenRepliesForTests,
  isLiveTailReplyId,
  markAssistantIdSpoken,
  resolveSpokenReply,
  spokenReplyOf,
  spokenSegmentKey
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

  it('follows a live-tail rewrite when tool rows change the assistant ordinal', () => {
    const before = [user('u1'), assistant('narration'), assistant('tool-segment'), assistant('assistant-stream-s')]

    markAssistantIdSpoken('s', before, 'assistant-stream-s')

    // Hydration folds the tool segments into one bubble. The assistant ordinal
    // moves; the user turn does not.
    const after = [user('u1'), assistant('42')]

    expect(resolveSpokenReply('s', after)?.id).toBe('42')
    expect(spokenReplyOf('s')?.id).toBe('42')
  })

  it('does not mark a later user turn spoken when the live tail vanished', () => {
    markAssistantIdSpoken('s', [user('u1'), assistant('assistant-stream-s')], 'assistant-stream-s')

    const after = [user('u1'), assistant('durable-1'), user('u2'), assistant('assistant-stream-next')]
    const spoken = resolveSpokenReply('s', after)

    expect(spoken?.id).not.toBe('assistant-stream-next')
    expect(after.findLast(message => message.role === 'assistant')?.id).not.toBe(spoken?.id)
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

describe('spokenSegmentKey', () => {
  it('survives a live-id rewrite: same turn, same text, same key', () => {
    expect(spokenSegmentKey('s1:0', 'the folded answer')).toBe(spokenSegmentKey('s1:0', 'the folded answer'))
  })

  it('separates the segments of one tool-run turn', () => {
    expect(spokenSegmentKey('s1:0', 'Looking into it.')).not.toBe(spokenSegmentKey('s1:0', 'Here is the final report.'))
  })

  it('separates two turns that say the same thing', () => {
    expect(spokenSegmentKey('s1:0', 'Done.')).not.toBe(spokenSegmentKey('s1:1', 'Done.'))
  })
})

// Hydration folds a tool-call turn's assistant rows into one bubble, so its full
// text is progress + final. A window that only sees that history must derive
// the same key as the window that spoke the live final bubble, or both read it.
describe('spokenSegmentKey across a hydration fold', () => {
  it("gives the hydrated bubble the live final bubble's key", () => {
    const stored: SessionMessage[] = [
      { role: 'user', content: 'fix it', timestamp: 1 },
      {
        role: 'assistant',
        content: 'Looking into it.',
        timestamp: 2,
        tool_calls: [{ id: 'tc', function: { name: 'terminal', arguments: '{}' } }]
      },
      { role: 'tool', tool_call_id: 'tc', content: 'ok', timestamp: 3 },
      { role: 'assistant', content: 'Here is the final report.', timestamp: 4 }
    ]

    const hydrated = toChatMessages(stored)
    const folded = hydrated.findLast(m => m.role === 'assistant')!

    // The fold really did merge both messages into one bubble.
    expect(hydrated.filter(m => m.role === 'assistant')).toHaveLength(1)

    const live: ChatMessage = {
      id: 'assistant-stream-2',
      parts: [assistantTextPart('Here is the final report.')],
      role: 'assistant'
    }

    expect(spokenSegmentKey('s1:0', lastTextPartText(folded))).toBe(spokenSegmentKey('s1:0', lastTextPartText(live)))
    expect(spokenSegmentKey('s1:0', lastTextPartText(folded))).not.toBe(spokenSegmentKey('s1:0', 'Looking into it.'))
  })
})

// An internal notification that ran a turn (`display_kind: internal_notification`
// without `completion_admission`) is a turn boundary like a user row: its reply
// is a new turn, not a live-id rewrite of the human turn before it. A recorded
// note is filed between turns and starts none. Rows come from real hydration so
// the provenance is the one the thread actually carries.
describe('notification turn identity', () => {
  const human = (content: string, timestamp: number): SessionMessage => ({ role: 'user', content, timestamp })
  const reply = (content: string, timestamp: number): SessionMessage => ({ role: 'assistant', content, timestamp })
  const notice = (content: string, timestamp: number): SessionMessage => ({
    role: 'user',
    content,
    display_kind: 'internal_notification',
    timestamp
  })
  const note = (content: string, timestamp: number): SessionMessage => ({
    ...notice(content, timestamp),
    display_metadata: { completion_admission: 'note-id' }
  })
  const replies = (messages: readonly ChatMessage[]) => messages.filter(m => m.role === 'assistant')

  it('gives a notification reply its own turn after the human turn that says the same thing', () => {
    const messages = toChatMessages([human('First question', 1), reply('Done.', 2), notice('Plugin notice', 3), reply('Done.', 4)])
    const [first, second] = replies(messages)

    expect(messages[2].internalNotification).toBe('turn')
    expect(assistantTurnIndex(messages, first.id)).toBe(0)
    expect(assistantTurnIndex(messages, second.id)).toBe(1)
    expect(spokenSegmentKey(assistantTurnKey('s', messages, second.id), 'Done.')).not.toBe(
      spokenSegmentKey(assistantTurnKey('s', messages, first.id), 'Done.')
    )
  })

  it('separates successive notification turns, even with identical reply text', () => {
    const messages = toChatMessages([
      human('First question', 1),
      reply('Done.', 2),
      notice('Cron fired', 3),
      reply('Done.', 4),
      notice('Cron fired', 5),
      reply('Done.', 6)
    ])
    const [, second, third] = replies(messages)

    expect(assistantTurnIndex(messages, second.id)).toBe(1)
    expect(assistantTurnIndex(messages, third.id)).toBe(2)
    expect(spokenSegmentKey(assistantTurnKey('s', messages, third.id), 'Done.')).not.toBe(
      spokenSegmentKey(assistantTurnKey('s', messages, second.id), 'Done.')
    )
  })

  it('does not count a recorded note as a turn boundary', () => {
    const messages = toChatMessages([human('First question', 1), reply('Done.', 2), note('Recorded note', 3), reply('Done.', 4)])
    const [first, second] = replies(messages)

    expect(messages[2].internalNotification).toBe('note')
    expect(assistantTurnIndex(messages, first.id)).toBe(0)
    expect(assistantTurnIndex(messages, second.id)).toBe(0)
  })

  it('still counts a hidden human row as a turn boundary', () => {
    const messages = [user('u1'), assistant('a1'), { hidden: true, id: 'u2', role: 'user' as const }, assistant('a2')]

    expect(assistantTurnIndex(messages, 'a1')).toBe(0)
    expect(assistantTurnIndex(messages, 'a2')).toBe(1)
  })

  it('does not absorb the next notification reply as the rewrite of a stale human-turn live id', () => {
    const before = toChatMessages([human('First question', 1), reply('Done.', 2)])
    before[1].id = 'assistant-stream-first'
    markAssistantIdSpoken('s', before, 'assistant-stream-first')

    const after = toChatMessages([human('First question', 1), reply('Done.', 2), notice('Plugin notice', 3), reply('Done.', 4)])
    const last = after.findLast(m => m.role === 'assistant')!
    const spoken = resolveSpokenReply('s', after)

    expect(spoken?.id).toBe('assistant-stream-first')
    expect(spoken?.id).not.toBe(last.id)
    expect(collectUnspokenTurnSpeech(after, spoken?.id ?? null)).toEqual({ id: last.id, pending: false, text: 'Done.' })
  })

  it('does not replay the human turn when nothing was spoken before a notification turn', () => {
    const after = toChatMessages([human('First question', 1), reply('Earlier reply', 2), notice('Plugin notice', 3), reply('Current reply', 4)])
    const last = after.findLast(m => m.role === 'assistant')!

    expect(resolveSpokenReply('s', after)).toBeNull()
    expect(collectUnspokenTurnSpeech(after, null)).toEqual({ id: last.id, pending: false, text: 'Current reply' })
  })

  it("follows a notification turn's own live-id rewrite onto its durable id", () => {
    const before = toChatMessages([human('First question', 1), reply('Done.', 2), notice('Plugin notice', 3), reply('Done.', 4)])
    before[3].id = 'assistant-stream-notice'
    markAssistantIdSpoken('s', before, 'assistant-stream-notice')

    const after = toChatMessages([human('First question', 1), reply('Done.', 2), notice('Plugin notice', 3), reply('Done.', 4)])
    const last = after.findLast(m => m.role === 'assistant')!

    expect(spokenReplyOf('s')?.turnIndex).toBe(1)
    expect(resolveSpokenReply('s', after)?.id).toBe(last.id)
    expect(collectUnspokenTurnSpeech(after, last.id)).toBeNull()
  })
})
