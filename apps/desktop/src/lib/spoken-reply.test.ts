import { afterEach, describe, expect, it } from 'vitest'

import {
  absorbSpokenReplyRewrite,
  adoptSpokenReplySession,
  assistantReplyOrdinal,
  clearSpokenRepliesForTests,
  isLiveTailReplyId,
  markAssistantIdSpoken,
  markTurnSpokenThrough,
  nextUnspokenTurnPiece,
  releaseUnplayedSpokenReply,
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

describe('nextUnspokenTurnPiece', () => {
  interface Row {
    hidden?: boolean
    id: string
    pending?: boolean
    role: 'assistant' | 'user'
    text: string
  }

  const said = (id: string, text: string, pending = false): Row => ({ id, pending, role: 'assistant', text })
  const asked = (id: string): Row => ({ id, role: 'user', text: 'what time is it?' })
  const textOf = (row: Row) => row.text
  const next = (rows: Row[]) => nextUnspokenTurnPiece('s', rows, textOf)

  const speak = (rows: Row[]) => {
    const piece = next(rows)

    if (piece) {
      markTurnSpokenThrough('s', rows, piece.id, piece.through)
    }

    return piece?.text ?? null
  }

  it('reads a tool turn bubble by bubble, in order, each once', () => {
    const rows = [
      asked('u1'),
      said('assistant-stream-1', 'Let me check the clock.'),
      said('assistant-stream-2', 'Now the host name.'),
      said('assistant-stream-3', 'All done.')
    ]

    expect(speak(rows)).toBe('Let me check the clock.')
    expect(speak(rows)).toBe('Now the host name.')
    expect(speak(rows)).toBe('All done.')
    expect(next(rows)).toBeNull()
  })

  it('offers a sealed narration while a newer bubble is still streaming', () => {
    const rows = [asked('u1'), said('assistant-stream-1', 'Let me check.'), said('assistant-stream-2', '', true)]

    expect(next(rows)).toMatchObject({ id: 'assistant-stream-1', pending: false, text: 'Let me check.' })
  })

  it('waits for a bubble that is still streaming', () => {
    expect(next([asked('u1'), said('assistant-stream-1', 'Let me', true)])).toMatchObject({ pending: true })
  })

  it('reads the rest of a turn after hydration folds its bubbles into one row', () => {
    const live = [
      asked('u1'),
      said('assistant-stream-1', 'Let me check the clock.'),
      said('assistant-stream-2', 'Now the host name.'),
      said('assistant-stream-3', 'All done.')
    ]

    expect(speak(live)).toBe('Let me check the clock.')

    // The spoken live row is gone, and the ordinal moved. The answer was never
    // read, so the fold must not count as "already spoken".
    const folded = [asked('1-user'), said('2-assistant', 'Let me check the clock.Now the host name.All done.')]

    expect(speak(folded)).toBe('Now the host name.All done.')
    expect(next(folded)).toBeNull()
  })

  it('stays silent after the fold when the whole turn was already read', () => {
    const live = [asked('u1'), said('assistant-stream-1', 'Checking.'), said('assistant-stream-2', 'Done.')]

    speak(live)
    speak(live)

    expect(next([asked('1-user'), said('2-assistant', 'Checking.\n\nDone.')])).toBeNull()
  })

  it('never re-reads a turn whose stored text no longer lines up', () => {
    speak([asked('u1'), said('assistant-stream-1', 'Checking.'), said('assistant-stream-2', 'Done.', true)])

    expect(next([asked('1-user'), said('2-assistant', 'Something else entirely.')])).toBeNull()
  })

  it('treats a whole-bubble mark as the turn read through that bubble', () => {
    const rows = [asked('u1'), said('a1', 'Checking.'), said('a2', 'Done.')]

    markAssistantIdSpoken('s', rows, 'a2')

    expect(next(rows)).toBeNull()
  })

  it('reads a new turn from its start', () => {
    const first = [asked('u1'), said('a1', 'Done.')]

    markAssistantIdSpoken('s', first, 'a1')

    expect(next([...first, asked('u2'), said('assistant-stream-2', 'Done.')])).toMatchObject({
      start: 0,
      text: 'Done.'
    })
  })
})

describe('releaseUnplayedSpokenReply', () => {
  it('rolls back to the previous mark, keeping the pieces already read', () => {
    const rows = [user('u1'), assistant('a1'), assistant('a2')]

    markTurnSpokenThrough('s', rows, 'a1', 'Checking.')
    const previous = spokenReplyOf('s')

    markTurnSpokenThrough('s', rows, 'a2', 'Checking.Done.')
    releaseUnplayedSpokenReply('s', spokenReplyOf('s'), previous)

    expect(spokenReplyOf('s')).toEqual(previous)
  })
})
