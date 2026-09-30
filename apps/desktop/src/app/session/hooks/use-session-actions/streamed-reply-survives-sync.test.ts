import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const tool = (id: string): ChatMessagePart =>
  ({ type: 'tool-call', toolCallId: id, toolName: 'terminal', result: 'done' }) as ChatMessagePart

const message = (
  id: string,
  role: ChatMessage['role'],
  parts: ChatMessagePart[],
  extra: Partial<ChatMessage> = {}
): ChatMessage => ({ id, role, parts, ...extra }) as ChatMessage

const liveUser = message('user-live', 'user', [textPart('check the model')])

const settledStream = (parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}) =>
  message('assistant-stream-live', 'assistant', parts, { interim: false, pending: false, ...extra })

const storedCall = { id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{}' } }

// #123047: the store committed the turn as a tool round with content = '' and
// no prose row after it, while the renderer streamed and settled the answer.
describe('preserveLocalPendingTurnMessages — settled reply over an empty tool shell (#123047)', () => {
  it('keeps the streamed final reply when the stored turn hydrates as a text-less tool shell', () => {
    const next = toChatMessages([
      { id: 196700, role: 'user', content: 'check the model', timestamp: 1 },
      { id: 196701, role: 'assistant', content: '', tool_calls: [storedCall], timestamp: 2 },
      { id: 196702, role: 'tool', content: 'done', tool_call_id: 'call-1', timestamp: 3 }
    ] as never)

    expect(next.map(row => chatMessageText(row))).toEqual(['check the model', ''])

    const previous = [liveUser, settledStream([textPart('I checked it.'), tool('call-1'), textPart('Model is up.')])]
    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toHaveLength(2)
    expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', pending: false })
    expect(chatMessageText(merged[1])).toContain('Model is up.')
  })

  it('carries the shell row identity onto the kept reply', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-shell', 'assistant', [tool('call-1')], { pending: false, rowId: 196701 })
    ]

    const merged = preserveLocalPendingTurnMessages(next, [liveUser, settledStream([textPart('Model is up.')])])

    expect(merged).toHaveLength(2)
    expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', pending: false, rowId: 196701 })
  })

  it('yields to the committed answer once a prose row folds into the tool round (no duplicate)', () => {
    const next = toChatMessages([
      { id: 196700, role: 'user', content: 'check the model', timestamp: 1 },
      { id: 196701, role: 'assistant', content: '', tool_calls: [storedCall], timestamp: 2 },
      { id: 196702, role: 'tool', content: 'done', tool_call_id: 'call-1', timestamp: 3 },
      { id: 196703, role: 'assistant', content: 'Model is up and serving.', timestamp: 4 }
    ] as never)

    const previous = [liveUser, settledStream([textPart('I checked it.'), tool('call-1'), textPart('Model is up.')])]
    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toEqual(next)
  })

  it('does not let settled interim narration replace the shell', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-shell', 'assistant', [tool('call-1')], { pending: false, rowId: 196701 })
    ]

    const previous = [liveUser, settledStream([textPart('Checking the logs…')], { interim: true })]

    expect(preserveLocalPendingTurnMessages(next, previous)).toEqual(next)
  })

  it('does not let a structural-only local row replace the shell', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-shell', 'assistant', [tool('call-stored')], { pending: false, rowId: 196701 })
    ]

    expect(preserveLocalPendingTurnMessages(next, [liveUser, settledStream([tool('call-local')])])).toEqual(next)
  })

  it('never repaints a retained error row from the local partial', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-error', 'assistant', [tool('call-1')], {
        error: 'provider failed',
        pending: false,
        rowId: 196701
      } as Partial<ChatMessage>)
    ]

    expect(preserveLocalPendingTurnMessages(next, [liveUser, settledStream([textPart('Model is up.')])])).toEqual(next)
  })
})

// #80151: the backend persists segments while the turn runs, so switching away
// mid-stream and back hydrates the store's flat mid-turn partial next to the
// still-streaming local copy. That partial need not be a byte prefix of the
// streamed text, so the strict-prefix pairing dropped the pre-switch content.
describe('preserveLocalPendingTurnMessages — divergent same-turn store partial (#80151)', () => {
  it('keeps a pending streamed reply when the store holds a same-turn partial whose flat text is not a byte prefix', () => {
    const previous = [
      message('user-live', 'user', [textPart('inspect the logs')]),
      message(
        'assistant-stream-live',
        'assistant',
        [textPart('Reading the logs now.'), tool('call-1'), textPart('Found the stale lock.')],
        { pending: true }
      )
    ]

    const next = [
      message('user-stored', 'user', [textPart('inspect the logs')], { rowId: 300 }),
      message('assistant-stored-partial', 'assistant', [textPart('Reading the logs now.\n\nFound the stale')], {
        pending: false,
        rowId: 301
      })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toHaveLength(2)
    expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', rowId: 301 })
    expect(merged[1].parts.some(part => part.type === 'tool-call')).toBe(true)
    expect(chatMessageText(merged[1])).toContain('Found the stale lock.')
  })

  it('keeps a pending streamed reply over a committed same-turn partial proven by shared tool-call ids', () => {
    // A tool-heavy turn: the store committed the tool round plus the segment
    // after it while the turn was still running. The committed text is not a
    // prefix of the streamed text (the stream also holds the pre-tool
    // commentary), but the shared tool-call id proves the same turn.
    const previous = [
      message('user-live', 'user', [textPart('inspect the logs')]),
      message(
        'assistant-stream-live',
        'assistant',
        [textPart('Reading the logs now.'), tool('call-9'), textPart('Removed the stale lock.')],
        { pending: true }
      )
    ]

    const next = [
      message('user-stored', 'user', [textPart('inspect the logs')], { rowId: 310 }),
      message('assistant-stored-partial', 'assistant', [tool('call-9'), textPart('Removed the stale lock.')], {
        pending: false,
        rowId: 311
      })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toHaveLength(2)
    expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', rowId: 311 })
    expect(chatMessageText(merged[1])).toContain('Reading the logs now.')
    expect(merged[1].parts.some(part => part.type === 'tool-call')).toBe(true)
  })

  it('does not let a pending stream claim a different turn\'s committed answer at the same ordinal', () => {
    // No shared tool-call ids and no folded prefix relation: the committed row
    // is an earlier answer to the same resent prompt. The widening must not
    // let the unrelated pending stream take its slot, and the stale-copy rule
    // keeps dropping the local row.
    const previous = [
      message('user-live', 'user', [textPart('inspect the logs')]),
      message('assistant-stream-live', 'assistant', [textPart('Fresh streaming reply, different wording.')], {
        pending: true
      })
    ]

    const next = [
      message('user-stored', 'user', [textPart('inspect the logs')], { rowId: 320 }),
      message('assistant-stored-old-answer', 'assistant', [textPart('An older, unrelated answer text.')], {
        pending: false,
        rowId: 321
      })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toEqual(next)
  })

  it('does not trade a committed partial for a pending stream that is behind its text', () => {
    // Same turn (shared tool-call id, folded prefix), but the local stream has
    // LESS text than the store already persisted — replacing would lose visible
    // content, so the committed partial keeps the slot.
    const previous = [
      message('user-live', 'user', [textPart('inspect the logs')]),
      message('assistant-stream-live', 'assistant', [textPart('Reading the'), tool('call-4')], { pending: true })
    ]

    const next = [
      message('user-stored', 'user', [textPart('inspect the logs')], { rowId: 330 }),
      message('assistant-stored-partial', 'assistant', [tool('call-4'), textPart('Reading the logs now. Done.')], {
        pending: false,
        rowId: 331
      })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged[1]).toMatchObject({ id: 'assistant-stored-partial', rowId: 331 })
    expect(chatMessageText(merged[1])).toContain('Reading the logs now. Done.')
  })
})
