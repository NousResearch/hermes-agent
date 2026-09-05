import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { appendLiveSessionProjection, chatMessagesEquivalent, overlayConcurrentMessageChanges } from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { persistInFlightTurnState, readInFlightTurnJournal, recoverInFlightTurnJournal, resetInFlightTurnJournalStateForTests } from '@/lib/inflight-turn-journal'

const oldHistory = () => toChatMessages([
  { role: 'user', content: 'earlier task', user_originated: true, timestamp: 1 },
  { role: 'assistant', content: 'earlier completed answer', timestamp: 2 }
])

const snapshot = (displayKind = 'hidden', id = 'runtime', start = 10) => ({
  session_id: id, turn_started_at: start,
  inflight: { user: 'runtime wake', display_kind: displayKind, user_originated: false, assistant: 'new partial answer', streaming: true }
})

const textRows = (messages: ChatMessage[], value: string) => messages.filter(message => chatMessageText(message) === value)

const withTool = (messages: ChatMessage[]) => messages.map((message, index) => index === messages.length - 1 ? {
  ...message, parts: [{ type: 'tool-call' as const, toolName: 'terminal', toolCallId: 'runtime-call', args: {} }, ...message.parts]
} : message)

function record(messages: ChatMessage[], streamId: string | null = messages.findLast(message => message.role === 'assistant')?.id ?? null) {
  persistInFlightTurnState({
    storedSessionId: 'stored', messages, streamId, busy: true, awaitingResponse: false,
    // Deliberately unrelated renderer clock. Runtime identity comes only from
    // the backend projection, not this UI timer or ChatMessage.timestamp.
    turnStartedAt: 9876543210000
  })
  vi.advanceTimersByTime(400)
}

beforeEach(() => {
  vi.useFakeTimers()
  resetInFlightTurnJournalStateForTests()
  window.localStorage.clear()
})
afterEach(() => {
  resetInFlightTurnJournalStateForTests()
  vi.useRealTimers()
})

it.each(['hidden', 'auto_continue', 'internal_notification'])('recovers %s runtime output after an older completed human turn', kind => {
  const base = oldHistory()
  record(appendLiveSessionProjection(base, snapshot(kind)))
  const saved = readInFlightTurnJournal('stored')!
  expect(textRows(saved.messages, 'earlier task')).toHaveLength(0)
  expect(textRows(saved.messages, 'earlier completed answer')[0]?.runtimeTurnStartedAt).toBeUndefined()
  const recovered = recoverInFlightTurnJournal('stored', base)
  expect(textRows(recovered.messages, 'new partial answer')).toHaveLength(1)
  expect(recovered.streamId).toBeNull()
  expect(textRows(recovered.messages, 'runtime wake')).toHaveLength(kind === 'internal_notification' ? 1 : 0)
  expect(textRows(recovered.messages, 'resumed interrupted turn')).toHaveLength(kind === 'auto_continue' ? 1 : 0)
})

it.each(['hidden', 'auto_continue'])('overlays journaled %s structure onto the same runtime in a new session binding', kind => {
  const live = withTool(appendLiveSessionProjection(oldHistory(), snapshot(kind, 'before')))
  record(live)
  const base = appendLiveSessionProjection(oldHistory(), snapshot(kind, 'after'))
  const recovered = recoverInFlightTurnJournal('stored', base, { keepPending: true })
  const answer = textRows(recovered.messages, 'new partial answer')
  expect(answer).toHaveLength(1)
  expect(answer[0].parts.some(part => part.type === 'tool-call')).toBe(true)
  expect(recovered.streamId).toBe('assistant-stream-after')
  const repeated = recoverInFlightTurnJournal('stored', recovered.messages, { keepPending: true })
  expect(textRows(repeated.messages, 'new partial answer')).toHaveLength(1)
  expect(new Set(repeated.messages.map(message => message.id)).size).toBe(repeated.messages.length)
})

it.each(['human', 'runtime'])('does not bind an old hidden runtime to a later %s turn with the same stream id', laterKind => {
  const old = withTool(appendLiveSessionProjection(oldHistory(), snapshot()))
  record(old)

  const laterProjection = laterKind === 'runtime'
    ? { ...snapshot('hidden', 'runtime', 20), inflight: { ...snapshot().inflight, assistant: 'new partial answer' } }
    : { session_id: 'runtime', turn_started_at: 20, inflight: { user: 'later human', user_originated: true, assistant: 'new partial answer', streaming: true } }

  const base = appendLiveSessionProjection(oldHistory(), laterProjection)
  const recovered = recoverInFlightTurnJournal('stored', base, { keepPending: true })
  const current = recovered.messages.find(message => message.id === 'assistant-stream-runtime')!
  expect(current.parts.some(part => part.type === 'tool-call')).toBe(false)
  const orphan = recovered.messages.filter(message => message.parts.some(part => part.type === 'tool-call'))
  expect(orphan).toHaveLength(1)
  expect(orphan[0].pending).toBe(false)
  expect(orphan[0].id).not.toBe(current.id)
  expect(recovered.streamId).toBe(current.id)
  expect(recovered.turnStartedAt).toBeNull()
  const repeated = recoverInFlightTurnJournal('stored', recovered.messages, { keepPending: true })
  expect(repeated.messages.filter(message => message.parts.some(part => part.type === 'tool-call'))).toHaveLength(1)
  expect(new Set(repeated.messages.map(message => message.id)).size).toBe(repeated.messages.length)
})

it('keeps human corrections and split runtime output exactly once through repeated recovery', () => {
  const live = {
    ...snapshot('auto_continue', 'before'),
    inflight: { ...snapshot('auto_continue').inflight, assistant: 'before correctionafter correction', corrections: ['human correction'], correction_offsets: [17] }
  }

  const messages = withTool(appendLiveSessionProjection(oldHistory(), live))
  record(messages)
  const base = appendLiveSessionProjection(oldHistory(), { ...live, session_id: 'after' })
  const recovered = recoverInFlightTurnJournal('stored', base, { keepPending: true })
  const repeated = recoverInFlightTurnJournal('stored', recovered.messages, { keepPending: true })

  for (const value of ['before correction', 'human correction', 'after correction']) {
    expect(textRows(repeated.messages, value)).toHaveLength(1)
  }

  expect(textRows(repeated.messages, 'human correction')[0].role).toBe('user')
  expect(textRows(repeated.messages, 'human correction')[0].userOriginated).not.toBe(false)
  expect(repeated.messages.findIndex(message => chatMessageText(message) === 'human correction')).toBeGreaterThan(
    repeated.messages.findIndex(message => chatMessageText(message) === 'before correction')
  )
})

it('marks a hydrated structured runtime reply and journals later streamed output behind it', () => {
  const base = [...oldHistory(), ...toChatMessages([
    { role: 'assistant', content: 'first response', reasoning: 'planning', timestamp: 11 }
  ])]

  const live = snapshot()
  live.inflight.assistant = 'first response'
  const projected = appendLiveSessionProjection(base, live)
  const structured = textRows(projected, 'first response')[0]
  expect(structured.runtimeTurnStartedAt).toBe(10)
  const later: ChatMessage = { id: 'assistant-stream-new', role: 'assistant', parts: [{ type: 'text', text: 'later response' }], pending: true }
  record([...projected, later], later.id)
  const journal = readInFlightTurnJournal('stored')!
  expect(textRows(journal.messages, 'earlier task')).toHaveLength(0)
  expect(textRows(journal.messages, 'later response')).toHaveLength(1)
  const recovered = recoverInFlightTurnJournal('stored', oldHistory())
  expect(textRows(recovered.messages, 'first response')).toHaveLength(1)
  expect(textRows(recovered.messages, 'later response')).toHaveLength(1)
  expect(textRows(recovered.messages, 'first response')[0].parts.some(part => part.type === 'reasoning')).toBe(true)
})

it('leaves a later human journal boundary unchanged even after a runtime turn', () => {
  const previous = appendLiveSessionProjection(oldHistory(), snapshot()).map(message => ({ ...message, pending: false }))
  const human: ChatMessage = { id: 'user-new', role: 'user', parts: [{ type: 'text', text: 'new human prompt' }] }
  const assistant: ChatMessage = { id: 'assistant-stream-new', role: 'assistant', parts: [{ type: 'text', text: 'human response' }], pending: true }
  record([...previous, human, assistant], assistant.id)
  expect(readInFlightTurnJournal('stored')?.messages.map(message => message.id)).toEqual([human.id, assistant.id])
})

it('compares and preserves runtime identity when a concurrent edit uses the older same-id row', () => {
  const before: ChatMessage = { id: 'same', role: 'assistant', parts: [{ type: 'text', text: 'partial' }] }
  const after = { ...before, runtimeTurnStartedAt: 10 }
  expect(chatMessagesEquivalent(before, after)).toBe(false)
  const current = { ...before, pending: true }
  expect(overlayConcurrentMessageChanges([after], [before], [current])[0].runtimeTurnStartedAt).toBe(10)
})

it.each(['hidden', 'auto_continue'])('settles a %s journal after persisted history covers its anchored runtime tail', kind => {
  record(appendLiveSessionProjection(oldHistory(), snapshot(kind)))

  const base = [...oldHistory(), ...toChatMessages([
    { role: 'user', content: 'runtime wake', display_kind: kind, user_originated: false, timestamp: 11 },
    { role: 'assistant', content: 'new partial answer', timestamp: 12 }
  ])]

  const recovered = recoverInFlightTurnJournal('stored', base, { keepPending: false })
  expect(recovered.applied).toBe(false)
  expect(recovered.caughtUp).toBe(true)
  expect(recovered.messages).toBe(base)
  expect(recovered.streamId).toBeNull()
  expect(readInFlightTurnJournal('stored')).toBeNull()
  expect(recoverInFlightTurnJournal('stored', base).messages).toBe(base)
})

it('does not resurrect a completed marked runtime response as pending', () => {
  const live = appendLiveSessionProjection(oldHistory(), snapshot())
  record(live)

  const completed = live.map(message => message.role === 'assistant' && message.runtimeTurnStartedAt !== undefined
    ? { ...message, pending: false, completedAt: 12 }
    : message)

  const recovered = recoverInFlightTurnJournal('stored', completed, { keepPending: true })
  expect(recovered.caughtUp).toBe(true)
  expect(recovered.messages).toBe(completed)
  expect(recovered.messages.some(message => message.pending)).toBe(false)
  expect(recovered.streamId).toBeNull()
})

it('keeps a later journal segment separate from an earlier sealed runtime assistant', () => {
  const live = snapshot()
  record(withTool(appendLiveSessionProjection(oldHistory(), live)))

  const base = [...oldHistory(), {
    id: 'sealed-earlier', role: 'assistant' as const, runtimeTurnStartedAt: 10,
    parts: [{ type: 'text' as const, text: 'earlier runtime segment' }], pending: false, interim: true
  }]

  const recovered = recoverInFlightTurnJournal('stored', base)
  const sealed = recovered.messages.find(message => message.id === 'sealed-earlier')!
  expect(chatMessageText(sealed)).toBe('earlier runtime segment')
  expect(sealed.parts.some(part => part.type === 'tool-call')).toBe(false)
  expect(textRows(recovered.messages, 'new partial answer')).toHaveLength(1)
})

it('allocates an unused recovery id when both the stream id and first recovery id already exist', () => {
  record(withTool(appendLiveSessionProjection(oldHistory(), snapshot())))

  const laterHuman = appendLiveSessionProjection(oldHistory(), {
    session_id: 'runtime', turn_started_at: 20,
    inflight: { user: 'later human', user_originated: true, assistant: 'later answer', streaming: true }
  })

  laterHuman.push({
    id: 'runtime-recovery-10-0-0-assistant-stream-runtime', role: 'assistant',
    parts: [{ type: 'text', text: 'unrelated existing row' }], runtimeTurnStartedAt: 30
  })
  const recovered = recoverInFlightTurnJournal('stored', laterHuman, { keepPending: true })
  expect(new Set(recovered.messages.map(message => message.id)).size).toBe(recovered.messages.length)
  expect(textRows(recovered.messages, 'unrelated existing row')).toHaveLength(1)
  const repeated = recoverInFlightTurnJournal('stored', recovered.messages, { keepPending: true })
  expect(textRows(repeated.messages, 'new partial answer')).toHaveLength(1)
  expect(new Set(repeated.messages.map(message => message.id)).size).toBe(repeated.messages.length)
})

it('retains an unanchored hidden runtime tail when only unrelated same-text committed output exists', () => {
  record(withTool(appendLiveSessionProjection([], snapshot())))

  const base = toChatMessages([
    { role: 'user', content: 'another human', user_originated: true, timestamp: 50 },
    { role: 'assistant', content: 'new partial answer', timestamp: 51 }
  ])

  const recovered = recoverInFlightTurnJournal('stored', base)
  expect(recovered.caughtUp).toBe(false)
  expect(recovered.messages.some(message => message.parts.some(part => part.type === 'tool-call'))).toBe(true)
  expect(recovered.streamId).toBeNull()
})

it('does not treat a later completed human answer as anchored runtime catch-up', () => {
  record(appendLiveSessionProjection(oldHistory(), snapshot()))

  const base = [...oldHistory(), ...toChatMessages([
    { role: 'user', content: 'later human', user_originated: true, timestamp: 20 },
    { role: 'assistant', content: 'new partial answer', timestamp: 21 }
  ])]

  const recovered = recoverInFlightTurnJournal('stored', base)
  expect(recovered.caughtUp).toBe(false)
  expect(textRows(recovered.messages, 'new partial answer')).toHaveLength(2)
  expect(recovered.messages.some(message => message.pending)).toBe(false)
})

it('keeps an accepted queued human prompt outside the runtime boundary without duplicating it', () => {
  const live = { ...snapshot('hidden', 'before'), queued: { user: 'accepted next prompt' } }
  record(appendLiveSessionProjection(oldHistory(), live), 'assistant-stream-before')
  const base = appendLiveSessionProjection(oldHistory(), { ...live, session_id: 'after' })
  const recovered = recoverInFlightTurnJournal('stored', base, { keepPending: true })
  const repeated = recoverInFlightTurnJournal('stored', recovered.messages, { keepPending: true })
  expect(textRows(repeated.messages, 'accepted next prompt')).toHaveLength(1)
  expect(textRows(repeated.messages, 'accepted next prompt')[0].runtimeTurnStartedAt).toBeUndefined()
})

it('retains a durable row id for catch-up when its renderer id changes and it has no timestamp', () => {
  const anchor: ChatMessage = { id: 'old-id', rowId: 42, role: 'assistant', parts: [{ type: 'text', text: 'anchor' }] }
  record(appendLiveSessionProjection([anchor], snapshot()))
  expect(readInFlightTurnJournal('stored')!.messages[0].rowId).toBe(42)

  const base = [{ ...anchor, id: 'new-id' }, ...toChatMessages([
    { role: 'assistant', content: 'new partial answer', timestamp: 11 }
  ])]

  expect(recoverInFlightTurnJournal('stored', base).caughtUp).toBe(true)
})

it('keeps an earlier completed runtime marker out of the next runtime journal anchor', () => {
  const firstLive = snapshot()
  firstLive.inflight.assistant = 'first runtime answer'

  const first = appendLiveSessionProjection([...oldHistory(), ...toChatMessages([
    { role: 'assistant', content: 'first runtime answer', reasoning: 'first plan', timestamp: 11 }
  ])], firstLive).map(message => message.runtimeTurnStartedAt === 10 ? { ...message, pending: false, completedAt: 12 } : message)

  const second = appendLiveSessionProjection(first, snapshot('hidden', 'second', 20))
  record(second, 'assistant-stream-second')
  const saved = readInFlightTurnJournal('stored')!
  expect(saved.messages[0].runtimeTurnStartedAt).toBeUndefined()
  expect(saved.messages.find(message => message.runtimeTurnStartedAt !== undefined)?.runtimeTurnStartedAt).toBe(20)
  const recovered = recoverInFlightTurnJournal('stored', first)
  expect(textRows(recovered.messages, 'first runtime answer')).toHaveLength(1)
  expect(textRows(recovered.messages, 'new partial answer')).toHaveLength(1)
  expect(textRows(recovered.messages, 'first runtime answer')[0].runtimeTurnStartedAt).toBe(10)
  expect(textRows(recovered.messages, 'new partial answer')[0].runtimeTurnStartedAt).toBe(20)
})

it('settles a throttled runtime partial when the anchored persisted final answer extends it', () => {
  const live = snapshot()
  live.inflight.assistant = 'new partial'
  record(appendLiveSessionProjection(oldHistory(), live))

  const base = [...oldHistory(), ...toChatMessages([
    { role: 'assistant', content: 'new partial answer, finished', timestamp: 12 }
  ])]

  const recovered = recoverInFlightTurnJournal('stored', base)
  expect(recovered.caughtUp).toBe(true)
  expect(recovered.messages).toBe(base)
  expect(readInFlightTurnJournal('stored')).toBeNull()
})

it('retains unmatched runtime failure and tool-only progress beside a committed text answer', () => {
  const runtime: ChatMessage = {
    id: 'assistant-stream-runtime', role: 'assistant', runtimeTurnStartedAt: 10,
    parts: [{ type: 'tool-call', toolName: 'terminal', toolCallId: 'uncommitted-call', args: {} }],
    error: 'runtime operation failed', pending: true
  }

  record([...oldHistory(), runtime])

  const base = [...oldHistory(), ...toChatMessages([
    { role: 'assistant', content: 'completed some other operation', timestamp: 12 }
  ])]

  const recovered = recoverInFlightTurnJournal('stored', base)
  expect(recovered.caughtUp).toBe(false)
  expect(recovered.messages.some(message => message.error === runtime.error)).toBe(true)
})

it('does not let an earlier stale pending assistant take ownership from the current marked runtime', () => {
  record(withTool(appendLiveSessionProjection(oldHistory(), snapshot())))

  const stale: ChatMessage = {
    id: 'assistant-stream-stale', role: 'assistant', parts: [{ type: 'text', text: 'stale pending reply' }], pending: true
  }

  const base = appendLiveSessionProjection([...oldHistory(), stale], snapshot())
  const recovered = recoverInFlightTurnJournal('stored', base, { keepPending: true })
  expect(recovered.streamId).toBe('assistant-stream-runtime')
  expect(recovered.streamId).not.toBe(stale.id)
  expect(textRows(recovered.messages, 'new partial answer')[0].pending).toBe(true)
})
