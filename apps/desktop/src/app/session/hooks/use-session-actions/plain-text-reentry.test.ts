import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { reconcileActiveTranscript } from '@/app/contrib/hooks/use-background-sync'
import { getLatestSessionMessages, getOlderSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type { SessionMessage, SessionMessagesResponse } from '@/types/hermes'

import { preserveLocalPendingTurnMessages } from './utils'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(),
  getOlderSessionMessages: vi.fn()
}))

// SYNTHETIC FIXTURES ONLY: plain strings, no attachments or live history/RPC.
const SID = 'synthetic-text-runtime'
const STORED = 'synthetic-text-stored'
const prompt = 'Synthetic plain prompt.'
const answer = 'Synthetic plain answer.'

const prefix: SessionMessage[] = [
  { id: 101, role: 'user', timestamp: 101, content: 'Synthetic prior prompt.' },
  { id: 102, role: 'assistant', timestamp: 102, content: 'Synthetic prior answer.' }
]

const older: SessionMessage[] = [
  { id: 99, role: 'user', timestamp: 99, content: 'Synthetic older prompt.' },
  { id: 100, role: 'assistant', timestamp: 100, content: 'Synthetic older answer.' }
]

const committed: SessionMessage[] = [
  { id: 153, role: 'user', timestamp: 103, content: prompt },
  { id: 154, role: 'assistant', timestamp: 104, content: answer }
]

const localPrompt: ChatMessage = {
  id: 'user-synthetic-retained',
  role: 'user',
  timestamp: 103,
  parts: [textPart(prompt)]
}

const localReply: ChatMessage = {
  id: 'assistant-stream-synthetic-retained',
  role: 'assistant',
  pending: false,
  parts: [textPart(answer)]
}

function page(messages: SessionMessage[], paginated = false): SessionMessagesResponse {
  return {
    session_id: STORED,
    messages,
    ...(paginated
      ? { pagination: { limit: messages.length, offset: 0, order: 'latest' as const, returned: messages.length } }
      : {})
  }
}

const retained = () => [...toChatMessages(prefix), localPrompt, localReply]
const labels = (messages: ChatMessage[]) => messages.map(message => `${message.role}:${chatMessageText(message)}`)

beforeEach(() => {
  // An unreferenced runtime is evicted on the second publish, which would drop
  // the transcript the refresh is reconciling.
  $activeSessionId.set(SID)
  $selectedStoredSessionId.set(STORED)
})

afterEach(() => {
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  clearAllSessionStates()
  vi.resetAllMocks()
})

async function heldReentry(responsePage: SessionMessagesResponse, previous = retained()) {
  let release!: (page: SessionMessagesResponse) => void

  const response = new Promise<SessionMessagesResponse>(resolve => {
    release = resolve
  })

  vi.mocked(getLatestSessionMessages).mockReturnValueOnce(response)
  let state = { ...createClientSessionState(STORED), messages: previous }
  publishSessionState(SID, state)

  const refresh = reconcileActiveTranscript({
    activeSessionIdRef: { current: SID },
    selectedStoredSessionIdRef: { current: STORED },
    busyRef: { current: false },
    requestSequenceRef: { current: 0 },
    signatureRef: { current: new Map() },
    resolveSession: () => ({ profile: 'default' }),
    updateSessionState: (_, update) => {
      state = update(state)
      publishSessionState(SID, state)

      return state
    }
  })

  try {
    expect(getLatestSessionMessages).toHaveBeenCalledOnce()
    expect(state.messages).toBe(previous)
  } finally {
    release(responsePage)
    await refresh
  }

  return state.messages
}

it('does not append plain retained text when the refreshed page starts at the first cached durable row', async () => {
  const rows = [...prefix, ...committed]
  expect(labels(await heldReentry(page(rows)))).toEqual(labels(toChatMessages(rows)))
})

it('does not append plain retained text when an overlapping page starts before the cached durable window', async () => {
  const rows = [...older, ...prefix, ...committed]
  // The acknowledgement/candidate reducer alone handles exact plain text.
  // Re-entry runs the real backfill graft before that reducer.
  expect(labels(preserveLocalPendingTurnMessages(toChatMessages(rows), retained()))).toEqual(
    labels(toChatMessages(rows))
  )
  expect(labels(await heldReentry(page(rows)))).toEqual(labels(toChatMessages(rows)))
})

it('does not append an old plain turn after a paginated tail extends past the cached first row', async () => {
  const newest: SessionMessage[] = [
    ...committed,
    { id: 155, role: 'user', timestamp: 105, content: 'Synthetic later prompt.' },
    { id: 156, role: 'assistant', timestamp: 106, content: 'Synthetic later answer.' }
  ]

  const preceding = [...older, ...prefix]
  const olderPage = page(preceding, true)
  olderPage.pagination!.offset = newest.length
  vi.mocked(getOlderSessionMessages).mockResolvedValueOnce(olderPage)
  const messages = await heldReentry(page(newest, true))
  expect(getOlderSessionMessages).toHaveBeenCalledExactlyOnceWith(STORED, expect.anything(), newest.length)
  expect(labels(messages)).toEqual(labels(toChatMessages([...preceding, ...newest])))
})

it('deduplicates plain text after a full physical-row rewrite with no durable overlap', async () => {
  const rows = [...prefix.map(row => ({ ...row, id: row.id! + 50 })), ...committed]
  expect(labels(await heldReentry(page(rows)))).toEqual(labels(toChatMessages(rows)))
})

it.each([undefined, 200])('keeps a genuinely newer repeat after the committed answer (rowId=%s)', async rowId => {
  const rows = [...older, ...prefix, ...committed]
  const repeat = { ...localPrompt, id: 'user-synthetic-new-repeat', timestamp: 105, rowId }
  const messages = await heldReentry(page(rows), [...toChatMessages([...prefix, ...committed]), repeat])
  expect(messages.filter(message => chatMessageText(message) === prompt)).toHaveLength(2)
  expect(messages.at(-1)?.id).toBe(repeat.id)
})

it('keeps the settled uncommitted stream answer when the refreshed page ends at its prompt', async () => {
  const rows = [...prefix, committed[0]]
  const messages = await heldReentry(page(rows))
  expect(labels(messages)).toEqual([...labels(toChatMessages(rows)), `assistant:${answer}`])
  expect(messages.at(-1)?.id).toBe(localReply.id)
})

it.each(['prompt', 'earlier'] as const)(
  'keeps an uncommitted turn behind a carried local notice when history ends at %s',
  async boundary => {
    const rows = boundary === 'prompt' ? [...prefix, committed[0]] : [...prefix]

    const notice: ChatMessage = {
      id: 'fallback-switch-synthetic',
      role: 'system',
      parts: [textPart('Synthetic fallback notice.')]
    }

    const messages = await heldReentry(page(rows), [
      ...toChatMessages(prefix).slice(1),
      localPrompt,
      localReply,
      notice
    ])

    expect(messages.some(message => message.id === localReply.id && chatMessageText(message) === answer)).toBe(true)
    expect(messages.some(message => message.id === notice.id)).toBe(true)
    expect(messages.filter(message => message.role === 'user')).toHaveLength(2)
  }
)
