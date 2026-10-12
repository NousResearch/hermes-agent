import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { resetLiveSync } from '@/store/live-sync'
import {
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId
} from '@/store/session'
import { $sessionStates, clearAllSessionStates } from '@/store/session-states'
import { clearSessionTodos } from '@/store/todos'
import type { SessionMessage } from '@/types/hermes'

import { reconcileActiveTranscript } from './use-background-sync'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))

const RUNTIME = 'folded-runtime'
const STORED = 'folded-stored'
const INTERIM = 'Let me look at the files first.'
const FINAL = 'Here is the summary table.\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nEverything checks out.'
const noop = async () => undefined

const history: SessionMessage[] = [
  { id: 1, role: 'user', content: 'Earlier question', timestamp: 1 },
  { id: 2, role: 'assistant', content: 'Earlier answer', timestamp: 2 },
  { id: 3, role: 'user', content: 'Read the files and summarize', timestamp: 3 }
]

const tools = ['read-1', 'term-1', 'term-2']

const toolCalls = (ids: string[]) =>
  ids.map(id => ({ id, type: 'function', function: { name: 'terminal', arguments: '{}' } }))

// Durable rows for the turn: interim + 3 tool calls, 3 tool results, final.
// toChatMessages folds them into ONE assistant bubble (serverRowSpan > 1).
const durableTurn: SessionMessage[] = [
  ...history,
  {
    id: 4,
    role: 'assistant',
    content: INTERIM,
    timestamp: 4,
    tool_calls: toolCalls(tools)
  },
  ...tools.map((id, index) => ({
    id: 5 + index,
    role: 'tool' as const,
    content: `out ${index}`,
    timestamp: 5 + index,
    tool_call_id: id,
    tool_name: 'terminal'
  })),
  { id: 8, role: 'assistant', content: FINAL, timestamp: 8 }
]

let cache: ReturnType<typeof useSessionStateCache>
let stream: ReturnType<typeof useMessageStream>
let refresh: () => Promise<void>

function Harness() {
  const busyRef = useRef(false)
  const queryClient = useRef(new QueryClient()).current
  const requestSequenceRef = useRef(0)
  const signatureRef = useRef(new Map<string, string>())
  cache = useSessionStateCache({
    activeSessionId: RUNTIME,
    selectedStoredSessionId: STORED,
    busyRef,
    setBusy,
    setAwaitingResponse,
    setMessages
  })
  stream = useMessageStream({
    ...cache,
    queryClient,
    hydrateFromStoredSession: noop,
    refreshHermesConfig: noop,
    refreshSessions: noop
  })
  refresh = () =>
    reconcileActiveTranscript({
      activeSessionIdRef: cache.activeSessionIdRef,
      selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
      updateSessionState: cache.updateSessionState,
      busyRef,
      requestSequenceRef,
      signatureRef,
      resolveSession: () => ({ profile: 'default' })
    })

  return null
}

function send(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) {
  act(() => stream.handleGatewayEvent({ session_id: RUNTIME, type, payload }))
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.mocked(getLatestSessionMessages).mockReset()
  setActiveSessionId(RUNTIME)
  setSelectedStoredSessionId(STORED)
})

afterEach(() => {
  cleanup()
  clearSessionTodos(RUNTIME)
  clearAllSessionStates()
  resetLiveSync()
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
  localStorage.clear()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

async function streamToolTurn(receipt: 'complete' | 'partial' | 'none') {
  render(<Harness />)
  act(() => cache.updateSessionState(RUNTIME, state => ({ ...state, messages: toChatMessages(history) }), STORED))

  send('message.start')
  send('message.delta', { text: INTERIM })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(100)
  })
  send('message.interim', { text: INTERIM, already_streamed: true })

  for (const id of tools) {
    send('tool.start', { name: 'terminal', tool_id: id, args: {} })
    send('tool.complete', { name: 'terminal', tool_id: id, result: 'ok' })
  }

  send('reasoning.delta', { text: 'Compose the table.' })

  for (const chunk of FINAL.match(/[\s\S]{1,12}/g) ?? []) {
    send('message.delta', { text: chunk })
  }

  await act(async () => {
    await vi.advanceTimersByTimeAsync(100)
  })
  // `complete` is only true when no compaction/redirect touched the turn; a
  // compacted session (or an older backend) settles with a partial or no receipt.
  send('message.complete', {
    text: FINAL,
    persisted_turn:
      receipt === 'none'
        ? undefined
        : { row_ids: [3, 4, 5, 6, 7, 8], user_row_id: 3, final_assistant_row_id: 8, complete: receipt === 'complete' }
  })
}

async function refreshWith(messages: SessionMessage[]) {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: STORED, messages })
  await act(async () => {
    await refresh()
  })

  return $sessionStates.get()[RUNTIME].messages.map(chatMessageText)
}

it.each(['complete', 'partial', 'none'] as const)(
  'paints a tool-interleaved reply once after the refresh folds its durable rows (receipt: %s)',
  async receipt => {
    await streamToolTurn(receipt)
    const texts = await refreshWith(durableTurn)

    expect(texts.filter(text => text.includes('Everything checks out.'))).toHaveLength(1)
    expect(texts).toContain('Earlier answer')
  }
)

it('keeps the settled live reply when the fold lacks one of its tool rounds', async () => {
  await streamToolTurn('none')

  // The newest page stops before the last tool round and the final answer.
  const texts = await refreshWith(
    durableTurn.slice(0, -2).map(row => (row.id === 4 ? { ...row, tool_calls: toolCalls(tools.slice(0, 2)) } : row))
  )

  expect(texts.filter(text => text.includes('Everything checks out.'))).toHaveLength(1)
})
