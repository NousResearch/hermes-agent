import type { PersistedTurn } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { bindStoppedTurnReceipt } from '@/app/session/hooks/use-message-stream/interrupted-reply'
import { finalizeStoppedMessages } from '@/app/session/hooks/use-prompt-actions/rewind'
import {
  clearSessionRecentlyInterrupted,
  markSessionRecentlyInterrupted
} from '@/app/session/hooks/use-prompt-actions/utils'
import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates, clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type { SessionMessage } from '@/types/hermes'

import { reconcileActiveTranscript } from './use-background-sync'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getLatestSessionMessages: vi.fn(),
  getOlderSessionMessages: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')

const RUNTIME = 'runtime-stopped'
const STORED = 'stored-stopped'

/** The four rows a turn stopped after its first tool round stores. */
function storedStoppedTurn(n: number, first: number, at: number): SessionMessage[] {
  const call = `call-${n}`

  return [
    { id: first, role: 'user', content: `U${n} stop me`, timestamp: at },
    {
      id: first + 1,
      role: 'assistant',
      content: `A${n}i1 looking`,
      timestamp: at + 0.2,
      tool_calls: [{ id: call, type: 'function', function: { name: 'terminal', arguments: '{"command":"echo"}' } }]
    },
    {
      id: first + 2,
      role: 'tool',
      content: '{"output": "ok"}',
      tool_call_id: call,
      tool_name: 'terminal',
      timestamp: at + 0.3
    },
    {
      id: first + 3,
      role: 'assistant',
      content: `A${n} partial`,
      timestamp: at + 1.2,
      display_metadata: { interrupted: true }
    }
  ] as SessionMessage[]
}

const count = (needle: string) =>
  ($sessionStates.get()[RUNTIME]?.messages ?? []).filter(message => chatMessageText(message).includes(needle)).length

function refresh() {
  return reconcileActiveTranscript({
    activeSessionIdRef: { current: RUNTIME },
    busyRef: { current: false },
    requestSequenceRef: { current: 0 },
    resolveSession: () => ({ profile: 'default' }),
    selectedStoredSessionIdRef: { current: STORED },
    signatureRef: { current: new Map<string, string>() },
    updateSessionState: (sessionId, updater) => {
      const next = updater($sessionStates.get()[sessionId] ?? createClientSessionState(STORED))
      publishSessionState(sessionId, next)

      return next
    }
  })
}

beforeEach(() => {
  // An inactive runtime's transcript is evicted on publish; this one is open.
  $activeSessionId.set(RUNTIME)
  $selectedStoredSessionId.set(STORED)
})

afterEach(() => {
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  clearAllSessionStates()
  clearSessionRecentlyInterrupted()
  vi.mocked(getLatestSessionMessages).mockReset()
})

describe('a page read right after Stop', () => {
  it('draws the stopped turn once when the page lands before the interrupted completion (#144899 N2)', async () => {
    const receipt: PersistedTurn = {
      row_ids: [145, 146, 147, 148],
      user_row_ids: [145],
      submission_id: 'stop-10',
      complete: false
    }

    const liveReply: ChatMessage = {
      id: 'assistant-stream-10',
      role: 'assistant',
      pending: true,
      timestamp: 1101.19,
      parts: [
        { type: 'text', text: 'A10i1 looking' },
        {
          type: 'tool-call',
          toolCallId: 'call-10',
          toolName: 'terminal',
          args: { command: 'echo' },
          argsText: '{"command":"echo"}',
          result: '{"output": "ok"}'
        },
        { type: 'text', text: '\n\nA10 ' }
      ]
    }

    // Stop sealed the reply at its first chunk and idled the view: the stopped
    // turn's rows are live and unstored, below an earlier stopped turn.
    const stopped: ClientSessionState = {
      ...createClientSessionState(STORED),
      interrupted: true,
      messages: finalizeStoppedMessages(
        [
          ...toChatMessages(storedStoppedTurn(9, 141, 1090.29)),
          { id: 'user-stop-10', role: 'user', parts: [{ type: 'text', text: 'U10 stop me' }], timestamp: 1100.93 },
          liveReply
        ],
        liveReply.id
      )
    }

    publishSessionState(RUNTIME, stopped)
    markSessionRecentlyInterrupted(RUNTIME)

    // The newest page already holds the stored stopped turn.
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        ...storedStoppedTurn(8, 137, 1079.93),
        ...storedStoppedTurn(9, 141, 1090.29),
        ...storedStoppedTurn(10, 145, 1101.04)
      ],
      session_id: STORED
    } as never)

    await refresh()

    expect(count('U10 stop me')).toBe(1)
    expect(count('A10i1 looking')).toBe(1)

    // The interrupted completion binds the receipt; the retried read then lands.
    const settled = $sessionStates.get()[RUNTIME]!
    publishSessionState(RUNTIME, { ...settled, messages: bindStoppedTurnReceipt(settled.messages, receipt) })

    await refresh()

    expect(getLatestSessionMessages).toHaveBeenCalled()
    expect(count('U10 stop me')).toBe(1)
    expect(count('A10i1 looking')).toBe(1)
  })
})
