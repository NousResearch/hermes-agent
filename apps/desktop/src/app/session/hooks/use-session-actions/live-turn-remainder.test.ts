import { expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { mergeLiveAssistantRun } from './live-turn-remainder'
import { reconcilePersistedLiveTurn } from './persisted-live-turn'

const assistant = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [{ type: 'text', text }],
  ...extra
})

it('extends one response across arbitrary chunk cuts without changing its tool or text occurrences', () => {
  const tool = {
    type: 'tool-call' as const,
    toolCallId: 'local-tool',
    toolName: 'read_file',
    args: {},
    argsText: '{}',
    result: 'fixture'
  }

  const remote = [assistant('snapshot', 'Checking.\n\nThe result.', { pending: true })]

  for (let cut = 1; cut < 'The result.'.length; cut++) {
    let local = [
      assistant('live', '', {
        parts: [
          { type: 'text', text: 'Checking.' },
          tool,
          { type: 'text', text: `\n\n${'The result.'.slice(0, cut)}` }
        ],
        pending: true
      })
    ]

    for (let resume = 0; resume < 3; resume++) {
      local = mergeLiveAssistantRun(remote, local)
      expect(local.map(chatMessageText)).toEqual(['Checking.\n\nThe result.'])
      expect(local.flatMap(row => row.parts.filter(part => part.type === 'tool-call'))).toEqual([tool])
      expect(local.map(row => row.id)).toEqual(['live'])
    }
  }

  const sealed = assistant('sealed', '', {
    parts: [{ type: 'text', text: 'The res', completedAt: 10 }]
  })

  const terminal = [assistant('snapshot', 'The result.', { error: 'Connection reset' })]
  const settled = mergeLiveAssistantRun(terminal, [sealed])
  expect(settled.map(chatMessageText)).toEqual(['The result.'])
  expect(mergeLiveAssistantRun(terminal, settled)).toEqual(settled)
  expect(settled[0].parts[0].completedAt).toBe(10)
})

it('settles a matching partial error without discarding richer local parts or distinct failures', () => {
  const tool = { type: 'tool-call' as const, toolCallId: 'local-tool', toolName: 'read_file', args: {}, argsText: '{}' }
  const surface = { layer: 'streaming' as const, code: 'stream_drop', retryable: true }

  const failed = assistant('snapshot', 'The result.', {
    error: 'Connection reset',
    errorSurface: surface,
    pending: false
  })

  const local = assistant('live', '', {
    parts: [tool, { type: 'text', text: 'The result. More local detail.' }],
    pending: true
  })

  let rows = mergeLiveAssistantRun([failed], [local])

  for (let resume = 0; resume < 3; resume++) {
    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      id: local.id,
      parts: local.parts,
      error: failed.error,
      errorSurface: surface,
      pending: false
    })
    rows = mergeLiveAssistantRun([failed], rows)
  }

  const earlier = assistant('earlier-failure', 'A different partial reply', { error: 'Read failed', pending: false })
  rows = mergeLiveAssistantRun([failed], [earlier])
  expect(rows).toEqual([earlier, failed])
  expect(mergeLiveAssistantRun([failed], rows)).toEqual(rows)

  const equalTextDifferentFailure = assistant('another-failure', chatMessageText(failed), {
    error: 'Permission denied'
  })

  expect(mergeLiveAssistantRun([failed], [equalTextDifferentFailure])).toEqual([equalTextDifferentFailure, failed])
  expect(mergeLiveAssistantRun([assistant('distinct', 'Unrelated reply')], [local])).toEqual([
    local,
    assistant('distinct', 'Unrelated reply')
  ])
})

it('folds a narration-led turn in either direction so one `clarify` staging renders once', () => {
  const userRow: SessionMessage = { id: 1, role: 'user', content: 'Help', timestamp: 1 }

  const clarifyRow = (extra: Partial<SessionMessage> = {}): SessionMessage => ({
    id: 2,
    role: 'assistant',
    content: 'Choose the plan.',
    timestamp: 2,
    tool_calls: [
      {
        id: 'call-clarify-one',
        type: 'function',
        function: {
          name: 'clarify',
          arguments: JSON.stringify({ questions: [{ question: 'A?' }, { question: 'B?' }] })
        }
      }
    ],
    ...extra
  })

  const projection: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'> = {
    session_id: 'runtime',
    inflight: { user: 'Help', assistant: 'Choose the plan.', streaming: true }
  }

  // The same open staging must never render twice, whichever side the
  // narration asymmetry sits on: a window that attached mid-turn missed the
  // `reasoning.delta` frames, and `appendReasoningDelta`'s replace branch can
  // attach reasoning to the live row before any text arrives.
  const runCase = (rows: SessionMessage[], liveParts: (stored: ChatMessage[]) => ChatMessagePart[]) => {
    const stored = toChatMessages(rows)

    let current: ChatMessage[] = [
      stored[0],
      { ...stored[1], id: 'assistant-stream-live', pending: true, parts: liveParts(stored) }
    ]

    for (let resume = 0; resume < 3; resume++) {
      current = reconcilePersistedLiveTurn(toChatMessages(rows), current, rows, projection)!

      expect(
        current.flatMap(row => row.parts).filter(part => part.type === 'tool-call' && part.toolCallId === 'call-clarify-one')
      ).toHaveLength(1)
    }
  }

  runCase([userRow, clarifyRow()], stored => [
    { type: 'reasoning', text: 'Need user input.' },
    ...stored[1].parts
  ])

  runCase([userRow, clarifyRow({ reasoning: 'Internal narration.' })], stored =>
    stored[1].parts.filter(part => part.type !== 'reasoning')
  )
})

it('pairs only the queue projection, preserving equal corrections and different local queued occurrences', () => {
  const prompt = 'Inspect this file'
  const correction = 'Also inspect its tests'
  const rows: SessionMessage[] = [{ id: 1, role: 'user', content: prompt }]

  const projection: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'> = {
    session_id: 'runtime',
    inflight: { user: prompt, assistant: '', corrections: [correction], correction_offsets: [0], streaming: true },
    queued: { user: correction }
  }

  const reconcile = (previous: ChatMessage[]) =>
    reconcilePersistedLiveTurn(toChatMessages(rows), previous, rows, projection)!

  let current = reconcile([])

  const unrelatedUser: ChatMessage = { id: 'local-user', role: 'user', parts: [{ type: 'text', text: correction }] }

  const oldQueue: ChatMessage = {
    id: 'user-queued-older-runtime',
    role: 'user',
    parts: [{ type: 'text', text: 'A different queued request' }]
  }

  const response = assistant('local-response', 'An unrepresented later reply')
  current.push(unrelatedUser, oldQueue, response)

  for (let resume = 0; resume < 3; resume++) {
    current = reconcile(current)
    expect(current.filter(row => row.role === 'user').map(chatMessageText)).toEqual([
      prompt,
      correction,
      correction,
      correction,
      chatMessageText(oldQueue)
    ])
    expect(current.filter(row => row.id === 'user-queued-runtime')).toHaveLength(1)
    expect(current).toContainEqual(unrelatedUser)
    expect(current).toContainEqual(oldQueue)
    expect(current).toContainEqual(response)
    expect(new Set(current.map(row => row.id)).size).toBe(current.length)
  }
})
