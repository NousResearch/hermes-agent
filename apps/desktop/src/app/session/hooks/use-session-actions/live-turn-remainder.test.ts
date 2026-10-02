import { expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { mergeLiveAssistantRun } from './live-turn-remainder'
import { reconcilePersistedLiveTurn } from './persisted-live-turn'
import { runningProjectionStreamId } from './utils'

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

it('reconciles the marked queue once after the source-anchored active turn and its corrections', () => {
  for (const structured of [false, true]) {
    const rows: SessionMessage[] = [
      { id: 1, role: 'user', content: 'repeat this', timestamp: 1 },
      { id: 2, role: 'assistant', content: 'Old answer', timestamp: 2 },
      { id: 3, role: 'user', content: 'repeat this', timestamp: 3 },
      {
        id: 4,
        role: 'user',
        content: '@file:/tmp/input.txt\n\nrepeat this',
        timestamp: 4,
        display_metadata: JSON.stringify({ _queued_prompt: true }) as never
      }
    ]
    if (structured)
      rows.push(
        {
          id: 5,
          role: 'assistant',
          content: 'Checking.',
          timestamp: 5,
          tool_calls: [{ id: 'inspect', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
        },
        { id: 6, role: 'tool', content: 'File content', tool_call_id: 'inspect', timestamp: 6 },
        { id: 7, role: 'user', content: 'Try tests', display_kind: 'steer', timestamp: 7 }
      )
    const persisted = toChatMessages(rows)
    const before = structuredClone(persisted)
    const projection = {
      session_id: 'runtime',
      queued: { user: 'repeat this' },
      inflight: {
        user: 'repeat this',
        assistant: structured ? 'Checking.\n\nHello' : 'Hello',
        streaming: true,
        ...(structured ? { corrections: ['Try tests'], correction_offsets: ['Checking.\n\n'.length] } : {})
      }
    }
    let current = [...persisted, assistant('assistant-stream-local', 'Hello + local delta', { pending: true })]
    for (let resume = 0; resume < 3; resume++) {
      current = reconcilePersistedLiveTurn(persisted, current, rows, projection)!
      expect(current).not.toBeNull()
      expect(current.map(chatMessageText)).toEqual(
        structured
          ? ['repeat this', 'Old answer', 'repeat this', 'Checking.', 'Try tests', 'Hello + local delta', 'repeat this']
          : ['repeat this', 'Old answer', 'repeat this', 'Hello + local delta', 'repeat this']
      )
      expect(current.filter(message => message.rowId === 4)).toHaveLength(1)
      expect(runningProjectionStreamId(current, true)).toBe('assistant-stream-runtime')
      expect(current.at(-1)).toMatchObject({
        id: 'user-queued-runtime',
        rowId: 4,
        queuedPrompt: true,
        attachmentRefs: ['@file:/tmp/input.txt']
      })
      expect(current.filter(message => chatMessageText(message).includes('Hello'))).toHaveLength(1)
      expect(new Set(current.map(message => message.id)).size).toBe(current.length)
      if (structured) {
        expect(current.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toEqual([
          expect.objectContaining({ toolCallId: 'inspect', result: 'File content' })
        ])
        expect(current.flatMap(message => message.parts)).toContainEqual(
          expect.objectContaining({ type: 'text', text: 'Checking.', sourceRowId: 5 })
        )
      }
    }

    let divergent = [...persisted, assistant('assistant-stream-runtime', 'Different local output', { pending: true })]
    for (let replay = 0; replay < 3; replay++) {
      divergent = reconcilePersistedLiveTurn(persisted, divergent, rows, projection)!
      expect(divergent.filter(message => chatMessageText(message) === 'Different local output')).toHaveLength(1)
      expect(divergent.filter(message => chatMessageText(message) === 'Hello')).toHaveLength(1)
      expect(new Set(divergent.map(message => message.id)).size).toBe(divergent.length)
      expect(runningProjectionStreamId(divergent, true)).toBe('assistant-stream-runtime')
    }
    expect(persisted).toEqual(before)
    if (structured) {
      // Equal commentary prose without its source occurrence cannot prove coverage.
      const uncertain = persisted.map(message => ({
        ...message,
        parts: message.parts.map(part =>
          part.type === 'text' && part.sourceRowId === 5 ? { ...part, sourceRowId: 99 } : part
        )
      }))
      expect(reconcilePersistedLiveTurn(uncertain, [], rows, projection)).toBeNull()
    }
  }
})
