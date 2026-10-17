import { describe, expect, it } from 'vitest'

import { chatMessagesEquivalent } from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import legacyCache from './legacy-folded-cache.json'
import { graftRefreshedTailOntoBackfill, mergeOlderTranscriptPage } from './transcript-backfill'

// Deliberately sparse database ids: a folded row count is NOT an id interval.
// Reasoning is essential: without it these become a different tool-only fold.
const rows: SessionMessage[] = [
  { id: 10, role: 'user', content: 'Inspect the sample', timestamp: 10 },
  {
    id: 30,
    role: 'assistant',
    content: '',
    reasoning: 'Inspect first',
    timestamp: 30,
    tool_calls: [{ id: 'first', function: { name: 'read_file', arguments: '{}' } }]
  },
  { id: 90, role: 'tool', tool_call_id: 'first', content: 'First result', timestamp: 90 },
  {
    id: 150,
    role: 'assistant',
    content: '',
    reasoning: 'Inspect again',
    timestamp: 150,
    tool_calls: [{ id: 'second', function: { name: 'read_file', arguments: '{}' } }]
  },
  { id: 210, role: 'tool', tool_call_id: 'second', content: 'Second result', timestamp: 210 },
  { id: 400, role: 'assistant', content: 'The sample is ready.', timestamp: 400 },
  { id: 900, role: 'user', content: 'Repeat the answer', timestamp: 900 },
  { id: 950, role: 'assistant', content: 'The sample is ready.', timestamp: 950 }
]

function fixture() {
  const full = toChatMessages(rows)
  const narrow = toChatMessages(rows.slice(3))

  const live: ChatMessage = {
    id: 'assistant-stream-sample',
    rowId: 400,
    role: 'assistant',
    durableComplete: true,
    completedAt: 401,
    parts: [
      { type: 'text', text: 'The sample is ready.' },
      {
        type: 'tool-call',
        toolCallId: 'live-only',
        toolName: 'read_file',
        args: {},
        argsText: '{}',
        result: 'Live result'
      }
    ]
  }

  return { full, narrow, local: [full[0], live, ...full.slice(2)] }
}

function expectLossless(messages: ChatMessage[]) {
  const assistants = messages.filter(message => message.role === 'assistant')
  // Equal answers on DIFFERENT durable rows remain distinct occurrences.
  expect(assistants).toHaveLength(2)
  const parts = assistants[0].parts
  expect(parts.filter(part => part.type === 'text').map(part => part.text)).toEqual(['The sample is ready.'])
  expect(parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual([
    'Inspect first',
    'Inspect again'
  ])
  expect(parts.filter(part => part.type === 'tool-call').map(part => [part.toolCallId, part.result])).toEqual([
    ['first', 'First result'],
    ['second', 'Second result'],
    ['live-only', 'Live result']
  ])
  expect(messages.filter(message => message.role === 'user').map(message => message.rowId)).toEqual([10, 900])
}

const seams = [
  ['tail', (cached: ChatMessage[], page: ChatMessage[]) => graftRefreshedTailOntoBackfill(page, cached)],
  ['older', mergeOlderTranscriptPage]
] as const

describe('folded transcript source identity', () => {
  // JSON captured by executing hydration.ts + tool-parts.ts from 9084d41393^,
  // not by stripping metadata from today's hydration output.
  it.each(seams)('upgrades real pre-change cached folds without losing live activity (%s)', (_, merge) => {
    const full = toChatMessages(legacyCache.rows as SessionMessage[])
    const partial = toChatMessages((legacyCache.rows as SessionMessage[]).slice(2))

    const live: ChatMessage = {
      id: 'assistant-stream-live',
      rowId: 400,
      role: 'assistant',
      parts: [
        { type: 'reasoning', text: 'Inspect first', timestamp: 401 },
        { type: 'text', text: 'Done' },
        { type: 'tool-call', toolCallId: 'live-only', toolName: 'read_file', args: {}, argsText: '{}' }
      ]
    }

    let cached: ChatMessage[] = JSON.parse(JSON.stringify([...legacyCache.full, ...legacyCache.partial, live]))
    expect(cached[0].parts.find(part => part.type === 'reasoning')?.sourceRowId).toBeUndefined()

    for (const page of [partial, full, partial, full]) {
      cached = merge(cached, page)
      cached = JSON.parse(JSON.stringify(cached))
    }

    expect(cached).toHaveLength(1)
    expect(cached[0].parts.map(part => [part.type, part.sourceRowId])).toEqual([
      ['reasoning', 30],
      ['tool-call', 30],
      ['reasoning', 150],
      ['tool-call', 150],
      ['text', 400],
      ['reasoning', undefined],
      ['tool-call', undefined]
    ])
    expect(cached[0].parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual([
      'Inspect first',
      'Inspect first',
      'Inspect first'
    ])
    expect(cached[0].parts.at(-1)).toMatchObject({ toolCallId: 'live-only' })
  })
  it.each(seams)('retains distinct live reasoning beside an already persisted tool (%s)', (_, merge) => {
    const full = toChatMessages(legacyCache.rows as SessionMessage[])

    const live: ChatMessage = {
      id: 'assistant-stream-live',
      rowId: 400,
      role: 'assistant',
      parts: [
        { type: 'reasoning', text: 'A distinct live observation', timestamp: 30 },
        { type: 'tool-call', toolCallId: 'first', toolName: 'read_file', timestamp: 30, args: { path: 'abc' } },
        { type: 'text', text: 'Done' }
      ]
    }

    const merged = merge([live], full)
    expect(merged[0].parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual([
      'Inspect first',
      'Inspect first',
      'A distinct live observation'
    ])
  })
  it.each(seams)('restores call metadata through repeated partial/full/partial merges (%s)', (_, merge) => {
    const source: SessionMessage[] = [
      {
        id: 30,
        role: 'assistant',
        content: '',
        reasoning: 'Inspect first',
        timestamp: 30,
        tool_calls: [{ id: 'first', function: { name: 'read_file', arguments: '{"path":"abc"}' } }]
      },
      { id: 90, role: 'tool', tool_call_id: 'first', content: 'result', timestamp: 90 },
      { id: 400, role: 'assistant', content: 'Done', timestamp: 400 }
    ]

    const full = toChatMessages(source)
    const partial = toChatMessages(source.slice(1))
    let cached = partial

    for (let cycle = 0; cycle < 3; cycle++) {
      cached = merge(cached, full)
      expect(cached[0].parts.map(part => part.type)).toEqual(['reasoning', 'tool-call', 'text'])
      const tool = cached[0].parts.find(part => part.type === 'tool-call')!
      expect(tool.unpairedStoredToolResult).toBeFalsy()
      expect(tool).toMatchObject({
        toolName: 'read_file',
        args: { path: 'abc' },
        argsText: '{"path":"abc"}',
        sourceRowId: 30,
        resultRowId: 90,
        timestamp: 30,
        result: 'result'
      })
      cached = merge(JSON.parse(JSON.stringify(cached)), partial)
      expect(cached[0].parts.find(part => part.type === 'tool-call')).toEqual(tool)
    }
  })
  it('anchors an unfinished tool-only fold without dropping the backfilled prompt', () => {
    const source = rows.slice(0, 5).map(row => ({ ...row, reasoning: undefined }))
    const full = toChatMessages(source)
    const tail = toChatMessages(source.slice(3))
    const merged = graftRefreshedTailOntoBackfill(tail, full)
    expect(merged.map(message => message.role)).toEqual(['user', 'assistant'])
    expect(merged[1].parts.filter(part => part.type === 'tool-call').map(part => part.result)).toEqual([
      'First result',
      'Second result'
    ])
  })

  it('publishes durable coverage changes even when the visible transcript is unchanged', () => {
    const message = fixture().full[1]
    const old = { ...message, sourceRowIds: undefined }
    expect(chatMessagesEquivalent(old, message)).toBe(false)
    expect(chatMessagesEquivalent(message, JSON.parse(JSON.stringify(message)))).toBe(true)
  })
  it('keeps reused tool-call ids attached to their own durable result occurrences', () => {
    const source = rows.slice(1, 6).map(row => ({
      ...row,
      ...(row.tool_calls ? { tool_calls: [{ id: 'reused', function: { name: 'read_file', arguments: '{}' } }] } : {}),
      ...(row.role === 'tool' ? { tool_call_id: 'reused' } : {})
    }))

    const full = toChatMessages(source)
    const tail = toChatMessages(source.slice(3))

    for (const merged of [graftRefreshedTailOntoBackfill(tail, full), mergeOlderTranscriptPage(tail, full)]) {
      const tools = merged[0].parts.filter(part => part.type === 'tool-call')
      expect(tools.map(part => [part.sourceRowId, part.result])).toEqual([
        [30, 'First result'],
        [150, 'Second result']
      ])
      expect(new Set(tools.map(part => part.toolCallId)).size).toBe(tools.length)
    }
  })

  it.each([
    { storedFirst: true, toolName: 'write_file' },
    { storedFirst: false, toolName: 'write_file' },
    { storedFirst: true, toolName: 'read_file' },
    { storedFirst: false, toolName: 'read_file' },
    { storedFirst: true, toolName: 'read_file', sameArgs: true },
    { storedFirst: false, toolName: 'read_file', sameArgs: true }
  ])('does not overwrite a distinct live call with a reused tool id (%o)', ({ storedFirst, toolName, sameArgs }) => {
    const stored = toChatMessages(rows.slice(1, 6))

    const live: ChatMessage = {
      id: 'assistant-stream-live',
      rowId: 400,
      role: 'assistant',
      parts: [
        {
          type: 'tool-call',
          toolCallId: 'first',
          toolName,
          args: sameArgs ? {} : { path: 'different' },
          timestamp: 999,
          result: 'live result'
        },
        { type: 'text', text: 'The sample is ready.' }
      ]
    }

    const merged = storedFirst
      ? graftRefreshedTailOntoBackfill([live], stored)
      : graftRefreshedTailOntoBackfill(stored, [live])

    const calls = merged.flatMap(message => message.parts.filter(part => part.type === 'tool-call'))
    expect(calls.map(part => [part.toolName, part.result])).toEqual([
      ['read_file', 'First result'],
      ['read_file', 'Second result'],
      [toolName, 'live result']
    ])
  })

  it.each([true, false])('preserves exact row membership across result-only page cuts (reasoning=%s)', reasoning => {
    const source = rows.slice(1, 6).map(row => (reasoning ? row : { ...row, reasoning: undefined }))
    const full = toChatMessages(source)
    const resultTail = toChatMessages(source.slice(3))
    expect(full[0].sourceRowIds).toEqual(source.map(row => row.id))

    for (const merged of [
      graftRefreshedTailOntoBackfill(resultTail, full),
      mergeOlderTranscriptPage(resultTail, full)
    ]) {
      expect(merged).toHaveLength(1)
      expect(merged[0].sourceRowIds).toEqual(source.map(row => row.id))
      expect(
        merged[0].parts
          .filter(part => part.type === 'tool-call')
          .map(part => [part.toolCallId, part.toolName, part.args])
      ).toEqual([
        ['first', 'read_file', {}],
        ['second', 'read_file', {}]
      ])
      expect(merged[0].parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual(
        reasoning ? ['Inspect first', 'Inspect again'] : []
      )
    }
  })

  it('reconciles older-page folds including already duplicated cached representations', () => {
    const { full, narrow, local } = fixture()
    const merged = mergeOlderTranscriptPage(mergeOlderTranscriptPage(local, narrow), full)
    expectLossless(merged)
    expect(mergeOlderTranscriptPage(merged, full)).toBe(merged)
    const stale = JSON.parse(JSON.stringify([full[0], full[1], narrow[0], ...local.slice(1)])) as ChatMessage[]
    expectLossless(mergeOlderTranscriptPage(stale, narrow))
    expectLossless(graftRefreshedTailOntoBackfill(narrow, stale))
  })

  it('reconciles partial tail folds without losing source occurrences or live tools', () => {
    const { full, narrow, local } = fixture()
    const wider = graftRefreshedTailOntoBackfill(full, local)
    const merged = graftRefreshedTailOntoBackfill(narrow, wider)
    expectLossless(merged)
    expect(merged[1].timestamp).toBe(full[1].timestamp)
    const incomplete = toChatMessages(rows.slice(1, 5))
    const retained = graftRefreshedTailOntoBackfill(incomplete, [full[1]])
    expect(retained[0].durableComplete).toBe(true)
    // A cold cache round-trip and repeated narrower/full refreshes are idempotent.
    const restored = JSON.parse(JSON.stringify(merged)) as ChatMessage[]
    expectLossless(graftRefreshedTailOntoBackfill(full, graftRefreshedTailOntoBackfill(narrow, restored)))
  })
})
