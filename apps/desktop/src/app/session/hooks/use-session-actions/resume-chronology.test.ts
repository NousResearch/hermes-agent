import { expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { appendLiveSessionProjection, reconcileResumeMessages } from './utils'

const text = (message: ChatMessage): string =>
  message.parts.flatMap(part => (part.type === 'text' ? [part.text] : [])).join('')

it('does not attach a later cached phase to an earlier projected segment by assistant ordinal', () => {
  const late: ChatMessage = {
    id: 'assistant-stream-late',
    role: 'assistant',
    pending: true,
    parts: [
      { type: 'reasoning', text: 'Later planning', timestamp: 5 },
      { type: 'tool-call', toolCallId: 'late', toolName: 'terminal', timestamp: 9 }
    ]
  }

  for (const parts of [
    [{ type: 'tool-call' as const, toolCallId: 'early', toolName: 'skill_view', timestamp: 1 }],
    [{ type: 'text' as const, text: 'Earlier narration', timestamp: 1 }]
  ]) {
    const next: ChatMessage[] = [
      { id: 'stored-early', role: 'assistant', parts },
      { ...late, id: 'stored-late', pending: false }
    ]

    expect.soft(reconcileResumeMessages(next, [late])).toEqual(next)
  }

  const projection = {
    session_id: 'resume',
    inflight: {
      user: 'ship the fix',
      assistant: 'Earlier narration and later output',
      streaming: true,
      corrections: ['use the safer path'],
      correction_offsets: ['Earlier narration'.length]
    }
  }

  for (const cachedParts of [
    [
      { type: 'reasoning' as const, text: 'Later planning', timestamp: 5 },
      { type: 'tool-call' as const, toolCallId: 'late', toolName: 'terminal', timestamp: 9 }
    ],
    [
      { type: 'reasoning' as const, text: 'Later planning', timestamp: 5 },
      { type: 'tool-call' as const, toolCallId: 'late', toolName: 'terminal', timestamp: 9 },
      { type: 'text' as const, text: 'Earlier narration and later output', timestamp: 10 }
    ]
  ]) {
    const cached: ChatMessage[] = [
      { id: 'user-live', role: 'user', parts: [{ type: 'text', text: 'ship the fix' }] },
      { id: 'assistant-stream-local', role: 'assistant', pending: true, parts: cachedParts }
    ]

    const projected = appendLiveSessionProjection([], projection)
    const result = reconcileResumeMessages(projected, cached)
    const earlier = result.find(message => message.id === 'inflight-assistant-segment-0-resume')

    expect.soft(earlier?.parts).toEqual([{ type: 'text', text: 'Earlier narration' }])
    expect.soft(result.flatMap(message => message.parts.map(part => part.timestamp)).filter(Boolean)).toEqual([])
  }
})

it('carries cached structure only across identified rows or corresponding current live tails', () => {
  const structure = [{ type: 'tool-call' as const, toolCallId: 'owned', toolName: 'terminal', timestamp: 2 }]

  const identifiedCached: ChatMessage = {
    id: 'assistant-stream-local',
    rowId: 42,
    role: 'assistant',
    pending: true,
    parts: structure
  }

  const cachedTail: ChatMessage = {
    ...identifiedCached,
    parts: [...structure, { type: 'text', text: 'local answer' }]
  }

  const user: ChatMessage = { id: 'user-live', role: 'user', parts: [{ type: 'text', text: 'ship the fix' }] }

  const correction: ChatMessage = {
    id: 'user-correction',
    role: 'user',
    parts: [{ type: 'text', text: 'use the safer path' }]
  }

  const anonymousProjection = appendLiveSessionProjection([], {
    session_id: 'resume',
    inflight: {
      user: 'ship the fix',
      assistant: 'Earlier narration and flat nonextending dump',
      streaming: true,
      corrections: ['use the safer path'],
      correction_offsets: ['Earlier narration'.length]
    },
    queued: { user: 'queued follow-up' }
  })

  const image = `data:image/png;base64,${'A'.repeat(128)}`

  const settled: ChatMessage = {
    ...cachedTail,
    pending: false,
    reactions: [{ emoji: '👍', author: 'user', at: 3 }],
    parts: [...structure, { type: 'text', text: `local answer${image}` }]
  }

  const scenarios: Array<{ name: string; next: ChatMessage[]; previous: ChatMessage[]; expectedText: string }> = [
    {
      name: 'same id',
      next: [{ ...identifiedCached, pending: false, parts: [] }],
      previous: [identifiedCached],
      expectedText: ''
    },
    {
      name: 'same durable row id',
      next: [{ ...identifiedCached, id: 'stored-owned', pending: false, parts: [] }],
      previous: [identifiedCached],
      expectedText: ''
    },
    {
      name: 'settled same answer under its new durable id',
      next: [{ id: 'stored-reply', role: 'assistant', parts: [{ type: 'text', text: 'local answer' }] }],
      previous: [settled],
      expectedText: `local answer\n${image}`
    },
    {
      name: 'anonymous current flat projection with a queued following user',
      next: anonymousProjection,
      previous: [user, correction, cachedTail],
      expectedText: 'local answer'
    }
  ]

  for (const scenario of scenarios) {
    const result = reconcileResumeMessages(scenario.next, scenario.previous)
    const carried = result.find(message => message.parts.some(part => part.type === 'tool-call'))

    expect.soft(carried, scenario.name).toBeDefined()
    expect
      .soft(
        carried?.parts.filter(part => part.type === 'tool-call'),
        scenario.name
      )
      .toEqual(structure)
    expect.soft(carried ? text(carried) : '', scenario.name).toBe(scenario.expectedText)
    expect.soft(carried?.rowId, scenario.name).toBe(scenario.previous.at(-1)?.rowId)
    expect.soft(carried?.reactions, scenario.name).toEqual(scenario.previous.at(-1)?.reactions)
  }
})
