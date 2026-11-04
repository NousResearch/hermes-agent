import { expect, it } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { reconcileAuthoritativeMessages } from './resume-reconcile'

const prompt = 'Inspect the current run'
const first = 'Checking the run.'
const second = 'Still checking.'

const round = (id: number, text: string): SessionMessage => ({
  id,
  role: 'assistant',
  content: text,
  timestamp: id,
  tool_calls: [{ id: `call-${id}`, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
})

it.each([
  { warm: false, repeat: false },
  { warm: true, repeat: false },
  { warm: false, repeat: true },
  { warm: true, repeat: true }
])('keeps the running turn and every durable next-turn prompt once ($warm, repeat=$repeat)', ({ warm, repeat }) => {
  const next = repeat ? prompt : 'Check the tests next'

  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    round(2, first),
    { id: 3, role: 'user', content: next, timestamp: 3, display_metadata: '{"_queued_prompt":true}' },
    round(4, second),
    {
      id: 5,
      role: 'user',
      content: 'Then report the result',
      timestamp: 5,
      display_metadata: '{"_queued_prompt":true}'
    }
  ]

  const projection = {
    session_id: 'queue-runtime',
    inflight: { user: prompt, assistant: `${first}\n\n${second}`, streaming: true },
    queued: { user: next }
  }

  let current = reconcileAuthoritativeMessages(rows, [], projection)

  if (warm) {
    current = reconcileAuthoritativeMessages(rows, current, projection)
  }

  expect(current.filter(message => message.role === 'user').map(chatMessageText)).toEqual([
    prompt,
    next,
    'Then report the result'
  ])
  expect(current.filter(message => message.role === 'user').map(message => message.rowId)).toEqual([1, 3, 5])
  expect(
    current.flatMap(message =>
      message.role === 'assistant' ? message.parts.filter(part => part.type === 'text').map(part => part.text) : []
    )
  ).toEqual([first, second])
  expect(new Set(current.map(message => message.id)).size).toBe(current.length)
})

it('refreshes changed queue text in the same warm slot beside a durable queued prompt', () => {
  // Arrange
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    round(2, first),
    { id: 3, role: 'user', content: 'Durable queue', timestamp: 3, display_metadata: '{"_queued_prompt":true}' }
  ]

  const projection = {
    session_id: 'queue-runtime',
    inflight: { user: prompt, assistant: first, streaming: true },
    queued: { user: 'Old next slot' }
  }

  // Act
  const once = reconcileAuthoritativeMessages(rows, [], projection)

  // Assert
  expect(once.filter(message => message.role === 'user').map(chatMessageText)).toEqual([
    prompt,
    'Durable queue',
    'Old next slot'
  ])

  // Act
  const updated = { ...projection, queued: { user: 'Updated next slot' } }
  const twice = reconcileAuthoritativeMessages(rows, once, updated)

  // Assert
  expect(new Set(twice.map(message => message.id)).size).toBe(twice.length)
  expect(twice.filter(message => message.role === 'user').map(chatMessageText)).toEqual([
    prompt,
    'Durable queue',
    'Updated next slot'
  ])
  expect(twice.find(message => message.rowId === 3)).toEqual(once.find(message => message.rowId === 3))
  expect(twice.find(message => message.id === 'user-queued-queue-runtime')).toMatchObject({
    role: 'user',
    parts: [{ type: 'text', text: 'Updated next slot' }]
  })
})

it('does not re-project an already durable queue after the running turn settles', () => {
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    { id: 2, role: 'assistant', content: 'Done.', timestamp: 2 },
    { id: 3, role: 'user', content: 'Run this next', timestamp: 3, display_metadata: '{"_queued_prompt":true}' }
  ]

  const projection = { session_id: 'queue-runtime', queued: { user: 'Run this next' } }
  const once = reconcileAuthoritativeMessages(rows, [], projection)
  const twice = reconcileAuthoritativeMessages(rows, once, projection)
  expect(twice.map(chatMessageText)).toEqual([prompt, 'Done.', 'Run this next'])
})

it.each([
  ['@file:/fixture/a.txt', '@file:/fixture/b.txt'],
  ['@image:/fixture/a.png', '@image:/fixture/b.png']
])('preserves a newer queued attachment after the older durable queue (%s)', (older, newer) => {
  // Arrange
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    round(2, first),
    {
      id: 3,
      role: 'user',
      content: `${older}\n\nReview this`,
      timestamp: 3,
      display_metadata: '{"_queued_prompt":true}'
    },
    round(4, second)
  ]

  const projection = {
    session_id: 'queue-runtime',
    inflight: { user: prompt, assistant: `${first}\n\n${second}`, streaming: true },
    queued: { user: `${newer}\n\nReview this` }
  }

  // Act
  const once = reconcileAuthoritativeMessages(rows, [], projection)
  const twice = reconcileAuthoritativeMessages(rows, once, projection)

  // Assert
  for (const messages of [once, twice]) {
    const users = messages.filter(message => message.role === 'user')
    const visible = users.map(message => [...(message.attachmentRefs ?? []), chatMessageText(message)].join('\n'))
    expect(users).toHaveLength(3)
    expect(visible[1]).toContain(older)
    expect(visible[2]).toContain(newer)
  }
})

it.each(['@file:/fixture/', '@image:/fixture/'])(
  'refreshes queued attachments without consuming accepted same-caption users (%s)',
  reference => {
    // Arrange
    const rows: SessionMessage[] = [
      { id: 1, role: 'user', content: prompt, timestamp: 1 },
      round(2, first),
      {
        id: 3,
        role: 'user',
        content: `${reference}durable\n\nReview this`,
        timestamp: 3,
        display_metadata: '{"_queued_prompt":true}'
      }
    ]

    const projection = {
      session_id: 'queue-runtime',
      inflight: { user: prompt, assistant: first, streaming: true },
      queued: { user: `${reference}old\n\nReview this` }
    }

    const accepted: ChatMessage = {
      id: 'user-accepted',
      role: 'user',
      parts: [{ type: 'text', text: 'Review this' }],
      attachmentRefs: [`${reference}accepted`]
    }

    let current = [...reconcileAuthoritativeMessages(rows, [], projection), accepted]

    for (const attachment of ['updated', 'updated', 'latest']) {
      // Act
      current = reconcileAuthoritativeMessages(rows, current, {
        ...projection,
        queued: { user: `${reference}${attachment}\n\nReview this` }
      })

      // Assert
      expect(new Set(current.map(message => message.id)).size).toBe(current.length)
      expect(current.filter(message => message.role === 'user')).toHaveLength(4)
      expect(current).toContainEqual(accepted)
      expect(current.find(message => message.rowId === 3)).toMatchObject({
        attachmentRefs: [`${reference}durable`]
      })
      expect(chatMessageText(current.find(message => message.id === 'user-queued-queue-runtime')!)).toBe(
        `${reference}${attachment}\n\nReview this`
      )
    }
  }
)

it('preserves a consumed queue occurrence under its own identity when the warm slot advances', () => {
  // Arrange
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    round(2, first),
    { id: 3, role: 'user', content: 'Durable queue', timestamp: 3, display_metadata: '{"_queued_prompt":true}' }
  ]

  const projection = {
    session_id: 'queue-runtime',
    inflight: { user: prompt, assistant: first, streaming: true },
    queued: { user: 'Accepted next turn' }
  }

  const response: ChatMessage = {
    id: 'accepted-response',
    role: 'assistant',
    parts: [{ type: 'text', text: 'Accepted turn reply' }]
  }

  let current = [...reconcileAuthoritativeMessages(rows, [], projection), response]
  let acceptedId: string | undefined

  for (const queued of ['Updated next slot', 'Updated next slot', 'Latest next slot']) {
    // Act
    current = reconcileAuthoritativeMessages(rows, current, { ...projection, queued: { user: queued } })

    // Assert
    expect(new Set(current.map(message => message.id)).size).toBe(current.length)
    expect(current.filter(message => message.role === 'user')).toHaveLength(4)
    expect(current.filter(message => message.role === 'user').map(chatMessageText)).toEqual(
      expect.arrayContaining([prompt, 'Accepted next turn', 'Durable queue', queued])
    )

    const accepted = current.find(
      message => message.role === 'user' && chatMessageText(message) === 'Accepted next turn'
    )!

    expect(accepted.id).not.toBe('user-queued-queue-runtime')
    expect(accepted.id).toBe(acceptedId ?? accepted.id)
    acceptedId = accepted.id
    expect(current).toContainEqual(response)
  }
})
