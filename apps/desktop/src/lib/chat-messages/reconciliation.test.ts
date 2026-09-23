import { expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, textPart, toChatMessages } from './index'

const row = (id: string, role: 'user' | 'assistant', text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role,
  parts: [textPart(text)],
  ...extra
})

it('reconciles only the represented failed tail, retaining its structured error and omitted segments', () => {
  const hydrated = toChatMessages([
    { role: 'user', content: 'read it', timestamp: 1 },
    {
      role: 'assistant',
      content: '',
      timestamp: 2,
      tool_calls: [{ id: 'call', function: { name: 'read_file', arguments: '{"path":"README.md"}' } }]
    },
    { role: 'tool', tool_call_id: 'call', tool_name: 'read_file', content: 'contents', timestamp: 3 },
    { role: 'assistant', content: 'Done.', timestamp: 4 }
  ])

  const storedAssistant = hydrated.find(message => message.role === 'assistant')!

  const failed = row('local-failure', 'assistant', 'Done.', {
    parts: storedAssistant.parts,
    error: 'connection lost after completion',
    errorSurface: { code: 'transport_lost', layer: 'streaming', retryable: true }
  })

  for (const id of [failed.id, storedAssistant.id]) {
    const merged = preserveLocalAssistantErrors(hydrated, [row('local-user', 'user', 'read it'), { ...failed, id }])
    expect(merged.filter(message => message.role === 'assistant')).toHaveLength(1)
    expect(merged.at(-1)).toMatchObject({
      id: storedAssistant.id,
      error: failed.error,
      errorSurface: failed.errorSurface,
      pending: false
    })
  }

  const user = row('user', 'user', 'read it')
  const earlier = row('earlier', 'assistant', 'Done.')
  const laterUser = row('later-user', 'user', 'read it')

  const cases: { name: string; stored: ChatMessage[]; local: ChatMessage[] }[] = [
    {
      name: 'omitted continuation',
      stored: [user, earlier],
      local: [user, earlier, row('hidden', 'user', 'Continue.', { hidden: true }), failed]
    },
    { name: 'older failed segment', stored: [user, earlier], local: [user, failed, earlier] },
    { name: 'repeated prompt', stored: [user, earlier], local: [user, earlier, laterUser, failed] },
    { name: 'older identical text', stored: [user, earlier, laterUser], local: [user, earlier, laterUser, failed] },
    {
      name: 'different attachments',
      stored: [{ ...user, attachmentRefs: ['a.png'] }, earlier],
      local: [{ ...user, attachmentRefs: ['b.png'] }, failed]
    },
    {
      name: 'different durable row',
      stored: [user, { ...earlier, rowId: 10 }],
      local: [user, { ...failed, rowId: 20 }]
    },
    {
      name: 'reused tool id across turns',
      stored: [user, storedAssistant, laterUser],
      local: [user, storedAssistant, laterUser, failed]
    }
  ]

  for (const fixture of cases) {
    const merged = preserveLocalAssistantErrors(fixture.stored, fixture.local)
    expect(merged.find(message => message.id === failed.id)?.error, fixture.name).toBe(failed.error)

    for (const stored of fixture.stored.filter(message => message.role === 'assistant')) {
      expect(merged.find(message => message.id === stored.id)?.error, fixture.name).toBeUndefined()
    }
  }

  const current = [user, row('earlier-live', 'assistant', 'First.'), failed]
  const stored = [user, row('earlier-stored', 'assistant', 'First.'), earlier]
  const merged = preserveLocalAssistantErrors(stored, current)
  expect(merged.map(message => message.id)).toEqual(stored.map(message => message.id))
  expect(merged.at(-1)).toMatchObject({ error: failed.error, errorSurface: failed.errorSurface })
})

const turnLabels = (messages: ChatMessage[]) => messages.map(message => `${message.role}:${chatMessageText(message)}`)

it('drops a local errored turn the refreshed transcript already stored under new ids', () => {
  const stored = [
    row('s1', 'user', 'old q', { rowId: 1 }),
    row('s2', 'assistant', 'old a', { rowId: 2 }),
    row('s3', 'user', 'new q', { rowId: 3 }),
    row('s4', 'assistant', 'new a', { rowId: 4 })
  ]

  const current = [
    row('user-1-x', 'user', 'old q'),
    row('assistant-stream-1-0', 'assistant', 'old a', { error: 'boom' }),
    row('s3', 'user', 'new q'),
    row('s4', 'assistant', 'new a')
  ]

  expect(turnLabels(preserveLocalAssistantErrors(stored, current))).toEqual(turnLabels(stored))
})

it('keeps an unstored errored turn at its original position instead of after newer turns', () => {
  const stored = [
    row('s1', 'user', 'first q'),
    row('s2', 'assistant', 'first a'),
    row('s3', 'user', 'later q'),
    row('s4', 'assistant', 'later a')
  ]

  const current = [
    row('s1', 'user', 'first q'),
    row('s2', 'assistant', 'first a'),
    row('user-9-x', 'user', 'failed q'),
    row('assistant-stream-9-0', 'assistant', '', { error: 'boom' }),
    row('s3', 'user', 'later q'),
    row('s4', 'assistant', 'later a')
  ]

  expect(turnLabels(preserveLocalAssistantErrors(stored, current))).toEqual([
    'user:first q',
    'assistant:first a',
    'user:failed q',
    'assistant:',
    'user:later q',
    'assistant:later a'
  ])
})
