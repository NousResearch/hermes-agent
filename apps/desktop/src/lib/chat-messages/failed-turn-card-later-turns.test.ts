import { describe, expect, it } from 'vitest'

import { type ChatMessage, preserveLocalAssistantErrors } from './index'

describe('persisted failed-turn card after later turns', () => {
  const surface = { code: 'billing', layer: 'billing', provider: 'test-provider', retryable: false } as const

  const stored: ChatMessage[] = [
    { id: 'stored-user-1', parts: [{ text: 'review the draft', type: 'text' }], role: 'user', rowId: 1 },
    { error: 'Credits exhausted.', errorSurface: surface, id: 'stored-error', parts: [], role: 'assistant', rowId: 2 },
    { id: 'stored-notice', parts: [{ text: 'Your request was not processed.', type: 'text' }], role: 'system' },
    { id: 'stored-user-2', parts: [{ text: 'try again', type: 'text' }], role: 'user', rowId: 3 },
    { id: 'stored-reply-2', parts: [{ text: 'Here is the review.', type: 'text' }], role: 'assistant', rowId: 4 }
  ]

  const local: ChatMessage[] = [
    stored[0],
    { error: 'Credits exhausted.', errorSurface: surface, id: 'local-error', parts: [], role: 'assistant' },
    stored[3],
    stored[4]
  ]

  it('keeps the stored card once and in its own turn after a later turn', () => {
    const merged = preserveLocalAssistantErrors(stored, local)

    expect(merged.filter(message => message.error).map(message => message.id)).toEqual(['stored-error'])
    expect(merged.map(message => message.id)).toEqual(stored.map(message => message.id))
  })

  it('stays at one card across repeated refreshes', () => {
    let current = local

    for (let refresh = 0; refresh < 3; refresh += 1) {
      current = preserveLocalAssistantErrors(stored, current)
    }

    expect(current.filter(message => message.error)).toHaveLength(1)
  })

  it('does not fold an old local error into a matching error in a later turn', () => {
    const newerError = { ...stored[1], id: 'newer-error', rowId: 5 }
    const history = [stored[0], stored[3], newerError]
    const merged = preserveLocalAssistantErrors(history, local)

    expect(merged.filter(message => message.error).map(message => message.id)).toEqual(['local-error', 'newer-error'])
  })

  it('keeps the local card when the older user row has no durable rowId', () => {
    const withoutRow = [{ ...local[0], rowId: undefined }, ...local.slice(1)]
    const merged = preserveLocalAssistantErrors(stored, withoutRow)

    expect(merged.some(message => message.id === 'local-error')).toBe(true)
  })

  it('keeps the local card when the older durable user row is outside the refreshed page', () => {
    const history = stored.map(message => (message.id === 'stored-user-1' ? { ...message, rowId: 99 } : message))
    const merged = preserveLocalAssistantErrors(history, local)

    expect(merged.some(message => message.id === 'local-error')).toBe(true)
  })

  it('keeps a different local failure within the same durable turn', () => {
    const history = stored.map(message =>
      message.id === 'stored-error' ? { ...message, errorSurface: { ...surface, code: 'transport_lost' } } : message
    )

    const merged = preserveLocalAssistantErrors(history, local)

    expect(merged.some(message => message.id === 'local-error')).toBe(true)
  })

  it('does not treat a hidden user directive as a new visible turn boundary', () => {
    const hidden: ChatMessage = { id: 'directive', role: 'user', hidden: true, rowId: 10, parts: [] }
    const history = [stored[0], hidden, ...stored.slice(1)]
    const current = [local[0], hidden, ...local.slice(1)]
    const merged = preserveLocalAssistantErrors(history, current)

    expect(merged.filter(message => message.error).map(message => message.id)).toEqual(['stored-error'])
  })
})
