import { describe, expect, it } from 'vitest'

import {
  createUserInputDraftStore,
  draftKey,
  type UserInputDraftRequest
} from './user-input-drafts'

const request = (sessionId: string, requestId: string, options = ['a', 'b']): UserInputDraftRequest => ({
  questions: [{
    allowFreeText: false,
    defaultValue: 'a',
    id: 'choice',
    options,
    text: 'Pick one'
  }],
  requestId,
  sessionId
})

describe('native user-input draft store', () => {
  it('keeps independent drafts keyed by session and request', () => {
    const store = createUserInputDraftStore()
    const first = request('session-1', 'request-1')
    const second = request('session-2', 'request-1')

    expect(draftKey(first)).toBe('session-1:request-1')
    expect(store.get(first)).toEqual({ choice: 'a' })
    store.set(first, { choice: 'b' })
    store.set(second, { choice: 'a' })

    expect(store.get(first)).toEqual({ choice: 'b' })
    expect(store.get(second)).toEqual({ choice: 'a' })
  })

  it('retains valid values but drops closed options removed by a schema update', () => {
    const store = createUserInputDraftStore()
    const original = request('session-1', 'request-1')
    store.set(original, { choice: 'b' })

    expect(store.get(request('session-1', 'request-1', ['b', 'c']))).toEqual({ choice: 'b' })
    expect(store.get(request('session-1', 'request-1', ['a', 'c']))).toEqual({})
  })

  it('deletes only the terminal request draft', () => {
    const store = createUserInputDraftStore()
    const first = request('session-1', 'request-1')
    const second = request('session-1', 'request-2')
    store.set(first, { choice: 'b' })
    store.set(second, { choice: 'a' })

    store.delete(first)

    expect(store.get(first)).toEqual({ choice: 'a' })
    expect(store.get(second)).toEqual({ choice: 'a' })
  })
})
