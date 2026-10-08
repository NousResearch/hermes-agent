import { describe, expect, it } from 'vitest'

import { type AccountOperation, accountOperationFor } from './account-operations'

const operation = (opId: string, scope: AccountOperation['scope'], settled = false): AccountOperation => ({
  connectors: ['gmail'],
  deadlineAt: 1_800_000_000,
  opId,
  scope,
  seq: 0,
  settled,
  settledBy: null,
  targets: []
})

describe('accountOperationFor', () => {
  it('answers only with an operation started in the same profile scope', () => {
    const operations = {
      a: operation('a', 'work-profile'),
      b: operation('b', { connectionId: 'remote', profile: 'home' })
    }

    expect(accountOperationFor(operations, 'gmail', 'work-profile')?.opId).toBe('a')
    expect(accountOperationFor(operations, 'gmail', { connectionId: 'remote', profile: 'home' })?.opId).toBe('b')
    expect(accountOperationFor(operations, 'gmail', 'other')).toBeNull()
  })

  it('treats a bare profile and the same profile pinned to the active connection as one scope', () => {
    const operations = { a: operation('a', 'home') }

    expect(accountOperationFor(operations, 'gmail', { connectionId: 'local', profile: 'home' })?.opId).toBe('a')
  })
})
