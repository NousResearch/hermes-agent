import { describe, expect, it } from 'vitest'

import type { ConnectionTarget } from '@/store/connection-request'

import { type AccountOperation, accountOperationFor, retryableAccount } from './account-operations'

const operation = (opId: string, scope: AccountOperation['scope'], settled = false): AccountOperation => ({
  connectors: ['gmail'],
  deadlineAt: 1_800_000_000,
  opId,
  request: { reconnect: false },
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

describe('retryableAccount', () => {
  const target = (state: ConnectionTarget['state']): ConnectionTarget => ({
    action: 'reconnect',
    connectUrl: null,
    connectionId: 'ca_new',
    detail: '',
    discoveryError: null,
    instructions: null,
    kind: 'connector',
    name: 'gmail',
    requiredEnv: [],
    state,
    tools: []
  })

  const repair = (state: ConnectionTarget['state']): AccountOperation => ({
    ...operation('a', 'home'),
    request: { connectionId: 'ca_work', reconnect: true },
    targets: [target(state)]
  })

  it('names the account a failed or expired sign-in repairs, so only that row may retry', () => {
    expect(retryableAccount(repair('failed'))).toBe('ca_work')
    expect(retryableAccount(repair('expired'))).toBe('ca_work')
  })

  it('names nothing while a sign-in still waits, or for an add', () => {
    expect(retryableAccount(repair('initiated'))).toBeNull()
    expect(retryableAccount({ ...repair('failed'), request: { alias: 'work', reconnect: false } })).toBeNull()
    expect(retryableAccount(null)).toBeNull()
  })
})
