import { afterEach, describe, expect, it } from 'vitest'

import {
  $hiddenRecoveryEphemeral,
  $hiddenRecoveryRows,
  type RecoveryMessage,
  type RecoveryScope,
  recoveryStatuses,
  recoveryVisibilityKey,
  setRecoveryHidden
} from './message-recovery'

const user = (id: string, text = 'hello'): RecoveryMessage => ({
  id,
  role: 'user',
  content: text,
  metadata: { custom: { rowId: 1 } }
})

const assistant = (id: string, content: unknown, custom = {}): RecoveryMessage => ({
  id,
  role: 'assistant',
  content,
  metadata: { custom }
})

afterEach(() => {
  $hiddenRecoveryRows.set([])
  $hiddenRecoveryEphemeral.set([])
})

describe('message recovery presentation', () => {
  it('distinguishes missing and unfinished turns without treating notes, queued rows or errors as failed sends', () => {
    const messages: RecoveryMessage[] = [
      user('missing'),
      user('unfinished'),
      assistant('thought', [{ type: 'reasoning', text: 'checking' }]),
      user('notification', '[IMPORTANT: Background process finished]'),
      assistant('interim', 'I will check', { interim: true }),
      user('answered'),
      assistant('answer', 'A partial visible answer'),
      user('error'),
      { ...assistant('failed', []), status: { type: 'incomplete', reason: 'error' } },
      user('running'),
      { ...assistant('stream', []), status: { type: 'running' } },
      user('hidden'),
      { ...user('scaffold'), metadata: { custom: { displayKind: 'hidden' } } },
      user('delivery', 'Message from 🤖 Helper: ready'),
      user('user-queued-1')
    ]

    expect([...recoveryStatuses(messages)]).toEqual([
      ['hidden', 'missing'],
      ['unfinished', 'unfinished'],
      ['missing', 'missing']
    ])
  })

  it('hides stable rows only in their connection/profile/session and leaves the transcript unchanged', () => {
    const scope: RecoveryScope = {
      connection: 'local',
      profile: 'work',
      session: 'A',
      durable: true,
      ready: true,
      pending: false
    }

    const history = [user('row')]
    const snapshot = JSON.stringify(history)
    const identity = recoveryVisibilityKey(scope, history[0])!
    setRecoveryHidden(identity, true)
    expect($hiddenRecoveryRows.get()).toEqual([identity.key])

    for (const change of [{ session: 'B' }, { profile: 'personal' }, { connection: 'remote' }]) {
      expect($hiddenRecoveryRows.get()).not.toContain(recoveryVisibilityKey({ ...scope, ...change }, history[0])!.key)
    }

    const ephemeral = recoveryVisibilityKey(scope, { ...history[0], metadata: {} })!
    expect(ephemeral.persistent).toBe(false)
    setRecoveryHidden(ephemeral, true)
    expect($hiddenRecoveryEphemeral.get()).toContain(ephemeral.key)
    expect(JSON.stringify(history)).toBe(snapshot)
    setRecoveryHidden(identity, false)
    expect($hiddenRecoveryRows.get()).toEqual([])
    expect(recoveryVisibilityKey({ ...scope, profile: null }, history[0])?.persistent).toBe(false)
  })
})
