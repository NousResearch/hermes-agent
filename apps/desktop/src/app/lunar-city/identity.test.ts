import { describe, expect, it } from 'vitest'

import { entityKey } from './identity'
import type { EntityIdentity } from './model'

describe('entityKey', () => {
  it('includes the complete source identity so similarly named workers on separate connections cannot collide', () => {
    const identities: EntityIdentity[] = [
      { kind: 'profile', connectionId: 'local', profile: 'worker' },
      { kind: 'profile', connectionId: 'ssh-1', profile: 'worker' },
      { kind: 'session', connectionId: 'local', profile: 'worker', sessionId: 'same-session' },
      {
        kind: 'subagent',
        connectionId: 'local',
        profile: 'worker',
        sessionId: 'same-session',
        subagentId: 'child'
      },
      {
        kind: 'kanban',
        connectionId: 'local',
        profile: 'worker',
        board: 'default',
        taskId: 'same-session',
        workerId: 'child'
      }
    ]

    expect(new Set(identities.map(entityKey)).size).toBe(identities.length)
  })

  it('includes the Kanban profile owner when a connection serves multiple profiles', () => {
    const defaultProfile = {
      board: 'default',
      connectionId: 'shared-gateway',
      kind: 'kanban',
      profile: 'default',
      taskId: 'same-task'
    } as unknown as EntityIdentity

    const researchProfile = {
      ...defaultProfile,
      profile: 'research'
    } as unknown as EntityIdentity

    expect(entityKey(defaultProfile)).not.toBe(entityKey(researchProfile))
  })

  it('uses tagged escaped fields so delimiters, unicode, and optional empty identifiers cannot create a collision', () => {
    const special = entityKey({
      kind: 'kanban',
      connectionId: 'a:b/% ✓',
      board: 'board:one',
      profile: 'worker',
      taskId: 'task/one',
      runId: '',
      workerId: ''
    })

    const ambiguous = entityKey({
      kind: 'kanban',
      connectionId: 'a',
      board: 'b/% ✓:board',
      profile: 'worker',
      taskId: 'one:task/one',
      runId: undefined,
      workerId: undefined
    })

    expect(special).not.toBe(ambiguous)
    expect(special).toContain('connection:string:')
    expect(special).toContain('run:string:0:')
    expect(ambiguous).toContain('run:undefined')
  })

  it('fails closed when a required canonical identity field is missing instead of using a display name', () => {
    expect(() =>
      entityKey({ kind: 'session', connectionId: 'local', profile: 'worker', sessionId: ' ' } as EntityIdentity)
    ).toThrow(/sessionId is required/i)
  })

  it('distinguishes absent, empty, literal legacy sentinels, delimiters, and Unicode optional IDs', () => {
    const base = {
      board: 'board:=/✓',
      connectionId: 'connection:=/✓',
      kind: 'kanban' as const,
      profile: 'worker',
      taskId: 'task:=/✓'
    }

    const keys = [
      entityKey(base),
      entityKey({ ...base, runId: '', workerId: '' }),
      entityKey({ ...base, runId: '@absent' }),
      entityKey({ ...base, workerId: '@absent' }),
      entityKey({ ...base, runId: '@empty' }),
      entityKey({ ...base, workerId: '@empty' }),
      entityKey({ ...base, runId: '::=/% ✓', workerId: '::=/% ✓' })
    ]

    expect(new Set(keys).size).toBe(keys.length)
    expect(keys.every(key => key.includes(':'))).toBe(true)
  })
})
