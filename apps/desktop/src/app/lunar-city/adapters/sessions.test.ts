import { describe, expect, it } from 'vitest'

import { normalizeOwnedSubagents, normalizeSessions, normalizeSubagents, ownerObservationKey } from './sessions'

const session = (overrides: Record<string, unknown> = {}) => ({
  archived: false,
  cwd: '/workspace/not-an-identity',
  ended_at: null,
  id: 'session-1',
  input_tokens: 0,
  is_active: true,
  last_active: 20,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 10,
  title: 'same display title',
  tool_call_count: 0,
  ...overrides
})

describe('session live adapter', () => {
  it('does not collapse duplicate session ids from distinct connection-tagged owners', () => {
    const normalized = normalizeSessions(
      [
        session({ connection_id: 'local', git_repo_root: '/projects/alpha', profile: 'worker' }),
        session({ connection_id: 'ssh-1', git_repo_root: '/projects/alpha', profile: 'worker' })
      ],
      { observedAt: 42 }
    )

    expect(normalized.entities.map(entity => entity.identity.connectionId)).toEqual(['local', 'ssh-1'])
    expect(new Set(normalized.entities.map(entity => entity.key)).size).toBe(2)
    expect(normalized.entities.map(entity => entity.projectId)).toEqual(['/projects/alpha', '/projects/alpha'])
  })

  it('applies freshness and the last authoritative timestamp per exact connection source', () => {
    const normalized = normalizeSessions(
      [session({ connection_id: 'local', profile: 'worker' }), session({ connection_id: 'ssh-1', profile: 'worker' })],
      {
        observedAt: 999,
        sourceObservations: new Map([
          [ownerObservationKey('local', 'worker'), { fresh: true, generation: 4, observedAt: 80 }],
          [ownerObservationKey('ssh-1', 'worker'), { fresh: false, generation: 2, observedAt: 40 }]
        ])
      }
    )

    expect(
      normalized.entities.map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'authoritative', 80],
      ['ssh-1', 'stale', 40]
    ])
    expect(normalized.sources).toEqual([
      { authority: 'authoritative', observedAt: 80, source: 'session:local:worker' },
      { authority: 'stale', observedAt: 40, source: 'session:ssh-1:worker' }
    ])
  })

  it('fails closed for untagged sessions in a registry topology', () => {
    const normalized = normalizeSessions([session({ profile: 'worker' })], { observedAt: 42 })

    expect(normalized.entities).toEqual([])
  })

  it('allows a legacy session only when local ownership was explicitly proven', () => {
    const normalized = normalizeSessions([session({ git_repo_root: null, profile: 'worker' })], {
      legacyConnectionId: 'local',
      legacySingleBackend: true,
      observedAt: 42
    })

    expect(normalized.entities).toHaveLength(1)
    expect(normalized.entities[0]?.identity).toEqual({
      connectionId: 'local',
      kind: 'session',
      profile: 'worker',
      sessionId: 'session-1'
    })
    expect(normalized.entities[0]?.projectId).toBeUndefined()
  })

  it('joins a terminal child only through a unique exact-owner parent session', () => {
    const sessions = normalizeSessions([session({ connection_id: 'local', id: 'parent', profile: 'worker' })], {
      observedAt: 42
    })

    const normalized = normalizeSubagents(
      {
        parent: [
          {
            filesRead: [],
            filesWritten: [],
            goal: 'this must not become identity',
            id: 'sub-1',
            parentId: 'parent',
            startedAt: 10,
            status: 'completed',
            stream: [{ at: 20, kind: 'summary', text: 'secret stream text' }],
            taskCount: 1,
            taskIndex: 0,
            updatedAt: 20
          }
        ]
      },
      sessions.entities,
      { observedAt: 42 }
    )

    expect(normalized.entities).toHaveLength(1)
    expect(normalized.entities[0]).toMatchObject({
      animation: 'done',
      authority: 'authoritative',
      destination: 'project',
      identity: {
        connectionId: 'local',
        kind: 'subagent',
        profile: 'worker',
        sessionId: 'parent',
        subagentId: 'sub-1'
      }
    })
    expect(normalized.entities[0]).not.toHaveProperty('stream')
  })

  it('omits a child when matching parent ids have ambiguous source owners', () => {
    const parents = normalizeSessions(
      [
        session({ connection_id: 'local', id: 'parent', profile: 'worker' }),
        session({ connection_id: 'ssh-1', id: 'parent', profile: 'worker' })
      ],
      { observedAt: 42 }
    )

    const normalized = normalizeSubagents(
      {
        parent: [
          {
            filesRead: [],
            filesWritten: [],
            goal: 'work',
            id: 'sub-1',
            parentId: 'parent',
            startedAt: 10,
            status: 'running',
            stream: [],
            taskCount: 1,
            taskIndex: 0,
            updatedAt: 20
          }
        ]
      },
      parents.entities,
      { observedAt: 42 }
    )

    expect(normalized.entities).toEqual([])
  })

  it('normalizes per-owner child batches without collapsing duplicate parent or child ids', () => {
    const localOwner = {
      connectionId: 'local',
      kind: 'session' as const,
      profile: 'worker',
      sessionId: 'parent'
    }

    const remoteOwner = { ...localOwner, connectionId: 'ssh-1' }

    const row = {
      filesRead: [],
      filesWritten: [],
      goal: 'work',
      id: 'sub-1',
      parentId: 'parent',
      startedAt: 10,
      status: 'running' as const,
      stream: [],
      taskCount: 1,
      taskIndex: 0,
      updatedAt: 20
    }

    const normalized = normalizeOwnedSubagents(
      [
        { owner: localOwner, rows: [row] },
        { owner: remoteOwner, rows: [row] }
      ],
      {
        observedAt: 0,
        sourceObservations: new Map([
          [ownerObservationKey('local', 'worker'), { fresh: true, generation: 3, observedAt: 80 }],
          [ownerObservationKey('ssh-1', 'worker'), { fresh: false, generation: 1, observedAt: 30 }]
        ])
      }
    )

    expect(
      normalized.entities.map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'authoritative', 80],
      ['ssh-1', 'stale', 30]
    ])
    expect(new Set(normalized.entities.map(entity => entity.key)).size).toBe(2)
  })

  it('fails closed when an owned child batch exceeds its explicit bounds', () => {
    const owner = {
      connectionId: 'local',
      kind: 'session' as const,
      profile: 'worker',
      sessionId: 'parent'
    }

    const rows = ['one', 'two'].map(id => ({
      filesRead: [],
      filesWritten: [],
      goal: 'work',
      id,
      parentId: 'parent',
      startedAt: 10,
      status: 'running' as const,
      stream: [],
      taskCount: 1,
      taskIndex: 0,
      updatedAt: 20
    }))

    expect(
      normalizeOwnedSubagents([{ owner, rows }], { observedAt: 20 }, { maxOwners: 1, maxRows: 1, maxRowsPerOwner: 1 })
        .entities
    ).toEqual([])
  })

  it('marks conflicting observations for one exact child partial regardless of input order', () => {
    const owner = {
      connectionId: 'local',
      kind: 'session' as const,
      profile: 'worker',
      sessionId: 'parent'
    }

    const row = (status: 'completed' | 'running') => ({
      filesRead: [],
      filesWritten: [],
      goal: 'work',
      id: 'child',
      parentId: 'parent',
      startedAt: 10,
      status,
      stream: [],
      taskCount: 1,
      taskIndex: 0,
      updatedAt: 20
    })

    const options = { maxOwners: 2, maxRows: 4, maxRowsPerOwner: 2 }

    const forward = normalizeOwnedSubagents(
      [{ owner, rows: [row('running'), row('completed')] }],
      { observedAt: 20 },
      options
    )

    const reversed = normalizeOwnedSubagents(
      [{ owner, rows: [row('completed'), row('running')] }],
      { observedAt: 20 },
      options
    )

    expect(forward).toEqual(reversed)
    expect(forward.entities[0]).toMatchObject({
      animation: 'unavailable',
      authority: 'partial',
      destination: 'unknown'
    })
  })

  it('fails closed when a subagent row claims a different parent than its map key', () => {
    const parents = normalizeSessions([session({ connection_id: 'local', id: 'parent', profile: 'worker' })], {
      observedAt: 42
    })

    const normalized = normalizeSubagents(
      {
        parent: [
          {
            filesRead: [],
            filesWritten: [],
            goal: 'work',
            id: 'sub-1',
            parentId: 'other-parent',
            startedAt: 10,
            status: 'running',
            stream: [],
            taskCount: 1,
            taskIndex: 0,
            updatedAt: 20
          }
        ]
      },
      parents.entities,
      { observedAt: 42 }
    )

    expect(normalized.entities).toEqual([])
  })
})
