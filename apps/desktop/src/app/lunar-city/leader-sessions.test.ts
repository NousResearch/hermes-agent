import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  createLeaderSessionResolver,
  type LeaderOwner,
  leaderOwnerKey,
  type LeaderSessionDependencies,
  type LeaderSessionPersistence
} from './leader-sessions'

const OWL: LeaderOwner = { connectionId: 'a', profile: 'owl' }

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function harness(initial: unknown = null) {
  let persisted = initial

  const request = vi.fn<LeaderSessionDependencies['request']>(async (_owner, method) => {
    if (method === 'session.resume') {
      return { session_id: 'r1' }
    }

    return { session_id: 'r-created', stored_session_id: 's-created' }
  })

  const findStoredSession = vi.fn<LeaderSessionDependencies['findStoredSession']>(async (_owner, storedId) => ({
    connection_id: 'a',
    id: storedId,
    profile: 'owl'
  }))

  const recordOwnerHint = vi.fn<LeaderSessionDependencies['recordOwnerHint']>()
  const retainOwner = vi.fn<LeaderSessionDependencies['retainOwner']>(async () => () => undefined)

  const writePersistence = vi.fn<LeaderSessionDependencies['writePersistence']>(next => {
    persisted = next
  })

  const dependencies = {
    findStoredSession,
    readPersistence: () => persisted,
    recordOwnerHint,
    retainOwner,
    request,
    writePersistence
  } satisfies LeaderSessionDependencies & {
    retainOwner(owner: LeaderOwner): Promise<() => void>
  }

  return {
    dependencies,
    findStoredSession,
    get persisted() {
      return persisted
    },
    recordOwnerHint,
    retainOwner,
    request,
    writePersistence
  }
}

function mapping(owner: LeaderOwner, storedId = 's1'): LeaderSessionPersistence {
  return { leaders: { [leaderOwnerKey(owner)]: { storedId } }, version: 1 }
}

describe('leader session ownership', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('resumes the mapped session only through its exact connection and canonical profile', async () => {
    const test = harness(mapping(OWL))
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession({ connectionId: ' a ', profile: ' owl ' })).resolves.toEqual({
      runtimeId: 'r1',
      storedId: 's1'
    })

    expect(test.findStoredSession).toHaveBeenCalledWith(OWL, 's1')
    expect(test.request).toHaveBeenCalledTimes(1)
    expect(test.request).toHaveBeenCalledWith(OWL, 'session.resume', {
      cols: 96,
      profile: 'owl',
      session_id: 's1',
      source: 'desktop'
    })
    expect(test.request.mock.calls.some(([owner]) => owner.connectionId === 'b')).toBe(false)
    expect(test.recordOwnerHint).toHaveBeenCalledWith('s1', OWL)
  })

  it('uses collision-safe keys for connection and profile pairs', () => {
    expect(leaderOwnerKey({ connectionId: 'a::b', profile: 'c' })).toBe('a%3A%3Ab::c')
    expect(leaderOwnerKey({ connectionId: 'a', profile: 'b::c' })).toBe('a::b%3A%3Ac')
    expect(leaderOwnerKey({ connectionId: 'a::b', profile: 'c' })).not.toBe(
      leaderOwnerKey({ connectionId: 'a', profile: 'b::c' })
    )
    expect(leaderOwnerKey({ connectionId: ' moon base ', profile: ' ' })).toBe('moon%20base::default')
  })

  it.each([
    ['malformed JSON result', null],
    ['unknown version', { leaders: {}, version: 2 }],
    ['missing leaders record', { version: 1 }],
    ['array leaders value', { leaders: [], version: 1 }],
    ['empty durable id', { leaders: { [leaderOwnerKey(OWL)]: { storedId: '' } }, version: 1 }],
    [
      'runtime id in a leader entry',
      { leaders: { [leaderOwnerKey(OWL)]: { runtimeId: 'r1', storedId: 's1' } }, version: 1 }
    ]
  ])('treats %s persistence as empty and creates an ordinary routed session', async (_name, persisted) => {
    const test = harness(persisted)
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).resolves.toEqual({ runtimeId: 'r-created', storedId: 's-created' })

    expect(test.findStoredSession).not.toHaveBeenCalled()
    expect(test.request).toHaveBeenCalledWith(OWL, 'session.create', {
      cols: 96,
      profile: 'owl',
      source: 'desktop'
    })
  })

  it.each([
    ['deleted row', null],
    ['mismatched profile', { connection_id: 'a', id: 's1', profile: 'fox' }],
    ['mismatched connection', { connection_id: 'b', id: 's1', profile: 'owl' }],
    ['mismatched durable id', { connection_id: 'a', id: 'foreign-s1', profile: 'owl' }]
  ])('recreates after a %s without adopting a similarly named foreign row', async (_name, row) => {
    const test = harness(mapping(OWL))
    test.findStoredSession.mockResolvedValue(row)
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).resolves.toEqual({ runtimeId: 'r-created', storedId: 's-created' })

    expect(test.request.mock.calls.map(([, method]) => method)).toEqual(['session.create'])
    expect(test.persisted).toEqual(mapping(OWL, 's-created'))
  })

  it('recreates only after an exact-owner resume returns no runtime id', async () => {
    const test = harness(mapping(OWL))
    test.request.mockImplementation(async (_owner, method) =>
      method === 'session.resume' ? { resumed: 's1' } : { session_id: 'r-created', stored_session_id: 's-created' }
    )
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).resolves.toEqual({ runtimeId: 'r-created', storedId: 's-created' })

    expect(test.request.mock.calls.map(([, method]) => method)).toEqual(['session.resume', 'session.create'])
    expect(test.persisted).toEqual(mapping(OWL, 's-created'))
  })

  it.each([
    ['a rejected missing-session error', new Error('4001: session not found')],
    ['an explicit missing-session result', { error: 'Stored session not found', status: 'error' }]
  ])('recreates after %s races the verified row read', async (_label, missingResult) => {
    const test = harness(mapping(OWL))

    test.request.mockImplementation(async (_owner, method) => {
      if (method === 'session.resume') {
        if (missingResult instanceof Error) {
          throw missingResult
        }

        return missingResult
      }

      return { session_id: 'r-recreated', stored_session_id: 's-recreated' }
    })
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).resolves.toEqual({
      runtimeId: 'r-recreated',
      storedId: 's-recreated'
    })
    expect(test.request.mock.calls.map(([, method]) => method)).toEqual(['session.resume', 'session.create'])
    expect(test.persisted).toEqual(mapping(OWL, 's-recreated'))
  })

  it('propagates an explicit resume authorization error without creating or replacing the mapping', async () => {
    const test = harness(mapping(OWL))
    test.request.mockResolvedValue({ error: '403: unauthorized for owl', status: 'error' })
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).rejects.toThrow('403: unauthorized for owl')

    expect(test.request.mock.calls.map(([, method]) => method)).toEqual(['session.resume'])
    expect(test.writePersistence).not.toHaveBeenCalled()
    expect(test.persisted).toEqual(mapping(OWL))
  })

  it('persists only the durable id after successful exact-owner resolution', async () => {
    const test = harness(mapping(OWL))
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await resolveLeaderSession(OWL)

    expect(test.writePersistence).toHaveBeenLastCalledWith(mapping(OWL))
    expect(JSON.stringify(test.persisted)).not.toContain('r1')
    expect(JSON.stringify(test.persisted)).not.toContain('runtime')
  })

  it('retains the exact owner socket through create, owner-hint publication, and persistence', async () => {
    const events: string[] = []
    const test = harness({ leaders: {}, version: 1 })

    test.retainOwner.mockImplementation(async owner => {
      events.push(`retain:${owner.connectionId}:${owner.profile}`)

      return () => {
        events.push('release')
      }
    })
    test.request.mockImplementation(async (_owner, method) => {
      events.push(method)

      return { session_id: 'r-created', stored_session_id: 's-created' }
    })
    test.recordOwnerHint.mockImplementation(() => events.push('hint'))
    test.writePersistence.mockImplementation(() => events.push('persist'))
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await resolveLeaderSession(OWL)

    expect(events).toEqual(['retain:a:owl', 'session.create', 'hint', 'persist', 'release'])
  })

  it('single-flights concurrent resolution for the same exact owner', async () => {
    const gate = deferred<unknown>()
    const test = harness(mapping(OWL))
    test.request.mockImplementation(async (_owner, method) => {
      expect(method).toBe('session.resume')

      return gate.promise
    })
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    const first = resolveLeaderSession(OWL)
    const second = resolveLeaderSession({ ...OWL })

    await vi.waitFor(() => expect(test.request).toHaveBeenCalledTimes(1))
    gate.resolve({ session_id: 'r1' })

    await expect(Promise.all([first, second])).resolves.toEqual([
      { runtimeId: 'r1', storedId: 's1' },
      { runtimeId: 'r1', storedId: 's1' }
    ])
    expect(test.findStoredSession).toHaveBeenCalledTimes(1)
  })

  it('preserves both mappings when different owners finish resolution concurrently', async () => {
    const owlCreate = deferred<unknown>()
    const foxCreate = deferred<unknown>()
    const test = harness({ leaders: {}, version: 1 })

    test.request.mockImplementation(async owner => (owner.profile === 'owl' ? owlCreate.promise : foxCreate.promise))
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)
    const fox: LeaderOwner = { connectionId: 'a', profile: 'fox' }
    const owlResolution = resolveLeaderSession(OWL)
    const foxResolution = resolveLeaderSession(fox)

    owlCreate.resolve({ session_id: 'r-owl', stored_session_id: 's-owl' })
    await owlResolution
    foxCreate.resolve({ session_id: 'r-fox', stored_session_id: 's-fox' })
    await foxResolution

    expect(test.persisted).toEqual({
      leaders: {
        [leaderOwnerKey(fox)]: { storedId: 's-fox' },
        [leaderOwnerKey(OWL)]: { storedId: 's-owl' }
      },
      version: 1
    })
  })

  it('propagates routed failures without an ambient fallback or mapping replacement', async () => {
    const test = harness(mapping(OWL))
    test.request.mockRejectedValue(new Error('connection a is offline'))
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).rejects.toThrow('connection a is offline')

    expect(test.request).toHaveBeenCalledTimes(1)
    expect(test.writePersistence).not.toHaveBeenCalled()
    expect(test.persisted).toEqual(mapping(OWL))
  })

  it('does not replace a stale mapping when exact-owner creation lacks either durable or runtime identity', async () => {
    const test = harness(mapping(OWL))
    test.findStoredSession.mockResolvedValue(null)
    test.request.mockResolvedValue({ session_id: 'r-created' })
    const resolveLeaderSession = createLeaderSessionResolver(test.dependencies)

    await expect(resolveLeaderSession(OWL)).rejects.toThrow(/durable stored session id/i)

    expect(test.writePersistence).not.toHaveBeenCalled()
    expect(test.recordOwnerHint).not.toHaveBeenCalled()
    expect(test.persisted).toEqual(mapping(OWL))
  })
})
