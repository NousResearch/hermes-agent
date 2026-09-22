import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { $sessionStates, $sessionTiles } from '@/store/session-states'

import { type CommandPlanningSnapshot, CommandRejectedError, executeCommand, planCommand } from './command-broker'
import { createLunarCityCommandExecutors } from './command-executors'
import { entityKey } from './identity'
import type { EntityIdentity, LunarCitySnapshot, LunarEntity, SourceHealth } from './model'

const { pluginRest, requestForSessionProfile } = vi.hoisted(() => ({
  pluginRest: vi.fn(),
  requestForSessionProfile: vi.fn()
}))

vi.mock('@/api/plugins', () => ({ pluginRest }))
vi.mock('@/store/session-request-router', () => ({ requestForSessionProfile }))

function frozenMap<K, V>(entries: readonly (readonly [K, V])[]): ReadonlyMap<K, V> {
  return new Map(entries)
}

function runtimeState(
  storedSessionId: string,
  owner: Readonly<{ connectionId: string; profile: string }>
): ClientSessionState {
  return {
    storedSessionId,
    transcriptProvenance: {
      connectionId: owner.connectionId,
      coverage: 'latest-page',
      lineageRootId: null,
      profile: owner.profile,
      source: 'persisted-display',
      storedSessionId
    }
  } as ClientSessionState
}

function planning(identity: EntityIdentity, state = 'working'): CommandPlanningSnapshot {
  const key = entityKey(identity)
  const observedAt = 100

  const source: SourceHealth = {
    authority: 'authoritative',
    observedAt,
    source:
      identity.kind === 'kanban'
        ? `kanban:${encodeURIComponent(identity.connectionId)}:${encodeURIComponent(identity.profile)}`
        : `session:${identity.connectionId}`
  }

  const entity: LunarEntity = {
    animation: state,
    authority: 'authoritative',
    destination: 'project',
    identity,
    key,
    observedAt
  }

  const city: LunarCitySnapshot = {
    entities: frozenMap([[key, entity]]),
    observedAt,
    revision: 4,
    sources: [source]
  }

  const owner = { connectionId: identity.connectionId, profile: identity.profile }

  return {
    city,
    targets: frozenMap([
      [
        key,
        {
          availableOperations:
            identity.kind === 'session'
              ? (['send-guidance'] as const)
              : (['inspect-evidence', 'reclaim-task', 'terminate-run', 'change-task-state'] as const),
          entity,
          observedState: {
            animation: state,
            authority: 'authoritative' as const,
            destination: 'project' as const,
            observedAt,
            source: source.source,
            value: state
          },
          ownerCandidates: [owner],
          readbackCapabilities:
            identity.kind === 'kanban' ? (['kanban-task', 'kanban-run'] as const) : (['session'] as const),
          source,
          sourceOwner: owner
        }
      ]
    ])
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  $sessionStates.set({})
  $sessionTiles.set([])
})

describe('createLunarCityCommandExecutors', () => {
  it('sends session guidance only through the immutable exact owner route', async () => {
    requestForSessionProfile.mockResolvedValue({ status: 'queued' })
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'same' } as const
    const snapshot = planning(identity)
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'send-guidance', text: 'Check.' }, snapshot)

    await createLunarCityCommandExecutors({
      resolveLiveRuntime: (owner, storedId) =>
        owner.connectionId === 'source-b' && storedId === 'same' ? 'runtime-b' : undefined
    }).session.send(plan)

    expect(requestForSessionProfile).toHaveBeenCalledOnce()
    expect(requestForSessionProfile.mock.calls[0]?.[0]).toEqual({ connectionId: 'source-b', profile: 'builder' })
    expect(requestForSessionProfile.mock.calls[0]?.[2]).toBe('prompt.submit')
    expect(requestForSessionProfile.mock.calls[0]?.[3]).toEqual({ session_id: 'runtime-b', text: 'Check.' })
  })

  it.each([
    ['send-guidance', false, 'prompt.submit'],
    ['interrupt-session', true, 'session.interrupt']
  ] as const)(
    'proves exact current authority before %s reaches the production route',
    async (kind, confirmed, method) => {
      requestForSessionProfile
        .mockResolvedValueOnce({ sessions: [{ id: 'same' }] })
        .mockResolvedValueOnce({ status: 'queued' })
      const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'same' } as const
      const snapshot = planning(identity)

      const plan = planCommand(
        kind === 'send-guidance'
          ? { entityKey: entityKey(identity), kind, text: 'Check.' }
          : { entityKey: entityKey(identity), kind },
        {
          ...snapshot,
          targets: new Map([
            [
              entityKey(identity),
              {
                ...snapshot.targets.get(entityKey(identity))!,
                availableOperations: [kind]
              }
            ]
          ])
        }
      )

      const executors = createLunarCityCommandExecutors({ resolveLiveRuntime: () => 'runtime-b' })

      const receipt = await executeCommand(plan, executors, {
        confirmed,
        latestSnapshot: () => ({
          ...snapshot,
          targets: new Map([
            [
              entityKey(identity),
              {
                ...snapshot.targets.get(entityKey(identity))!,
                availableOperations: [kind]
              }
            ]
          ])
        })
      })

      expect(receipt.verification).toBe('verification_required')
      expect(requestForSessionProfile.mock.calls.map(call => call[2])).toEqual(['session.list', method])
      expect(requestForSessionProfile.mock.calls[1]?.[3]).toMatchObject({ session_id: 'runtime-b' })
    }
  )

  it('does not send when the exact authority list is stale, foreign, or unreachable', async () => {
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'same' } as const
    const snapshot = planning(identity)
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'send-guidance', text: 'Check.' }, snapshot)

    for (const response of [{ sessions: [{ id: 'foreign' }] }, new Error('unreachable')]) {
      requestForSessionProfile.mockReset()

      if (response instanceof Error) {
        requestForSessionProfile.mockRejectedValueOnce(response)
      } else {
        requestForSessionProfile.mockResolvedValueOnce(response)
      }

      const receipt = await executeCommand(
        plan,
        createLunarCityCommandExecutors({ resolveLiveRuntime: () => 'runtime-b' }),
        { latestSnapshot: () => snapshot }
      )

      expect(receipt.verification).toBe('rejected')
      expect(requestForSessionProfile).toHaveBeenCalledOnce()
      expect(requestForSessionProfile.mock.calls[0]?.[2]).toBe('session.list')
    }
  })

  it.each([
    ['send-guidance', false, 'subagent.steer'],
    ['interrupt-subagent', true, 'subagent.interrupt']
  ] as const)('proves the exact live child before %s reaches the production route', async (kind, confirmed, method) => {
    requestForSessionProfile
      .mockResolvedValueOnce({ sessions: [{ id: 'parent-session' }] })
      .mockResolvedValueOnce({
        found: true,
        subagent: { generation: 'generation-1', status: 'running', subagent_id: 'child-same' }
      })
      .mockResolvedValueOnce({ found: true, status: 'queued', subagent_id: 'child-same' })

    const identity = {
      connectionId: 'source-b',
      kind: 'subagent',
      profile: 'builder',
      sessionId: 'parent-session',
      subagentId: 'child-same'
    } as const

    const base = planning(identity)

    const snapshot: CommandPlanningSnapshot = {
      ...base,
      targets: new Map([
        [
          entityKey(identity),
          {
            ...base.targets.get(entityKey(identity))!,
            availableOperations: [kind],
            readbackCapabilities: ['subagent']
          }
        ]
      ])
    }

    const plan = planCommand(
      kind === 'send-guidance'
        ? { entityKey: entityKey(identity), kind, text: 'Check the exact child.' }
        : { entityKey: entityKey(identity), kind },
      snapshot
    )

    const receipt = await executeCommand(
      plan,
      createLunarCityCommandExecutors({ resolveLiveRuntime: () => 'parent-runtime-b' }),
      { confirmed, latestSnapshot: () => snapshot }
    )

    expect(receipt.verification).toBe('verification_required')
    expect(requestForSessionProfile.mock.calls.map(call => call[2])).toEqual([
      'session.list',
      'subagent.status',
      method
    ])
    expect(requestForSessionProfile.mock.calls[1]?.[0]).toEqual({ connectionId: 'source-b', profile: 'builder' })
    expect(requestForSessionProfile.mock.calls[1]?.[3]).toEqual({
      session_id: 'parent-runtime-b',
      subagent_id: 'child-same'
    })
    expect(requestForSessionProfile.mock.calls[2]?.[3]).toMatchObject({
      expected_generation: 'generation-1',
      session_id: 'parent-runtime-b',
      subagent_id: 'child-same'
    })
  })

  it.each([
    ['missing child', { found: false, subagent: null }],
    ['foreign child', { found: true, subagent: { status: 'running', subagent_id: 'child-on-other-connection' } }],
    ['stale child', { found: true, subagent: { status: 'completed', subagent_id: 'child-same' } }]
  ])('rejects a %s before any production subagent mutation is sent', async (_label, childStatus) => {
    requestForSessionProfile
      .mockResolvedValueOnce({ sessions: [{ id: 'parent-session' }] })
      .mockResolvedValueOnce(childStatus)

    const identity = {
      connectionId: 'source-b',
      kind: 'subagent',
      profile: 'builder',
      sessionId: 'parent-session',
      subagentId: 'child-same'
    } as const

    const base = planning(identity)

    const snapshot: CommandPlanningSnapshot = {
      ...base,
      targets: new Map([
        [
          entityKey(identity),
          {
            ...base.targets.get(entityKey(identity))!,
            availableOperations: ['send-guidance'],
            readbackCapabilities: ['subagent']
          }
        ]
      ])
    }

    const plan = planCommand(
      { entityKey: entityKey(identity), kind: 'send-guidance', text: 'Do not misroute.' },
      snapshot
    )

    const receipt = await executeCommand(
      plan,
      createLunarCityCommandExecutors({ resolveLiveRuntime: () => 'parent-runtime-b' }),
      { latestSnapshot: () => snapshot }
    )

    expect(receipt.verification).toBe('rejected')
    expect(requestForSessionProfile.mock.calls.map(call => call[2])).toEqual(['session.list', 'subagent.status'])
  })

  it('rejects an unreachable exact child authority source before send', async () => {
    requestForSessionProfile
      .mockResolvedValueOnce({ sessions: [{ id: 'parent-session' }] })
      .mockRejectedValueOnce(new Error('child authority unreachable'))

    const identity = {
      connectionId: 'source-b',
      kind: 'subagent',
      profile: 'builder',
      sessionId: 'parent-session',
      subagentId: 'child-same'
    } as const

    const base = planning(identity)

    const snapshot: CommandPlanningSnapshot = {
      ...base,
      targets: new Map([
        [
          entityKey(identity),
          {
            ...base.targets.get(entityKey(identity))!,
            availableOperations: ['send-guidance'],
            readbackCapabilities: ['subagent']
          }
        ]
      ])
    }

    const plan = planCommand({ entityKey: entityKey(identity), kind: 'send-guidance', text: 'Do not send.' }, snapshot)

    const receipt = await executeCommand(
      plan,
      createLunarCityCommandExecutors({ resolveLiveRuntime: () => 'parent-runtime-b' }),
      { latestSnapshot: () => snapshot }
    )

    expect(receipt.verification).toBe('rejected')
    expect(requestForSessionProfile.mock.calls.map(call => call[2])).toEqual(['session.list', 'subagent.status'])
  })

  it('does not let a duplicate child id on another connection satisfy exact-owner authority', async () => {
    requestForSessionProfile.mockImplementation(async (owner, _ambient, method) => {
      if (method === 'session.list') {
        return { sessions: [{ id: 'parent-session' }] }
      }

      if (method === 'subagent.status' && owner.connectionId === 'source-a') {
        return { found: true, subagent: { status: 'running', subagent_id: 'child-same' } }
      }

      return { found: false, subagent: null }
    })

    const identity = {
      connectionId: 'source-b',
      kind: 'subagent',
      profile: 'builder',
      sessionId: 'parent-session',
      subagentId: 'child-same'
    } as const

    const base = planning(identity)

    const snapshot: CommandPlanningSnapshot = {
      ...base,
      targets: new Map([
        [
          entityKey(identity),
          {
            ...base.targets.get(entityKey(identity))!,
            availableOperations: ['send-guidance'],
            readbackCapabilities: ['subagent']
          }
        ]
      ])
    }

    const plan = planCommand(
      { entityKey: entityKey(identity), kind: 'send-guidance', text: 'Stay isolated.' },
      snapshot
    )

    const receipt = await executeCommand(
      plan,
      createLunarCityCommandExecutors({ resolveLiveRuntime: () => 'parent-runtime-b' }),
      { latestSnapshot: () => snapshot }
    )

    expect(receipt.verification).toBe('rejected')
    expect(requestForSessionProfile).toHaveBeenCalledTimes(2)
    expect(requestForSessionProfile.mock.calls.map(call => call[0])).toEqual([
      { connectionId: 'source-b', profile: 'builder' },
      { connectionId: 'source-b', profile: 'builder' }
    ])
  })

  it('does not send a stored DB id when no exact-owner live runtime is authoritative', async () => {
    const identity = {
      connectionId: 'source-b',
      kind: 'session',
      profile: 'builder',
      sessionId: 'stored-only'
    } as const

    const snapshot = planning(identity)
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'send-guidance', text: 'Check.' }, snapshot)

    await expect(
      createLunarCityCommandExecutors({ resolveLiveRuntime: () => undefined }).session.send(plan)
    ).rejects.toThrow(/live runtime.*unavailable/i)
    expect(requestForSessionProfile).not.toHaveBeenCalled()
  })

  it('production-default routing resolves duplicate stored ids by exact owner-tagged runtime provenance', async () => {
    requestForSessionProfile.mockResolvedValue({ status: 'queued' })
    $sessionStates.set({
      'runtime-a': runtimeState('same', { connectionId: 'source-a', profile: 'builder' }),
      'runtime-b': runtimeState('same', { connectionId: 'source-b', profile: 'builder' })
    })
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'same' } as const

    const plan = planCommand(
      { entityKey: entityKey(identity), kind: 'send-guidance', text: 'Exact.' },
      planning(identity)
    )

    await createLunarCityCommandExecutors().session.send(plan)

    expect(requestForSessionProfile.mock.calls[0]?.[3]).toEqual({ session_id: 'runtime-b', text: 'Exact.' })
  })

  it('production-default routing rejects a lone foreign runtime despite an owner hint for the stored id', async () => {
    $sessionStates.set({
      'runtime-a': runtimeState('same', { connectionId: 'source-a', profile: 'builder' })
    })
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'source-b', profile: 'builder' },
        storedSessionId: 'same'
      }
    ])
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'same' } as const

    const plan = planCommand(
      { entityKey: entityKey(identity), kind: 'send-guidance', text: 'Exact.' },
      planning(identity)
    )

    await expect(createLunarCityCommandExecutors().session.send(plan)).rejects.toThrow(/live runtime.*unavailable/i)
    expect(requestForSessionProfile).not.toHaveBeenCalled()
  })

  it('production-default routing accepts an exact owner-tagged live tile binding', async () => {
    requestForSessionProfile.mockResolvedValue({ status: 'queued' })
    $sessionStates.set({
      'runtime-b': { storedSessionId: 'same' } as ClientSessionState
    })
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'source-b', profile: 'builder' },
        runtimeId: 'runtime-b',
        storedSessionId: 'same'
      }
    ])
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'same' } as const

    const plan = planCommand(
      { entityKey: entityKey(identity), kind: 'send-guidance', text: 'Exact.' },
      planning(identity)
    )

    await createLunarCityCommandExecutors().session.send(plan)

    expect(requestForSessionProfile.mock.calls[0]?.[3]).toEqual({ session_id: 'runtime-b', text: 'Exact.' })
  })

  it('uses only exact scoped Kanban endpoints and verifies from authoritative task readback', async () => {
    pluginRest
      .mockResolvedValueOnce({ ok: true, task_id: 'T-9' })
      .mockResolvedValueOnce({ task: { current_run_id: null, id: 'T-9', status: 'ready' } })

    const identity = {
      board: 'alpha/beta',
      connectionId: 'source-b',
      kind: 'kanban',
      profile: 'builder',
      runId: '7',
      taskId: 'T-9'
    } as const

    const snapshot = planning(identity, 'running')
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'reclaim-task' }, snapshot)
    const executor = createLunarCityCommandExecutors().kanbanTask

    await executor.send(plan)
    const readback = await executor.readback(plan)

    expect(pluginRest.mock.calls).toEqual([
      [
        'kanban',
        '/tasks/T-9/reclaim?board=alpha%2Fbeta',
        {
          body: { reason: 'Operator confirmed in Lunar City' },
          method: 'POST',
          scope: { connectionId: 'source-b', profile: 'builder' }
        }
      ],
      [
        'kanban',
        '/tasks/T-9?board=alpha%2Fbeta',
        { method: 'GET', scope: { connectionId: 'source-b', profile: 'builder' } }
      ]
    ])
    expect(readback).toMatchObject({ effect: plan.readback.expectedEffect, outcome: 'verified' })
  })

  it('does not retry an ambiguous Kanban mutation failure', async () => {
    pluginRest.mockRejectedValue(new Error('request timed out'))

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-2'
    } as const

    const snapshot = planning(identity, 'running')
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'reclaim-task' }, snapshot)

    await expect(createLunarCityCommandExecutors().kanbanTask.send(plan)).rejects.toThrow('request timed out')
    expect(pluginRest).toHaveBeenCalledOnce()
  })

  it('reads Kanban evidence through the exact task endpoint without issuing a write', async () => {
    pluginRest.mockResolvedValue({ task: { id: 'T-4', status: 'blocked' } })

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-4'
    } as const

    const snapshot = planning(identity, 'blocked')

    const plan = planCommand(
      { entityKey: entityKey(identity), evidence: 'diagnostics', kind: 'inspect-evidence' },
      snapshot
    )

    const executor = createLunarCityCommandExecutors().kanbanTask

    await executor.send(plan)
    const readback = await executor.readback(plan)

    expect(pluginRest).toHaveBeenCalledOnce()
    expect(pluginRest).toHaveBeenCalledWith('kanban', '/tasks/T-4?board=main', {
      method: 'GET',
      scope: { connectionId: 'source-a', profile: 'default' }
    })
    expect(readback).toBeNull()
  })

  it('does not verify or replace malformed exact-source evidence', async () => {
    pluginRest.mockResolvedValue({ task: { diagnostics: ['wrong'], id: 'T-other', status: 'blocked' } })

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-4'
    } as const

    const snapshot = planning(identity, 'blocked')

    const plan = planCommand(
      { entityKey: entityKey(identity), evidence: 'diagnostics', kind: 'inspect-evidence' },
      snapshot
    )

    const executor = createLunarCityCommandExecutors().kanbanTask

    await expect(executor.send(plan)).rejects.toThrow(/exact-task evidence.*unavailable/i)
    await expect(executor.readback(plan)).resolves.toBeNull()
  })

  it('validates the bounded top-level worker-log contract for the exact task', async () => {
    pluginRest.mockResolvedValue({
      content: 'exact worker tail',
      exists: true,
      path: '/tmp/T-4.log',
      size_bytes: 17,
      task_id: 'T-4',
      truncated: false
    })

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-4'
    } as const

    const plan = planCommand(
      { entityKey: entityKey(identity), evidence: 'log', kind: 'inspect-evidence' },
      planning(identity, 'blocked')
    )

    await expect(createLunarCityCommandExecutors().kanbanTask.send(plan)).resolves.toMatchObject({
      content: 'exact worker tail',
      task_id: 'T-4'
    })
    expect(pluginRest).toHaveBeenCalledWith('kanban', '/tasks/T-4/log?board=main&tail=65536', {
      method: 'GET',
      scope: { connectionId: 'source-a', profile: 'default' }
    })
  })

  it.each([
    ['wrong task', { content: 'tail', exists: true, size_bytes: 4, task_id: 'T-other', truncated: false }],
    [
      'oversized tail',
      { content: 'x'.repeat(65_537), exists: true, size_bytes: 65_537, task_id: 'T-4', truncated: true }
    ],
    ['untyped tail', { content: 9, exists: true, size_bytes: 1, task_id: 'T-4', truncated: false }]
  ])('rejects malformed worker-log evidence with %s', async (_label, response) => {
    pluginRest.mockResolvedValue(response)

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-4'
    } as const

    const plan = planCommand(
      { entityKey: entityKey(identity), evidence: 'log', kind: 'inspect-evidence' },
      planning(identity, 'blocked')
    )

    await expect(createLunarCityCommandExecutors().kanbanTask.send(plan)).rejects.toThrow(/worker-log evidence/i)
  })

  it('classifies an authoritative Kanban conflict as rejected instead of ambiguous', async () => {
    pluginRest.mockRejectedValue({ message: 'already ended', status: 409 })

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-5'
    } as const

    const snapshot = planning(identity, 'work')
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'reclaim-task' }, snapshot)

    await expect(createLunarCityCommandExecutors().kanbanTask.send(plan)).rejects.toBeInstanceOf(CommandRejectedError)
    expect(pluginRest).toHaveBeenCalledOnce()
  })

  it.each([
    ['empty payload', {}],
    ['missing task id', { task: { current_run_id: null, status: 'ready' } }],
    ['mismatched task id', { task: { current_run_id: null, id: 'T-other', status: 'ready' } }],
    ['missing terminal run field', { task: { id: 'T-6', status: 'ready' } }],
    ['wrong terminal status type', { task: { current_run_id: null, id: 'T-6', status: 7 } }]
  ])('does not verify reclaim from %s', async (_label, malformed) => {
    pluginRest.mockResolvedValueOnce({ ok: true, task_id: 'T-6' }).mockResolvedValueOnce(malformed)

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      runId: '8',
      taskId: 'T-6'
    } as const

    const snapshot = planning(identity, 'work')
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'reclaim-task' }, snapshot)
    const executor = createLunarCityCommandExecutors().kanbanTask

    await executor.send(plan)
    const readback = await executor.readback(plan)

    expect(readback).not.toMatchObject({ effect: plan.readback.expectedEffect, outcome: 'verified' })
  })

  it('does not verify reclaim from an arbitrary non-running status string', async () => {
    pluginRest
      .mockResolvedValueOnce({ ok: true, task_id: 'T-6' })
      .mockResolvedValueOnce({ task: { current_run_id: null, id: 'T-6', status: 'definitely-finished' } })

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      runId: '8',
      taskId: 'T-6'
    } as const

    const plan = planCommand({ entityKey: entityKey(identity), kind: 'reclaim-task' }, planning(identity, 'work'))
    const executor = createLunarCityCommandExecutors().kanbanTask

    await executor.send(plan)
    await expect(executor.readback(plan)).resolves.not.toMatchObject({ outcome: 'verified' })
  })

  it.each([
    ['mismatched run id', { run: { ended_at: 200, id: 99, outcome: 'reclaimed', task_id: 'T-7' } }],
    ['mismatched task id', { run: { ended_at: 200, id: 9, outcome: 'reclaimed', task_id: 'T-other' } }],
    ['missing terminal timestamp', { run: { id: 9, outcome: 'reclaimed', task_id: 'T-7' } }],
    ['wrong terminal outcome', { run: { ended_at: 200, id: 9, outcome: 'running', task_id: 'T-7' } }]
  ])('does not verify terminate-run from %s', async (_label, malformed) => {
    pluginRest.mockResolvedValueOnce({ ok: true, run_id: 9 }).mockResolvedValueOnce(malformed)

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban',
      profile: 'default',
      runId: '9',
      taskId: 'T-7'
    } as const

    const snapshot = planning(identity, 'work')
    const plan = planCommand({ entityKey: entityKey(identity), kind: 'terminate-run' }, snapshot)
    const executor = createLunarCityCommandExecutors().kanbanRun

    await executor.send(plan)
    const readback = await executor.readback(plan)

    expect(readback).not.toMatchObject({ effect: plan.readback.expectedEffect, outcome: 'verified' })
  })

  it('does not manufacture verified session evidence without an authoritative evidence seam', async () => {
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'stored-8' } as const
    const snapshot = planning(identity)

    const plan = planCommand(
      { entityKey: entityKey(identity), evidence: 'diagnostics', kind: 'inspect-evidence' },
      {
        ...snapshot,
        targets: new Map([
          [
            entityKey(identity),
            { ...snapshot.targets.get(entityKey(identity))!, availableOperations: ['inspect-evidence'] }
          ]
        ])
      }
    )

    const executor = createLunarCityCommandExecutors().session

    await expect(executor.send(plan)).rejects.toThrow(/authoritative session evidence.*unavailable/i)
    await expect(executor.readback(plan)).resolves.toBeNull()
  })
})
