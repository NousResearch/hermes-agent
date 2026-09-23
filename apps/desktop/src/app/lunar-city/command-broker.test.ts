import { describe, expect, it, vi } from 'vitest'

import {
  type CommandEffect,
  type CommandExecutor,
  type CommandExecutors,
  type CommandIntent,
  type CommandOperation,
  type CommandPlan,
  type CommandPlanningSnapshot,
  type CommandReadback,
  CommandRejectedError,
  type CommandTargetState,
  CommandTimeoutError,
  executeCommand,
  planCommand,
  revalidateCommandPlan
} from './command-broker'
import { entityKey } from './identity'
import type { EntityIdentity, LunarCitySnapshot, LunarEntity, SourceHealth } from './model'

type IdentityKind = EntityIdentity['kind']

const OWNER = { connectionId: 'connection-a', profile: 'worker' }

const IDENTITIES: Readonly<Record<IdentityKind, EntityIdentity>> = {
  profile: { ...OWNER, kind: 'profile' },
  session: { ...OWNER, kind: 'session', sessionId: 'session-1' },
  subagent: { ...OWNER, kind: 'subagent', sessionId: 'session-1', subagentId: 'child-1' },
  kanban: {
    ...OWNER,
    board: 'primary',
    kind: 'kanban',
    runId: 'run-7',
    taskId: 'task-7',
    workerId: 'worker-7'
  }
}

const ALL_OPERATIONS: readonly CommandOperation[] = [
  'open-session',
  'inspect-evidence',
  'send-guidance',
  'interrupt-session',
  'interrupt-subagent',
  'terminate-run',
  'retry-task',
  'reclaim-task',
  'reassign-task',
  'dispatch-task',
  'change-task-state'
]

interface ValidCompatibility {
  identityKind: IdentityKind
  method: string
  operation: CommandOperation
  readback: 'kanban-run' | 'kanban-task' | 'session' | 'subagent'
}

const VALID_COMPATIBILITY: readonly ValidCompatibility[] = [
  { identityKind: 'session', method: 'session.open', operation: 'open-session', readback: 'session' },
  { identityKind: 'subagent', method: 'session.open', operation: 'open-session', readback: 'session' },
  { identityKind: 'session', method: 'evidence.inspect', operation: 'inspect-evidence', readback: 'session' },
  { identityKind: 'subagent', method: 'evidence.inspect', operation: 'inspect-evidence', readback: 'subagent' },
  { identityKind: 'kanban', method: 'evidence.inspect', operation: 'inspect-evidence', readback: 'kanban-task' },
  { identityKind: 'session', method: 'session.steer', operation: 'send-guidance', readback: 'session' },
  { identityKind: 'subagent', method: 'subagent.steer', operation: 'send-guidance', readback: 'subagent' },
  { identityKind: 'session', method: 'session.interrupt', operation: 'interrupt-session', readback: 'session' },
  {
    identityKind: 'subagent',
    method: 'subagent.interrupt',
    operation: 'interrupt-subagent',
    readback: 'subagent'
  },
  { identityKind: 'kanban', method: 'kanban.run.terminate', operation: 'terminate-run', readback: 'kanban-run' },
  { identityKind: 'kanban', method: 'kanban.task.retry', operation: 'retry-task', readback: 'kanban-task' },
  { identityKind: 'kanban', method: 'kanban.task.reclaim', operation: 'reclaim-task', readback: 'kanban-task' },
  { identityKind: 'kanban', method: 'kanban.task.reassign', operation: 'reassign-task', readback: 'kanban-task' },
  { identityKind: 'kanban', method: 'kanban.task.dispatch', operation: 'dispatch-task', readback: 'kanban-task' },
  {
    identityKind: 'kanban',
    method: 'kanban.task.patch',
    operation: 'change-task-state',
    readback: 'kanban-task'
  }
]

const EXPECTED_EFFECTS: Readonly<Record<CommandOperation, CommandEffect>> = {
  'open-session': { kind: 'session-present', targetId: 'session-1' },
  'inspect-evidence': { kind: 'evidence-present', targetId: 'session-1', value: 'diagnostics' },
  'send-guidance': { kind: 'guidance-recorded', targetId: 'session-1' },
  'interrupt-session': { kind: 'session-interrupted', targetId: 'session-1' },
  'interrupt-subagent': { kind: 'subagent-interrupted', targetId: 'child-1' },
  'terminate-run': { kind: 'run-terminated', targetId: 'run-7' },
  'retry-task': { kind: 'task-retried', targetId: 'task-7' },
  'reclaim-task': { kind: 'task-reclaimed', targetId: 'task-7' },
  'reassign-task': { kind: 'task-reassigned', targetId: 'task-7', value: 'reviewer' },
  'dispatch-task': { kind: 'task-dispatched', targetId: 'task-7' },
  'change-task-state': { kind: 'task-state-changed', targetId: 'task-7', value: 'review' }
}

interface SnapshotOptions {
  authority?: LunarEntity['authority']
  revision?: number
  sourceOwner?: { connectionId: string; profile: string }
  sources?: readonly SourceHealth[]
  state?: string
  stateAnimation?: string
  stateSource?: string
  targetEntity?: LunarEntity
  time?: number
}

function makeEntity(
  identity: EntityIdentity,
  time: number,
  authority: LunarEntity['authority'] = 'authoritative'
): LunarEntity {
  return {
    animation: 'work',
    authority,
    destination: 'project',
    identity,
    key: entityKey(identity),
    observedAt: time,
    position: { x: 1, y: 0, z: 2 },
    projectId: 'project-1',
    variant: 'blue'
  }
}

function snapshot(identity: EntityIdentity, options: SnapshotOptions = {}): CommandPlanningSnapshot {
  const time = options.time ?? 2_000
  const revision = options.revision ?? 3
  const authority = options.authority ?? 'authoritative'
  const cityEntity = makeEntity(identity, time, authority)

  const source: SourceHealth = {
    authority,
    observedAt: time,
    source: identity.kind === 'kanban' ? 'connection-a/kanban/primary' : 'connection-a/sessions'
  }

  const targetEntity = options.targetEntity ?? cityEntity

  const target: CommandTargetState = {
    availableOperations: ALL_OPERATIONS,
    canonicalProjectId: 'project-1',
    entity: targetEntity,
    observedState: {
      animation: options.stateAnimation ?? targetEntity.animation,
      authority: targetEntity.authority,
      destination: targetEntity.destination,
      observedAt: targetEntity.observedAt,
      source: options.stateSource ?? source.source,
      value: options.state ?? 'running'
    },
    ownerCandidates: [OWNER],
    readbackCapabilities: ['kanban-run', 'kanban-task', 'session', 'subagent'],
    repositoryId: 'repo/hermes',
    source,
    sourceOwner: options.sourceOwner ?? OWNER
  }

  const city: LunarCitySnapshot = {
    entities: new Map([[cityEntity.key, cityEntity]]),
    observedAt: time,
    revision,
    sources: options.sources ?? [source]
  }

  return { city, targets: new Map([[targetEntity.key, target]]) }
}

function replaceTarget(
  value: CommandPlanningSnapshot,
  update: (target: CommandTargetState) => CommandTargetState
): CommandPlanningSnapshot {
  const key = [...value.targets.keys()][0]
  const target = value.targets.get(key)!

  return { ...value, targets: new Map([[key, update(target)]]) }
}

function intent(operation: CommandOperation, identity: EntityIdentity): CommandIntent {
  const key = entityKey(identity)

  switch (operation) {
    case 'send-guidance':
      return { entityKey: key, kind: operation, text: 'Please inspect the failing test.' }

    case 'reassign-task':
      return { assignee: 'reviewer', entityKey: key, kind: operation }

    case 'change-task-state':
      return { entityKey: key, kind: operation, state: 'review' }

    case 'inspect-evidence':
      return { entityKey: key, evidence: 'diagnostics', kind: operation }

    default:
      return { entityKey: key, kind: operation }
  }
}

function matchingRow(operation: CommandOperation, identityKind: IdentityKind): ValidCompatibility | undefined {
  return VALID_COMPATIBILITY.find(row => row.operation === operation && row.identityKind === identityKind)
}

function bridge(readback: CommandReadback | null) {
  const currentAuthority: NonNullable<CommandExecutor['currentAuthority']> = async plan => ({
    authority: 'authoritative',
    identity: plan.identity,
    observedAt: plan.plannedAt,
    owner: plan.owner
  })

  return {
    currentAuthority: vi.fn(currentAuthority),
    readback: vi.fn().mockResolvedValue(readback),
    send: vi.fn().mockResolvedValue({ accepted: true })
  }
}

function executors(selected: ReturnType<typeof bridge>, kind: CommandPlan['readback']['kind']): CommandExecutors {
  const unused = bridge(null)

  return {
    kanbanRun: kind === 'kanban-run' ? selected : unused,
    kanbanTask: kind === 'kanban-task' ? selected : unused,
    session: kind === 'session' ? selected : unused,
    subagent: kind === 'subagent' ? selected : unused
  }
}

function readback(plan: CommandPlan, overrides: Partial<CommandReadback> = {}): CommandReadback {
  return {
    authority: 'authoritative',
    effect: plan.readback.expectedEffect,
    identity: plan.identity,
    observedAt: plan.plannedAt + 1,
    operation: plan.operation,
    outcome: 'verified',
    owner: plan.owner,
    revision: plan.plannedRevision + 1,
    ...overrides
  }
}

async function run(
  plan: CommandPlan,
  planningSnapshot: CommandPlanningSnapshot,
  result: CommandReadback | null,
  confirmed = true
) {
  const selected = bridge(result)

  const receipt = await executeCommand(plan, executors(selected, plan.readback.kind), {
    confirmed,
    latestSnapshot: () => planningSnapshot
  })

  return { receipt, selected }
}

describe('LunarCityCommandBroker immutable target binding', () => {
  it.each([
    ['authority', (entity: LunarEntity) => ({ ...entity, authority: 'stale' as const })],
    ['observedAt', (entity: LunarEntity) => ({ ...entity, observedAt: entity.observedAt - 1 })],
    ['animation', (entity: LunarEntity) => ({ ...entity, animation: 'idle' })],
    ['destination', (entity: LunarEntity) => ({ ...entity, destination: 'garden' as const })],
    ['position', (entity: LunarEntity) => ({ ...entity, position: { x: 99, y: 0, z: 2 } })],
    ['project', (entity: LunarEntity) => ({ ...entity, projectId: 'foreign-project' })]
  ])('rejects target entity %s that differs from the immutable city entity', (_field, change) => {
    const identity = IDENTITIES.session
    const value = snapshot(identity)
    const cityEntity = value.city.entities.get(entityKey(identity))!
    const mismatched = replaceTarget(value, target => ({ ...target, entity: change(cityEntity) }))

    expect(() => planCommand(intent('interrupt-session', identity), mismatched)).toThrow(/target-binding-mismatch/)
  })

  it.each([
    ['source owner', { sourceOwner: { connectionId: 'connection-b', profile: 'worker' } }],
    ['state source', { stateSource: 'connection-b/sessions' }],
    ['state animation', { stateAnimation: 'idle' }]
  ])('rejects a mismatched %s binding', (_label, options) => {
    const identity = IDENTITIES.session

    expect(() => planCommand(intent('interrupt-session', identity), snapshot(identity, options))).toThrow(
      /target-binding-mismatch|owner-is-ambiguous/
    )
  })

  it('revalidates the exact latest target at confirmation and immediately before send', async () => {
    const identity = IDENTITIES.session
    const initial = snapshot(identity)
    const plan = planCommand(intent('interrupt-session', identity), initial)
    const changed = snapshot(identity, { revision: 4, state: 'completed', time: 2_100 })

    const valid = revalidateCommandPlan(plan, initial)
    const invalid = revalidateCommandPlan(plan, changed)

    expect(valid.ok).toBe(true)
    expect(invalid.ok).toBe(false)

    if (invalid.ok) {
      throw new Error('Expected changed target revalidation to fail')
    }

    expect(invalid.error).toMatch(/target-changed-since-plan/)

    const selected = bridge(readback(plan))

    const receipt = await executeCommand(plan, executors(selected, plan.readback.kind), {
      confirmed: true,
      latestSnapshot: () => changed
    })

    expect(receipt.verification).toBe('rejected')
    expect(receipt.error).toContain('target-changed-since-plan')
    expect(selected.send).not.toHaveBeenCalled()
    expect(selected.readback).not.toHaveBeenCalled()
  })

  it.each(['deleted', 'ambiguous'] as const)('blocks a %s target before send', async kind => {
    const identity = IDENTITIES.session
    const initial = snapshot(identity)
    const plan = planCommand(intent('interrupt-session', identity), initial)

    const latest =
      kind === 'deleted'
        ? { ...initial, city: { ...initial.city, entities: new Map() }, targets: new Map() }
        : replaceTarget(initial, target => ({
            ...target,
            ownerCandidates: [OWNER, { connectionId: 'connection-b', profile: 'worker' }]
          }))

    const selected = bridge(readback(plan))

    const receipt = await executeCommand(plan, executors(selected, plan.readback.kind), {
      confirmed: true,
      latestSnapshot: () => latest
    })

    expect(receipt.verification).toBe('rejected')
    expect(selected.send).not.toHaveBeenCalled()
  })
})

describe('LunarCityCommandBroker compatibility matrix', () => {
  it.each(
    ALL_OPERATIONS.flatMap(operation =>
      (Object.keys(IDENTITIES) as IdentityKind[]).map(identityKind => ({ identityKind, operation }))
    )
  )('enforces $operation × $identityKind regardless of advertised capabilities', ({ identityKind, operation }) => {
    const identity = IDENTITIES[identityKind]
    const expected = matchingRow(operation, identityKind)
    const act = () => planCommand(intent(operation, identity), snapshot(identity))

    if (!expected) {
      expect(act).toThrow(/operation-identity-incompatible/)

      return
    }

    const plan = act()

    expect(plan.method).toBe(expected.method)
    expect(plan.readback.kind).toBe(expected.readback)
  })

  it('uses prompt.submit only for an idle standard session and never for a subagent', () => {
    const sessionPlan = planCommand(
      intent('send-guidance', IDENTITIES.session),
      snapshot(IDENTITIES.session, { state: 'idle' })
    )

    const subagentPlan = planCommand(
      intent('send-guidance', IDENTITIES.subagent),
      snapshot(IDENTITIES.subagent, { state: 'idle' })
    )

    expect(sessionPlan.method).toBe('prompt.submit')
    expect(subagentPlan.method).toBe('subagent.steer')
  })

  it.each(ALL_OPERATIONS)('rejects a forged %s method/readback combination before send', async operation => {
    const valid = VALID_COMPATIBILITY.find(row => row.operation === operation)!
    const identity = IDENTITIES[valid.identityKind]
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent(operation, identity), planningSnapshot)

    const forged = {
      ...plan,
      method: plan.method === 'session.interrupt' ? 'prompt.submit' : 'session.interrupt',
      readback: { ...plan.readback, kind: plan.readback.kind === 'session' ? 'kanban-task' : 'session' }
    } as CommandPlan

    const selected = bridge(readback(plan))

    const receipt = await executeCommand(forged, executors(selected, plan.readback.kind), {
      confirmed: true,
      latestSnapshot: () => planningSnapshot
    })

    expect(receipt.verification).toBe('rejected')
    expect(receipt.error).toContain('Invalid command plan')
    expect(selected.send).not.toHaveBeenCalled()
  })
})

describe('LunarCityCommandBroker canonical plan integrity', () => {
  it('returns a newly rebuilt deeply frozen canonical plan', () => {
    const identity = IDENTITIES.kanban
    const planningSnapshot = snapshot(identity)
    const callerPlan = structuredClone(planCommand(intent('reassign-task', identity), planningSnapshot)) as CommandPlan
    const result = revalidateCommandPlan(callerPlan, planningSnapshot)

    expect(result.ok).toBe(true)

    if (!result.ok) {
      throw new Error(result.error)
    }

    expect(result.canonicalPlan).toEqual(callerPlan)
    expect(result.canonicalPlan).not.toBe(callerPlan)
    expect(Object.isFrozen(result.canonicalPlan)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.identity)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.intent)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.owner)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.params)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.context)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.context.source)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.context.sourceOwner)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.readback)).toBe(true)
    expect(Object.isFrozen(result.canonicalPlan.readback.expectedEffect)).toBe(true)
  })

  it('uses only the rebuilt canonical plan after revalidation while the caller mutates its aliases', async () => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const original = planCommand(intent('interrupt-session', identity), planningSnapshot)
    const callerPlan = structuredClone(original) as CommandPlan
    let releaseSend!: () => void
    let markSendStarted!: () => void

    const sendStarted = new Promise<void>(resolve => {
      markSendStarted = resolve
    })

    const sendGate = new Promise<void>(resolve => {
      releaseSend = resolve
    })

    const selected = {
      currentAuthority: vi.fn(async (canonicalPlan: CommandPlan) => ({
        authority: 'authoritative' as const,
        identity: canonicalPlan.identity,
        observedAt: canonicalPlan.plannedAt,
        owner: canonicalPlan.owner
      })),
      readback: vi.fn(async (canonicalPlan: CommandPlan) => readback(canonicalPlan)),
      send: vi.fn(async (_canonicalPlan: CommandPlan) => {
        markSendStarted()
        await sendGate

        return { accepted: true }
      })
    }

    const execution = executeCommand(callerPlan, executors(selected, original.readback.kind), {
      confirmed: true,
      latestSnapshot: () => planningSnapshot
    })

    await sendStarted

    const mutable = callerPlan as unknown as {
      identity: { sessionId: string }
      method: string
      params: Record<string, unknown>
      readback: { expectedEffect: { targetId: string }; kind: string }
    }

    mutable.method = 'kanban.task.patch'
    mutable.params.session_id = 'foreign-session'
    mutable.readback.kind = 'kanban-task'
    mutable.readback.expectedEffect.targetId = 'foreign-session'
    mutable.identity.sessionId = 'foreign-session'
    releaseSend()

    const receipt = await execution
    const sentPlan = selected.send.mock.calls[0]?.[0]

    expect(receipt.verification).toBe('verified')
    expect(sentPlan).toEqual(original)
    expect(sentPlan).not.toBe(callerPlan)
    expect(Object.isFrozen(sentPlan)).toBe(true)
    expect(Object.isFrozen(sentPlan?.identity)).toBe(true)
    expect(Object.isFrozen(sentPlan?.params)).toBe(true)
    expect(Object.isFrozen(sentPlan?.readback.expectedEffect)).toBe(true)
    expect(selected.readback).toHaveBeenCalledWith(sentPlan)
    expect(receipt.identity).toEqual(original.identity)
  })

  it('snapshots a hostile proxy once deterministically before canonical execution', async () => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const original = planCommand(intent('interrupt-session', identity), planningSnapshot)
    const reads = new Map<PropertyKey, number>()

    const hostile = new Proxy(structuredClone(original) as CommandPlan, {
      get(target, property, receiver) {
        const count = (reads.get(property) ?? 0) + 1
        reads.set(property, count)

        if (count > 1) {
          throw new Error(`caller plan ${String(property)} was read more than once`)
        }

        return Reflect.get(target, property, receiver)
      }
    })

    const selected = bridge(readback(original))

    const receipt = await executeCommand(hostile, executors(selected, original.readback.kind), {
      confirmed: true,
      latestSnapshot: () => planningSnapshot
    })

    expect(receipt.verification).toBe('verified')
    expect([...reads.values()].every(count => count === 1)).toBe(true)
    expect(selected.send.mock.calls[0]?.[0]).toEqual(original)
    expect(selected.send.mock.calls[0]?.[0] === hostile).toBe(false)
  })

  it.each([
    ['consequence', (plan: CommandPlan) => ({ ...plan, consequence: 'Everything is already complete.' })],
    ['current state', (plan: CommandPlan) => ({ ...plan, context: { ...plan.context, currentState: 'done' } })],
    [
      'source',
      (plan: CommandPlan) => ({
        ...plan,
        context: { ...plan.context, source: { ...plan.context.source, source: 'connection-b/sessions' } }
      })
    ],
    [
      'authority',
      (plan: CommandPlan) => ({
        ...plan,
        context: { ...plan.context, source: { ...plan.context.source, authority: 'stale' as const } }
      })
    ],
    ['intent', (plan: CommandPlan) => ({ ...plan, intent: { ...plan.intent, kind: 'open-session' as const } })],
    ['digest', (plan: CommandPlan) => ({ ...plan, digest: 'forged' })]
  ])('rejects a spread copy with forged %s before send', async (_field, forge) => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('interrupt-session', identity), planningSnapshot)
    const forged = forge(plan) as CommandPlan
    const selected = bridge(readback(plan))

    const receipt = await executeCommand(forged, executors(selected, plan.readback.kind), {
      confirmed: true,
      latestSnapshot: () => planningSnapshot
    })

    expect(receipt.verification).toBe('rejected')
    expect(selected.send).not.toHaveBeenCalled()
  })
})

describe('LunarCityCommandBroker causal readback', () => {
  it.each(ALL_OPERATIONS)('requires a strictly newer canonical %s effect', async operation => {
    const valid = VALID_COMPATIBILITY.find(row => row.operation === operation)!
    const identity = IDENTITIES[valid.identityKind]
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent(operation, identity), planningSnapshot)

    const expectedEffect = EXPECTED_EFFECTS[operation]

    expect(plan.readback.expectedEffect).toEqual(expectedEffect)

    const cached = await run(
      plan,
      planningSnapshot,
      readback(plan, { effect: expectedEffect, observedAt: plan.plannedAt, revision: plan.plannedRevision })
    )

    expect(cached.receipt.verification).toBe('verification_required')

    const fresh = await run(plan, planningSnapshot, readback(plan, { effect: expectedEffect }))

    expect(fresh.receipt.verification).toBe('verified')
    expect(fresh.selected.send).toHaveBeenCalledTimes(1)
    expect(fresh.selected.readback).toHaveBeenCalledOnce()
  })

  it.each(ALL_OPERATIONS)('does not verify %s from a boolean outcome without its canonical effect', async operation => {
    const valid = VALID_COMPATIBILITY.find(row => row.operation === operation)!
    const identity = IDENTITIES[valid.identityKind]
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent(operation, identity), planningSnapshot)
    const result = await run(plan, planningSnapshot, readback(plan, { effect: undefined }))

    expect(result.receipt.verification).toBe('verification_required')
  })

  it('accepts a matching authoritative causal receipt when monotonic clocks are unavailable', async () => {
    const identity = IDENTITIES.kanban
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('reassign-task', identity), planningSnapshot)

    const result = await run(
      plan,
      planningSnapshot,
      readback(plan, {
        observedAt: plan.plannedAt,
        receipt: { authority: 'authoritative', planDigest: plan.digest },
        revision: plan.plannedRevision
      })
    )

    expect(result.receipt.verification).toBe('verified')
  })

  it('rejects the wrong operation-specific effect even when identity and clocks match', async () => {
    const identity = IDENTITIES.kanban
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('reassign-task', identity), planningSnapshot)
    const wrong: CommandEffect = { kind: 'task-reassigned', targetId: 'task-7', value: 'foreign-worker' }
    const result = await run(plan, planningSnapshot, readback(plan, { effect: wrong }))

    expect(result.receipt.verification).toBe('verification_required')
  })

  it('returns an authoritative causal rejection before evaluating an expected effect', async () => {
    const identity = IDENTITIES.kanban
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('change-task-state', identity), planningSnapshot)
    const result = await run(plan, planningSnapshot, readback(plan, { effect: undefined, outcome: 'rejected' }))

    expect(result.receipt.verification).toBe('rejected')
  })
})

describe('LunarCityCommandBroker failure boundaries', () => {
  it('does not send session guidance until an exact current authority read succeeds', async () => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('send-guidance', identity), planningSnapshot)
    const selected = bridge(null)
    selected.currentAuthority.mockResolvedValueOnce(null)

    const blocked = await executeCommand(plan, executors(selected, plan.readback.kind), {
      latestSnapshot: () => planningSnapshot
    })

    expect(blocked.verification).toBe('rejected')
    expect(blocked.error).toContain('Exact current authority')
    expect(selected.send).not.toHaveBeenCalled()

    selected.currentAuthority.mockResolvedValueOnce({
      authority: 'authoritative',
      identity: plan.identity,
      observedAt: plan.plannedAt,
      owner: plan.owner
    })
    selected.readback.mockResolvedValueOnce(readback(plan))

    const restored = await executeCommand(plan, executors(selected, plan.readback.kind), {
      latestSnapshot: () => planningSnapshot
    })

    expect(restored.verification).toBe('verified')
    expect(selected.send).toHaveBeenCalledOnce()
  })

  it('rejects a current-authority proof from a duplicate session id on another connection', async () => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('interrupt-session', identity), planningSnapshot)
    const foreignIdentity = { ...identity, connectionId: 'connection-b' }
    const selected = bridge(readback(plan))
    selected.currentAuthority.mockResolvedValueOnce({
      authority: 'authoritative',
      identity: foreignIdentity,
      observedAt: plan.plannedAt,
      owner: { connectionId: 'connection-b', profile: 'worker' }
    })

    const receipt = await executeCommand(plan, executors(selected, plan.readback.kind), {
      confirmed: true,
      latestSnapshot: () => planningSnapshot
    })

    expect(receipt.verification).toBe('rejected')
    expect(selected.send).not.toHaveBeenCalled()
  })

  it('sends exactly once through the typed executor and never uses a blind retry', async () => {
    const identity = IDENTITIES.subagent
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('interrupt-subagent', identity), planningSnapshot)
    const result = await run(plan, planningSnapshot, readback(plan))

    expect(result.selected.send).toHaveBeenCalledTimes(1)
    expect(result.selected.send).toHaveBeenCalledWith(plan)
    expect(result.receipt.identity).toEqual(identity)
  })

  it('refuses an unconfirmed disruptive plan without reading the latest target or sending', async () => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('interrupt-session', identity), planningSnapshot)
    const latestSnapshot = vi.fn(() => planningSnapshot)
    const selected = bridge(readback(plan))

    const receipt = await executeCommand(plan, executors(selected, plan.readback.kind), {
      confirmed: false,
      latestSnapshot
    })

    expect(receipt.verification).toBe('rejected')
    expect(latestSnapshot).not.toHaveBeenCalled()
    expect(selected.send).not.toHaveBeenCalled()
  })

  it.each([
    [new CommandRejectedError('backend refused'), 'rejected'],
    [new CommandTimeoutError('connection unavailable', false), 'timed_out'],
    [new CommandTimeoutError('ACK timed out', true), 'verification_required']
  ] as const)('classifies %s without retrying', async (error, verification) => {
    const identity = IDENTITIES.session
    const planningSnapshot = snapshot(identity)
    const plan = planCommand(intent('interrupt-session', identity), planningSnapshot)
    const selected = bridge(null)
    selected.send.mockRejectedValueOnce(error)

    const receipt = await executeCommand(plan, executors(selected, plan.readback.kind), {
      confirmed: true,
      latestSnapshot: () => planningSnapshot
    })

    expect(receipt.verification).toBe(verification)
    expect(selected.send).toHaveBeenCalledTimes(1)
    expect(selected.currentAuthority).toHaveBeenCalledOnce()
    expect(selected.readback).not.toHaveBeenCalled()
  })
})
