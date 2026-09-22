// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CommandRejectedError, CommandTimeoutError } from './command-broker'
import type {
  CommandExecutor,
  CommandExecutors,
  CommandPlan,
  CommandReadback,
  CommandVerification
} from './command-broker'
import { buildLunarCityCommandSnapshot, LunarCityCommandRuntime } from './command-runtime'
import { entityKey } from './identity'
import type { EntityIdentity, LunarCitySnapshot, LunarEntity, SourceHealth } from './model'
import { $lunarCitySnapshot, applyLunarDelta, createLunarCitySnapshot } from './store'

const source = (connectionId: string, authority: SourceHealth['authority'] = 'authoritative'): SourceHealth => ({
  authority,
  observedAt: 100,
  source: `session:${connectionId}`
})

function sessionEntity(connectionId: string, authority: LunarEntity['authority'] = 'authoritative'): LunarEntity {
  const identity: EntityIdentity = { connectionId, kind: 'session', profile: 'default', sessionId: 'same-session' }

  return {
    animation: 'working',
    authority,
    destination: 'project',
    identity,
    key: entityKey(identity),
    observedAt: 100
  }
}

function publish(entities: readonly LunarEntity[], sources: readonly SourceHealth[], revision = 1): LunarCitySnapshot {
  $lunarCitySnapshot.set(createLunarCitySnapshot())

  return applyLunarDelta({ observedAt: 100, removals: [], revision, sources, upserts: entities })
}

function verifiedReadback(plan: CommandPlan, verification: CommandVerification = 'verified'): CommandReadback | null {
  if (verification === 'timed_out') {
    return null
  }

  return {
    authority: 'authoritative',
    effect: verification === 'verified' ? plan.readback.expectedEffect : undefined,
    identity: plan.identity,
    observedAt: plan.plannedAt + 1,
    operation: plan.operation,
    outcome: verification === 'rejected' ? 'rejected' : 'verified',
    owner: plan.owner,
    receipt: { authority: 'authoritative', planDigest: plan.digest },
    revision: plan.plannedRevision + 1
  }
}

function executors(send: CommandExecutor['send'] = vi.fn(async () => ({ accepted: true }))): CommandExecutors {
  const executor: CommandExecutor = {
    currentAuthority: vi.fn(async plan => ({
      authority: 'authoritative' as const,
      identity: plan.identity,
      observedAt: plan.plannedAt,
      owner: plan.owner
    })),
    readback: vi.fn(async plan => verifiedReadback(plan)),
    send
  }

  return { kanbanRun: executor, kanbanTask: executor, session: executor, subagent: executor }
}

afterEach(() => {
  cleanup()
  $lunarCitySnapshot.set(createLunarCitySnapshot())
  vi.clearAllMocks()
})

describe('LunarCityCommandRuntime', () => {
  it('routes duplicate session ids to the exact selected connection without ambient lookup', async () => {
    const left = sessionEntity('connection-a')
    const right = sessionEntity('connection-b')
    publish([left, right], [source('connection-a'), source('connection-b')])
    const send = vi.fn(async (_plan: CommandPlan) => ({ accepted: true }))

    render(<LunarCityCommandRuntime executors={executors(send)} selectedEntityKey={right.key} />)

    fireEvent.change(screen.getByLabelText('Guidance for selected entity'), { target: { value: 'Check the receipt.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send guidance' }))

    await waitFor(() => expect(screen.getByRole('status').textContent).toContain('Verified'))
    const plan = send.mock.calls[0]?.[0]
    expect(plan.owner).toEqual({ connectionId: 'connection-b', profile: 'default' })
    expect(plan.identity).toEqual(right.identity)
    expect(plan.params).toEqual({ session_id: 'same-session', text: 'Check the receipt.' })
  })

  it('keeps compatible writes visible but disabled with a truthful stale reason', () => {
    const entity = sessionEntity('connection-a', 'stale')
    publish([entity], [source('connection-a', 'stale')])

    render(<LunarCityCommandRuntime executors={executors()} selectedEntityKey={entity.key} />)

    const guidance = screen.getByRole('button', { name: 'Send guidance' }) as HTMLButtonElement
    const interrupt = screen.getByRole('button', { name: 'Interrupt session' }) as HTMLButtonElement
    expect(guidance.disabled).toBe(true)
    expect(interrupt.disabled).toBe(true)
    expect(screen.getByText(/refresh exact authoritative state/i)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Reclaim task' })).toBeNull()
  })

  it('does not advertise unsupported session evidence inspection', () => {
    const entity = sessionEntity('connection-a')
    const city = publish([entity], [source('connection-a')])

    render(<LunarCityCommandRuntime executors={executors()} selectedEntityKey={entity.key} />)

    expect(buildLunarCityCommandSnapshot(city).targets.get(entity.key)?.availableOperations).not.toContain(
      'inspect-evidence'
    )
    expect(screen.queryByRole('button', { name: /inspect evidence/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /inspect diagnostics/i })).toBeNull()
  })

  it('shows only Kanban mutations compatible with the exact current authoritative state', () => {
    const identity = {
      board: 'main',
      connectionId: 'connection-a',
      kind: 'kanban',
      profile: 'default',
      runId: '7',
      taskId: 'T-7'
    } as const

    const entity: LunarEntity = {
      animation: 'done',
      authority: 'authoritative',
      destination: 'project',
      identity,
      key: entityKey(identity),
      observedAt: 100,
      sourceState: 'done'
    }

    publish([entity], [{ authority: 'authoritative', observedAt: 100, source: 'kanban:connection-a:default' }])

    render(<LunarCityCommandRuntime executors={executors()} selectedEntityKey={entity.key} />)

    expect(screen.getByRole('button', { name: 'Move task to ready' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Reclaim task' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Terminate run' })).toBeNull()
  })

  it('uses canonical Kanban source state instead of lossy animation to advertise commands', () => {
    const identity = {
      board: 'main',
      connectionId: 'connection-a',
      kind: 'kanban',
      profile: 'default',
      runId: '7',
      taskId: 'T-7'
    } as const

    const entity: LunarEntity = {
      animation: 'work',
      authority: 'authoritative',
      destination: 'project',
      identity,
      key: entityKey(identity),
      observedAt: 100,
      sourceState: 'review'
    }

    publish([entity], [{ authority: 'authoritative', observedAt: 100, source: 'kanban:connection-a:default' }])
    render(<LunarCityCommandRuntime executors={executors()} selectedEntityKey={entity.key} />)

    expect(screen.getByRole('button', { name: 'Move task to ready' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Reclaim task' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Terminate run' })).toBeNull()
  })

  it('does not advertise a no-op move when the exact Kanban state is already ready', () => {
    const identity = {
      board: 'main',
      connectionId: 'connection-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-ready'
    } as const

    const entity: LunarEntity = {
      animation: 'idle',
      authority: 'authoritative',
      destination: 'project',
      identity,
      key: entityKey(identity),
      observedAt: 100,
      sourceState: 'ready'
    }

    publish([entity], [{ authority: 'authoritative', observedAt: 100, source: 'kanban:connection-a:default' }])
    render(<LunarCityCommandRuntime executors={executors()} selectedEntityKey={entity.key} />)

    expect(screen.queryByRole('button', { name: 'Move task to ready' })).toBeNull()
  })

  it('renders the exact authoritative evidence response instead of a manufactured receipt', async () => {
    const identity = {
      board: 'main',
      connectionId: 'connection-a',
      kind: 'kanban',
      profile: 'default',
      taskId: 'T-8'
    } as const

    const entity: LunarEntity = {
      animation: 'idle',
      authority: 'authoritative',
      destination: 'project',
      identity,
      key: entityKey(identity),
      observedAt: 100,
      sourceState: 'blocked'
    }

    const response = { task: { diagnostics: ['exact-source'], id: 'T-8', status: 'blocked' } }

    const executor: CommandExecutor = {
      readback: vi.fn(async () => null),
      send: vi.fn(async () => response)
    }

    publish([entity], [{ authority: 'authoritative', observedAt: 100, source: 'kanban:connection-a:default' }])
    render(
      <LunarCityCommandRuntime
        executors={{ kanbanRun: executor, kanbanTask: executor, session: executor, subagent: executor }}
        selectedEntityKey={entity.key}
      />
    )
    fireEvent.click(screen.getByRole('button', { name: 'Inspect evidence' }))

    await waitFor(() => expect(screen.getByLabelText('Exact source evidence').textContent).toContain('exact-source'))
    expect(screen.getByRole('status').textContent).toContain('Verification required')
  })

  it('invalidates a staged disruptive confirmation when the source revision changes', async () => {
    const entity = sessionEntity('connection-a')
    publish([entity], [source('connection-a')])

    render(<LunarCityCommandRuntime executors={executors()} selectedEntityKey={entity.key} />)
    fireEvent.click(screen.getByRole('button', { name: 'Interrupt session' }))
    expect(screen.getByRole('dialog', { name: 'Confirm Lunar City command' })).toBeTruthy()

    await act(async () => {
      applyLunarDelta({
        observedAt: 101,
        removals: [],
        revision: 2,
        sources: [{ ...source('connection-a'), observedAt: 101 }],
        upserts: [{ ...entity, observedAt: 101 }]
      })
    })

    expect(screen.queryByRole('dialog', { name: 'Confirm Lunar City command' })).toBeNull()
    expect(screen.getByText(/selection or source changed/i)).toBeTruthy()
  })

  it.each([
    ['Rejected', new CommandRejectedError('owner rejected')],
    ['Timed out', new CommandTimeoutError('deadline elapsed', false)],
    ['Verification required', new Error('connection closed after send')]
  ])('shows the causal %s receipt state without optimistic world mutation', async (label, error) => {
    const entity = sessionEntity('connection-a')
    const before = publish([entity], [source('connection-a')])

    const executor: CommandExecutor = {
      currentAuthority: vi.fn(async plan => ({
        authority: 'authoritative' as const,
        identity: plan.identity,
        observedAt: plan.plannedAt,
        owner: plan.owner
      })),
      readback: vi.fn(async plan => verifiedReadback(plan)),
      send: vi.fn(async () => {
        throw error
      })
    }

    render(
      <LunarCityCommandRuntime
        executors={{ kanbanRun: executor, kanbanTask: executor, session: executor, subagent: executor }}
        selectedEntityKey={entity.key}
      />
    )
    fireEvent.change(screen.getByLabelText('Guidance for selected entity'), { target: { value: 'Check.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send guidance' }))

    await waitFor(() => expect(screen.getByRole('status').textContent).toContain(label))
    expect($lunarCitySnapshot.get()).toBe(before)
  })

  it('ignores a late command completion after route teardown', async () => {
    const entity = sessionEntity('connection-a')
    publish([entity], [source('connection-a')])
    let resolveSend!: (value: unknown) => void
    const send = vi.fn((_plan: CommandPlan) => new Promise(resolve => (resolveSend = resolve)))
    const view = render(<LunarCityCommandRuntime executors={executors(send)} selectedEntityKey={entity.key} />)

    fireEvent.change(screen.getByLabelText('Guidance for selected entity'), { target: { value: 'Wait.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send guidance' }))
    await waitFor(() => expect(send).toHaveBeenCalledOnce())
    view.unmount()

    await act(async () => resolveSend({ accepted: true }))
    expect(globalThis.document.body.textContent).not.toContain('Verified')
  })
})
