// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { type CommandPlan, type CommandPlanningSnapshot, type CommandTargetState, planCommand } from '../command-broker'
import { entityKey } from '../identity'
import type { EntityIdentity, LunarCitySnapshot, LunarEntity, SourceHealth } from '../model'

import { CommandConfirmation } from './command-confirmation'

const IDENTITY: EntityIdentity = {
  board: 'primary',
  connectionId: 'connection-a',
  kind: 'kanban',
  profile: 'worker',
  runId: 'run-7',
  taskId: 'task-7',
  workerId: 'worker-7'
}

function planningSnapshot(state = 'running', time = 2_000, revision = 3): CommandPlanningSnapshot {
  const source: SourceHealth = {
    authority: 'authoritative',
    observedAt: time,
    source: 'connection-a/kanban/primary'
  }

  const entity: LunarEntity = {
    animation: 'work',
    authority: 'authoritative',
    destination: 'project',
    identity: IDENTITY,
    key: entityKey(IDENTITY),
    observedAt: time,
    projectId: 'project-1'
  }

  const target: CommandTargetState = {
    availableOperations: ['reassign-task'],
    canonicalProjectId: 'project-1',
    entity,
    observedState: {
      animation: entity.animation,
      authority: entity.authority,
      destination: entity.destination,
      observedAt: time,
      source: source.source,
      value: state
    },
    ownerCandidates: [{ connectionId: 'connection-a', profile: 'worker' }],
    readbackCapabilities: ['kanban-task'],
    repositoryId: 'repo/hermes',
    source,
    sourceOwner: { connectionId: 'connection-a', profile: 'worker' }
  }

  const city: LunarCitySnapshot = {
    entities: new Map([[entity.key, entity]]),
    observedAt: time,
    revision,
    sources: [source]
  }

  return { city, targets: new Map([[entity.key, target]]) }
}

function setup() {
  const latest = planningSnapshot()
  const plan = planCommand({ assignee: 'reviewer', entityKey: entityKey(IDENTITY), kind: 'reassign-task' }, latest)

  return { getLatestSnapshot: () => latest, plan }
}

describe('CommandConfirmation', () => {
  it('renders a real accessible dialog with the complete exact identity, state, operation, and canonical consequence', () => {
    const { getLatestSnapshot, plan } = setup()

    render(
      <CommandConfirmation
        getLatestSnapshot={getLatestSnapshot}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
        open
        plan={plan}
      />
    )

    const dialog = screen.getByRole('dialog', { name: 'Confirm Lunar City command' })

    for (const value of [
      'connection-a',
      'worker',
      'project-1',
      'repo/hermes',
      'primary',
      'task-7',
      'run-7',
      'worker-7',
      'running',
      'Reassign task',
      'Reassign the exact task to reviewer'
    ]) {
      expect(dialog.textContent).toContain(value)
    }
  })

  it('focuses Cancel first, traps focus, and maps Escape to the safe cancel path', async () => {
    const { getLatestSnapshot, plan } = setup()
    const onCancel = vi.fn()
    const outside = window.document.createElement('button')
    outside.textContent = 'Outside'
    window.document.body.append(outside)

    render(
      <CommandConfirmation
        getLatestSnapshot={getLatestSnapshot}
        onCancel={onCancel}
        onConfirm={vi.fn()}
        open
        plan={plan}
      />
    )

    const dialog = screen.getByRole('dialog', { name: 'Confirm Lunar City command' })
    const cancel = screen.getByRole('button', { name: 'Cancel command' })

    await waitFor(() => expect(window.document.activeElement).toBe(cancel))
    outside.focus()
    await waitFor(() => expect(dialog.contains(window.document.activeElement)).toBe(true))

    fireEvent.keyDown(window.document, { key: 'Escape' })
    await waitFor(() => expect(onCancel).toHaveBeenCalledTimes(1))
    outside.remove()
  })

  it('revalidates the latest target on render and again at the confirmation click', () => {
    const initial = planningSnapshot()
    let latest = initial
    const plan = planCommand({ assignee: 'reviewer', entityKey: entityKey(IDENTITY), kind: 'reassign-task' }, initial)
    const onConfirm = vi.fn()

    const view = render(
      <CommandConfirmation getLatestSnapshot={() => latest} onCancel={vi.fn()} onConfirm={onConfirm} open plan={plan} />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Confirm Reassign task' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)

    latest = planningSnapshot('done', 2_100, 4)
    view.rerender(
      <CommandConfirmation getLatestSnapshot={() => latest} onCancel={vi.fn()} onConfirm={onConfirm} open plan={plan} />
    )

    expect((screen.getByRole('button', { name: 'Confirm command' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('target-changed-since-plan')
    fireEvent.click(screen.getByRole('button', { name: 'Confirm command' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['consequence', (plan: CommandPlan) => ({ ...plan, consequence: 'No consequence.' })],
    ['state', (plan: CommandPlan) => ({ ...plan, context: { ...plan.context, currentState: 'done' } })],
    [
      'source',
      (plan: CommandPlan) => ({
        ...plan,
        context: { ...plan.context, source: { ...plan.context.source, source: 'foreign' } }
      })
    ],
    [
      'authority',
      (plan: CommandPlan) => ({
        ...plan,
        context: { ...plan.context, source: { ...plan.context.source, authority: 'stale' as const } }
      })
    ]
  ])('disables a forged %s even when disabledReason is empty', (_label, forge) => {
    const { getLatestSnapshot, plan } = setup()

    render(
      <CommandConfirmation
        disabledReason="   "
        getLatestSnapshot={getLatestSnapshot}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
        open
        plan={forge(plan) as CommandPlan}
      />
    )

    expect((screen.getByRole('button', { name: 'Confirm command' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('Invalid command plan')
    expect(screen.queryByText('No consequence.')).toBeNull()
    expect(screen.queryByText('done')).toBeNull()
    expect(screen.queryByText('foreign')).toBeNull()
  })
})
