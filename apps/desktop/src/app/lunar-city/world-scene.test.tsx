import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { WorldProjection } from '@/store/lunar-city'

import { WorldScene } from './world-scene'

function projection(): WorldProjection {
  return {
    conditions: [
      {
        active: true,
        facts: {},
        id: 'condition-task-7',
        kind: 'task.blocked',
        scope: 'task',
        severity: 'warning',
        source: 'kanban',
        sourceRef: { taskId: 'task-7' },
        title: 'Fix authentication'
      }
    ],
    recentEvents: [
      {
        actionKinds: ['inspect'],
        facts: {},
        id: 'pull_request:pr-9',
        kind: 'pr.merged_stable',
        occurredAt: 1,
        receivedAt: 1,
        scope: 'city',
        severity: 'success',
        source: 'pull_request',
        sourceRef: { prId: 'pr-9' },
        title: 'Stable merge',
        transition: true
      }
    ],
    sourceError: null,
    stale: false,
    transitions: []
  }
}

describe('WorldScene', () => {
  it('renders active conditions and recent source events as selectable scenes', () => {
    render(<WorldScene projection={projection()} />)

    expect(screen.getByText('Fix authentication')).toBeTruthy()
    expect(screen.getByRole('img', { name: /Blender baseline/i })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Download 3D scene' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Download class roster' })).toBeTruthy()
    expect(screen.getByText('pr.merged_stable')).toBeTruthy()
    expect(screen.getByTestId('world-scene-celebration.citywide')).toBeTruthy()
    expect(screen.getByTestId('world-npc-condition-task-7-panicking')).toBeTruthy()
    expect(screen.getByTestId('world-npc-condition-task-7-repairing')).toBeTruthy()
    expect(screen.getByText(/celebrating ·/i)).toBeTruthy()
  })

  it('shows stale source status and recap transitions', () => {
    const value = projection()
    value.stale = true
    value.sourceError = 'Gateway unavailable'
    value.transitions = [value.recentEvents[0]]

    render(<WorldScene projection={value} />)

    expect(screen.getByText('Stale source')).toBeTruthy()
    expect(screen.getByText('Gateway unavailable')).toBeTruthy()
    expect(screen.getByText(/while you were away/i)).toBeTruthy()
  })
})
