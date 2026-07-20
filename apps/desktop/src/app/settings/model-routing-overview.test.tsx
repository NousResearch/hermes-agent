import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { AuxiliaryModelsResponse } from '@/types/hermes'

import { AUX_MODEL_TASK_KEYS, buildModelRoutingLanes, ModelRoutingOverview } from './model-routing-overview'

afterEach(cleanup)

const auxiliary: AuxiliaryModelsResponse = {
  main: { provider: 'nous', model: 'hermes-4' },
  tasks: [
    { base_url: '', model: '', provider: 'auto', task: 'vision' },
    { base_url: '', model: 'claude-sonnet-4.6', provider: 'anthropic', task: 'compression' },
    { base_url: '', model: '', provider: 'openrouter', task: 'approval' }
  ]
}

describe('buildModelRoutingLanes', () => {
  it('resolves automatic tasks to the applied main route and preserves dedicated overrides', () => {
    const lanes = buildModelRoutingLanes({ provider: 'nous', model: 'hermes-4' }, auxiliary)

    expect(lanes[0]).toEqual({
      inherited: false,
      key: 'main',
      model: 'hermes-4',
      provider: 'nous'
    })
    expect(lanes.find(lane => lane.key === 'vision')).toEqual({
      inherited: true,
      key: 'vision',
      model: 'hermes-4',
      provider: 'nous'
    })
    expect(lanes.find(lane => lane.key === 'compression')).toEqual({
      inherited: false,
      key: 'compression',
      model: 'claude-sonnet-4.6',
      provider: 'anthropic'
    })
    expect(lanes.find(lane => lane.key === 'approval')).toEqual({
      inherited: false,
      key: 'approval',
      model: '',
      provider: 'openrouter'
    })
  })

  it('falls back to the backend main route while the applied main state is unavailable', () => {
    const lanes = buildModelRoutingLanes(null, auxiliary)

    expect(lanes[0]).toMatchObject({ model: 'hermes-4', provider: 'nous' })
    expect(lanes.find(lane => lane.key === 'vision')).toMatchObject({
      inherited: true,
      model: 'hermes-4',
      provider: 'nous'
    })
  })
})

describe('ModelRoutingOverview', () => {
  it('renders every lane with its resolved provider, model and routing mode', () => {
    render(<ModelRoutingOverview auxiliary={auxiliary} mainModel={{ provider: 'nous', model: 'hermes-4' }} />)

    const overview = screen.getByRole('region', { name: 'Model routing' })

    expect(overview.querySelectorAll('[data-routing-lane]')).toHaveLength(AUX_MODEL_TASK_KEYS.length + 1)
    expect(screen.getByText('Main chat')).toBeTruthy()
    expect(screen.getAllByText('Kanban decomposer').length).toBeGreaterThan(0)
    expect(screen.getAllByText('nous').length).toBeGreaterThan(1)
    expect(screen.getByText('anthropic')).toBeTruthy()
    expect(screen.getByText('claude-sonnet-4.6')).toBeTruthy()
    expect(screen.getByText('(provider default)')).toBeTruthy()
    expect(screen.getAllByText('Uses main').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Dedicated').length).toBe(2)
  })
})
