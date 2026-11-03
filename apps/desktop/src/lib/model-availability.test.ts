import type { ModelOptionProvider } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { availabilityRank, contextLabel } from './model-availability'

const provider = (patch: Partial<ModelOptionProvider> = {}) =>
  ({ slug: 'test', name: 'Test', models: [], authenticated: true, ...patch }) as ModelOptionProvider

describe('model capacity and availability', () => {
  it('keeps unknown capacity unknown and formats known capacity', () => {
    expect(contextLabel(undefined)).toBe('Context unknown')
    expect(contextLabel(0)).toBe('Context unknown')
    expect(contextLabel(1_000_000)).toBe('1M context')
    expect(contextLabel(200_000)).toBe('200K context')
  })
  it('sorts exhausted accounts after usable and unknown accounts', () => {
    expect(availabilityRank(provider({ limit: { scope: 'account' } }), 'm')).toBeGreaterThan(
      availabilityRank(provider(), 'm')
    )
  })
  it('keeps unavailable models below usable models', () => {
    expect(availabilityRank(provider({ unavailable_models: ['m'] }), 'm')).toBeGreaterThan(
      availabilityRank(provider(), 'other')
    )
  })
})
