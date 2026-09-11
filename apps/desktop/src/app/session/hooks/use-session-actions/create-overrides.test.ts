import { describe, expect, it } from 'vitest'

import { sessionCreateOverrideParams } from './create-overrides'

describe('session create overrides', () => {
  it('carries reasoning and title without choosing a model or provider', () => {
    const params = sessionCreateOverrideParams({ reasoningEffort: 'minimal', title: 'x' })

    expect(params).toEqual({ reasoning_effort: 'minimal', title: 'x' })
    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
  })

  it('adds no params for absent overrides and empty seeds', () => {
    expect(sessionCreateOverrideParams(undefined, [])).toEqual({})
  })
})
