import { describe, expect, it } from 'vitest'

import { getSkillHubQueryState } from './hub-query-state'

describe('getSkillHubQueryState', () => {
  it('shows featured skills for an empty input', () => {
    expect(getSkillHubQueryState('  ', 'previous')).toEqual({
      pending: false,
      showLanding: true,
      showResults: false
    })
  })

  it('shows results only after the debounced term matches the input', () => {
    expect(getSkillHubQueryState('matrix', 'matrix')).toEqual({
      pending: false,
      showLanding: false,
      showResults: true
    })
  })

  it('hides stale results while a changed query is still debouncing', () => {
    expect(getSkillHubQueryState('telegram', 'matrix')).toEqual({
      pending: true,
      showLanding: false,
      showResults: false
    })
  })
})
