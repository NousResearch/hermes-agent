import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RolloutSummary } from './rollout-summary'

describe('managed rollout summary', () => {
  it('keeps archive, exclusion, and unresolved state truthful', () => {
    render(<RolloutSummary summary={{ phase: 'completed-with-exclusions', excluded: 1, unresolved: 1, archived: true, reason: 'operator stopped' }} />)
    expect(screen.getByText(/completed-with-exclusions/)).toBeTruthy(); expect(screen.getByText(/Unresolved fences: 1/)).toBeTruthy(); expect(screen.getByText(/archived/)).toBeTruthy()
  })
})
