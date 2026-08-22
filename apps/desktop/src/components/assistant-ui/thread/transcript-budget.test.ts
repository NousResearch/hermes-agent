import { describe, expect, it } from 'vitest'

import { renderBudgetToRevealGroup, shouldClampTranscriptBudget } from './transcript-budget'

describe('shouldClampTranscriptBudget', () => {
  it('never snaps a visible pane back after Show earlier / timeline reveal', () => {
    expect(shouldClampTranscriptBudget(false, 1200, 600)).toBe(false)
    expect(shouldClampTranscriptBudget(false, 40, 600)).toBe(false)
  })

  it('snaps only a hot-hidden pane that outgrew the retention budget', () => {
    expect(shouldClampTranscriptBudget(true, 1200, 40)).toBe(true)
    expect(shouldClampTranscriptBudget(true, 40, 40)).toBe(false)
  })
})

describe('renderBudgetToRevealGroup', () => {
  it('sums newest-first weight through the target group', () => {
    const groups = [
      { id: 'u1', weight: 10 },
      { id: 'u2', weight: 20 },
      { id: 'u3', weight: 5 }
    ]

    expect(renderBudgetToRevealGroup(groups, 'u3')).toBe(5)
    expect(renderBudgetToRevealGroup(groups, 'u2')).toBe(25)
    expect(renderBudgetToRevealGroup(groups, 'u1')).toBe(35)
  })

  it('returns null when the prompt is not in the materialized groups', () => {
    expect(renderBudgetToRevealGroup([{ id: 'u1', weight: 10 }], 'missing')).toBeNull()
  })
})
