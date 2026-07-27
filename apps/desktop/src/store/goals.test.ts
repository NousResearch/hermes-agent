import { afterEach, describe, expect, it, vi } from 'vitest'

import { $goalsBySession, applyGoalStatusText, clearSessionGoal } from './goals'

describe('goal store', () => {
  afterEach(() => {
    vi.useRealTimers()
    $goalsBySession.set({})
  })

  it('stores active goals from /goal output', () => {
    applyGoalStatusText('s1', '⊙ Goal set (20-turn budget): ship the feature')

    expect($goalsBySession.get().s1).toMatchObject({
      status: 'active',
      title: 'ship the feature'
    })
  })

  it('keeps the current title for continuation and pause messages', () => {
    applyGoalStatusText('s1', '⊙ Goal set (20-turn budget): ship the feature')
    applyGoalStatusText('s1', '↻ Continuing toward goal (1/20): next step is tests')

    expect($goalsBySession.get().s1).toMatchObject({
      detail: 'Continuing toward goal (1/20): next step is tests',
      status: 'active',
      title: 'ship the feature'
    })

    applyGoalStatusText('s1', '⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going.')

    expect($goalsBySession.get().s1).toMatchObject({
      status: 'paused',
      title: 'ship the feature'
    })
  })

  it('lingers done goals before clearing them', () => {
    vi.useFakeTimers()

    applyGoalStatusText('s1', '⊙ Goal set (20-turn budget): ship the feature')
    applyGoalStatusText('s1', '✓ Goal achieved: tests pass')

    expect($goalsBySession.get().s1).toMatchObject({ status: 'done' })

    vi.advanceTimersByTime(7_999)
    expect($goalsBySession.get().s1).toBeTruthy()

    vi.advanceTimersByTime(1)
    expect($goalsBySession.get().s1).toBeUndefined()
  })

  it('clears on no-goal output', () => {
    applyGoalStatusText('s1', '⊙ Goal set (20-turn budget): ship another feature')
    applyGoalStatusText('s1', 'No active goal. Set one with /goal <text>.')

    expect($goalsBySession.get().s1).toBeUndefined()
  })

  it('cancels pending done clears when replacing a goal', () => {
    vi.useFakeTimers()

    applyGoalStatusText('s1', '⊙ Goal set: first')
    applyGoalStatusText('s1', '✓ Goal achieved: first done')
    applyGoalStatusText('s1', '⊙ Goal set: second')

    vi.advanceTimersByTime(8_000)

    expect($goalsBySession.get().s1).toMatchObject({ status: 'active', title: 'second' })

    clearSessionGoal('s1')
  })
})
