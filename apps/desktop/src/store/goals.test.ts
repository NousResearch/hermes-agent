import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from './gateway'
import { $goalsBySession, applyGoalStatusText, clearSessionGoal, refreshSessionGoal } from './goals'
import { resetBackgroundPollingGuard } from './runtime-gone'

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

  it('clears immediately on /goal clear output', () => {
    applyGoalStatusText('s1', '⊙ Goal set (20-turn budget): ship another feature')
    applyGoalStatusText('s1', '⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going.')
    applyGoalStatusText('s1', '✓ Goal cleared.')

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

  it('does not resurrect a done goal on hydration', () => {
    // `/goal status` reports "✓ Goal done" forever after completion (the DB
    // keeps terminal state). Hydrating that on session open must not bring
    // the completed chip back — Bot Mode's single endless session would show
    // it for the rest of time.
    applyGoalStatusText('s1', '✓ Goal done (12/20 turns): ship the feature', { hydrate: true })

    expect($goalsBySession.get().s1).toBeUndefined()
  })

  it('hydration drops a lingering done chip already on screen', () => {
    vi.useFakeTimers()

    applyGoalStatusText('s1', '✓ Goal achieved: tests pass')
    applyGoalStatusText('s1', '✓ Goal done (12/20 turns): tests pass', { hydrate: true })

    expect($goalsBySession.get().s1).toBeUndefined()
  })

  it('hydration still surfaces non-terminal goals', () => {
    applyGoalStatusText('s1', '⊙ Goal (active, 3/20 turns): ship the feature', { hydrate: true })

    expect($goalsBySession.get().s1).toMatchObject({ status: 'active', title: 'ship the feature' })

    applyGoalStatusText('s2', '⏸ Goal (paused, 20/20 turns): other work', { hydrate: true })

    expect($goalsBySession.get().s2).toMatchObject({ status: 'paused', title: 'other work' })
  })
})

describe('refreshSessionGoal dead-session guard', () => {
  afterEach(() => {
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
  })

  it('stops re-asking a runtime the gateway no longer holds', async () => {
    const request = vi.fn(async () => {
      throw new Error('session not found')
    })

    $gateway.set({ request } as never)

    await refreshSessionGoal('dead-1')
    await refreshSessionGoal('dead-1')
    await refreshSessionGoal('dead-1')

    expect(request).toHaveBeenCalledTimes(1)
  })

  it('does not latch on a transient failure', async () => {
    const request = vi.fn(async () => {
      throw new Error('request timed out after 30s: slash.exec')
    })

    $gateway.set({ request } as never)

    await refreshSessionGoal('s1')
    await refreshSessionGoal('s1')

    expect(request).toHaveBeenCalledTimes(2)
  })
})
