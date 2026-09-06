import { describe, expect, it } from 'vitest'

import { createUserInputReplayGuard } from './user-input-replay'

describe('native user-input replay ownership', () => {
  it('accepts only the newest same-session generation', () => {
    const guard = createUserInputReplayGuard()
    const oldToken = guard.begin('s1', 'gateway-a')
    const newToken = guard.begin('s1', 'gateway-a')

    expect(guard.isCurrent(oldToken)).toBe(false)
    expect(guard.isCurrent(newToken)).toBe(true)
  })

  it('keeps independent session replays current at the same time', () => {
    const guard = createUserInputReplayGuard()
    const first = guard.begin('s1', 'gateway-a')
    const second = guard.begin('s2', 'gateway-a')

    expect(guard.isCurrent(first)).toBe(true)
    expect(guard.isCurrent(second)).toBe(true)
  })

  it('rejects a switched session and a live request invalidation', () => {
    const guard = createUserInputReplayGuard()
    const oldToken = guard.begin('s1', 'gateway-a')
    guard.begin('s2', 'gateway-a')
    expect(guard.isCurrent(oldToken)).toBe(true)
    guard.invalidate('s1')
    expect(guard.isCurrent(oldToken)).toBe(false)

    const current = guard.begin('s2', 'gateway-a')
    guard.invalidate('s2')
    expect(guard.isCurrent(current)).toBe(false)
  })

  it('rejects a response from an old gateway owner', () => {
    const guard = createUserInputReplayGuard()
    const oldToken = guard.begin('s1', 'gateway-a')
    const newToken = guard.begin('s1', 'gateway-b')

    expect(guard.isCurrent(oldToken)).toBe(false)
    expect(guard.isCurrent(newToken)).toBe(true)
  })
})
